/*!
Tree-based Gaussian factor graph solver.

The two-pass (build–solve) procedure corresponds to an upward-filtering
downward-deciding (UFDD) in a Gaussian factor graph.

### References
- **Notations & Gaussian message-passing rules:**
  H.-A. Loeliger et al., *"On sparsity by NUV-EM, Gaussian message passing,
  and Kalman smoothing,"* ITA, 2016.
- **Filtering–deciding algorithmic framework:**
  Y. P. Li and H.-A. Loeliger, *"Backward filtering forward deciding in linear
  non-Gaussian state space models,"* AISTATS, 2024.
*/

use anyhow::{Context, Result};
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use dyn_stack::MemBuffer;
use dyn_stack::MemStack;

use faer::linalg::matmul::matmul;
// use faer::linalg::solvers::Svd;
use faer::diag::Diag;
use faer::linalg::svd;
use faer::prelude::*;
use faer::{Accum, ColMut, ColRef, Par, Spec};
use faer_traits::RealField;

use super::numeric::{ConcurrentStateBuffer, NumericTree};
use super::symbolic::SymbolicFactor;
use crate::factor::gaussian::GaussianFactor;
use crate::variable::gaussian::VarBlock;
use crate::variable::{VarId, dictionary::VarDict};

/// The main variable X and separator variable S are associated with the matrices A and B, respectively.
/// For convenience, the virtual variables Y := A X + Z and Z := B S are introduced.
/// The node contains several workspaces to avoid repeated memory reallocation.
pub struct GaussianNode<T> {
    pub main: VarId,
    pub separator: Vec<VarId>,

    pub children: Vec<GaussianNode<T>>, // Gaussian node children in the tree

    pub dim_x: usize,
    pub dim_y: usize,

    pub blk_x: VarBlock<T>,
    pub blk_s: Vec<VarBlock<T>>,

    pub xiby: Col<T>, // backward weighted mean vector for y -- shape (dim_y)
    pub wby: Mat<T>,  // backward precision matrix for y -- shape (dim_y, dim_y)

    // ---- Workspaces ----
    wby_a: Mat<T>, // (dim_y, dim_x)
    xitx: Col<T>,  // (dim_x)
    wtx: Mat<T>,   // (dim_x, dim_x)
    wtx_s: Diag<T>,
    wtx_u: Mat<T>,
    wtx_v: Mat<T>,
    vtx: Mat<T>, // (dim_x, dim_x) pseudoinverse of wtx computed by singular value decomposition
    vtx_xitx: Col<T>, // (dim_x)
    vtx_at_wby: Mat<T>, // (dim_x, dim_y)
    b_s: Col<T>, // (dim_y)
    mem_buf: MemBuffer,
}

impl<T> GaussianNode<T>
where
    T: RealField + Copy + Send + Sync + Debug + Display,
{
    // pub fn get_memory_layout(&self, var: VarId) -> Option<(usize, usize)> {
    //     self.b_ctx
    //         .iter()
    //         .find(|(v, _, _)| *v == var)
    //         .map(|(_, offset, size)| (*offset, *size))
    // }

    /// Create a new [`GaussianNode`].
    /// Note: all workspaces are initialized with zeros.
    pub fn new(
        main: VarId,
        separator: Vec<VarId>,
        children: Vec<Self>,
        dim_x: usize,
        dim_y: usize,
        blk_x: VarBlock<T>,
        blk_s: Vec<VarBlock<T>>,
        // separator: Vec<VarId>,
        // dim_x: usize,
        // dim_y: usize,
        // dim_s: usize,
        // a: Mat<T>,
        // b: Vec<Mat<T>>,
        // b: Mat<T>,
        // b_ctx: Vec<(VarId, usize, usize)>,
        xiby: Col<T>,
        wby: Mat<T>,
    ) -> Self {
        // Allocate workspaces
        let wby_a = Mat::zeros(dim_y, dim_x);
        let xitx = Col::zeros(dim_x);
        let wtx = Mat::zeros(dim_x, dim_x);
        let wtx_s = Diag::zeros(dim_x);
        let wtx_u = Mat::zeros(dim_x, dim_x);
        let wtx_v = Mat::zeros(dim_x, dim_x);
        let vtx = Mat::zeros(dim_x, dim_x);
        let b_s = Col::zeros(dim_y);
        let vtx_xitx = Col::zeros(dim_x);
        let vtx_at_wby = Mat::zeros(dim_x, dim_y);

        let svd_memory = svd::svd_scratch::<T>(
            dim_x,
            dim_x,
            svd::ComputeSvdVectors::Full,
            svd::ComputeSvdVectors::Full,
            Par::Seq,
            Spec::default(),
        );
        let pseudo_inverse_memory =
            svd::pseudoinverse_from_svd_scratch::<T>(dim_x, dim_x, Par::Seq);
        let mem_buf = MemBuffer::new(svd_memory.or(pseudo_inverse_memory));

        let node = Self {
            main,
            separator,
            children,
            dim_x,
            dim_y,
            blk_x,
            blk_s,
            xiby,
            wby,
            wby_a,
            xitx,
            wtx,
            wtx_s,
            wtx_u,
            wtx_v,
            vtx,
            b_s,
            vtx_xitx,
            vtx_at_wby,
            mem_buf,
        };

        node
    }

    /// Create a [`GaussianFactor`] by eliminating the [`GaussianNode`]'s main variable.
    /// The backward messages are computed via the SVD decomposition of `wtx` which is the
    /// most stable method to solve linear systems wtx c = d.
    /// At the end of this method call, the following fields are loaded:
    /// `wby_a`, `xitx`, `wtx`, `wtx_s`, `wtx_u`, `wtx_v`, `vtx`,
    fn eliminate(&mut self) -> Result<GaussianFactor<T>> {
        // Load wby a
        matmul(
            self.wby_a.as_mut(),
            Accum::Replace,
            self.wby.as_ref(),
            self.blk_x.a.as_ref(),
            T::one(),
            Par::Seq,
        );

        // Load xitx = a.t xiby
        matmul(
            self.xitx.as_mat_mut(),
            Accum::Replace,
            self.blk_x.a.transpose(),
            self.xiby.as_mat(),
            T::one(),
            Par::Seq,
        );

        // Load wtx = a.t wby a
        matmul(
            self.wtx.as_mut(),
            Accum::Replace,
            self.blk_x.a.transpose(),
            self.wby_a.as_ref(),
            T::one(),
            Par::Seq,
        );

        // SVD-based pseudo inverse of wtx
        let mut stack = MemStack::new(&mut self.mem_buf);
        // Load wtx_s, wtx_u, and wtx_v
        svd::svd(
            self.wtx.as_ref(),
            self.wtx_s.as_mut(),
            Some(self.wtx_u.as_mut()),
            Some(self.wtx_v.as_mut()),
            Par::Seq,
            &mut stack,
            default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to compute SVD: {:?}", e))?;

        // Load vtx
        svd::pseudoinverse_from_svd(
            self.vtx.as_mut(),
            self.wtx_s.as_ref(),
            self.wtx_u.as_ref(),
            self.wtx_v.as_ref(),
            Par::Seq,
            &mut stack,
        );
        // svd::pseudoinverse_from_svd_with_tolerance(
        //     self.vtx.as_mut(),
        //     self.wtx_s.as_ref(),
        //     self.wtx_u.as_ref(),
        //     self.wtx_v.as_ref(),
        //     abs_tol,
        //     rel_tol,
        //     Par::Seq,
        //     &mut stack,
        // );

        // Compute xibz = xiby - wby a vtx xitx where vtx is the (pseudo) inverse of wtx
        matmul(
            self.vtx_xitx.as_mat_mut(),
            Accum::Replace,
            self.vtx.as_ref(),
            self.xitx.as_mat(),
            T::one(),
            Par::Seq,
        ); // vtx xitx
        let mut xibz = self.xiby.clone();
        matmul(
            xibz.as_mat_mut(),
            Accum::Add,
            self.wby_a.as_ref(),
            self.vtx_xitx.as_mat(),
            T::one().neg(),
            Par::Seq,
        ); // xibz = xiby - wby a vtx xitx

        // Compute wbz = wby - wby a vtx a.t wby where vtx is the (pseudo) inverse of wtx
        matmul(
            self.vtx_at_wby.as_mut(),
            Accum::Replace,
            self.vtx.as_ref(),
            self.wby_a.transpose(),
            T::one(),
            Par::Seq,
        ); // vtx a.t wby
        let mut wbz = self.wby.clone();
        matmul(
            wbz.as_mut(),
            Accum::Add,
            self.wby_a.as_ref(),
            self.vtx_at_wby.as_ref(),
            T::one().neg(),
            Par::Seq,
        ); // wbz = wby - wby a vtx a.t wby

        let msg = GaussianFactor {
            vars: self.separator.clone(),
            blk_x: self.blk_s.clone(),
            dim_y: self.dim_y,
            xiby: xibz,
            wby: wbz,
        };

        Ok(msg)
    }

    /// Decides the value of the node's variables using the current state of the tree.
    /// Warning:
    /// - `tmp_y` must be preloaded with b s!
    /// - A few fields must be up-to-date: `xitx`, `wtx_svd`, and `wby_a`.
    /// - `x` must be a mutable slice of the same length as the node's x dimension.
    pub fn decide(&mut self, x: &mut [T]) {
        let mut x_col = ColMut::from_slice_mut(x);
        x_col.copy_from(self.vtx_xitx.as_ref());
        matmul(
            x_col.as_mut(),
            Accum::Add,
            self.vtx_at_wby.as_ref(),
            self.b_s.as_ref(),
            T::one().neg(),
            Par::Seq,
        );

        // x = vtx ( xitx - a.t wby b s) = vtx xitx - vtx a.t wby b s
        // from eliminate: tmp_xy = vtx a.t wby and // tmp_x = vtx xitx
        //
        //         x_col.copy_from(&self.xitx);
        //
        //         llt::solve::solve_in_place(self.rtx.as_ref(),

        // let mut x_col = ColMut::from_slice_mut(x);
        // x_col.copy_from(&self.xitx);
        // matmul(
        //     x_col.rb_mut(),
        //     Accum::Add,
        //     self.wby_a.transpose(),
        //     &self.tmp_y.as_ref(),
        //     T::one().neg(),
        //     Par::Seq,
        // );
        // matmul(
        //     self.tmp_x.as_mat_mut(),
        //     Accum::Replace,
        //     self.vtx.as_ref(),
        //     self.xitx.as_mat(),
        //     T::one(),
        //     Par::Seq,
        // );
        // self.wtx_svd.solve_in_place(x_col)
    }

    /// Helper method to recursively format the tree with ASCII branches
    fn fmt_recursive(&self, f: &mut Formatter<'_>, prefix: &str, is_last: bool) -> FmtResult {
        let branch_marker = if is_last { "└── " } else { "├── " };
        writeln!(
            f,
            "{}{}[Var: {}, Separator: {:?}, Wby: {:?}, xiby: {:?}]",
            prefix,
            branch_marker,
            self.blk_x.id,
            self.blk_s.iter().map(|blk| blk.id).collect::<Vec<VarId>>(),
            self.wby,
            self.xiby
        )?;

        let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
        let num_children = self.children.len();

        for (i, child) in self.children.iter().enumerate() {
            child.fmt_recursive(f, &child_prefix, i == num_children - 1)?;
        }

        Ok(())
    }
}

/// A tree-structure Gaussian factor graph defined by its roots.
/// Nodes are defined recursively.
pub struct GaussianTree<T> {
    pub roots: Vec<GaussianNode<T>>,
}

impl<T> SymbolicFactor for GaussianFactor<T> {
    fn vars(&self) -> &[VarId] {
        &self.vars
    }
}

impl<T> NumericTree for GaussianTree<T>
where
    T: Send + Sync + Copy + RealField + Display + Debug,
{
    /// The concrete node type.
    type Node = GaussianNode<T>;
    /// The original raw numerical factor.
    type Factor = GaussianFactor<T>;
    /// The mathematical message passed upward during elimination.
    type Message = GaussianFactor<T>;
    /// The type of variable values
    type Value = T;
    /// User-provided context passed up during the build and the solve.
    type Context = VarDict;

    // ==========================================
    // REQUIRED METHODS (Math & Accessors)
    // ==========================================

    fn from_roots(roots: Vec<Self::Node>) -> Self {
        Self { roots }
    }

    fn roots(&self) -> &[Self::Node] {
        &self.roots
    }

    fn roots_mut(&mut self) -> &mut [Self::Node] {
        &mut self.roots
    }

    fn node_children(node: &Self::Node) -> &[Self::Node] {
        &node.children
    }

    fn node_children_mut(node: &mut Self::Node) -> &mut [Self::Node] {
        &mut node.children
    }

    fn build_node(
        main: VarId,
        separator: Vec<VarId>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self::Node>,
        ctx: &Self::Context,
    ) -> Result<Self::Node> {
        // Create the context for the separator variables
        // let mut b_ctx = Vec::with_capacity(separator.len());
        // let mut start = 0;
        // for &sep in &separator {
        //     let dim = ctx
        //         .get_size(sep)
        //         .with_context(|| format!("Missing size for variable with id: {}", sep))?;
        //     b_ctx.push((sep, start, dim));
        //     start += dim;
        // }
        // let dim_s = start; // the total dimension of the separator variables

        // Init observation states
        let dim_y = assigned_factors.iter().map(|f| f.dim_y).sum::<usize>()
            + incoming_messages.iter().map(|m| m.dim_y).sum::<usize>();
        let mut xiby = Col::zeros(dim_y);
        let mut wby = Mat::zeros(dim_y, dim_y);

        // Init main variable block
        let dim_x = ctx
            .get_size(main)
            .with_context(|| format!("Missing size for variable with id: {}", main))?;
        let mut blk_x = VarBlock {
            id: main,
            a: Mat::zeros(dim_y, dim_x),
        };

        // Init separator variables blocks
        let mut blk_s = Vec::with_capacity(separator.len());
        for &sep in &separator {
            let dim_u = ctx
                .get_size(sep)
                .with_context(|| format!("Missing size for variable with id: {}", sep))?;
            blk_s.push(VarBlock {
                id: sep,
                a: Mat::zeros(dim_y, dim_u),
            });
        }

        let mut offset_obs = 0;
        for factor in assigned_factors.into_iter().chain(incoming_messages.iter()) {
            let size_obs = factor.dim_y;

            // Populate xiby
            xiby.as_mut()
                .subrows_mut(offset_obs, size_obs)
                .copy_from(&factor.xiby);

            // Populate wby
            wby.submatrix_mut(offset_obs, offset_obs, size_obs, size_obs)
                .copy_from(&factor.wby);

            // Populate a
            if let Some(fblk) = factor.get_blk(main) {
                blk_x.a.subrows_mut(offset_obs, size_obs).copy_from(&fblk.a);
            }

            // Populate b
            for blk_u in blk_s.iter_mut() {
                if let Some(fblk) = factor.get_blk(blk_u.id) {
                    blk_u.a.subrows_mut(offset_obs, size_obs).copy_from(&fblk.a);
                }
            }

            offset_obs += size_obs;
        }

        let node = GaussianNode::new(
            main, separator, children, dim_x, dim_y, blk_x, blk_s, xiby, wby,
        );

        Ok(node)
    }

    fn build_out_message(node: &mut Self::Node, _ctx: &Self::Context) -> Result<Self::Message> {
        node.eliminate()
    }

    /// Decide the node value by reading separator values from the buffer.
    fn decide_node(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()> {
        // Gather separator values from the global buffer directly into the node's workspace (tmp_y)
        node.b_s.fill(T::zero());
        for blk in &node.blk_s {
            let (offset, size) = ctx
                .get_memory_layout(blk.id)
                .with_context(|| format!("Missing offset for variable with id: {}", blk.id))?;

            unsafe {
                let s = ColRef::from_slice(buffer.read_slice(offset, size));
                matmul(node.b_s.as_mut(), Accum::Add, &blk.a, s, T::one(), Par::Seq);
            }
        }

        // Obtain a direct mutable view into the global array and solve in-place
        let (offset, size) = ctx.get_memory_layout(node.main).with_context(|| {
            format!("Missing memory layout for variable with id: {}", node.main)
        })?;
        unsafe {
            let x_slice = buffer.get_mut_slice(offset, size);
            node.decide(x_slice);
        }

        Ok(())
    }
}

impl<T> Display for GaussianTree<T>
where
    T: Display + Debug + Send + Sync + Copy + RealField,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if self.roots.is_empty() {
            return write!(f, "Empty GaussianTree");
        }

        for (i, root) in self.roots.iter().enumerate() {
            // Print the root node
            writeln!(
                f,
                "Root [Var: {}, Separator: {:?}, Observation Dimension: {}]",
                root.main, root.separator, root.dim_y
            )?;

            // Print all children recursively
            let num_children = root.children.len();
            for (j, child) in root.children.iter().enumerate() {
                child.fmt_recursive(f, "", j == num_children - 1)?;
            }

            // Add a blank line between multiple roots
            if i < self.roots.len() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

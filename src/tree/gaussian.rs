/*!
Implementation of Gaussian numeric trees.

The two-pass (build–solve) procedure can be interpreted as
upward-filtering downward-deciding in a Gaussian factor graph.

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
use crate::factor::gaussian::{GaussianFactor, LinVar};
use crate::variable::dictionary::VarDict;

/// A node in a tree-structured Gaussian factor graph.
///
/// Encapsulates a main variable `X` and separator variables `S` coupled
/// via linear observations `Y := A X + B S`. The observation vector `Y`
/// is assigned a Gaussian prior defined by a weighted mean and a precision matrix.
///
/// The node owns its children in the tree.
/// Internal workspaces are preallocated to eliminate heap churn during inference.
pub struct GaussianNode<T> {
    /// The main variable index.
    pub main: usize,
    /// The separator variables indices.
    pub separator: Vec<usize>,
    /// The node children.
    pub children: Vec<GaussianNode<T>>,
    /// The dimension of the main variable.
    pub dim_main: usize,
    /// The dimension of the observations.
    pub dim_obs: usize,
    /// The linear contribution of the main variable into the observation.
    pub main_lin_var: LinVar<T>,
    /// The linear contribution of the separator variables into the observation.
    pub sep_lin_vars: Vec<LinVar<T>>,
    /// The weighted mean vector of the observation prior with shape `(dim_obs,)`.
    pub xib_obs: Col<T>, // (dim_obs)
    /// The precision matrix of the observation prior with shape `(dim_obs, dim_obs)`.
    pub wb_obs: Mat<T>, // (dim_obs, dim_obs)

    // ---- Workspaces ----
    wby_a: Mat<T>, // (dim_obs, dim_main)
    xitx: Col<T>,  // (dim_main)
    wtx: Mat<T>,   // (dim_main, dim_main)
    wtx_s: Diag<T>,
    wtx_u: Mat<T>,
    wtx_v: Mat<T>,
    vtx: Mat<T>, // (dim_main, dim_main) pseudoinverse of wtx computed by singular value decomposition
    vtx_xitx: Col<T>, // (dim_main)
    vtx_at_wby: Mat<T>, // (dim_main, dim_obs)
    b_s: Col<T>, // (dim_obs)
    mem_buf: MemBuffer,
}

impl<T> GaussianNode<T>
where
    T: RealField + Copy + Send + Sync + Debug + Display,
{
    /// Create a new [`GaussianNode`] with all workspace filled with zero.
    pub fn new(
        main: usize,
        separator: Vec<usize>,
        children: Vec<Self>,
        dim_main: usize,
        dim_obs: usize,
        main_lin_var: LinVar<T>,
        sep_lin_vars: Vec<LinVar<T>>,
        xib_obs: Col<T>,
        wb_obs: Mat<T>,
    ) -> Self {
        // Allocate workspaces
        let wby_a = Mat::zeros(dim_obs, dim_main);
        let xitx = Col::zeros(dim_main);
        let wtx = Mat::zeros(dim_main, dim_main);
        let wtx_s = Diag::zeros(dim_main);
        let wtx_u = Mat::zeros(dim_main, dim_main);
        let wtx_v = Mat::zeros(dim_main, dim_main);
        let vtx = Mat::zeros(dim_main, dim_main);
        let b_s = Col::zeros(dim_obs);
        let vtx_xitx = Col::zeros(dim_main);
        let vtx_at_wby = Mat::zeros(dim_main, dim_obs);

        let svd_memory = svd::svd_scratch::<T>(
            dim_main,
            dim_main,
            svd::ComputeSvdVectors::Full,
            svd::ComputeSvdVectors::Full,
            Par::Seq,
            Spec::default(),
        );
        let pseudo_inverse_memory =
            svd::pseudoinverse_from_svd_scratch::<T>(dim_main, dim_main, Par::Seq);
        let mem_buf = MemBuffer::new(svd_memory.or(pseudo_inverse_memory));

        let node = Self {
            main,
            separator,
            children,
            dim_main,
            dim_obs,
            main_lin_var,
            sep_lin_vars,
            xib_obs,
            wb_obs,
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

    /// Eliminates the node's main variable to produce an outgoing marginal [`GaussianFactor`].
    ///
    /// Updates internal workspaces during elimination, notably:
    /// - `wby_a`: intermediate factor workspace.
    /// - `xitx`, `wtx`: total weighted mean and precision of the eliminated variable.
    /// - `wtx_u`, `wtx_s`, `wtx_v`: SVD components of `wtx`.
    /// - `vtx`: Moore–Penrose pseudo-inverse of `wtx`, derived via SVD (implementation detail subject to change).
    ///
    /// # Errors
    ///
    /// Returns an error if numerical factorization (such as the SVD of `wtx`) fails.
    fn eliminate(&mut self) -> Result<GaussianFactor<T>> {
        // Load wby_a
        matmul(
            self.wby_a.as_mut(),
            Accum::Replace,
            self.wb_obs.as_ref(),
            self.main_lin_var.mat.as_ref(),
            T::one(),
            Par::Seq,
        );

        // Load xitx = a.t xiby
        matmul(
            self.xitx.as_mat_mut(),
            Accum::Replace,
            self.main_lin_var.mat.transpose(),
            self.xib_obs.as_mat(),
            T::one(),
            Par::Seq,
        );

        // Load wtx = a.t wby a
        matmul(
            self.wtx.as_mut(),
            Accum::Replace,
            self.main_lin_var.mat.transpose(),
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
        let mut xibz = self.xib_obs.clone();
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
        let mut wbz = self.wb_obs.clone();
        matmul(
            wbz.as_mut(),
            Accum::Add,
            self.wby_a.as_ref(),
            self.vtx_at_wby.as_ref(),
            T::one().neg(),
            Par::Seq,
        ); // wbz = wby - wby a vtx a.t wby

        let msg = GaussianFactor::new(self.sep_lin_vars.clone(), xibz, wbz);

        Ok(msg)
    }

    /// Decides the value of the main variable based on the values of its separator variables.
    /// Warning: before calling, make sure the (private) fields `vtx_xitx`, `vtx_at_wby`, and `b_s` are up-to-date.
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
            "{}{}[Var: {}, Separator: {:?}]",
            prefix, branch_marker, self.main, self.separator,
        )?;

        let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
        let num_children = self.children.len();

        for (i, child) in self.children.iter().enumerate() {
            child.fmt_recursive(f, &child_prefix, i == num_children - 1)?;
        }

        Ok(())
    }
}

/// An elimination tree (forest) for a Gaussian factor graph.
///
/// Encapsulates the numerical factor graph organized into one or more rooted
/// trees, where each subtree represents a recursive elimination clique.
pub struct GaussianTree<T> {
    /// Root nodes corresponding to the independent elimination trees in the forest.
    pub roots: Vec<GaussianNode<T>>,
}

impl<T> NumericTree for GaussianTree<T>
where
    T: Send + Sync + Copy + RealField + Display + Debug,
{
    /// Tree node type encapsulating the elimination clique and its workspaces.
    type Node = GaussianNode<T>;

    /// Input factor type representing local Gaussian observation constraints.
    type Factor = GaussianFactor<T>;

    /// Upward marginal message type passed from children to parents during elimination.
    type Message = GaussianFactor<T>;

    /// Scalar element type for variable vectors and numerical operations.
    type Value = T;

    /// Shared variable index map context threaded through build and solve passes.
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
        main: usize,
        separator: Vec<usize>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self::Node>,
        ctx: &Self::Context,
    ) -> Result<Self::Node> {
        // Init observation states
        let dim_obs = assigned_factors.iter().map(|f| f.dim_obs).sum::<usize>()
            + incoming_messages.iter().map(|m| m.dim_obs).sum::<usize>();
        let mut xiby = Col::zeros(dim_obs);
        let mut wby = Mat::zeros(dim_obs, dim_obs);

        // Init main variable block
        let dim_main = ctx
            .get_size(main)
            .with_context(|| format!("Missing size for variable with id: {}", main))?;
        let mut main_lin_var = LinVar {
            id: main,
            mat: Mat::zeros(dim_obs, dim_main),
        };

        // Init separator variables blocks
        let mut sep_lin_vars = Vec::with_capacity(separator.len());
        for &sep in &separator {
            let dim_u = ctx
                .get_size(sep)
                .with_context(|| format!("Missing size for variable with id: {}", sep))?;
            sep_lin_vars.push(LinVar {
                id: sep,
                mat: Mat::zeros(dim_obs, dim_u),
            });
        }

        let mut offset_obs = 0;
        for factor in assigned_factors.into_iter().chain(incoming_messages.iter()) {
            let size_obs = factor.dim_obs;

            // Populate xiby
            xiby.as_mut()
                .subrows_mut(offset_obs, size_obs)
                .copy_from(&factor.xib_obs);

            // Populate wby
            wby.submatrix_mut(offset_obs, offset_obs, size_obs, size_obs)
                .copy_from(&factor.wb_obs);

            // Populate a
            if let Some(lin_var) = factor.get_lin_var(main) {
                main_lin_var
                    .mat
                    .subrows_mut(offset_obs, size_obs)
                    .copy_from(&lin_var.mat);
            }

            // Populate b
            for sep_lin_var in sep_lin_vars.iter_mut() {
                if let Some(lin_var) = factor.get_lin_var(sep_lin_var.id) {
                    sep_lin_var
                        .mat
                        .subrows_mut(offset_obs, size_obs)
                        .copy_from(&lin_var.mat);
                }
            }

            offset_obs += size_obs;
        }

        let node = GaussianNode::new(
            main,
            separator,
            children,
            dim_main,
            dim_obs,
            main_lin_var,
            sep_lin_vars,
            xiby,
            wby,
        );

        Ok(node)
    }

    fn build_out_message(node: &mut Self::Node, _ctx: &Self::Context) -> Result<Self::Message> {
        node.eliminate()
    }

    /// Solve the main value from the separator values read from the buffer.
    fn decide_node(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()> {
        // Gather separator values from the global buffer directly into the node's workspace (b_s)
        node.b_s.fill(T::zero());
        for sep_lin_var in &node.sep_lin_vars {
            let (offset, size) = ctx.get_memory_layout(sep_lin_var.id).with_context(|| {
                format!("Missing offset for variable with id: {}", sep_lin_var.id)
            })?;

            unsafe {
                let s = ColRef::from_slice(buffer.read_slice(offset, size));
                matmul(
                    node.b_s.as_mut(),
                    Accum::Add,
                    &sep_lin_var.mat,
                    s,
                    T::one(),
                    Par::Seq,
                );
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
                root.main, root.separator, root.dim_obs
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

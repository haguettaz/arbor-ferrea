//! Gaussian elimination trees for factor graphs.
//!
//! Implements upward-filtering (elimination) and downward-deciding (back-substitution)
//! message passing over tree-structured Gaussian factor graphs.
//!
//! # References
//! - H.-A. Loeliger et al., *"On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing,"* ITA, 2016.
//! - Yunpeng Li and H.-A. Loeliger, *"Backward filtering forward deciding in linear non-Gaussian state space models,"* AISTATS, 2024.

use anyhow::{Context, Result};
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use dyn_stack::MemBuffer;
use dyn_stack::MemStack;

use faer::diag::Diag;
use faer::linalg::matmul::matmul;
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
/// via linear observations `Y = AX + BS`. The observation vector `Y`
/// is assigned a Gaussian prior defined by a weighted mean and a precision matrix.
///
/// The node owns its children in the tree.
/// Internal workspaces are preallocated to eliminate heap churn during inference.
pub struct GaussianNode<T> {
    /// The main variable identifier.
    pub main: usize,
    /// The separator variables identifiers.
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
    pub xib_obs: Col<T>,
    /// The precision matrix of the observation prior with shape `(dim_obs, dim_obs)`.
    pub wb_obs: Mat<T>,

    // ---- Workspaces ----
    /// Intermediate workspace, shape `(dim_obs, dim_main)`.
    wby_a: Mat<T>,
    /// Dual weighted mean for the main variable, shape `(dim_main,)`.
    xitx: Col<T>,
    /// Dual precision for the main variable, shape `(dim_main, dim_main)`.
    wtx: Mat<T>,
    /// Singular values of the dual precision.
    wtx_s: Diag<T>,
    /// Left singular vectors of the dual precision.
    wtx_u: Mat<T>,
    /// Right singular vectors of the dual precision.
    wtx_v: Mat<T>,
    /// Moore–Penrose pseudoinverse of the dual precision computed via SVD, shape `(dim_main, dim_main)`.
    vtx: Mat<T>,
    /// Intermediate workspace, shape `(dim_main,)`.
    vtx_xitx: Col<T>,
    /// Intermediate workspace, shape `(dim_main, dim_obs)`.
    vtx_at_wby: Mat<T>,
    /// Separator contribution into observation space, shape `(dim_obs,)`.
    b_s: Col<T>,
    /// Preallocated scratchpad memory for SVD and pseudoinverse calculations.
    mem_buf: MemBuffer,
}

impl<T> GaussianNode<T>
where
    T: RealField + Copy + Send + Sync + Debug + Display,
{
    /// Creates a new [`GaussianNode`] with all workspaces zero-allocated.
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

        Self {
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
        }
    }

    /// Eliminates the node's main variable to produce an outgoing marginal [`GaussianFactor`].
    ///
    /// Updates internal workspaces during elimination, notably:
    /// - `wby_a`: intermediate factor workspace.
    /// - `xitx`, `wtx`: total weighted mean and precision of the eliminated variable.
    /// - `wtx_u`, `wtx_s`, `wtx_v`: SVD components of `wtx`.
    /// - `vtx`: Moore–Penrose pseudo-inverse of `wtx`, derived via SVD.
    ///
    /// # Errors
    ///
    /// Returns an error if SVD computation of `wtx` fails.
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

        // Compute xibz = xiby - wby a vtx xitx
        matmul(
            self.vtx_xitx.as_mat_mut(),
            Accum::Replace,
            self.vtx.as_ref(),
            self.xitx.as_mat(),
            T::one(),
            Par::Seq,
        );
        let mut xibz = self.xib_obs.clone();
        matmul(
            xibz.as_mat_mut(),
            Accum::Add,
            self.wby_a.as_ref(),
            self.vtx_xitx.as_mat(),
            T::one().neg(),
            Par::Seq,
        );

        // Compute wbz = wby - wby a vtx a.t wby
        matmul(
            self.vtx_at_wby.as_mut(),
            Accum::Replace,
            self.vtx.as_ref(),
            self.wby_a.transpose(),
            T::one(),
            Par::Seq,
        );
        let mut wbz = self.wb_obs.clone();
        matmul(
            wbz.as_mut(),
            Accum::Add,
            self.wby_a.as_ref(),
            self.vtx_at_wby.as_ref(),
            T::one().neg(),
            Par::Seq,
        );

        let msg = GaussianFactor::new(self.sep_lin_vars.clone(), xibz, wbz);

        Ok(msg)
    }

    /// Decides the value of the main variable based on separator variables.
    ///
    /// Computes `vtx xitx - vtx at wby b s` into `x`.
    /// Requires `vtx_xitx`, `vtx_at_wby`, and `b_s` to be populated prior to calling.
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
    }

    /// Recursively formats the tree with ASCII branches.
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

/// An elimination tree forest for Gaussian factor graphs.
///
/// Encapsulates the numerical factor graph organized into one or more rooted
/// trees, where each subtree represents a recursive elimination clique.
pub struct GaussianTree<T> {
    /// Root nodes corresponding to independent elimination trees in the forest.
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

    /// Instantiates a forest from root nodes.
    fn from_roots(roots: Vec<Self::Node>) -> Self {
        Self { roots }
    }

    /// Returns a slice of the forest's root nodes.
    fn roots(&self) -> &[Self::Node] {
        &self.roots
    }

    /// Returns a mutable slice of the forest's root nodes.
    fn roots_mut(&mut self) -> &mut [Self::Node] {
        &mut self.roots
    }

    /// Returns the children of a given node.
    fn node_children(node: &Self::Node) -> &[Self::Node] {
        &node.children
    }

    /// Returns a mutable slice of children of a given node.
    fn node_children_mut(node: &mut Self::Node) -> &mut [Self::Node] {
        &mut node.children
    }

    /// Assembles an elimination node by concatenating assigned factor constraints and incoming messages.
    fn build_node(
        main: usize,
        separator: Vec<usize>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self::Node>,
        ctx: &Self::Context,
    ) -> Result<Self::Node> {
        let dim_obs = assigned_factors.iter().map(|f| f.dim_obs).sum::<usize>()
            + incoming_messages.iter().map(|m| m.dim_obs).sum::<usize>();
        let mut xiby = Col::zeros(dim_obs);
        let mut wby = Mat::zeros(dim_obs, dim_obs);

        let dim_main = ctx
            .size(main)
            .with_context(|| format!("Missing size for variable with id: {}", main))?;
        let mut main_lin_var = LinVar {
            id: main,
            mat: Mat::zeros(dim_obs, dim_main),
        };

        let mut sep_lin_vars = Vec::with_capacity(separator.len());
        for &sep in &separator {
            let dim_u = ctx
                .size(sep)
                .with_context(|| format!("Missing size for variable with id: {}", sep))?;
            sep_lin_vars.push(LinVar {
                id: sep,
                mat: Mat::zeros(dim_obs, dim_u),
            });
        }

        let mut offset_obs = 0;
        for factor in assigned_factors.into_iter().chain(incoming_messages.iter()) {
            let size_obs = factor.dim_obs;

            xiby.as_mut()
                .subrows_mut(offset_obs, size_obs)
                .copy_from(&factor.xib_obs);

            wby.submatrix_mut(offset_obs, offset_obs, size_obs, size_obs)
                .copy_from(&factor.wb_obs);

            if let Some(lin_var) = factor.lin_var(main) {
                anyhow::ensure!(
                    lin_var.mat.ncols() == dim_main,
                    format!(
                        "Variable {} has {} columns, but the registered size is {}.",
                        main,
                        lin_var.mat.ncols(),
                        dim_main
                    )
                );
                main_lin_var
                    .mat
                    .subrows_mut(offset_obs, size_obs)
                    .copy_from(&lin_var.mat);
            }

            for sep_lin_var in sep_lin_vars.iter_mut() {
                if let Some(lin_var) = factor.lin_var(sep_lin_var.id) {
                    anyhow::ensure!(
                        lin_var.mat.ncols() == sep_lin_var.mat.ncols(),
                        format!(
                            "Separator variable {} has {} columns, but the registered size is {}.",
                            sep_lin_var.id,
                            lin_var.mat.ncols(),
                            sep_lin_var.mat.ncols()
                        )
                    );
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

    /// Computes the upward message factor by marginalizing out the node's main variable.
    fn build_out_message(node: &mut Self::Node, _ctx: &Self::Context) -> Result<Self::Message> {
        node.eliminate()
    }

    /// Back-substitutes separator states to decide the main variable value in-place.
    fn decide_node(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()> {
        buffer.ensure_len(ctx.total_size())?;

        node.b_s.fill(T::zero());
        for sep_lin_var in &node.sep_lin_vars {
            let (offset, size) = ctx.memory_layout(sep_lin_var.id).with_context(|| {
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

        let (offset, size) = ctx.memory_layout(node.main).with_context(|| {
            format!("Missing memory layout for variable with id: {}", node.main)
        })?;
        anyhow::ensure!(
            size == node.dim_main,
            "Dimension mismatch for variable {}: expected {}, got {}",
            node.main,
            node.dim_main,
            size
        );
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
            writeln!(
                f,
                "Root [Var: {}, Separator: {:?}, Observation Dimension: {}]",
                root.main, root.separator, root.dim_obs
            )?;

            let num_children = root.children.len();
            for (j, child) in root.children.iter().enumerate() {
                child.fmt_recursive(f, "", j == num_children - 1)?;
            }

            if i < self.roots.len() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

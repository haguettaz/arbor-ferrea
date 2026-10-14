//! Elimination trees for exact inference on Gaussian factor graphs.
//!
//! Implements two-pass message passing—upward filtering (variable elimination) and
//! downward deciding (back-substitution)—over tree-structured Gaussian models.
//!
//! For theoretical background, refer to the
//! [lecture notes](https://doi.org/10.3929/ethz-c-000785981)
//! of the class *Model-Based Estimation and Signal Analysis*
//! (Prof. H.-A. Loeliger, ETH Zürich, 2025).

use anyhow::{Context, Result, ensure};
use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView, DVectorViewMut};
use std::iter::zip;

use crate::{
    numeric::{NumericClique, NumericContext, NumericTree},
    symbolic::SymbolicFactor,
};

/// Diagonal regularizer added before computing the Cholesky decomposition in
/// [`GaussianClique::build_and_close`].
/// Warning: to high value can lead to numerical imprecision.
const CHOL_REG: f64 = 1e-15;

/// A forest of elimination trees for exact inference on Gaussian factor graphs.
///
/// This structure organizes the numerical factor graph into independent rooted trees of cliques.
/// It orchestrates a two-pass solver: a bottom-up elimination pass (upward filtering) to
/// marginalize variables, and a top-down back-substitution pass (downward deciding) to
/// compute their exact assignments.
///
/// Both sequential and parallel execution modes are provided via the [`NumericTree`] trait.
pub struct GaussianTree {
    /// Root nodes corresponding to independent elimination trees in the forest.
    pub roots: Vec<GaussianClique>,
}

impl NumericTree for GaussianTree {
    /// Tree node type encapsulating the Gaussian clique and its workspaces.
    type Clique = GaussianClique;

    /// Instantiates a forest from root nodes.
    fn from_roots(roots: Vec<Self::Clique>) -> Self {
        Self { roots }
    }

    /// Returns a slice of the forest's root cliques.
    fn roots(&self) -> &[Self::Clique] {
        &self.roots
    }

    /// Returns a mutable slice of the forest's root cliques.
    fn roots_mut(&mut self) -> &mut [Self::Clique] {
        &mut self.roots
    }

    /// Returns the children of a given clique.
    fn children(clique: &Self::Clique) -> &[Self::Clique] {
        &clique.children
    }

    /// Returns a mutable slice of children of a given clique.
    fn children_mut(clique: &mut Self::Clique) -> &mut [Self::Clique] {
        &mut clique.children
    }
}

/// A Gaussian clique eliminating one frontal variable `X` given its separator variables `S`.
///
/// The clique gathers its factors and incoming messages into a single Gaussian observation
/// `Y := A X + B S` parameterized by a weighted mean vector and a precision matrix.
/// Eliminating `X` yields a new Gaussian factor on `Z := B S` which is propagated to the parent clique.
pub struct GaussianClique {
    /// Child cliques, eliminated before this one.
    children: Vec<GaussianClique>,

    /// Key of the frontal variable `x`.
    front_key: usize,
    /// Keys of the separator variables `s`.
    sep_keys: Vec<usize>,

    /// Matrices `A` of shape `(dim_y, dim_x)`, coupling the frontal variable to the observations.
    _a: DMatrix<f64>,
    /// Matrix `B` of shape `(dim_y, dim_s)`, coupling the separator variables to the observations.
    b: DMatrix<f64>,
    /// Context of the separator variable as tuples `(key, col range)` aligned with `B`.
    sep_ctx: Vec<(usize, (usize, usize))>,

    /// Observation weighted mean vector of shape (dim_y,)
    _xiby: DVector<f64>,
    /// Observation precision matrix of shape (dim_y, dim_y)
    _wby: DMatrix<f64>,

    /// Conditional mean `m_x = V Aᵀ ξ` of the frontal variable at `s = 0`, of shape `(dim_x,)`.
    mtx: DVector<f64>,
    /// Gain `V Aᵀ W` of shape `(dim_x, dim_y)`.
    vtx_at_wby: DMatrix<f64>,
    /// Workspace for `Σ_k B_k s_k`, of shape `(dim_y,)`.
    b_s: DVector<f64>,
    /// Workspace for the separator values
    s: DVector<f64>,
}

impl GaussianClique {
    /// Updates the clique after the matrices of its factors change, e.g., after relinearization.
    ///
    /// # Panics
    ///
    /// Always panics, as it is not implemented yet.
    pub fn update() -> Result<()> {
        todo!("Implement update, e.g., when jacobians are updated.");
    }
}

impl NumericClique for GaussianClique {
    /// Input factor type representing local Gaussian factors.
    type Factor = GaussianFactor;

    /// Upward Gaussian message type passed from children to parents during clique closure.
    type Message = GaussianFactor;

    /// Scalar element type for variable vectors and numerical operations.
    type Scalar = f64;

    // /// Shared contextual metadata threaded through the build and solve phases.
    // type Context = GaussianContext;

    /// Returns the child cliques.
    fn children(&self) -> &[Self] {
        &self.children
    }

    /// Returns the child cliques mutably.
    fn children_mut(&mut self) -> &mut [Self] {
        &mut self.children
    }

    /// Expose the main variable key so the tree can resolve its memory.
    fn front_key(&self) -> usize {
        self.front_key
    }

    /// Expose the separator keys so the tree can resolve their memory.
    fn sep_keys(&self) -> &[usize] {
        &self.sep_keys
    }

    /// Assembles the Gaussian clique for the specified variables and eliminates the frontal
    /// variable, yielding the resolved clique and the outgoing message to its parent.
    ///
    /// The resulting message is a Gaussian factor representing a linear observation over the
    /// separator variables, parameterized by a weighted mean and precision matrix.
    ///
    /// # Orchestration Invariants
    ///
    /// To ensure mathematical correctness, the caller (the tree) must
    /// guarantee the following preconditions:
    /// * The frontal and separator keys are strictly disjoint.
    /// * The separator keys contains no duplicate entries.
    /// * All provided factors and messages explicitly involve the frontal variable.
    /// * All provided factors and messages involve *only* the frontal and separator variables.
    ///
    /// # Errors
    ///
    /// Returns an error if a variable's memory layout cannot be resolved from the `ctx`,
    /// or if the internal Cholesky decomposition fails (e.g., due to an ill-conditioned matrix).
    fn build_and_close<C: NumericContext>(
        front_key: usize,
        sep_keys: Vec<usize>,
        factors: Vec<&GaussianFactor>,
        in_msgs: Vec<GaussianFactor>,
        children: Vec<GaussianClique>,
        ctx: &C,
    ) -> Result<(GaussianClique, GaussianFactor)> {
        let dim_y = factors.iter().map(|f| f.dim_y()).sum::<usize>()
            + in_msgs.iter().map(|f| f.dim_y()).sum::<usize>();
        let mut xiby = DVector::zeros(dim_y);
        let mut wby = DMatrix::zeros(dim_y, dim_y);

        let (_, dim_x) = ctx.memory_layout(front_key).with_context(|| {
            format!(
                "The dimension of the frontal variable {} cannot be found in the variable context",
                front_key
            )
        })?;
        let mut a = DMatrix::zeros(dim_y, dim_x);

        let mut sep_ctx = Vec::<(usize, (usize, usize))>::with_capacity(sep_keys.len());
        let mut icol = 0;
        for &key in &sep_keys {
            let (_, ncols) = ctx.memory_layout(key).with_context(|| format!("The dimension of the separator variable {} cannot be found in the variable context", key))?;
            sep_ctx.push((key, (icol, ncols)));
            icol += ncols;
        }
        let dim_s = icol;
        let mut b = DMatrix::zeros(dim_y, dim_s);

        // Init vectors and matrices
        let mut irow = 0;
        for f in factors.into_iter().chain(in_msgs.iter()) {
            let nrows = f.dim_y();
            if let Some(mat) = f.a_view(&front_key) {
                a.rows_mut(irow, nrows).copy_from(&mat);
            }
            for &(key, (icol, ncols)) in &sep_ctx {
                if let Some(mat) = f.a_view(&key) {
                    // b.view_range_mut(f_start..f_start + f_size, cols.clone())
                    b.view_mut((irow, icol), (nrows, ncols)).copy_from(&mat);
                }
            }
            xiby.rows_mut(irow, nrows).copy_from(&f.xiby_view());
            wby.view_mut((irow, irow), (nrows, nrows))
                .copy_from(&f.wby_view());
            irow += nrows;
        }

        let mut xitx = DVector::zeros(dim_x);
        a.tr_mul_to(&xiby, &mut xitx); // xitx = a.t xiby

        let mut at_wby = DMatrix::zeros(dim_x, dim_y);
        a.tr_mul_to(&wby, &mut at_wby);

        // let mut wtx = DMatrix::zeros(dim_x, dim_x);
        let mut wtx = DMatrix::from_diagonal_element(dim_x, dim_x, CHOL_REG); // ensure wtx is positive definite
        wtx.gemm(1.0, &at_wby, &a, 1.0); // wtx = a.t wby a
        let rtx = wtx.cholesky().with_context(|| {
            format!(
                "Cholesky decomposition for frontal variable {} failed",
                front_key
            )
        })?;

        let mut mtx = xitx.clone();
        rtx.solve_mut(&mut mtx); // wtx mtx = xitx

        let mut vtx_at_wby = at_wby.clone();
        rtx.solve_mut(&mut vtx_at_wby);

        let mut xibz = xiby.clone(); // xibz = xiby - wby a mtx
        xibz.gemv_tr(-1.0, &at_wby, &mtx, 1.0);

        let mut wbz = wby.clone(); // wbz = wby - wby a vtx a.t wby
        wbz.gemm_tr(-1.0, &at_wby, &vtx_at_wby, 1.0);

        let s = DVector::from_element(dim_s, f64::NAN);
        let b_s = DVector::from_element(dim_y, f64::NAN);

        let out_msg =
            GaussianFactor::new_unchecked(sep_keys.clone(), sep_ctx.clone(), b.clone(), xibz, wbz);
        let clique = GaussianClique {
            children,
            front_key,
            sep_keys,
            sep_ctx,
            _a: a,
            b,
            _xiby: xiby,
            _wby: wby,
            mtx,
            vtx_at_wby,
            s,
            b_s,
        };

        Ok((clique, out_msg))
    }

    /// Writes the conditional mean `x̂(s)` of the frontal variable into `out`, given the values
    /// `sep_vals` of the separator variables, aligned with `sep_keys`.
    fn decide(&mut self, out: &mut [Self::Scalar], sep_vals: &[&[Self::Scalar]]) -> Result<()> {
        // Load the separator values into self.s
        for (&(_, (irow, nrows)), sep_val) in zip(self.sep_ctx.iter(), sep_vals.iter()) {
            let mut s = self.s.rows_mut(irow, nrows);
            s.copy_from_slice(sep_val);
        }
        // Compute the product b s into the corresponding pre-allocated workspace.
        self.b.mul_to(&self.s, &mut self.b_s);

        // Compute x := mtx - vtx a.t wby b s into out.
        let mut x_view = DVectorViewMut::from_slice(out, out.len());
        x_view.copy_from(&self.mtx);
        x_view.gemv(-1.0, &self.vtx_at_wby, &self.b_s, 1.0);

        Ok(())
    }
}

/// A Gaussian factor over a linear observation parameterized by precision and a weighted mean.
pub struct GaussianFactor {
    /// Variable keys.
    keys: Vec<usize>,
    /// Context of the variables, i.e., tuples of keys and column range (icol, ncols) in `A`.
    ctx: Vec<(usize, (usize, usize))>,
    /// Observation model matrices of shape `(dim_y, dim_x)`
    a: DMatrix<f64>,
    /// Observation weighted mean of shape `(dim_y,)`.
    xiby: DVector<f64>,
    /// Observation precision of shape `(dim_y, dim_y)`.
    wby: DMatrix<f64>,
}

impl GaussianFactor {
    /// Creates a Gaussian factor without validating its parameters.
    ///
    /// The caller must ensure that `keys` and `ctx` contains the same keys,
    /// that `a` has shape `(dim_y, dim_x)` where `dim_x` is the total size of the keys,
    /// and that `xiby` and `wby` have shapes  `(dim_y,)` and `(dim_y, dim_y)`.
    pub(crate) fn new_unchecked(
        keys: Vec<usize>,
        ctx: Vec<(usize, (usize, usize))>,
        a: DMatrix<f64>,
        xiby: DVector<f64>,
        wby: DMatrix<f64>,
    ) -> Self {
        // Check that every key in ctx is in keys.
        debug_assert!(!ctx.iter().any(|(ck, _)| !keys.contains(ck)),);
        // Check that every key in keys is in ctx.
        debug_assert!(!keys.iter().any(|k| !ctx.iter().any(|(ck, _)| ck == k)),);

        // Check dimensions
        debug_assert!(a.ncols() == ctx.iter().map(|(_, (_, ncols))| ncols).sum());
        debug_assert!(a.nrows() == xiby.nrows());
        debug_assert!(a.nrows() == wby.nrows());
        debug_assert!(a.nrows() == wby.ncols());

        Self {
            keys,
            ctx,
            a,
            xiby,
            wby,
        }
    }

    pub fn try_new(
        ctx: Vec<(usize, usize)>, // (key, size)
        a: DMatrix<f64>,
        xiby: DVector<f64>,
        wby: DMatrix<f64>,
    ) -> Result<Self> {
        ensure!(
            a.ncols() == ctx.iter().map(|(_, ncols)| ncols).sum(),
            "The shape of a is not compatible with ctx."
        );
        ensure!(
            a.nrows() == xiby.nrows(),
            "The shape of a is not compatible with xiby."
        );
        ensure!(
            (a.nrows() == wby.nrows()) && (a.nrows() == wby.ncols()),
            "The shape of a is not compatible with wby."
        );

        let keys = ctx.iter().map(|&(key, _)| key).collect();

        let mut new_ctx = Vec::with_capacity(ctx.len());
        let mut icol = 0;
        for &(key, size) in &ctx {
            new_ctx.push((key, (icol, size)));
            icol += size;
        }

        Ok(Self {
            keys,
            ctx: new_ctx,
            a,
            xiby,
            wby,
        })
    }

    /// Returns the keys of the variables.
    pub fn keys(&self) -> &[(usize, (usize, usize))] {
        &self.ctx
    }

    /// Returns `true` if the factor involves the variable `key`.
    fn key(&self, key: &usize) -> Option<&(usize, (usize, usize))> {
        self.ctx.iter().find(|(k, _)| k == key)
    }

    /// Returns a view on the block column of the variable `key`, or `None` if absent.
    pub fn a_view(&self, key: &usize) -> Option<DMatrixView<'_, f64>> {
        self.key(key)
            .map(|&(_, (icol, ncols))| self.a.columns(icol, ncols))
    }

    /// Returns a view on the observation weighted mean vector.
    pub fn xiby_view(&self) -> DVectorView<'_, f64> {
        self.xiby.as_view()
    }

    /// Returns a view on the observation precision matrix.
    pub fn wby_view(&self) -> DMatrixView<'_, f64> {
        self.wby.as_view()
    }

    /// Returns the observation dimension.
    pub fn dim_y(&self) -> usize {
        self.xiby.nrows()
    }
}

impl SymbolicFactor for GaussianFactor {
    fn keys(&self) -> &[usize] {
        &self.keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_factor() {
        let res = GaussianFactor::try_new(
            vec![(0, 2), (1, 3)],
            DMatrix::zeros(4, 5),
            DVector::zeros(4),
            DMatrix::identity(4, 4),
        );
        assert!(res.is_ok());

        let res = GaussianFactor::try_new(
            vec![(0, 2), (1, 4)],
            DMatrix::zeros(4, 5),
            DVector::zeros(4),
            DMatrix::identity(4, 4),
        );
        assert!(res.is_err(), "Incompatible a and ctx");

        let res = GaussianFactor::try_new(
            vec![(0, 2), (1, 3)],
            DMatrix::zeros(4, 5),
            DVector::zeros(3),
            DMatrix::identity(4, 4),
        );
        assert!(res.is_err(), "Incompatible a and xiby");

        let res = GaussianFactor::try_new(
            vec![(0, 2), (1, 3)],
            DMatrix::zeros(4, 5),
            DVector::zeros(4),
            DMatrix::identity(2, 4),
        );
        assert!(res.is_err(), "Incompatible a and wby");
    }
}

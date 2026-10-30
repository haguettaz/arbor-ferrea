use std::collections::HashMap;
use std::fmt::Debug;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DMatrixView, DVector, RealField};

use anyhow::{Context, Result, bail, ensure};

/// Offsets of a sorted key set inside a stacked vector
#[derive(Debug)]
struct Layout {
    keys: Vec<usize>,
    offsets: Vec<usize>,
}

impl Layout {
    fn new(keys: Vec<usize>, size_of: impl Fn(usize) -> usize) -> Self {
        let mut offsets = Vec::with_capacity(keys.len() + 1);
        let mut acc = 0;
        offsets.push(0);
        for &k in &keys {
            acc += size_of(k);
            offsets.push(acc);
        }
        Self { keys, offsets }
    }

    fn dim(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }

    /// (offset, size) of `key`'s block, if present.
    fn find(&self, key: usize) -> Option<(usize, usize)> {
        let i = self.keys.binary_search(&key).ok()?;
        Some((self.offsets[i], self.offsets[i + 1] - self.offsets[i]))
    }
}

#[derive(Debug)]
pub struct Marginal<T: RealField> {
    pub key: usize,
    pub mean: DVector<T>,
    pub cov: DMatrix<T>,
}

impl<T: RealField> Marginal<T> {
    pub fn from(
        key: usize,
        prior_or_none: Option<&Prior<T>>,
        likelihood: &Likelihood<T>,
    ) -> Result<Self> {
        let dim_x = likelihood.dim_x();

        let (mut xi, mut w) = match prior_or_none {
            None => (DVector::zeros(dim_x), DMatrix::zeros(dim_x, dim_x)),
            Some(prior) => {
                ensure!(
                    likelihood.keys() == prior.keys(),
                    "prior and likelihood keys do not coincide"
                );
                ensure!(
                    likelihood.dim_x() == prior.dim_x(),
                    "prior and likelihood dimensions do not coincide"
                );
                (prior.xifx.clone(), prior.wfx.clone())
            }
        };
        xi.gemv_tr(T::one(), &likelihood.a, &likelihood.xiby, T::one());
        w.quadform(T::one(), &likelihood.wby, &likelihood.a, T::one());

        let chol_w =
            Cholesky::new(w)
                .with_context(||format!("Cholesky failed while computing the marginal. Consider adding dumb priors for variable with {key}"))?;

        let mean = chol_w.solve(&xi);
        let cov = chol_w.inverse();

        Ok(Marginal { key, mean, cov })
    }
}

#[derive(Debug)]
pub struct Prior<T: RealField> {
    key: usize,
    xifx: DVector<T>,
    wfx: DMatrix<T>,
}

impl<T: RealField> Prior<T> {
    pub fn keys(&self) -> &[usize] {
        std::slice::from_ref(&self.key)
    }

    pub fn has_key(&self, key: usize) -> bool {
        self.key == key
    }

    pub fn dim_x(&self) -> usize {
        self.xifx.nrows()
    }
}

#[derive(Debug)]
pub struct Likelihood<T: RealField> {
    layout: Layout,
    a: DMatrix<T>,
    xiby: DVector<T>,
    wby: DMatrix<T>,
}

impl<T: RealField> Likelihood<T> {
    pub fn keys(&self) -> &[usize] {
        &self.layout.keys
    }

    pub fn has_key(&self, key: usize) -> bool {
        self.layout.find(key).is_some()
    }

    /// Returns a view on the block column of the variable `key`, or `None` if absent.
    pub fn a_from_key(&self, key: usize) -> Option<DMatrixView<'_, T>> {
        self.layout
            .find(key)
            .map(|(icol, ncols)| self.a.columns(icol, ncols))
    }

    // pub fn layout(&self, key: &usize) -> Option<(usize, usize)> {
    //     match self.keys.binary_search(key) {
    //         Ok(i) => Some(self.layout[i]),
    //         Err(_) => None,
    //     }
    // }

    pub fn dim_x(&self) -> usize {
        self.a.ncols()
    }

    pub fn dim_y(&self) -> usize {
        self.a.nrows()
    }

    /// Create a new  likelihood by aggregating observation models and prior models and eliminating some variables.
    /// It is the caller responsability to check that the kept and cleared keys are compatible and valid with factors.
    /// This is equivalent to max-product and sum-product message passing for closing box.
    /// The likelihood factor is expressed with minimal observation dimension: if the total dimension of all input factors is larger than the total dimensions of the kept variables, the observation is expressed after projecting it in the kept variable space
    pub fn from_closure<'a, C>(
        factors: &[FactorRef<T>],
        kept_keys: Vec<usize>,
        cleared_keys: Vec<usize>,
        size_of: C,
    ) -> Result<Likelihood<T>>
    where
        C: Fn(usize) -> usize,
    {
        debug_assert!(cleared_keys.is_sorted(), "unsorted cleared keys");
        debug_assert!(kept_keys.is_sorted(), "unsorted kept keys");

        for f in factors.iter() {
            match f {
                FactorRef::Likelihood(f) => {
                    for key in f.keys() {
                        if let (Err(_), Err(_)) = (
                            kept_keys.binary_search(&key),
                            cleared_keys.binary_search(&key),
                        ) {
                            bail!("likelihood key {key} is ambiguous: neither kept nor cleared");
                        }
                    }
                }
                FactorRef::Prior(f) => {
                    let key = f.key;
                    if let (Err(_), Err(_)) = (
                        kept_keys.binary_search(&key),
                        cleared_keys.binary_search(&key),
                    ) {
                        bail!("prior key {key} is ambiguous: neither kept nor cleared");
                    }
                }
            }
        }

        // Compute dimensions and layout contexts
        let kept_layout = Layout::new(kept_keys, &size_of);
        let cleared_layout = Layout::new(cleared_keys, &size_of);
        let (dim_x, dim_u) = (kept_layout.dim(), cleared_layout.dim());
        let dim_y = factors
            .iter()
            .map(|f| match f {
                FactorRef::Likelihood(f) => f.dim_y(),
                FactorRef::Prior(f) => {
                    match kept_layout.find(f.key) {
                        Some(_) => f.dim_x(), // a prior on a kept variable is interpreted as a likelihood with identity matrix
                        None => 0,
                    }
                }
            })
            .sum();
        //         let dim_y: usize = factors.iter().map(|f| match f {
        //             FactorRef::Likelihood(l) => l.dim_y(),
        //             FactorRef::Prior(_) => 0,
        //         }).sum();
        //         //
        //         //
        //         let mut kept_layout: Vec<(usize, usize)> = Vec::with_capacity(kept_keys.len());
        //         let mut offset = 0;
        //         for key in &kept_keys {
        //             let size = ctx(*key);
        //             kept_layout.push((offset, size));
        //             offset += size;
        //         }
        //         let dim_x = offset;
        //
        //         let mut cleared_layout: Vec<(usize, usize)> = Vec::with_capacity(cleared_keys.len());
        //         let mut offset = 0;
        //         for key in &cleared_keys {
        //             let size = ctx(*key);
        //             cleared_layout.push((offset, size));
        //             offset += size;
        //         }
        //         let dim_u = offset;
        //
        //         // Compute total observation dimension

        // branch on the minimal observation model
        if dim_y <= dim_x {
            Self::from_closure_on_z(factors, kept_layout, cleared_layout, dim_x, dim_u, dim_y)
        } else {
            Self::from_closure_on_x(factors, kept_layout, cleared_layout, dim_x, dim_u, dim_y)
        }
    }

    fn from_closure_on_z(
        factors: &[FactorRef<T>],
        kept_layout: Layout,
        cleared_layout: Layout,
        dim_x: usize,
        dim_u: usize,
        dim_y: usize,
    ) -> Result<Likelihood<T>> {
        let mut a = DMatrix::zeros(dim_y, dim_x);
        let mut xiby = DVector::zeros(dim_y);
        let mut wby = DMatrix::zeros(dim_y, dim_y);
        let mut wby_b = DMatrix::zeros(dim_y, dim_u);

        let mut xifu_plus_bt_xiby = DVector::zeros(dim_u);
        let mut wfu_plus_bt_wby_b = DMatrix::zeros(dim_u, dim_u);

        let mut y_pos = 0;

        for f in factors {
            match f {
                FactorRef::Prior(f) => {
                    let key = f.key;
                    if let Some((x_pos, x_size)) = kept_layout.find(key) {
                        let y_size = f.dim_x();

                        // update xiby
                        let mut tmp = xiby.rows_mut(y_pos, y_size);
                        tmp.copy_from(&f.xifx);

                        // update wby
                        let mut tmp = wby.view_mut((y_pos, y_pos), (y_size, y_size));
                        tmp.copy_from(&f.wfx);

                        // update a
                        a.view_mut((y_pos, x_pos), (y_size, x_size))
                            .fill_diagonal(T::one());

                        y_pos += y_size;
                    } else if let Some((u_pos, u_size)) = cleared_layout.find(key) {
                        // update xifu_plus_bt_xiby
                        let mut tmp = xifu_plus_bt_xiby.rows_mut(u_pos, u_size);
                        tmp += &f.xifx;

                        // update wfu_plus_bt_wby_b
                        let mut tmp = wfu_plus_bt_wby_b.view_mut((u_pos, u_pos), (u_size, u_size));
                        tmp += &f.wfx;
                    } else {
                        bail!("prior key {key} is neither kept nor cleared");
                    }
                }
                FactorRef::Likelihood(f) => {
                    let y_size = f.dim_y();

                    // update xiby
                    let mut tmp = xiby.rows_mut(y_pos, y_size);
                    tmp.copy_from(&f.xiby);

                    // update wby
                    let mut tmp = wby.view_mut((y_pos, y_pos), (y_size, y_size));
                    tmp.copy_from(&f.wby);

                    for &key in f.keys() {
                        if let Some((x_pos, x_size)) = kept_layout.find(key) {
                            if let Some(a_view) = f.a_from_key(key) {
                                // update a
                                a.view_mut((y_pos, x_pos), (y_size, x_size))
                                    .copy_from(&a_view);
                            }
                        } else if let Some((u_pos, u_size)) = cleared_layout.find(key) {
                            if let Some(b_view) = f.a_from_key(key) {
                                // update wby_b
                                let mut tmp = wby_b.view_mut((y_pos, u_pos), (y_size, u_size));
                                tmp.gemm(T::one(), &f.wby, &b_view, T::one());

                                // update xifu_plus_bt_xiby
                                let mut tmp = xifu_plus_bt_xiby.rows_mut(u_pos, u_size);
                                tmp.gemv_tr(T::one(), &b_view, &f.xiby, T::one());

                                // update wfu_plus_bt_wby_b (only the lower triangular part)
                                for &other_key in f.keys() {
                                    if other_key >= key
                                        && let Some((u_other_pos, u_other_size)) =
                                            cleared_layout.find(other_key)
                                    {
                                        if let Some(b_view_other) = f.a_from_key(other_key) {
                                            let mut tmp = wfu_plus_bt_wby_b.view_mut(
                                                (u_other_pos, u_pos),
                                                (u_other_size, u_size),
                                            );
                                            tmp.gemm_tr(
                                                T::one(),
                                                &b_view_other,
                                                &wby_b.view((y_pos, u_pos), (y_size, u_size)),
                                                T::one(),
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            bail!("likelihood key {key} is neither kept nor cleared");
                        }
                    }
                    y_pos += y_size;
                }
            }
        }

        let chol_wfu_plus_bt_wby_b =
        Cholesky::new(wfu_plus_bt_wby_b)
            .with_context(|| format!("Cholesky failed: cleared variables are under-constrained. Consider adding dumb priors for variables {:?}", cleared_layout.keys))?;
        let l = chol_wfu_plus_bt_wby_b.l_dirty();

        let mut g = wby_b.transpose();
        let mut h = xifu_plus_bt_xiby;
        ensure!(
            l.solve_lower_triangular_mut(&mut g) && l.solve_lower_triangular_mut(&mut h),
            "Singular factor in elimination"
        );
        xiby.gemv_tr(-T::one(), &g, &h, T::one());
        wby.gemm_tr(-T::one(), &g, &g, T::one());

        Ok(Likelihood {
            layout: kept_layout,
            a,
            xiby,
            wby,
        })
    }

    /// kept_keys must be sorted and aligned with kept_layout
    /// cleared_keys must be sorted and aligned with cleared_layout
    /// it is the user responsability to ensure it is the case
    fn from_closure_on_x(
        factors: &[FactorRef<T>],
        kept_layout: Layout,
        cleared_layout: Layout,
        dim_x: usize,
        dim_u: usize,
        dim_y: usize,
    ) -> Result<Likelihood<T>> {
        let mut wby_a = DMatrix::zeros(dim_y, dim_x); // used for cross product terms in at_wby_a and bt_wby_a
        let mut wby_b = DMatrix::zeros(dim_y, dim_u); // used for cross product terms in wfu_plus_bt_wby_b
        let mut at_xiby = DVector::zeros(dim_x);
        let mut at_wby_a = DMatrix::zeros(dim_x, dim_x); // note: only the lower triangular part is valid
        let mut bt_wby_a = DMatrix::zeros(dim_u, dim_x);
        let mut xifu_plus_bt_xiby = DVector::zeros(dim_u);
        let mut wfu_plus_bt_wby_b = DMatrix::zeros(dim_u, dim_u); // note: only lower triangular part is valid

        let mut y_pos = 0;
        for f in factors {
            match f {
                FactorRef::Prior(f) => {
                    let key = f.key;
                    if let Some((x_pos, x_size)) = kept_layout.find(key) {
                        // update at_xiby
                        let mut tmp = at_xiby.rows_mut(x_pos, x_size);
                        tmp.copy_from(&f.xifx);

                        // update at_wby_a (no need to use wby_a since there is no cross-product terms in a prior)
                        let mut tmp = at_wby_a.view_mut((x_pos, x_pos), (x_size, x_size));
                        tmp.copy_from(&f.wfx);

                        y_pos += x_size;
                    } else if let Some((u_pos, u_size)) = cleared_layout.find(key) {
                        // update xifu_plus_bt_xiby
                        let mut tmp = xifu_plus_bt_xiby.rows_mut(u_pos, u_size);
                        tmp += &f.xifx;

                        // update wfu_plus_bt_wby_b (no need to use wby_b since there is no cross-product terms in a prior)
                        let mut tmp = wfu_plus_bt_wby_b.view_mut((u_pos, u_pos), (u_size, u_size));
                        tmp += &f.wfx;
                    } else {
                        bail!("prior key {key} is neither kept nor cleared");
                    }
                }
                FactorRef::Likelihood(f) => {
                    let y_size = f.dim_y();

                    for &key in f.keys() {
                        if let Some((x_pos, x_size)) = kept_layout.find(key) {
                            if let Some(a_view) = f.a_from_key(key) {
                                // update at_xiby
                                let mut tmp = at_xiby.rows_mut(x_pos, x_size);
                                tmp.gemv_tr(T::one(), &a_view, &f.xiby, T::one());

                                // update wby_a
                                let mut tmp = wby_a.view_mut((y_pos, x_pos), (y_size, x_size));
                                tmp.gemm(T::one(), &f.wby, &a_view, T::one());

                                for &other_key in f.keys() {
                                    if other_key >= key
                                        && let Some((x_other_pos, x_other_size)) =
                                            kept_layout.find(other_key)
                                    {
                                        if let Some(a_view_other) = f.a_from_key(other_key) {
                                            // update at_wby_a (only lower triangular part)
                                            let mut tmp = at_wby_a.view_mut(
                                                (x_other_pos, x_pos),
                                                (x_other_size, x_size),
                                            );
                                            tmp.gemm_tr(
                                                T::one(),
                                                &a_view_other,
                                                &wby_a.view((y_pos, x_pos), (y_size, x_size)),
                                                T::one(),
                                            );
                                        }
                                    } else if let Some((u_other_pos, u_other_size)) =
                                        cleared_layout.find(other_key)
                                    {
                                        if let Some(b_view_other) = f.a_from_key(other_key) {
                                            // update bt_wby_a
                                            let mut tmp = bt_wby_a.view_mut(
                                                (u_other_pos, x_pos),
                                                (u_other_size, x_size),
                                            );
                                            tmp.gemm_tr(
                                                T::one(),
                                                &b_view_other,
                                                &wby_a.view((y_pos, x_pos), (y_size, x_size)),
                                                T::one(),
                                            );
                                        }
                                    }
                                }
                            }
                        } else if let Some((u_pos, u_size)) = cleared_layout.find(key) {
                            if let Some(b_view) = f.a_from_key(key) {
                                // update wby_b
                                let mut tmp = wby_b.view_mut((y_pos, u_pos), (y_size, u_size));
                                tmp.gemm(T::one(), &f.wby, &b_view, T::one());

                                // update xifu_plus_bt_xiby
                                let mut tmp = xifu_plus_bt_xiby.rows_mut(u_pos, u_size);
                                tmp.gemv_tr(T::one(), &b_view, &f.xiby, T::one());

                                // update wfu_plus_bt_wby_b (only the lower triangular part)
                                for &other_key in f.keys() {
                                    if other_key >= key
                                        && let Some((u_other_pos, u_other_size)) =
                                            cleared_layout.find(other_key)
                                    {
                                        if let Some(b_view_other) = f.a_from_key(other_key) {
                                            let mut tmp = wfu_plus_bt_wby_b.view_mut(
                                                (u_other_pos, u_pos),
                                                (u_other_size, u_size),
                                            );
                                            tmp.gemm_tr(
                                                T::one(),
                                                &b_view_other,
                                                &wby_b.view((y_pos, u_pos), (y_size, u_size)),
                                                T::one(),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    y_pos += y_size;
                }
            }
        }

        let chol_wfu_plus_bt_wby_b =
        Cholesky::new(wfu_plus_bt_wby_b)
            .with_context(|| format!("Cholesky failed: cleared variables are under-constrained. Consider adding dumb priors for variables {:?}", cleared_layout.keys))?;
        let l = chol_wfu_plus_bt_wby_b.l_dirty();

        let (mut g, mut h) = (bt_wby_a, xifu_plus_bt_xiby);
        ensure!(
            l.solve_lower_triangular_mut(&mut g) && l.solve_lower_triangular_mut(&mut h),
            "Singular factor in elimination"
        );
        at_xiby.gemv_tr(-T::one(), &g, &h, T::one());
        at_wby_a.fill_upper_triangle_with_lower_triangle();
        at_wby_a.gemm_tr(-T::one(), &g, &g, T::one());

        Ok(Likelihood {
            layout: kept_layout,
            a: DMatrix::identity(dim_x, dim_x),
            xiby: at_xiby,
            wby: at_wby_a,
        })
    }
}

pub enum Factor<T: RealField> {
    Prior(Prior<T>),
    Likelihood(Likelihood<T>),
}

impl<T: RealField> Factor<T> {
    pub fn keys(&self) -> &[usize] {
        match self {
            Self::Prior(f) => f.keys(),
            Self::Likelihood(f) => f.keys(),
        }
    }

    pub fn has_key(&self, key: usize) -> bool {
        match self {
            Self::Prior(f) => f.has_key(key),
            Self::Likelihood(f) => f.has_key(key),
        }
    }
}

pub enum FactorRef<'a, T: RealField> {
    Prior(&'a Prior<T>),
    Likelihood(&'a Likelihood<T>),
}

impl<'a, T: RealField> FactorRef<'a, T> {
    pub fn keys(&self) -> &[usize] {
        match self {
            Self::Prior(f) => f.keys(),
            Self::Likelihood(f) => f.keys(),
        }
    }

    pub fn has_key(&self, key: usize) -> bool {
        match self {
            Self::Prior(f) => f.has_key(key),
            Self::Likelihood(f) => f.has_key(key),
        }
    }
}

/// Variable symbol
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub char, pub usize);

pub struct FactorGraph<T: RealField> {
    factors: Vec<Factor<T>>,
    key_to_symbol: Vec<Symbol>,
    key_to_size: Vec<usize>,
    symbol_to_key: HashMap<Symbol, usize>,
}

impl<T: RealField> FactorGraph<T> {
    pub fn new() -> Self {
        Self {
            factors: Vec::new(),
            key_to_symbol: Vec::new(),
            key_to_size: Vec::new(),
            symbol_to_key: HashMap::new(),
        }
    }

    pub fn n_keys(&self) -> usize {
        self.key_to_symbol.len()
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = usize> {
        0..self.key_to_symbol.len()
    }

    pub fn key(&self, symbol: &Symbol) -> Option<usize> {
        self.symbol_to_key.get(symbol).copied()
    }

    pub fn has_key(&self, key: usize) -> bool {
        key < self.key_to_symbol.len()
    }

    pub fn symbol(&self, key: usize) -> Option<&Symbol> {
        self.key_to_symbol.get(key)
    }

    pub fn size(&self, key: usize) -> Option<usize> {
        self.key_to_size.get(key).copied()
    }

    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    pub fn factors(&self) -> &[Factor<T>] {
        &self.factors
    }

    pub fn factor(&self, i: usize) -> Option<&Factor<T>> {
        self.factors.get(i)
    }

    pub fn add_variable(&mut self, symbol: Symbol, size: usize) -> Result<usize> {
        if let Some(_) = self.symbol_to_key.get(&symbol) {
            bail!("Variable {:?} has already been registered.", symbol)
        }

        let id = self.key_to_symbol.len();
        self.symbol_to_key.insert(symbol, id);
        self.key_to_symbol.push(symbol);
        self.key_to_size.push(size);
        Ok(id)
    }

    pub fn add_prior(&mut self, key: usize, mfx: DVector<T>, wfx: DMatrix<T>) -> Result<usize> {
        ensure!(
            self.has_key(key),
            "The provided prior concerns a variable that has not been registered yet.",
        );

        let dim_x = self
            .size(key)
            .with_context(|| format!("Size of {} not found in factor graph context.", key))?;

        ensure!(
            (mfx.nrows() == dim_x),
            format!(
                "Expected mean vector to have shape ({}, ), got ({}, ).",
                dim_x,
                mfx.nrows(),
            )
        );
        ensure!(
            (wfx.nrows() == dim_x) && (wfx.ncols() == dim_x),
            format!(
                "Expected precision matrix to have shape ({}, {}), got ({}, {}).",
                dim_x,
                dim_x,
                wfx.nrows(),
                wfx.ncols(),
            )
        );

        let id = self.factors().len();

        let xifx = &wfx * &mfx;
        self.factors.push(Factor::Prior(Prior { key, xifx, wfx }));

        Ok(id)
    }

    pub fn add_observation(
        &mut self,
        keys: Vec<usize>,
        a: DMatrix<T>,
        mby: DVector<T>,
        wby: DMatrix<T>,
    ) -> Result<usize> {
        ensure!(keys.is_sorted(), "Keys must be sorted.");
        ensure!(
            keys.iter().all(|&key| self.has_key(key)),
            "The provided observation concerns variables that have not been registered yet.",
        );

        let layout = Layout::new(keys, |k| self.size(k).unwrap_or(0));

        let dim_x = layout.dim();
        let dim_y = mby.nrows();

        // check wby shape
        ensure!(
            (wby.nrows() == dim_y) && (wby.ncols() == dim_y),
            format!(
                "Expected precision matrix to have shape ({}, {}), got ({}, {}).",
                dim_y,
                dim_y,
                wby.nrows(),
                wby.ncols(),
            )
        );

        // check a shape
        ensure!(
            (a.nrows() == dim_y) && (a.ncols() == dim_x),
            format!(
                "Expected A matrix to have shape ({}, {}), got ({}, {}).",
                dim_y,
                dim_x,
                a.nrows(),
                a.ncols(),
            )
        );

        let id = self.factors.len();

        let xiby = &wby * &mby;
        self.factors.push(Factor::Likelihood(Likelihood {
            layout,
            a,
            xiby,
            wby,
        }));

        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    const X1: Symbol = Symbol('x', 1);
    const X2: Symbol = Symbol('x', 2);
    const X3: Symbol = Symbol('x', 3);
    const X42: Symbol = Symbol('x', 42);

    #[test]
    fn test_factor_graph() {
        let mut factor_graph = FactorGraph::new();

        let x1 = factor_graph
            .add_variable(X1, 1)
            .expect("Adding a new variable must work.");
        let x2 = factor_graph
            .add_variable(X2, 3)
            .expect("Adding a new 3-dimensional variable must also work.");
        let x3 = factor_graph
            .add_variable(X3, 1_000_000)
            .expect("Adding a new, very high-dimension variable must also work.");
        let _ = factor_graph
            .add_variable(X2, 3)
            .expect_err("Adding an already registered variable must fail.");
        let not_registered_x42 = 42; // a not registered variable

        assert_eq!(
            factor_graph.n_keys(),
            3,
            "The factor graph has three registered variables: x1, x2, and x3"
        );

        let symbolx1 = factor_graph.symbol(x1);
        assert_eq!(
            symbolx1,
            Some(&X1),
            "A registered variable has an accessible symbol."
        );
        let not_registered_symbol = factor_graph.symbol(not_registered_x42);
        assert_eq!(
            not_registered_symbol, None,
            "A non-registered variable has no symbol."
        );

        let x2_from_its_symbol = factor_graph.key(&X2);
        assert_eq!(
            x2_from_its_symbol,
            Some(x2),
            "A registered variable can be accessed via its symbol."
        );
        let x42_from_its_symbol = factor_graph.key(&X42);
        assert_eq!(
            x42_from_its_symbol, None,
            "A non-registered variable cannot be accessed via its symbol."
        );

        let sizex3 = factor_graph.size(x3);
        assert_eq!(
            sizex3,
            Some(1_000_000),
            "The size of a registered variable is accessible from its key."
        );
        let size_not_registered_x42 = factor_graph.size(not_registered_x42);
        assert_eq!(
            size_not_registered_x42, None,
            "The size of a non-registered variable is not accessible."
        );

        factor_graph
            .add_prior(x1, DVector::zeros(1), DMatrix::identity(1, 1))
            .expect("Adding a prior on a registered variable must work.");
        factor_graph
            .add_prior(x2, DVector::zeros(2), DMatrix::identity(3, 3))
            .expect_err("Adding a prior with incorrect dimensions must fail.");
        factor_graph
            .add_prior(x2, DVector::zeros(3), DMatrix::identity(3, 2))
            .expect_err("Adding a prior with incorrect dimensions must fail.");
        factor_graph
            .add_prior(
                not_registered_x42,
                DVector::zeros(2),
                DMatrix::identity(2, 2),
            )
            .expect_err("Adding a prior on a non-registered variable must fail.");

        factor_graph
            .add_observation(
                vec![x1, x2],
                DMatrix::from_element(2, 4, 1.0),
                DVector::zeros(2),
                DMatrix::identity(2, 2),
            )
            .expect("Adding a relation between registered variables must work.");
        factor_graph
            .add_observation(
                vec![x2, x1],
                DMatrix::from_element(2, 4, 1.0),
                DVector::zeros(2),
                DMatrix::identity(2, 2),
            )
            .expect_err("Adding an observation with unsorted keys must fail.");
        factor_graph
            .add_observation(
                vec![x1, x2, not_registered_x42],
                DMatrix::from_element(3, 5, 1.0),
                DVector::zeros(3),
                DMatrix::identity(3, 3),
            )
            .expect_err("Adding an observation with unregistered variables must fail.");

        assert_eq!(
            factor_graph.n_factors(),
            2,
            "There should be 2 factors in total: 1 prior and 1 likelihood."
        );
    }

    #[test]
    fn test_likelihood_from_closure() {
        let sizes = vec![2, 2, 3, 2, 1];
        let size_of = |k| sizes[k];

        let key0 = 2;
        let wfx0 = dmatrix![1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, 1.0];
        let xifx0 = dvector![0.6, 0.6, 0.6];
        let f0 = Prior {
            key: key0,
            xifx: xifx0,
            wfx: wfx0,
        };

        let keys1 = vec![0, 1, 2];
        let layout1 = Layout::new(keys1, size_of);
        let a1 = dmatrix![
            1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.1;
            0.0, 1.0, 0.0, 0.0, 0.0, 1.2, -1.0;
            0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0
        ];
        let wby1 = dmatrix![1.0, 0.2, 0.0; 0.2, 1.0, 0.0; 0.0, 0.0, 1.0];
        let xiby1 = dvector![0.4, 1.04, -0.5];
        let f1 = Likelihood {
            layout: layout1,
            a: a1,
            wby: wby1,
            xiby: xiby1,
        };

        let keys2 = vec![1, 3, 4];
        let layout2 = Layout::new(keys2, size_of);
        let a2 = dmatrix![
            0.0, 0.2, 0.0, 0.0, 0.2;
            0.8, 0.0, 0.0, 1.0, 0.8
        ];
        let wby2 = dmatrix![0.8, 0.1; 0.1, 1.0];
        let xiby2 = dvector![0.05, 0.5];
        let f2 = Likelihood {
            layout: layout2,
            a: a2,
            wby: wby2,
            xiby: xiby2,
        };

        let factors = vec![
            FactorRef::Prior(&f0),
            FactorRef::Likelihood(&f1),
            FactorRef::Likelihood(&f2),
        ];
        let kept_keys = vec![0, 1, 3];
        let cleared_keys = vec![2, 4];
        let new_factor =
            Likelihood::from_closure(&factors, kept_keys, cleared_keys, size_of).unwrap();

        assert_eq!(new_factor.keys(), [0, 1, 3]);
        assert!(new_factor.dim_y() <= new_factor.dim_x());

        assert!((new_factor.xiby[0] - -0.18566446904015022f64).abs() < 1e-9);
        assert!((new_factor.xiby[1] - 0.2564704789814862f64).abs() < 1e-9);
        assert!((new_factor.xiby[2] - -0.5849271856215628f64).abs() < 1e-9);
        assert!((new_factor.xiby[3] - -0.08977272727272727f64).abs() < 1e-9);
        assert!((new_factor.xiby[4] - 0.02244318181818189f64).abs() < 1e-9);

        assert!((new_factor.wby[0] - 0.49401045386071685f64).abs() < 1e-9);
        assert!((new_factor.wby[2] - -0.005219289506583182).abs() < 1e-9);
        assert!((new_factor.wby[7] - -0.0347294345005616).abs() < 1e-9);
        assert!((new_factor.wby[13] - 0f64).abs() < 1e-9);
        assert!((new_factor.wby[19] - -0.17954545454545454f64).abs() < 1e-9);
        assert!((new_factor.wby[24] - 0.04488636363636378f64).abs() < 1e-9);
    }

    #[test]
    fn test_likelihood_from_closure_loop() {
        let sizes = vec![1, 1, 1];
        let size_of = |k| sizes[k];

        let f0 = Prior {
            key: 1,
            xifx: dvector![0.2],
            wfx: dmatrix![1.0],
        };

        let f1 = Likelihood {
            layout: Layout::new(vec![0, 1], size_of),
            a: dmatrix![1.0, 1.0],
            wby: dmatrix![0.6],
            xiby: dvector![0.06],
        };
        let f2 = Likelihood {
            layout: Layout::new(vec![0, 2], size_of),
            a: dmatrix![1.0, 1.0],
            wby: dmatrix![1.0],
            xiby: dvector![0.8],
        };

        let f3 = Likelihood {
            layout: Layout::new(vec![1, 2], size_of),
            a: dmatrix![1.0, 1.0],
            wby: dmatrix![0.5],
            xiby: dvector![0.5],
        };

        let factors = vec![
            FactorRef::Prior(&f0),
            FactorRef::Likelihood(&f1),
            FactorRef::Likelihood(&f2),
            FactorRef::Likelihood(&f3),
        ];
        let kept_keys = vec![0];
        let cleared_keys = vec![1, 2];

        // the total observation dimension is larger than the total dimension of the kept variables
        // the factor should close a
        let new_factor =
            Likelihood::from_closure(&factors, kept_keys, cleared_keys, size_of).unwrap();

        assert!(new_factor.dim_y() <= new_factor.dim_x());
        assert_eq!(new_factor.keys(), [0]);
        assert!((new_factor.a[0] - 1f64).abs() < 1e-9);
        assert!((new_factor.xiby[0] - -0.051724137931034475f64).abs() < 1e-9);
        assert!((new_factor.wby[0] - 0.8965517241379309f64).abs() < 1e-9);
    }
}

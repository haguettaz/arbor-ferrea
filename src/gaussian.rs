use std::collections::HashMap;
use std::fmt::Debug;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DMatrixView, DVector, RealField};

use anyhow::{Context, Result, bail, ensure};

#[derive(Debug)]
pub struct Marginal<T: RealField> {
    pub key: usize,
    pub m: DVector<T>,
    pub v: DMatrix<T>,
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
                .context(format!("Cholesky failed while computing the marginal. Consider adding dumb priors for variable with {key}"))?;

        let m = chol_w.solve(&xi);
        let v = chol_w.inverse();

        Ok(Marginal { key, m, v })
    }
}

#[derive(Debug)]
pub struct Prior<T: RealField> {
    pub key: usize,
    pub xifx: DVector<T>,
    pub wfx: DMatrix<T>,
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
    pub keys: Vec<usize>,
    pub layout: Vec<(usize, usize)>,
    pub a: DMatrix<T>,
    pub xiby: DVector<T>,
    pub wby: DMatrix<T>,
}

impl<T: RealField> Likelihood<T> {
    pub fn keys(&self) -> &[usize] {
        &self.keys
    }

    pub fn has_key(&self, key: usize) -> bool {
        match self.keys.binary_search(&key) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Returns a view on the block column of the variable `key`, or `None` if absent.
    pub fn a(&self, key: &usize) -> Option<DMatrixView<'_, T>> {
        match self.keys.binary_search(key) {
            Ok(i) => {
                let (icol, ncols) = self.layout[i];
                Some(self.a.columns(icol, ncols))
            }
            Err(_) => None,
        }
    }

    pub fn layout(&self, key: &usize) -> Option<(usize, usize)> {
        match self.keys.binary_search(key) {
            Ok(i) => Some(self.layout[i]),
            Err(_) => None,
        }
    }

    pub fn dim_x(&self) -> usize {
        self.a.ncols()
    }

    pub fn dim_y(&self) -> usize {
        self.a.nrows()
    }

    /// Create a new  likelihood by aggregating observation models and prior models and eliminating some variables.
    /// It is the caller responsability to check that the kept and cleared keys are compatible and valid with factors.
    /// This is equivalent to max-product and sum-product message passing for closing box.
    /// Warning: for performance reason, priors on kept variables are forbidden.
    /// Warning: if necessary, a weak Gaussian prior on eliminated variables is added (Tikhonov regularization)
    /// while computing the Cholesky decomposition.
    pub fn from_closure<'a, C>(
        factors: &[FactorRef<T>],
        kept_keys: &[usize],
        cleared_keys: &[usize],
        ctx: C,
    ) -> Result<Likelihood<T>>
    where
        C: Fn(usize) -> usize,
    {
        debug_assert!(cleared_keys.is_sorted(), "unsorted cleared keys");
        debug_assert!(kept_keys.is_sorted(), "unsorted kept keys");

        for f in factors.iter() {
            match f {
                FactorRef::Likelihood(likelihood) => {
                    for key in &likelihood.keys {
                        match (
                            kept_keys.binary_search(&key),
                            cleared_keys.binary_search(&key),
                        ) {
                            (Err(_), Err(_)) => bail!("Likelihood key {} does not exist.", key),
                            _ => (),
                        }
                    }
                }
                FactorRef::Prior(prior) => {
                    let key = prior.key;
                    match (
                        kept_keys.binary_search(&key),
                        cleared_keys.binary_search(&key),
                    ) {
                        (Ok(_), _) => bail!("Prior on a kept key is forbidden."),
                        (Err(_), Err(_)) => bail!("Prior key {} does not exist.", key),
                        _ => (),
                    }
                }
            }
        }

        // Setup dimensions and layout context
        let mut kept_layout: Vec<(usize, usize)> = Vec::with_capacity(kept_keys.len());
        let mut offset = 0;
        for key in kept_keys {
            let size = ctx(*key);
            kept_layout.push((offset, size));
            offset += size;
        }
        let kept_layout_ctx = |key| match kept_keys.binary_search(key) {
            Ok(i) => Some(kept_layout[i]),
            Err(_) => None,
        };
        let dim_x = offset;

        let mut cleared_layout: Vec<(usize, usize)> = Vec::with_capacity(cleared_keys.len());
        let mut offset = 0;
        for key in cleared_keys {
            let size = ctx(*key);
            cleared_layout.push((offset, size));
            offset += size;
        }
        let cleared_layout_ctx = |key| match cleared_keys.binary_search(key) {
            Ok(i) => Some(cleared_layout[i]),
            Err(_) => None,
        };
        let dim_u = offset;

        let dim_y = factors
            .iter()
            .map(|f| match f {
                FactorRef::Likelihood(likelihood) => likelihood.dim_y(),
                FactorRef::Prior(_) => 0,
            })
            .sum();

        let mut a = DMatrix::zeros(dim_y, dim_x);
        let mut xiby = DVector::zeros(dim_y);
        let mut wby = DMatrix::zeros(dim_y, dim_y);
        let mut wby_b = DMatrix::zeros(dim_y, dim_u);
        // let mut bt_wby = DMatrix::zeros(dim_u, dim_y);
        let mut xifu_plus_bt_xiby = DVector::zeros(dim_u);
        let mut wfu_plus_bt_wby_b = DMatrix::zeros(dim_u, dim_u);

        // rows = observations and columns = keys
        let mut ypos = 0;
        for f in factors {
            match f {
                FactorRef::Prior(prior) => {
                    let (icol, ncols) = cleared_layout_ctx(&prior.key).with_context(|| {
                        format!("Key {} not found in cleared_layout_ctx", prior.key)
                    })?;

                    let mut tmp = xifu_plus_bt_xiby.rows_mut(icol, ncols);
                    tmp += &prior.xifx;

                    let mut tmp = wfu_plus_bt_wby_b.view_mut((icol, icol), (ncols, ncols));
                    tmp += &prior.wfx;
                }
                FactorRef::Likelihood(likelihood) => {
                    let ysize = likelihood.dim_y();

                    // update xiby and wby
                    let mut tmp = xiby.rows_mut(ypos, ysize);
                    tmp.copy_from(&likelihood.xiby);
                    let mut tmp = wby.view_mut((ypos, ypos), (ysize, ysize));
                    tmp.copy_from(&likelihood.wby);

                    // update a (related to kept keys)
                    for key in kept_keys {
                        if let Some(a_view) = likelihood.a(key) {
                            let (icol, ncols) = kept_layout_ctx(&key).with_context(|| {
                                format!("Key {} not found in kept_layout_ctx", key)
                            })?;
                            a.view_mut((ypos, icol), (ysize, ncols)).copy_from(&a_view);
                        }
                    }

                    // update wby_b, xifu_plus_bt_xiby, and wfu_plus_bt_wby_b (related to cleared keys)
                    for ukey in cleared_keys {
                        if let Some(ub) = likelihood.a(ukey) {
                            let (upos, usize) = cleared_layout_ctx(&ukey).with_context(|| {
                                format!("Key {} not found in cleared_layout_ctx", ukey)
                            })?;

                            // update wby_b
                            let mut wby_ub = wby_b.view_mut((ypos, upos), (ysize, usize));
                            wby_ub.gemm(T::one(), &likelihood.wby, &ub, T::one());

                            // update wfu_plus_bt_wby_b (only the lower triangular part)
                            for vkey in cleared_keys {
                                if vkey >= ukey {
                                    if let Some(vb) = likelihood.a(vkey) {
                                        let (vpos, vsize) = cleared_layout_ctx(&vkey)
                                            .with_context(|| {
                                                format!(
                                                    "Key {} not found in cleared_layout_ctx",
                                                    vkey
                                                )
                                            })?;

                                        let mut tmp = wfu_plus_bt_wby_b
                                            .view_mut((vpos, upos), (vsize, usize));
                                        tmp.gemm_tr(T::one(), &vb, &wby_ub, T::one());
                                    }
                                }
                            }

                            // update xifu_plus_bt_xiby
                            let mut tmp = xifu_plus_bt_xiby.rows_mut(upos, usize);
                            tmp.gemv_tr(T::one(), &ub, &likelihood.xiby, T::one());
                        }
                    }

                    ypos += ysize;
                }
            }
        }

        let chol_wfu_plus_bt_wby_b =
            Cholesky::new(wfu_plus_bt_wby_b)
                .context(format!("Cholesky failed: cleared variables are under-constrained. Consider adding dumb priors for variables {:?}", cleared_keys))?;
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
            keys: Vec::from(kept_keys),
            layout: kept_layout,
            a,
            xiby,
            wby,
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

        let mut offset = 0;
        let mut layout = Vec::with_capacity(keys.len());
        for key in &keys {
            let size = self
                .size(*key)
                .with_context(|| format!("Size of {} not found in factor graph context.", key))?;
            layout.push((offset, size));
            offset += size;
        }
        let dim_x = offset;

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
            keys,
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
        let ctx = |k| sizes[k];

        let key0 = 2;
        let wfx0 = dmatrix![1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, 1.0];
        let xifx0 = dvector![0.6, 0.6, 0.6];
        let f0 = Prior {
            key: key0,
            xifx: xifx0,
            wfx: wfx0,
        };

        let keys1 = vec![0, 1, 2];
        let a1 = dmatrix![
            1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.1;
            0.0, 1.0, 0.0, 0.0, 0.0, 1.2, -1.0;
            0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0
        ];
        let layout1 = vec![(0, 2), (2, 2), (4, 3)];
        let wby1 = dmatrix![1.0, 0.2, 0.0; 0.2, 1.0, 0.0; 0.0, 0.0, 1.0];
        let xiby1 = dvector![0.4, 1.04, -0.5];
        let f1 = Likelihood {
            keys: keys1,
            layout: layout1,
            a: a1,
            wby: wby1,
            xiby: xiby1,
        };

        let keys2 = vec![1, 3, 4];
        let a2 = dmatrix![
            0.0, 0.2, 0.0, 0.0, 0.2;
            0.8, 0.0, 0.0, 1.0, 0.8
        ];
        let layout2 = vec![(0, 2), (2, 2), (4, 1)];
        let wby2 = dmatrix![0.8, 0.1; 0.1, 1.0];
        let xiby2 = dvector![0.05, 0.5];
        let f2 = Likelihood {
            keys: keys2,
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
            Likelihood::from_closure(&factors, &kept_keys, &cleared_keys, ctx).unwrap();

        assert_eq!(new_factor.keys, kept_keys);

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
        let ctx = |k| sizes[k];

        let f0 = Prior {
            key: 1,
            xifx: dvector![0.2],
            wfx: dmatrix![1.0],
        };

        let f1 = Likelihood {
            keys: vec![0, 1],
            layout: vec![(0, 1), (1, 1)],
            a: dmatrix![1.0, 1.0],
            wby: dmatrix![0.6],
            xiby: dvector![0.06],
        };
        let f2 = Likelihood {
            keys: vec![0, 2],
            layout: vec![(0, 1), (1, 1)],
            a: dmatrix![1.0, 1.0],
            wby: dmatrix![1.0],
            xiby: dvector![0.8],
        };

        let f3 = Likelihood {
            keys: vec![1, 2],
            layout: vec![(0, 1), (1, 1)],
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
        let new_factor =
            Likelihood::from_closure(&factors, &kept_keys, &cleared_keys, ctx).unwrap();

        assert_eq!(new_factor.keys, kept_keys);

        assert!((new_factor.xiby[0] - -0.04137931f64).abs() < 1e-6);
        assert!((new_factor.xiby[1] - -0.01034483f64).abs() < 1e-6);
        assert!((new_factor.xiby[2] - 0.01034483f64).abs() < 1e-6);

        assert!((new_factor.wby[0] - 0.4137931f64).abs() < 1e-6);
        assert!((new_factor.wby[2] - -0.10344828f64).abs() < 1e-6);
        assert!((new_factor.wby[4] - 0.27586207f64).abs() < 1e-6);
        assert!((new_factor.wby[7] - -0.27586207f64).abs() < 1e-6);
    }
}

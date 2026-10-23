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
use itertools::izip;
use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView, RealField};
use std::collections::{HashMap, HashSet};

use crate::core::*;

#[derive(Debug, PartialEq)]
pub struct GaussianFactor<T> {
    id: usize,
    keys: Vec<usize>,
    // key_to_size: HashMap<usize, usize>,
    a_vec: Vec<DMatrix<T>>,
    // a: DMatrix<T>,
    // layout: Vec<(usize, usize)>, // aligned with keys, indicate the location of the keys in the columns of a
    xiby: DVector<T>,
    wby: DMatrix<T>,
}

impl<T: RealField> GaussianFactor<T> {
    /// Returns the layout of the variable.
    // fn layout(&self, key: usize) -> Option<(usize, usize)> {
    //     let idx = self.keys.iter().position(|&k| k == key)?;
    //     self.layout.get(idx).copied()
    // }
    //

    pub fn key_to_pos(&self, key: usize) -> Option<usize> {
        self.keys.iter().position(|&k| k == key)
    }

    /// Returns a view on the block column of the variable `key`, or `None` if absent.
    pub fn a_view(&self, key: usize) -> Option<DMatrixView<'_, T>> {
        // self.layout(key)
        //     .map(|(icol, ncols)| self.a.columns(icol, ncols))
        self.key_to_pos(key).map(|idx| self.a_vec[idx].as_view())
        // Some(self.a[idx].as_view())
    }

    /// Returns a view on the observation weighted mean vector.
    pub fn xiby_view(&self) -> DVectorView<'_, T> {
        self.xiby.as_view()
    }

    /// Returns a view on the observation precision matrix.
    pub fn wby_view(&self) -> DMatrixView<'_, T> {
        self.wby.as_view()
    }

    /// Returns the observation dimension.
    pub fn dim_y(&self) -> usize {
        self.xiby.nrows()
    }

    // pub fn from() -> Result<GaussianFactor<T>> {}
}

impl<T> FactorBase for GaussianFactor<T> {
    fn keys(&self) -> &[usize] {
        &self.keys
    }
}

impl<T: RealField> Factor for GaussianFactor<T> {
    type TargetMessage = GaussianMessage<T>;
    fn id(&self) -> usize {
        self.id
    }
}

// pub struct LinearModel<T> {
//     keys: Vec<usize>,
//     mats: Vec<DMatrix<T>>,
// }
//
// impl<T> LinearModel<T> {
//     pub fn rmult(&self, other: &DMatrixView<T>, on_keys: &[usize]) -> Result<DMatrixView<T>> {
//         other * b
//     }
// }

#[derive(Clone, Debug)]
pub struct GaussianMessage<T> {
    keys: Vec<usize>,
    a_vec: Vec<DMatrix<T>>, // aligned with keys
    // a_layout: Vec<(usize, usize)>, // aligned with keys
    xiby: DVector<T>,
    wby: DMatrix<T>,
    closed_factors: Vec<usize>,
}

impl<T: RealField> GaussianMessage<T> {
    // /// Returns the layout of the variable.
    // fn layout(&self, key: usize) -> Option<(usize, usize)> {
    //     let idx = self.keys.iter().position(|&k| k == key)?;
    //     self.a_layout.get(idx).copied()
    // }
    pub fn key_to_pos(&self, key: usize) -> Option<usize> {
        self.keys.iter().position(|&k| k == key)
    }

    /// Returns a view on the matrix associated with the variable `key`, or `None` if absent.
    pub fn a_view(&self, key: usize) -> Option<DMatrixView<'_, T>> {
        self.key_to_pos(key).map(|idx| self.a_vec[idx].as_view())
        // self.layout(key)
        //     .map(|(icol, ncols)| self.a.columns(icol, ncols))
    }

    /// Create a new Gaussian message by aggregating observation models and prior models and eliminating some variables.
    pub fn from<C>(
        obs_models: &[&GaussianObservation<T>],
        prior_models: &[Option<&GaussianPrior<T>>],
        keys: &[usize],
        on_keys: &[usize],
        closed_factors: &[usize],
        size_ctx: C,
    ) -> Result<GaussianMessage<T>>
    where
        C: Fn(usize) -> usize,
    {
        let dim_u = on_keys.iter().map(|&key| size_ctx(key)).sum();
        let dim_y = obs_models.iter().map(|&obs_model| obs_model.dim_y()).sum();

        // Allocate temporary matrices and vectors
        let mut xifu_plus_bt_xiby = DVector::zeros(dim_u);
        let mut wfu_plus_bt_wby_b = DMatrix::zeros(dim_u, dim_u);

        // Load data from the prior models
        let mut pos_key = 0;
        for (key, prior_model) in izip!(on_keys.iter(), prior_models.iter()) {
            let size_key = size_ctx(*key);
            if let Some(prior) = prior_model {
                let xifu = prior.xifx_view();
                xifu_plus_bt_xiby
                    .rows_mut(pos_key, size_key)
                    .copy_from(xifu);

                let wfu = prior.wfx_view();
                wfu_plus_bt_wby_b
                    .view_mut((pos_key, pos_key), (size_key, size_key))
                    .copy_from(wfu);
            }
            pos_key += size_key;
        }

        // Collect the matrices of remaining variables
        let mut a_vec = Vec::with_capacity(keys.len());
        let (mut pos_obs, mut pos_key) = (0, 0);
        for key in keys.iter() {
            let size_key = size_ctx(*key);
            let mut a_key = DMatrix::zeros(dim_y, size_key);
            pos_obs = 0;
            for obs_model in obs_models.iter() {
                let size_obs = obs_model.dim_y();
                if let Some(a) = obs_model.a_view(key) {
                    a_key.rows_mut(pos_obs, size_obs).copy_from(a);
                }
                pos_obs += size_obs;
            }
            pos_key += size_key;
            a_vec.push(a_key);
        }

        // Load xiby from observation models
        let mut xiby = DVector::zeros(dim_y);
        let mut pos_obs = 0;
        for obs_model in obs_models.iter() {
            let size_obs = obs_model.dim_y();
            xiby.rows_mut(pos_obs, size_obs)
                .copy_from(obs_model.xiby_view());
            pos_obs += size_obs;
        }

        // Load wby_b from observation models
        let mut wby_b = DMatrix::zeros(dim_y, dim_u);
        let (mut pos_obs, mut pos_key) = (0, 0);
        for key in on_keys.iter() {
            let size_key = size_ctx(*key);
            pos_obs = 0;
            for obs_model in obs_models.iter() {
                let size_obs = obs_model.dim_y();
                if let Some(b) = obs_model.a_view(key) {
                    wby_b
                        .view_mut((pos_obs, pos_key), (size_obs, size_key))
                        .gemm(T::one(), obs_model.wby_view(), b, T::one());
                }
                pos_obs += size_obs;
            }
            pos_key += size_key;
        }

        // Add bt_xiby to allocated xifu_plus_bt_xiby already containing xifu
        // Add bt_wby_b to allocated wfu_plus_bt_wby_b already containing wfu
        let mut pos_key = 0;
        for key in on_keys.iter() {
            let size_key = size_ctx(*key);
            pos_obs = 0;
            for obs_model in obs_models.iter() {
                let size_obs = obs_model.dim_y();
                if let Some(b) = obs_model.a_view(key) {
                    xifu_plus_bt_xiby.rows_mut(pos_key, size_key).gemm_tr(
                        T::one(),
                        b,
                        &wby_b.rows(pos_obs, size_obs),
                        T::one(),
                    );
                    wfu_plus_bt_wby_b.rows_mut(pos_key, size_key).gemm_tr(
                        T::one(),
                        b,
                        &wby_b.rows(pos_obs, size_obs),
                        T::one(),
                    );
                }
                pos_obs += size_obs;
            }
            pos_key += size_key;
        }

        // Solve ( wfu + bt * wby * b ) * h = bt * wby via Cholesky decomposition
        let chol_wfu_plus_bt_wby_b = wfu_plus_bt_wby_b
            .cholesky()
            .with_context(|| format!("Cholesky decomposition failed"))?;

        let bt_wby = wby_b.transpose();
        let h = chol_wfu_plus_bt_wby_b.solve(&wby_b.transpose());

        // Compute:
        //  xibz = xiby - ht (xifu + bt xiby)
        //  wby = wby - ht bt wby
        let mut xibz = DVector::zeros(dim_y);
        let mut wbz = DMatrix::zeros(dim_y, dim_y);
        let mut pos_obs = 0;
        for obs_model in obs_models.iter() {
            let size_obs = obs_model.dim_y();
            xibz.rows_mut(pos_obs, size_obs)
                .copy_from(obs_model.xiby_view());
            wbz.view_mut((pos_obs, pos_obs), (size_obs, size_obs))
                .copy_from(obs_model.wby_view());
            pos_obs += size_obs;
        }
        xibz.gemm_tr(T::one().neg(), &h, &xifu_plus_bt_xiby, T::one());
        wbz.gemm_tr(T::one().neg(), &h, &bt_wby, T::one());

        let msg = GaussianMessage {
            keys: keys.to_vec(),
            a_vec,
            xiby: xibz,
            wby: wbz,
            closed_factors: closed_factors.to_vec(),
        };
        Ok(msg)
    }

    /// Returns a view on the observation weighted mean vector.
    pub fn xiby_view(&self) -> DVectorView<'_, T> {
        self.xiby.as_view()
    }

    /// Returns a view on the observation precision matrix.
    pub fn wby_view(&self) -> DMatrixView<'_, T> {
        self.wby.as_view()
    }

    /// Returns the observation dimension.
    pub fn dim_y(&self) -> usize {
        self.xiby.nrows()
    }
}

// Base implementation for Message
impl<T> FactorBase for GaussianMessage<T> {
    fn keys(&self) -> &[usize] {
        &self.keys
    }
}

// // Forward FactorBase methods through the enum
// impl<T> FactorBase for GaussianFactor<T> {
//     fn keys(&self) -> &[usize] {
//         match self {
//             Self::Prior(prior) => prior.keys(),
//             Self::Observation(obs) => obs.keys(),
//         }
//     }
// }
//
// // Forward Factor methods through the enum
// impl<T: RealField> Factor for GaussianFactor<T> {
//     type TargetMessage = GaussianMessage<T>;
//
//     fn id(&self) -> usize {
//         match self {
//             Self::Prior(prior) => prior.id(),
//             Self::Observation(obs) => obs.id(),
//         }
//     }
// }

impl<T: RealField> Message for GaussianMessage<T> {
    type TargetFactor = GaussianFactor<T>;

    fn closed_factors(&self) -> &[usize] {
        &self.closed_factors
    }

    fn close<'a, I, K, C>(factors: I, on_keys: K, ctx: C) -> Result<Self>
    where
        I: IntoIterator<Item = AnyFactor<'a, Self::TargetFactor>>,
        K: IntoIterator<Item = usize>,
        C: Fn(usize) -> usize,
        Self::TargetFactor: 'a,
    {
        let factors: Vec<_> = factors.into_iter().collect();
        let on_keys: HashSet<usize> = on_keys.into_iter().collect();
        let mut rem_keys = HashSet::new();

        let mut closed_factors = Vec::new();

        // Aggregate all keys and track which factors/messages are being closed
        for factor in &factors {
            for &key in factor.keys() {
                rem_keys.insert(key);
            }

            match factor {
                AnyFactor::BorrowedFactor(fct) => {
                    closed_factors.push(fct.id());
                }

                AnyFactor::OwnedMessage(msg) => {
                    closed_factors.extend_from_slice(msg.closed_factors());
                }
            }
        }

        // Remove the keys to eliminate
        for key in &on_keys {
            rem_keys.remove(key);
        }
        let mut rem_keys: Vec<usize> = rem_keys.into_iter().collect();
        rem_keys.sort_unstable();
        let mut on_keys: Vec<usize> = on_keys.into_iter().collect();
        on_keys.sort_unstable();

        // Compute the dimension of the observation (aggregated overall factors)
        let mut dim_y = 0;
        for factor in &factors {
            match factor {
                AnyFactor::BorrowedFactor(fct) => {
                    dim_y += fct.dim_y();
                }

                AnyFactor::OwnedMessage(msg) => {
                    closed_factors.extend_from_slice(msg.closed_factors());
                    dim_y += msg.dim_y();
                }
            }
        }

        let mut a_layout = Vec::with_capacity(rem_keys.len());
        let mut icol = 0;
        for &key in &rem_keys {
            let ncols = ctx(key);
            a_layout.push((icol, ncols));
            icol += ncols;
        }
        let dim_x = icol;
        let mut a = DMatrix::zeros(dim_y, dim_x);

        let mut b_layout = Vec::with_capacity(on_keys.len());
        let mut col = 0;
        for &key in &on_keys {
            let ncols = ctx(key);
            b_layout.push((col, ncols));
            col += ncols;
        }
        let dim_u = col;
        let mut b = DMatrix::zeros(dim_y, dim_u);

        // Initialize the observations and associated precisions
        let mut xiby = DVector::zeros(dim_y);
        let mut wby = DMatrix::zeros(dim_y, dim_y);

        let mut irow = 0;
        for factor in factors {
            match factor {
                AnyFactor::BorrowedFactor(fct) => {
                    let nrows = fct.dim_y();

                    // Fill the matrix A related to the remaining variables
                    for (key, (icol, ncols)) in izip!(rem_keys.iter(), a_layout.iter()) {
                        if let Some(mat) = fct.a_view(*key) {
                            ensure!(
                                mat.ncols() == *ncols,
                                "Factor block for variable {} has {} columns, expected {}",
                                key,
                                mat.ncols(),
                                ncols
                            );
                            a.view_mut((irow, *icol), (nrows, *ncols)).copy_from(&mat);
                        }
                    }

                    // Fill the matrix B related to the eliminated variables
                    for (key, (icol, ncols)) in izip!(on_keys.iter(), b_layout.iter()) {
                        if let Some(mat) = fct.a_view(*key) {
                            ensure!(
                                mat.ncols() == *ncols,
                                "Factor block for variable {} has {} columns, expected {}",
                                key,
                                mat.ncols(),
                                ncols
                            );
                            b.view_mut((irow, *icol), (nrows, *ncols)).copy_from(&mat);
                        }
                    }

                    // Fill the observation and precisions
                    xiby.rows_mut(irow, nrows).copy_from(&fct.xiby_view());
                    wby.view_mut((irow, irow), (nrows, nrows))
                        .copy_from(&fct.wby_view());
                    irow += nrows;
                }
                AnyFactor::OwnedMessage(msg) => {
                    let nrows = msg.dim_y();

                    // Fill the matrix A related to the remaining variables
                    for (key, (icol, ncols)) in izip!(rem_keys.iter(), a_layout.iter()) {
                        if let Some(mat) = msg.a_view(*key) {
                            ensure!(
                                mat.ncols() == *ncols,
                                "Factor block for variable {} has {} columns, expected {}",
                                key,
                                mat.ncols(),
                                ncols
                            );
                            a.view_mut((irow, *icol), (nrows, *ncols)).copy_from(&mat);
                        }
                    }

                    // Fill the matrix B related to the eliminated variables
                    for (key, (icol, ncols)) in izip!(on_keys.iter(), b_layout.iter()) {
                        if let Some(mat) = msg.a_view(*key) {
                            ensure!(
                                mat.ncols() == *ncols,
                                "Factor block for variable {} has {} columns, expected {}",
                                key,
                                mat.ncols(),
                                ncols
                            );
                            b.view_mut((irow, *icol), (nrows, *ncols)).copy_from(&mat);
                        }
                    }

                    // Fill the observation and precisions
                    xiby.rows_mut(irow, nrows).copy_from(&msg.xiby_view());
                    wby.view_mut((irow, irow), (nrows, nrows))
                        .copy_from(&msg.wby_view());
                    irow += nrows;
                }
            }
        }

        let mut xitu = DVector::zeros(dim_u);
        b.tr_mul_to(&xiby, &mut xitu); // xitu = b.t xiby

        let mut bt_wby = DMatrix::zeros(dim_u, dim_y);
        b.tr_mul_to(&wby, &mut bt_wby);

        let mut wtu = DMatrix::from_diagonal_element(dim_u, dim_u, T::default_epsilon()); // ensure wtx is positive definite
        wtu.gemm(T::one(), &bt_wby, &b, T::one()); // wtx = a.t wby a
        let rtu = wtu
            .cholesky()
            .with_context(|| format!("Cholesky decomposition failed"))?;

        let mut mtu = xitu.clone();
        rtu.solve_mut(&mut mtu); // wtu mtu = xitu

        let mut vtu_bt_wby = bt_wby.clone();
        rtu.solve_mut(&mut vtu_bt_wby);

        let mut xibz = xiby.clone(); // xibz = xiby - wby b mtu
        xibz.gemv_tr(T::one().neg(), &bt_wby, &mtu, T::one());

        let mut wbz = wby.clone(); // wbz = wby - wby b vtu b.t wby
        wbz.gemm_tr(T::one().neg(), &bt_wby, &vtu_bt_wby, T::one());

        let msg = GaussianMessage {
            keys: rem_keys,
            a,
            a_layout,
            xiby: xibz,
            wby: wbz,
            closed_factors,
        };

        Ok(msg)
    }
}

use anyhow::{Context, Result, ensure};
use nalgebra::{Cholesky, DMatrix, DVector, RealField};

use crate::graph::{FactorGraph, factor::Factor};

pub struct Marginals<T: RealField> {
    keys: Vec<usize>,
    means: Vec<DVector<T>>,
    covs: Vec<DMatrix<T>>,
}

impl<T: RealField> Marginals<T> {
    pub(crate) fn new() -> Self {
        let keys = Vec::new();
        let means = Vec::new();
        let covs = Vec::new();

        Self { keys, means, covs }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let keys = Vec::with_capacity(capacity);
        let means = Vec::with_capacity(capacity);
        let covs = Vec::with_capacity(capacity);

        Self { keys, means, covs }
    }

    pub(crate) fn add_from(&mut self, key: usize, factor: &Factor<T>) -> Result<()> {
        ensure!(
            factor.keys() == [key],
            "factor must depend on key only; got key {} and factor keys {:?}",
            key,
            factor.keys()
        );

        let (xibx, wbx) = (factor.xibx(), factor.wbx());

        let chol = Cholesky::new(wbx)
            .with_context(|| format!("cholesky failed: variable {key} is under-constrained",))?;

        let mx = chol.solve(&xibx);
        let vx = chol.inverse();

        self.keys.push(key);
        self.means.push(mx);
        self.covs.push(vx);

        Ok(())
    }
}

pub struct Marginal<T: RealField> {
    key: usize,
    mean: DVector<T>,
    cov: DMatrix<T>,
}

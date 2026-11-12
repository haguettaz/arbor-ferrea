use std::collections::HashMap;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, RealField};

use anyhow::{Context, Result, bail, ensure};

pub(crate) mod factor;
pub(crate) mod layout;
pub(crate) mod symbol;

use factor::Factor;
use layout::Layout;
use symbol::Symbol;

pub struct FactorGraph<T: RealField> {
    factors: Vec<Factor<T>>,
    symbols: Vec<Symbol>,
    sizes: Vec<usize>,
    symbol_to_key: HashMap<Symbol, usize>,
}

impl<T: RealField> FactorGraph<T> {
    pub fn new() -> Self {
        Self {
            factors: Vec::new(),
            symbols: Vec::new(),
            sizes: Vec::new(),
            symbol_to_key: HashMap::new(),
        }
    }

    pub fn n_keys(&self) -> usize {
        self.symbols.len()
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = usize> {
        0..self.symbols.len()
    }

    pub fn key(&self, symbol: &Symbol) -> Option<usize> {
        self.symbol_to_key.get(symbol).copied()
    }

    pub fn has_key(&self, key: usize) -> bool {
        key < self.symbols.len()
    }

    pub fn symbol_of(&self, key: usize) -> Option<&Symbol> {
        self.symbols.get(key)
    }

    pub fn size_of(&self, key: usize) -> Option<usize> {
        self.sizes.get(key).copied()
    }

    pub fn sizes(&self) -> impl DoubleEndedIterator<Item = usize> {
        self.sizes.iter().copied()
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

        let id = self.symbols.len();
        self.symbol_to_key.insert(symbol, id);
        self.symbols.push(symbol);
        self.sizes.push(size);
        Ok(id)
    }

    pub fn add_prior(&mut self, key: usize, mean: DVector<T>, cov: DMatrix<T>) -> Result<usize> {
        let dim = self
            .size_of(key)
            .with_context(|| format!("register variable {} before adding a prior on it.", key))?;

        ensure!(
            (mean.nrows() == dim),
            format!(
                "Expected mean vector to have shape ({}, ), got ({}, ).",
                dim,
                mean.nrows(),
            )
        );
        ensure!(
            (cov.nrows() == dim) && (cov.ncols() == dim),
            format!(
                "Expected precision matrix to have shape ({}, {}), got ({}, {}).",
                dim,
                dim,
                cov.nrows(),
                cov.ncols(),
            )
        );

        let layout = Layout::new(vec![key], |k| self.size_of(k).unwrap_or(0));
        let a = DMatrix::identity(dim, dim);

        let chol = Cholesky::new(cov).with_context(|| "Cholesky factorization failed")?;
        let xiby = chol.solve(&mean);
        let wby = chol.inverse();

        self.factors.push(Factor::new(layout, a, xiby, wby));

        Ok(self.factors().len() - 1)
    }

    pub fn add_observation(
        &mut self,
        keys: Vec<usize>,
        a: DMatrix<T>,
        mby: DVector<T>,
        vby: DMatrix<T>,
    ) -> Result<usize> {
        ensure!(keys.is_sorted(), "Keys must be sorted.");
        ensure!(
            keys.iter().all(|&key| self.has_key(key)),
            "The provided observation concerns variables that have not been registered yet.",
        );

        let layout = Layout::new(keys, |k| self.size_of(k).unwrap_or(0));

        let dim_x = layout.dim();
        let dim_y = mby.nrows();

        // check vby shape
        ensure!(
            (vby.nrows() == dim_y) && (vby.ncols() == dim_y),
            format!(
                "Expected precision matrix to have shape ({}, {}), got ({}, {}).",
                dim_y,
                dim_y,
                vby.nrows(),
                vby.ncols(),
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

        let chol_vby = Cholesky::new(vby).with_context(|| "Cholesky factorization failed")?;
        let xiby = chol_vby.solve(&mby);
        let wby = chol_vby.inverse();

        self.factors.push(Factor::new(layout, a, xiby, wby));

        Ok(self.factors.len() - 1)
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

        let symbolx1 = factor_graph.symbol_of(x1);
        assert_eq!(
            symbolx1,
            Some(&X1),
            "A registered variable has an accessible symbol."
        );
        let not_registered_symbol = factor_graph.symbol_of(not_registered_x42);
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

        let sizex3 = factor_graph.size_of(x3);
        assert_eq!(
            sizex3,
            Some(1_000_000),
            "The size of a registered variable is accessible from its key."
        );
        let size_not_registered_x42 = factor_graph.size_of(not_registered_x42);
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
}

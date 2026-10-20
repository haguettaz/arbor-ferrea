//! The main entry point to the crate.

use anyhow::{Context, Result, bail, ensure};
use nalgebra::{DMatrix, DVector};
use std::{
    collections::{HashMap, HashSet},
    iter::zip,
};

use crate::gaussian::{GaussianFactor, GaussianTree};
use crate::numeric::{NumericContext, NumericTree};
use crate::symbolic::SymbolicTree;

/// Variable symbol
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub char, pub usize);

/// Variable register
#[derive(Default, Debug, Clone)]
pub struct VarDict {
    sym_to_key: HashMap<Symbol, usize>,
    key_to_sym: Vec<Symbol>,
    key_to_size: Vec<usize>,
    key_to_offset: Vec<usize>,
    total_dimension: usize,
}

impl VarDict {
    /// Creates an empty variable dictionary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a variable and its dimension, returning its dense identifier.
    pub fn register(&mut self, sym: Symbol, size: usize) -> Result<usize> {
        if let Some(_) = self.sym_to_key.get(&sym) {
            bail!("Variable {:?} has already been registered.", sym)
        }

        let key = self.key_to_sym.len();
        self.sym_to_key.insert(sym, key);
        self.key_to_sym.push(sym);
        self.key_to_size.push(size);
        self.key_to_offset.push(self.total_dimension);
        self.total_dimension += size;
        Ok(key)
    }

    /// Returns the dense key corresponding to `sym`, if registered.
    pub fn key(&self, sym: Symbol) -> Option<usize> {
        self.sym_to_key.get(&sym).copied()
    }

    /// Returns the symbol corresponding to dense index `key`, if valid.
    pub fn symbol(&self, key: usize) -> Option<Symbol> {
        self.key_to_sym.get(key).copied()
    }

    /// Returns the dimension of the variable at `key`, if valid.
    pub fn size(&self, key: usize) -> Option<usize> {
        self.key_to_size.get(key).copied()
    }

    /// Returns the buffer start offset for the variable at `key`, if valid.
    pub fn offset(&self, key: usize) -> Option<usize> {
        self.key_to_offset.get(key).copied()
    }

    /// Returns the `(offset, size)` slice layout in the flat state buffer for `key`, if valid.
    pub fn memory_layout(&self, key: usize) -> Option<(usize, usize)> {
        match (self.key_to_offset.get(key), self.key_to_size.get(key)) {
            (Some(&offset), Some(&size)) => Some((offset, size)),
            _ => None,
        }
    }

    /// Returns the total scalar capacity required for the flat state vector.
    pub fn total_size(&self) -> usize {
        self.total_dimension
    }

    /// Returns the number of distinct variables registered.
    pub fn len(&self) -> usize {
        self.key_to_sym.len()
    }

    /// Returns `true` if no variables have been registered.
    pub fn is_empty(&self) -> bool {
        self.key_to_sym.is_empty()
    }
}

impl NumericContext for VarDict {
    fn memory_layout(&self, key: usize) -> Option<(usize, usize)> {
        self.memory_layout(key)
    }
}

// It can also be created from a more abstract FactorGraph, e.g., when using non-linear models requiring relinearization.
#[derive(Default)]
pub struct GaussianFactorGraph {
    /// The variable register
    var_dict: VarDict,
    /// The Gaussian factors
    factors: Vec<GaussianFactor>,
    /// The flat vector containing the optimal value of each variable
    values: Vec<f64>,
}

impl GaussianFactorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn factor(&self, key: usize) -> Option<&GaussianFactor> {
        self.factors.get(key)
    }

    pub fn add_variable(&mut self, sym: Symbol, size: usize) -> Result<usize> {
        let key = self.var_dict.register(sym, size)?;
        self.values.extend(vec![f64::NAN; size]);
        Ok(key)
    }

    pub fn add_factor(
        &mut self,
        keys: Vec<usize>,
        mats: Vec<DMatrix<f64>>,
        obs_m: DVector<f64>,
        obs_w: DMatrix<f64>,
    ) -> Result<usize> {
        let dim_obs = obs_m.nrows();
        ensure!(
            (obs_w.nrows() == dim_obs) && (obs_w.ncols() == dim_obs),
            format!(
                "The dimensions of obs_m ({},) and obs_w ({}, {}) must be compatible",
                dim_obs,
                obs_w.nrows(),
                obs_w.ncols()
            )
        );

        let mut ncols = 0;
        let mut layout = Vec::with_capacity(keys.len());
        for &key in &keys {
            let size = self
                .var_dict
                .size(key)
                .with_context(|| format!("Variable with key {} not found in the register", key))?;
            layout.push((key, ncols, size));
            ncols += size;
        }

        let mut a = DMatrix::zeros(dim_obs, ncols);
        for (&(key, icol, ncols), mat) in zip(layout.iter(), mats.iter()) {
            ensure!(
                (mat.nrows() == dim_obs) && (mat.ncols() == ncols),
                format!(
                    "The matrix shape for variable {} is not valid, expected ({}, {}) found ({}, {})",
                    key,
                    dim_obs,
                    ncols,
                    mat.nrows(),
                    mat.ncols()
                )
            );
            a.columns_mut(icol, ncols).copy_from(&mat);
        }

        let obs_xi = &obs_w * &obs_m;
        let fct = GaussianFactor::new_unchecked(keys, layout, a, obs_xi, obs_w);
        self.factors.push(fct);
        Ok(self.factors.len() - 1)
    }

    /// Returns the optimal value for the given variable
    pub fn value(&self, key: usize) -> Option<&[f64]> {
        self.var_dict
            .memory_layout(key)
            .map(|(offset, size)| &self.values[offset..offset + size])
    }

    /// Build the symbolic tree resulting from the given elimination order.
    pub fn build_symbolic_tree(&mut self, order: &[usize]) -> Result<SymbolicTree> {
        // Validate elimination order: check no duplicate
        let seen_in_order: HashSet<&usize> = order.iter().collect();
        if seen_in_order.len() != order.len() {
            bail!("Duplicate variable found in elimination order");
        }
        // Validate that all variables in the factors are accounted for in the order
        let seen_in_factors: HashSet<usize> = self
            .factors
            .iter()
            .flat_map(|f| f.keys().iter())
            .cloned()
            .collect();

        for key in &seen_in_factors {
            if !seen_in_order.contains(key) {
                bail!(
                    "Variable with key {} is present in factors but missing from elimination order",
                    key
                );
            }
        }

        for key in &seen_in_order {
            if !seen_in_factors.contains(key) {
                bail!(
                    "Variable with key {} is present in elimination order but missing from factors",
                    key
                );
            }
        }

        //
        let tree = SymbolicTree::build(&self.factors, &order);
        Ok(tree)
    }

    pub fn solve_seq(&mut self, symb_tree: &SymbolicTree) -> Result<&[f64]> {
        // Build pass (sequential)
        let mut gauss_tree =
            GaussianTree::from_symbolic_seq(symb_tree, &self.factors, &self.var_dict)?;

        // Solve pass (sequential)
        gauss_tree.solve_seq(&mut self.values, &self.var_dict)?;
        Ok(&self.values)
    }

    pub fn solve_par(&mut self, symb_tree: &SymbolicTree) -> Result<&[f64]> {
        // Build pass (parallel)
        let mut gauss_tree =
            GaussianTree::from_symbolic_par(symb_tree, &self.factors, &self.var_dict)?;

        // Solve pass (parallel)
        gauss_tree.solve_par(&mut self.values, &self.var_dict)?;
        Ok(&self.values)
    }
}

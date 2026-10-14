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
pub struct VarSymbol(pub char, pub usize);

/// Variable register
#[derive(Default, Debug, Clone)]
pub struct VarDict {
    sym_to_key: HashMap<VarSymbol, usize>,
    key_to_sym: Vec<VarSymbol>,
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
    pub fn register(&mut self, sym: VarSymbol, size: usize) -> Result<usize> {
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
    pub fn key(&self, sym: VarSymbol) -> Option<usize> {
        self.sym_to_key.get(&sym).copied()
    }

    /// Returns the symbol corresponding to dense index `key`, if valid.
    pub fn symbol(&self, key: usize) -> Option<VarSymbol> {
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
}

impl GaussianFactorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn factor(&self, key: usize) -> Option<&GaussianFactor> {
        self.factors.get(key)
    }

    pub fn add_variable(&mut self, sym: VarSymbol, size: usize) -> Result<usize> {
        self.var_dict.register(sym, size)
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

    /// Returns a buffer ready for filled in.
    pub fn buffer(&self, value: f64) -> Vec<f64> {
        vec![value; self.var_dict.total_size()]
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

    pub fn solve_seq(&mut self, symb_tree: &SymbolicTree) -> Result<Vec<f64>> {
        let mut buffer = self.buffer(f64::NAN);

        // Build pass (sequential)
        let mut gauss_tree =
            GaussianTree::from_symbolic_seq(symb_tree, &self.factors, &self.var_dict)?;

        // Solve pass (sequential)
        gauss_tree.solve_seq(&mut buffer, &self.var_dict)?;
        Ok(buffer)
    }

    pub fn solve_par(&mut self, symb_tree: &SymbolicTree) -> Result<Vec<f64>> {
        let mut buffer = self.buffer(f64::NAN);

        // Build pass (parallel)
        let mut gauss_tree =
            GaussianTree::from_symbolic_par(symb_tree, &self.factors, &self.var_dict)?;

        // Solve pass (parallel)
        gauss_tree.solve_par(&mut buffer, &self.var_dict)?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector};

    use super::*;

    // Reusable test helper fixtures
    const X1: VarSymbol = VarSymbol('x', 1);
    const X2: VarSymbol = VarSymbol('x', 2);
    const X3: VarSymbol = VarSymbol('x', 3);
    const L1: VarSymbol = VarSymbol('l', 1);
    const L2: VarSymbol = VarSymbol('l', 2);

    #[test]
    fn new_and_default_initialize_empty_state() {
        let from_new = VarDict::new();
        let from_default = VarDict::default();

        for reg in [&from_new, &from_default] {
            assert!(reg.is_empty());
            assert_eq!(reg.len(), 0);
            assert_eq!(reg.total_size(), 0);
        }
    }

    #[test]
    fn register_single_variable_updates_indices_and_layouts() {
        let mut reg = VarDict::new();
        let x1 = reg.register(X1, 3).expect("registration should succeed");

        assert_eq!(x1, 0);
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.total_size(), 3);

        // Verify bidirectional indexing
        assert_eq!(reg.key(X1), Some(x1));
        assert_eq!(reg.symbol(x1), Some(X1));

        // Verify metadata lookups
        assert_eq!(reg.size(x1), Some(3));
        assert_eq!(reg.offset(x1), Some(0));
        assert_eq!(reg.memory_layout(x1), Some((0, 3)));
    }

    #[test]
    fn register_multiple_variables_computes_contiguous_offsets_and_keys() {
        let mut reg = VarDict::new();

        let x1 = reg.register(X1, 6).expect("registering X1 should succeed"); // 6-DoF pose
        let l1 = reg.register(L1, 3).expect("registering L1 should succeed"); // 3-DoF landmark
        let x2 = reg.register(X2, 6).expect("registering X2 should succeed"); // next pose

        assert_eq!((x1, l1, x2), (0, 1, 2));
        assert_eq!(reg.len(), 3);
        assert_eq!(reg.total_size(), 15); // 6 + 3 + 6

        // Offsets must advance contiguously without overlaps or gaps
        assert_eq!(reg.memory_layout(x1), Some((0, 6)));
        assert_eq!(reg.memory_layout(l1), Some((6, 3)));
        assert_eq!(reg.memory_layout(x2), Some((9, 6)));

        // Individual getters match aggregate layout
        assert_eq!(reg.offset(l1), Some(6));
        assert_eq!(reg.size(l1), Some(3));
    }

    #[test]
    fn register_zero_dimension_variable_is_supported() {
        let mut reg = VarDict::new();

        let x1 = reg
            .register(X1, 0)
            .expect("zero-dimension registration should succeed");
        let x2 = reg.register(X2, 4).expect("registering X2 should succeed");

        assert_eq!(reg.len(), 2);
        assert_eq!(reg.total_size(), 4);
        assert_eq!(reg.memory_layout(x1), Some((0, 0)));
        assert_eq!(reg.memory_layout(x2), Some((0, 4)));
    }

    #[test]
    fn register_duplicate_symbol_fails_without_mutating_state() {
        let mut reg = VarDict::new();
        let x1 = reg
            .register(X1, 3)
            .expect("initial registration should succeed");

        // Attempting to register the exact same symbol again
        let _ = reg
            .register(X1, 5)
            .expect_err("duplicate symbol should fail");

        // Registry state must remain completely unaltered
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.total_size(), 3);
        assert_eq!(reg.key(X1), Some(x1));
        assert_eq!(reg.size(x1), Some(3));
    }

    #[test]
    fn lookups_for_unregistered_keys_and_symbols_return_none() {
        let mut reg = VarDict::new();
        reg.register(X1, 2).unwrap();

        // Non-existent symbol
        assert_eq!(reg.key(X3), None);

        // Out-of-bounds keys
        for invalid_key in [1, 2, usize::MAX] {
            assert_eq!(reg.symbol(invalid_key), None);
            assert_eq!(reg.size(invalid_key), None);
            assert_eq!(reg.offset(invalid_key), None);
            assert_eq!(reg.memory_layout(invalid_key), None);
        }
    }

    #[test]
    fn slice_layout_matches_vector_partitioning() {
        let mut reg = VarDict::new();
        let x1 = reg.register(X1, 2).unwrap();
        let x2 = reg.register(X2, 3).unwrap();

        let mut buffer = vec![0.0f64; reg.total_size()];
        assert_eq!(buffer.len(), 5);

        let (off_x1, len_x1) = reg.memory_layout(x1).unwrap();
        let (off_x2, len_x2) = reg.memory_layout(x2).unwrap();

        buffer[off_x1..off_x1 + len_x1].copy_from_slice(&[1.0, 2.0]);
        buffer[off_x2..off_x2 + len_x2].copy_from_slice(&[3.0, 4.0, 5.0]);

        assert_eq!(buffer, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn new_and_default_initialize_empty_graph() {
        let g1 = GaussianFactorGraph::new();
        let g2 = GaussianFactorGraph::default();

        assert_eq!(g1.buffer(0.0).len(), 0);
        assert_eq!(g2.buffer(0.0).len(), 0);
        assert!(g1.factors.is_empty());
        assert!(g2.factors.is_empty());
    }

    #[test]
    fn add_var_delegates_to_register_and_allocates_buffer() {
        let mut graph = GaussianFactorGraph::new();

        graph
            .add_variable(X1, 3)
            .expect("registering X1 should succeed");
        graph
            .add_variable(L1, 2)
            .expect("registering L1 should succeed");

        // Pre-allocated flat buffer should match the cumulative dimension
        let buf = graph.buffer(42.0);
        assert_eq!(buf, vec![42.0; 5]);

        // Duplicate symbol registration should bubble up an error
        graph
            .add_variable(X1, 3)
            .expect_err("registering another X1 should fail");
    }

    #[test]
    fn add_factor_single_variable_succeeds() {
        let mut graph = GaussianFactorGraph::new();
        let x1 = graph.add_variable(X1, 3).unwrap();

        let dim_obs = 2;
        let mats = vec![dmatrix![1.0, 0.0, 0.0; 0.0, 1.0, 1.0]];
        let obs_m = dvector![2.0, -1.0];
        // Weight matrix W = 2 * I
        let obs_w = DMatrix::from_diagonal(&DVector::from_element(dim_obs, 2.0));

        let f1 = graph.add_factor(vec![x1], mats, obs_m, obs_w).unwrap();
        assert_eq!(graph.factors.len(), 1);

        let factor = graph.factor(f1).unwrap();
        assert_eq!(factor.keys(), vec![x1]);
    }
}

// //! Helper functions.
//
// use std::collections::HashMap;
//
// use crate::numeric::NumericContext;
//
// /// A typed semantic variable key (e.g., `'x'` for pose, `'l'` for landmark).
// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
// pub struct VarSymbol(pub char, pub usize);
//
// /// Bidirectional map between [`VarSymbol`] keys, dense solver indices, and buffer offsets.
// #[derive(Default, Debug, Clone)]
// pub struct VarDict {
//     sym_to_key: HashMap<VarSymbol, usize>,
//     key_to_sym: Vec<VarSymbol>,
//     key_to_size: Vec<usize>,
//     key_to_offset: Vec<usize>,
//     total_dimension: usize,
// }
//
// impl VarDict {
//     /// Creates an empty variable dictionary.
//     pub fn new() -> Self {
//         Self::default()
//     }
//
//     /// Registers a variable and its dimension, returning its dense identifier.
//     ///
//     /// If `sym` is already registered, returns the existing identifier without modifying state.
//     pub fn register(&mut self, sym: VarSymbol, size: usize) -> usize {
//         if let Some(&key) = self.sym_to_key.get(&sym) {
//             return key;
//         }
//
//         let key = self.key_to_sym.len();
//         self.sym_to_key.insert(sym, key);
//         self.key_to_sym.push(sym);
//         self.key_to_size.push(size);
//
//         self.key_to_offset.push(self.total_dimension);
//         self.total_dimension += size;
//
//         key
//     }
//
//     /// Returns the dense key corresponding to `sym`, if registered.
//     pub fn key(&self, sym: VarSymbol) -> Option<usize> {
//         self.sym_to_key.get(&sym).copied()
//     }
//
//     /// Returns the symbol corresponding to dense index `key`, if valid.
//     pub fn symbol(&self, key: usize) -> Option<VarSymbol> {
//         self.key_to_sym.get(key).copied()
//     }
//
//     /// Returns the dimension of the variable at `key`, if valid.
//     pub fn size(&self, key: usize) -> Option<usize> {
//         self.key_to_size.get(key).copied()
//     }
//
//     /// Returns the buffer start offset for the variable at `key`, if valid.
//     pub fn offset(&self, key: usize) -> Option<usize> {
//         self.key_to_offset.get(key).copied()
//     }
//
//     /// Returns the `(offset, size)` slice layout in the flat state buffer for `key`, if valid.
//     pub fn memory_layout(&self, key: usize) -> Option<(usize, usize)> {
//         match (self.key_to_offset.get(key), self.key_to_size.get(key)) {
//             (Some(&offset), Some(&size)) => Some((offset, size)),
//             _ => None,
//         }
//     }
//
//     /// Returns the total scalar capacity required for the flat state vector.
//     pub fn total_size(&self) -> usize {
//         self.total_dimension
//     }
//
//     /// Returns the number of distinct variables registered.
//     pub fn len(&self) -> usize {
//         self.key_to_sym.len()
//     }
//
//     /// Returns `true` if no variables have been registered.
//     pub fn is_empty(&self) -> bool {
//         self.key_to_sym.is_empty()
//     }
// }
//
// impl NumericContext for VarDict {
//     fn memory_layout(&self, key: usize) -> Option<(usize, usize)> {
//         self.memory_layout(key)
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_variable_dictionary() {
//         let mut dict = VarDict::new();
//
//         // Register a 6D robot pose and a 3D landmark
//         let x1 = dict.register(VarSymbol('x', 1), 6);
//         let l5 = dict.register(VarSymbol('l', 5), 3);
//
//         assert_eq!(dict.total_size(), 9);
//         assert_eq!(x1, 0);
//         assert_eq!(l5, 1);
//         assert_eq!(dict.memory_layout(x1), Some((0, 6)));
//         assert_eq!(dict.memory_layout(l5), Some((6, 3)));
//     }
// }

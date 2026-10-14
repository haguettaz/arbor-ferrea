//! Helper functions.

use std::collections::HashMap;

use crate::numeric::NumericContext;

/// A typed semantic variable key (e.g., `'x'` for pose, `'l'` for landmark).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarSymbol(pub char, pub usize);

/// Bidirectional map between [`VarSymbol`] keys, dense solver indices, and buffer offsets.
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
    ///
    /// If `sym` is already registered, returns the existing identifier without modifying state.
    pub fn register(&mut self, sym: VarSymbol, size: usize) -> usize {
        if let Some(&key) = self.sym_to_key.get(&sym) {
            return key;
        }

        let key = self.key_to_sym.len();
        self.sym_to_key.insert(sym, key);
        self.key_to_sym.push(sym);
        self.key_to_size.push(size);

        self.key_to_offset.push(self.total_dimension);
        self.total_dimension += size;

        key
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_dictionary() {
        let mut dict = VarDict::new();

        // Register a 6D robot pose and a 3D landmark
        let x1 = dict.register(VarSymbol('x', 1), 6);
        let l5 = dict.register(VarSymbol('l', 5), 3);

        assert_eq!(dict.total_size(), 9);
        assert_eq!(x1, 0);
        assert_eq!(l5, 1);
        assert_eq!(dict.memory_layout(x1), Some((0, 6)));
        assert_eq!(dict.memory_layout(l5), Some((6, 3)));
    }
}

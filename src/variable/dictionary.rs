//! Variable indexing and contiguous memory layout mapping.

use std::collections::HashMap;

/// A typed semantic variable key (e.g., `'x'` for pose, `'l'` for landmark).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarSymbol(pub char, pub usize);

/// Bidirectional map between [`VarSymbol`] keys, dense solver indices, and buffer offsets.
#[derive(Default, Debug, Clone)]
pub struct VarDict {
    sym_to_id: HashMap<VarSymbol, usize>,
    id_to_sym: Vec<VarSymbol>,
    id_to_size: Vec<usize>,
    id_to_offset: Vec<usize>,
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
        if let Some(&id) = self.sym_to_id.get(&sym) {
            return id;
        }

        let id = self.id_to_sym.len();
        self.sym_to_id.insert(sym, id);
        self.id_to_sym.push(sym);
        self.id_to_size.push(size);

        self.id_to_offset.push(self.total_dimension);
        self.total_dimension += size;

        id
    }

    /// Returns the dense identifier corresponding to `sym`, if registered.
    pub fn get_id(&self, sym: VarSymbol) -> Option<usize> {
        self.sym_to_id.get(&sym).copied()
    }

    /// Returns the symbol corresponding to dense index `id`, if valid.
    pub fn get_symbol(&self, id: usize) -> Option<VarSymbol> {
        self.id_to_sym.get(id).copied()
    }

    /// Returns the dimension of the variable at `id`, if valid.
    pub fn get_size(&self, id: usize) -> Option<usize> {
        self.id_to_size.get(id).copied()
    }

    /// Returns the buffer start offset for the variable at `id`, if valid.
    pub fn get_offset(&self, id: usize) -> Option<usize> {
        self.id_to_offset.get(id).copied()
    }

    /// Returns the `(offset, size)` slice layout in the flat state buffer for `id`, if valid.
    pub fn get_memory_layout(&self, id: usize) -> Option<(usize, usize)> {
        match (self.id_to_offset.get(id), self.id_to_size.get(id)) {
            (Some(&offset), Some(&size)) => Some((offset, size)),
            _ => None,
        }
    }

    /// Returns the total scalar capacity required for the flat state vector.
    pub fn get_total_size(&self) -> usize {
        self.total_dimension
    }

    /// Returns the number of distinct variables registered.
    pub fn len(&self) -> usize {
        self.id_to_sym.len()
    }

    /// Returns `true` if no variables have been registered.
    pub fn is_empty(&self) -> bool {
        self.id_to_sym.is_empty()
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

        assert_eq!(dict.get_total_size(), 9);
        assert_eq!(x1, 0);
        assert_eq!(l5, 1);
        assert_eq!(dict.get_memory_layout(x1), Some((0, 6)));
        assert_eq!(dict.get_memory_layout(l5), Some((6, 3)));
    }
}

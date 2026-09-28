use std::collections::HashMap;

/// A lightweight semantic identifier (e.g., 'x'1, 'l'5)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub char, pub usize);

pub type VarId = usize;

/// Maps user-friendly Symbols to dense solver indices and tracks memory offsets.
#[derive(Default)]
pub struct VariableDictionary {
    // Forward lookup: Symbol -> Dense VarId
    sym_to_id: HashMap<Symbol, VarId>,

    // Reverse lookup: Dense VarId -> Symbol
    id_to_sym: Vec<Symbol>,

    // Physical dimension (e.g., 6 for Pose3, 3 for Point3)
    id_to_dim: Vec<usize>,

    // Pre-computed starting index in the flat global state array
    id_to_offset: Vec<usize>,

    total_dimension: usize,
}

impl VariableDictionary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a variable and its physical dimension.
    /// Returns the assigned dense VarId.
    pub fn register(&mut self, sym: Symbol, dim: usize) -> VarId {
        if let Some(&id) = self.sym_to_id.get(&sym) {
            return id;
        }

        let id = self.id_to_sym.len();
        self.sym_to_id.insert(sym, id);
        self.id_to_sym.push(sym);
        self.id_to_dim.push(dim);

        self.id_to_offset.push(self.total_dimension);
        self.total_dimension += dim;

        id
    }

    /// Translates a user Symbol to the solver's dense VarId
    pub fn get_id(&self, sym: Symbol) -> Option<VarId> {
        self.sym_to_id.get(&sym).copied()
    }

    /// Translates a solver's dense VarId back to the user Symbol
    pub fn get_symbol(&self, id: VarId) -> Option<Symbol> {
        self.id_to_sym.get(id).copied()
    }

    /// Returns the exact memory offset for the global flat state vector
    pub fn get_offset(&self, id: VarId) -> Option<usize> {
        self.id_to_offset.get(id).copied()
    }

    /// Total size needed to allocate the global state vector (Delta x)
    pub fn total_state_size(&self) -> usize {
        self.total_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_dictionary() {
        let mut dict = VariableDictionary::new();

        // Register a 6D robot pose and a 3D landmark
        let x1 = dict.register(Symbol('x', 1), 6);
        let l5 = dict.register(Symbol('l', 5), 3);

        assert_eq!(dict.total_state_size(), 9);
    }
}

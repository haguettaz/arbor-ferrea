/*!
A dictionary module to make the interface between the user and the solver.
*/

use std::collections::HashMap;

use super::VarId;

/// A lightweight semantic identifier (e.g., 'x'1, 'l'5)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarSymbol(pub char, pub usize);

/// Maps user-friendly VarSymbol to dense solver indices and tracks memory offsets.
#[derive(Default)]
pub struct VarDict {
    // Forward lookup: VarSymbol -> Dense VarId
    sym_to_id: HashMap<VarSymbol, VarId>,

    // Reverse lookup: Dense VarId -> VarSymbol
    id_to_sym: Vec<VarSymbol>,

    // Physical dimension (e.g., 6 for Pose3, 3 for Point3)
    id_to_size: Vec<usize>,

    // Pre-computed starting index in the flat global state array
    id_to_offset: Vec<usize>,

    total_dimension: usize,
}

impl VarDict {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a variable and its physical dimension.
    /// Returns the assigned dense VarId.
    pub fn register(&mut self, sym: VarSymbol, size: usize) -> VarId {
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

    /// Translates a user VarSymbol to the solver's dense VarId
    pub fn get_id(&self, sym: VarSymbol) -> Option<VarId> {
        self.sym_to_id.get(&sym).copied()
    }

    /// Translates a solver's dense VarId back to the user VarSymbol
    pub fn get_symbol(&self, id: VarId) -> Option<VarSymbol> {
        self.id_to_sym.get(id).copied()
    }

    /// Returns the size
    pub fn get_size(&self, id: VarId) -> Option<usize> {
        self.id_to_size.get(id).copied()
    }

    /// Returns the exact memory offset for the global flat state vector
    pub fn get_offset(&self, id: VarId) -> Option<usize> {
        self.id_to_offset.get(id).copied()
    }

    /// Returns the memory layout in the global flat state vector
    pub fn get_memory_layout(&self, id: VarId) -> Option<(usize, usize)> {
        match (self.id_to_offset.get(id), self.id_to_size.get(id)) {
            (Some(&offset), Some(&size)) => Some((offset, size)),
            _ => None,
        }
    }

    /// Total size needed to allocate the global state vector (Delta x)
    pub fn get_total_size(&self) -> usize {
        self.total_dimension
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
    }
}

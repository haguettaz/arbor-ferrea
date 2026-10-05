//! Traits for symbolic factors.

/// A factor defined purely by the variables it connects.
pub trait SymbolicFactor {
    fn vars(&self) -> &[usize];
}

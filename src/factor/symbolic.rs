//! Module for symbolic factors.

/// A symbolic factor.
pub trait SymbolicFactor {
    fn vars(&self) -> &[usize];
}

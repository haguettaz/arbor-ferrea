//! Factors that pin variables to fixed values.

/// A factor that constrains a variable to a predetermined state.
pub struct PinFactor<T> {
    pub id: usize,
    pub value: Vec<T>,
}

impl<T> PinFactor<T> {
    /// Creates a new pin factor for variable `id`.
    pub fn new(id: usize, value: Vec<T>) -> Self {
        Self { id, value }
    }
}

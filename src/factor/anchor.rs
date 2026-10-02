/// A factor that anchor a variable at a given value.
pub struct AnchorFactor<T> {
    pub id: usize,
    pub value: Vec<T>,
}

impl<T> AnchorFactor<T> {
    /// Create a new anchoring factor
    pub fn new(id: usize, value: Vec<T>) -> Self {
        Self { id, value }
    }
}

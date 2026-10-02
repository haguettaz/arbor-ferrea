use crate::variable::VarId;

/// A factor that anchor the variable X at a given value `mx`.
pub struct AnchorFactor<T> {
    pub id: VarId,
    // pub dim_x: usize,
    pub value: Vec<T>, // the anchor values for the variable X -- shape (dim_x)
}

impl<T> AnchorFactor<T> {
    pub fn new(id: VarId, value: Vec<T>) -> Self {
        Self {
            id,
            // dim_x: val.len(),
            value,
        }
    }
}

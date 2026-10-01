use crate::variable::VarId;

/// A factor that anchor the variable X at a given value `mx`.
pub struct AnchorFactor<T> {
    pub var: VarId,
    pub dim_x: usize,
    pub val: Vec<T>, // the anchor values for the variable X -- shape (dim_x)
}

impl<T> AnchorFactor<T> {
    pub fn new(var: VarId, val: Vec<T>) -> Self {
        Self {
            var,
            dim_x: val.len(),
            val,
        }
    }
}

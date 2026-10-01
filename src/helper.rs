use anyhow::{Context, Result};

use crate::factor::anchor::AnchorFactor;
use crate::variable::dictionary::VarDict;

/// Builds the values buffer for solving; anchored variable are initialized with their known values
/// and unknown variables are initialized with the provided default value.
pub fn build_buffer<T: Copy + Send + Sync>(
    anchor_factors: &[AnchorFactor<T>],
    context: &VarDict,
    default_value: T,
) -> Result<Vec<T>> {
    let mut buffer = vec![default_value; context.get_total_size()];

    let mut seen = std::collections::HashSet::with_capacity(anchor_factors.len());
    for factor in anchor_factors {
        let (offset, size) = context
            .get_memory_layout(factor.var)
            .with_context(|| format!("Missing layout for anchor variable {}", factor.var))?;
        anyhow::ensure!(
            factor.val.len() == size,
            "Anchor value length ({}) does not match variable size ({}) for variable {}",
            factor.val.len(),
            size,
            factor.var
        );
        anyhow::ensure!(
            seen.insert(factor.var),
            "Duplicate anchor for variable {}",
            factor.var
        );
        buffer[offset..offset + size].copy_from_slice(&factor.val);
    }
    Ok(buffer)
}

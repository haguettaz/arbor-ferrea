//! Helper functions for buffer initialization and layout.

use anyhow::{Context, Result};

use crate::factor::pin::PinFactor;
use crate::variable::dictionary::VarDict;

/// Initializes a values buffer from variable layouts, applying pinned values
/// and filling unpinned entries with `default_value`.
pub fn build_buffer<T: Copy + Send + Sync>(
    pin_factors: &[PinFactor<T>],
    context: &VarDict,
    default_value: T,
) -> Result<Vec<T>> {
    let mut buffer = vec![default_value; context.total_size()];

    let mut seen = std::collections::HashSet::with_capacity(pin_factors.len());
    for factor in pin_factors {
        let (offset, size) = context
            .memory_layout(factor.id)
            .with_context(|| format!("Missing layout for anchor variable {}", factor.id))?;
        anyhow::ensure!(
            factor.value.len() == size,
            "Anchor value length ({}) does not match variable size ({}) for variable {}",
            factor.value.len(),
            size,
            factor.id
        );
        anyhow::ensure!(
            seen.insert(factor.id),
            "Duplicate anchor for variable {}",
            factor.id
        );
        buffer[offset..offset + size].copy_from_slice(&factor.value);
    }
    Ok(buffer)
}

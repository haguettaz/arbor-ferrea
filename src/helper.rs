use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::factor::anchor::AnchorFactor;
use crate::tree::numeric::ConcurrentStateBuffer;
use crate::variable::dictionary::VarDict;

/// Warning: it is the caller's responsibility to ensure that the anchor factors are independent
/// and write to disjoint memory regions.
pub fn build_buffer<T: Copy + Send + Sync>(
    anchor_factors: &[AnchorFactor<T>],
    context: &VarDict,
    default_value: T,
) -> Result<Vec<T>> {
    let mut buffer = vec![default_value; context.get_total_size()];

    let state_buf = ConcurrentStateBuffer::new(&mut buffer);
    anchor_factors
        .par_iter()
        .try_for_each(|factor| -> Result<()> {
            let offset = context
                .get_offset(factor.var)
                .with_context(|| format!("Missing offset for anchor variable {}", factor.var))?;

            // SAFETY: The caller guarantees disjoint variable offsets across anchor factors.
            unsafe {
                state_buf.write_slice(offset, &factor.val);
            }

            Ok(())
        })?;

    Ok(buffer)
}

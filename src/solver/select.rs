use anyhow::{Context, Result, ensure};
use nalgebra::RealField;

use super::pass::{ClosedFactor, Slot};

/// Messages still alive after the first `n_steps` steps of a pass.
pub(crate) fn alive_messages<T: RealField>(
    messages: &[ClosedFactor<T>],
    n_steps: usize,
) -> Vec<usize> {
    // `step` is non-decreasing in message order, so the prefix is contiguous
    let m = messages.partition_point(|msg| msg.step <= n_steps);
    let mut consumed = vec![false; m];
    for msg in &messages[..m] {
        for s in &msg.content {
            if let Slot::Closed(j) = s {
                consumed[*j] = true;
            }
        }
    }
    (0..m).filter(|&j| !consumed[j]).collect()
}

/// Marks every original factor inside message `root`; errors on double coverage.
pub(crate) fn mark_leaves<T: RealField>(
    messages: &[ClosedFactor<T>],
    root: usize,
    covered: &mut [bool],
    stack: &mut Vec<usize>,
) -> Result<()> {
    stack.clear();
    stack.push(root);
    while let Some(j) = stack.pop() {
        for s in &messages[j].content {
            match *s {
                Slot::Original(i) => {
                    ensure!(!covered[i], "original factor {i} would be counted twice");
                    covered[i] = true;
                }
                Slot::Closed(c) => stack.push(c),
            }
        }
    }
    Ok(())
}

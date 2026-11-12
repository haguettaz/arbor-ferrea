use anyhow::{Context, Result, ensure};
use nalgebra::RealField;

use super::eliminate::eliminate;
use crate::graph::factor::Factor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Original(usize),
    Closed(usize),
}

impl Slot {
    fn keys<'a, T: RealField>(
        &self,
        original_factors: &'a [Factor<T>],
        processed_factors: &'a [ClosedFactor<T>],
    ) -> &'a [usize] {
        match self {
            Slot::Original(i) => original_factors[*i].keys(),
            Slot::Closed(j) => processed_factors[*j].keys(),
        }
    }
}

#[derive(Debug)]
pub struct ClosedFactor<T: RealField> {
    pub factor: Factor<T>,
    pub content: Vec<Slot>, // direct inputs only
    pub step: usize,        // the step in the pass at which the factors has been closed.
}

impl<T: RealField> ClosedFactor<T> {
    pub fn keys(&self) -> &[usize] {
        self.factor.keys()
    }

    // pub fn has_key(&self, key: usize) -> bool {
    //     self.factor.has_key(key)
    // }
}

pub(crate) fn full_pass<T: RealField>(
    n_keys: usize,
    factors: &[Factor<T>],
    order: impl DoubleEndedIterator<Item = usize>,
    size_of: impl Fn(usize) -> usize,
) -> Result<Vec<ClosedFactor<T>>> {
    let mut messages: Vec<ClosedFactor<T>> = Vec::with_capacity(n_keys);

    // Dense per-key state; the *_at arrays hold the step at which they were last set.
    let mut is_active = vec![false; n_keys];
    let mut coupled_at = vec![usize::MAX; n_keys];
    let mut cleared_at = vec![usize::MAX; n_keys];
    let mut seen_at = vec![usize::MAX; n_keys];

    // Scratch buffers reused across steps
    let mut active: Vec<Slot> = (0..factors.len()).map(Slot::Original).collect();
    let mut active_keys: Vec<usize> = Vec::new();
    let mut cleared_keys: Vec<usize> = Vec::new();
    let mut kept_keys: Vec<usize> = Vec::new();
    let mut cleared_slots: Vec<Slot> = Vec::new();

    for (step, key) in order.enumerate() {
        is_active[key] = true;
        active_keys.push(key);

        // Keys linked to a factor that still touches an unprocessed key
        for f in &active {
            let f_keys = f.keys(&factors, &messages);
            if f_keys.iter().any(|&k| !is_active[k]) {
                for &k in f_keys {
                    coupled_at[k] = step;
                }
            }
        }

        // Active keys that are neither `key` nor coupled get cleared
        cleared_keys.clear();
        active_keys.retain(|&k| {
            let keep = k == key || coupled_at[k] == step;
            if !keep {
                cleared_at[k] = step;
                is_active[k] = false;
                cleared_keys.push(k);
            }
            keep
        });
        if cleared_keys.is_empty() {
            continue;
        }

        // Move every factor touching a cleared key out of `active`
        cleared_slots.clear();
        let mut i = 0;
        while i < active.len() {
            if active[i]
                .keys(factors, &messages)
                .iter()
                .any(|&k| cleared_at[k] == step)
            {
                cleared_slots.push(active.swap_remove(i));
            } else {
                i += 1;
            }
        }

        // Kept keys: still-active keys appearing in the cleared factors
        kept_keys.clear();
        for f in &cleared_slots {
            for &k in f.keys(factors, &messages) {
                if is_active[k] && seen_at[k] != step {
                    seen_at[k] = step;
                    kept_keys.push(k);
                }
            }
        }
        kept_keys.sort_unstable();
        cleared_keys.sort_unstable();

        // Borrows of `messages` end with this block, so the push below is fine
        let factor = {
            let factor_refs = cleared_slots.iter().map(|s| match s {
                Slot::Original(i) => &factors[*i],
                Slot::Closed(j) => &messages[*j].factor,
            });

            eliminate(factor_refs, &kept_keys, &cleared_keys, &size_of)
                .with_context(|| format!("elimination failed for key {key}"))?
        };

        messages.push(ClosedFactor {
            factor,
            content: cleared_slots.clone(),
            step,
        });
        active.push(Slot::Closed(messages.len() - 1));
    }

    Ok(messages)
}

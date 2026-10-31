//! Solver (filter and smoother) for exact inference on Gaussian factor graphs.
//!
//! Implements Gaussian message-passing over factors graphs.
//!
//! For theoretical background, refer to the
//! [lecture notes](https://doi.org/10.3929/ethz-c-000785981)
//! of the class *Model-Based Estimation and Signal Analysis*
//! (Prof. H.-A. Loeliger, ETH Zürich, 2025).

use anyhow::{Context, Result, ensure};
use nalgebra::RealField;

use crate::gaussian::*;

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
    pub factor: Likelihood<T>,
    pub content: Vec<Slot>, // direct inputs only
    pub step: usize,        // the step at which the factors has been closed.
}

impl<T: RealField> ClosedFactor<T> {
    pub fn keys(&self) -> &[usize] {
        self.factor.keys()
    }

    pub fn has_key(&self, key: usize) -> bool {
        self.factor.has_key(key)
    }
}

/// Messages still alive after the first `n_steps` steps of a pass.
fn alive_messages<T: RealField>(messages: &[ClosedFactor<T>], n_steps: usize) -> Vec<usize> {
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
fn mark_leaves<T: RealField>(
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

fn full_pass<T, O, C>(
    n_keys: usize,
    factors: &[Factor<T>],
    order: O,
    ctx: C,
) -> Result<Vec<ClosedFactor<T>>>
where
    T: RealField,
    O: DoubleEndedIterator<Item = usize>,
    C: Fn(usize) -> usize,
{
    // let n_keys = n_keys;
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
            let refs = cleared_slots.iter().map(|s| match s {
                Slot::Original(i) => FactorView::from(&factors[*i]),
                Slot::Closed(j) => FactorView::from(&messages[*j].factor),
            });

            Likelihood::eliminate(refs, &kept_keys, &cleared_keys, &ctx)
                .with_context(|| format!("Failed to compute factor for key {key}"))?
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

pub struct Smoother<T: RealField> {
    factor_graph: FactorGraph<T>,
}

impl<T: RealField> Smoother<T> {
    fn n_keys(&self) -> usize {
        self.factor_graph.n_keys()
    }

    fn n_factors(&self) -> usize {
        self.factor_graph.n_factors()
    }

    fn factors(&self) -> &[Factor<T>] {
        self.factor_graph.factors()
    }

    fn factor(&self, i: usize) -> Option<&Factor<T>> {
        self.factor_graph.factor(i)
    }

    fn keys(&self) -> impl DoubleEndedIterator<Item = usize> {
        self.factor_graph.keys()
    }

    pub fn new(factor_graph: FactorGraph<T>) -> Self {
        Smoother { factor_graph }
    }

    pub fn run(&self) -> Result<Vec<Marginal<T>>> {
        let ctx = |k: usize| self.factor_graph.size(k).unwrap_or(0);
        let n = self.n_keys();

        let forward = full_pass(n, self.factors(), self.keys(), ctx)
            .context("error during the forward pass")?;
        let backward = full_pass(n, self.factors(), self.keys().rev(), ctx)
            .context("error during the backward pass")?;

        let mut marginals = Vec::with_capacity(n);
        let mut factors: Vec<FactorView<'_, T>> = Vec::new();
        let mut rem_keys: Vec<usize> = Vec::new();
        let mut covered = vec![false; self.n_factors()];
        let mut stack = Vec::new();

        for key in self.keys() {
            factors.clear();
            covered.fill(false);

            for i in alive_messages(&forward, key) {
                mark_leaves(&forward, i, &mut covered, &mut stack)?;
                factors.push(FactorView::from(&forward[i].factor));
            }
            for i in alive_messages(&backward, n - 1 - key) {
                mark_leaves(&backward, i, &mut covered, &mut stack)?;
                factors.push(FactorView::from(&backward[i].factor));
            }
            for (i, &covered_i) in covered.iter().enumerate() {
                if covered_i {
                    continue;
                }
                let f = self
                    .factor(i)
                    .with_context(|| format!("factor {i} not found"))?;
                factors.push(FactorView::from(f)); // priors on `key` included
            }

            rem_keys.clear();
            for f in &factors {
                rem_keys.extend(f.keys().iter().copied().filter(|&k| k != key));
            }
            rem_keys.sort_unstable();
            rem_keys.dedup();

            let likelihood = Likelihood::eliminate(factors.iter().copied(), &[key], &rem_keys, ctx)
                .with_context(|| format!("failed to eliminate around key {key}"))?;
            marginals.push(Marginal::from(key, None, &likelihood)?);
        }
        Ok(marginals)
    }
}

// pub struct Filter<P: Prior, R: Relation> {
//     factor_graph: FactorGraph<P, R>,
// }

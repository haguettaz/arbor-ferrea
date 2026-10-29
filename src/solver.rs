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
            let refs: Vec<FactorRef<T>> = cleared_slots
                .iter()
                .map(|s| match s {
                    Slot::Original(i) => match &factors[*i] {
                        Factor::Prior(p) => FactorRef::Prior(p),
                        Factor::Likelihood(l) => FactorRef::Likelihood(l),
                    },
                    Slot::Closed(j) => FactorRef::Likelihood(&messages[*j].factor),
                })
                .collect();
            Likelihood::from_closure(&refs, &kept_keys, &cleared_keys, &ctx)
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
        let ctx = |key| self.factor_graph.size(key).unwrap_or(0);

        // forward pass
        let forward_messages = full_pass(self.n_keys(), self.factors(), self.keys(), ctx)
            .with_context(|| "an error occured during the forward pass")?;

        // backward pass
        let backward_messages = full_pass(self.n_keys(), self.factors(), self.keys().rev(), ctx)
            .with_context(|| "an error occured during the backward pass")?;

        // compute marginals
        let mut marginals = Vec::with_capacity(self.n_keys());
        let mut factors = Vec::new();
        let mut rem_keys = Vec::new();

        for key in self.keys() {
            factors.clear();
            rem_keys.clear();

            let key_step_forward = key;
            let key_step_backward = self.n_keys() - 1 - key;

            let forward_ids = alive_messages(&forward_messages, key_step_forward);
            let backward_ids = alive_messages(&backward_messages, key_step_backward);

            let mut covered = vec![false; self.n_factors()];
            let mut stack = Vec::new();
            for &i in &forward_ids {
                mark_leaves(&forward_messages, i, &mut covered, &mut stack)?;
            }
            for &i in &backward_ids {
                mark_leaves(&backward_messages, i, &mut covered, &mut stack)?;
            }

            let original_ids = (0..self.n_factors())
                .filter(|&i| !covered[i])
                .collect::<Vec<_>>();

            for &i in &forward_ids {
                for &k in forward_messages[i].factor.keys() {
                    if k != key && !rem_keys.contains(&k) {
                        rem_keys.push(k);
                    }
                }
                factors.push(FactorRef::Likelihood(&forward_messages[i].factor));
            }
            for &i in &backward_ids {
                for &k in backward_messages[i].factor.keys() {
                    if k != key && !rem_keys.contains(&k) {
                        rem_keys.push(k);
                    }
                }
                factors.push(FactorRef::Likelihood(&backward_messages[i].factor));
            }

            let mut prior = None;
            for &i in &original_ids {
                let factor = self
                    .factor(i)
                    .with_context(|| format!("factor with id {} not found", i))?;
                match factor {
                    Factor::Likelihood(p) => {
                        factors.push(FactorRef::Likelihood(p));
                        for &k in p.keys() {
                            if k != key && !rem_keys.contains(&k) {
                                rem_keys.push(k);
                            }
                        }
                    }
                    Factor::Prior(p) => {
                        if p.has_key(key) {
                            prior = Some(p);
                        } else {
                            factors.push(FactorRef::Prior(p));
                            for &k in p.keys() {
                                if k != key && !rem_keys.contains(&k) {
                                    rem_keys.push(k);
                                }
                            }
                        }
                    }
                };
            }

            let likelihood = Likelihood::from_closure(&factors, &[key], &rem_keys, ctx)?;

            let marginal = Marginal::from(key, prior, &likelihood)?;

            marginals.push(marginal);
        }
        Ok(marginals)
    }
}

// pub struct Filter<P: Prior, R: Relation> {
//     factor_graph: FactorGraph<P, R>,
// }

use anyhow::{Context, Result};
use nalgebra::RealField;

use super::eliminate::eliminate;
use super::estimate::Marginals;
use super::pass::full_pass;
use super::select::{alive_messages, mark_leaves};
use crate::graph::{FactorGraph, factor::Factor};

pub(crate) fn smooth<T: RealField>(factor_graph: &FactorGraph<T>) -> Result<Marginals<T>> {
    let size_of = |key: usize| factor_graph.size_of(key).unwrap_or(0);

    let n_keys = factor_graph.n_keys();

    let forward_messages = full_pass(n_keys, factor_graph.factors(), factor_graph.keys(), size_of)
        .context("forward pass failed")?;

    let backward_messages = full_pass(
        n_keys,
        factor_graph.factors(),
        factor_graph.keys().rev(),
        size_of,
    )
    .context("backward pass failed")?;

    let mut marginals = Marginals::with_capacity(factor_graph.n_keys());

    let mut factors: Vec<&Factor<T>> = Vec::new();
    let mut rem_keys: Vec<usize> = Vec::new();
    let mut covered = vec![false; factor_graph.n_factors()];
    let mut stack = Vec::new();

    for key in factor_graph.keys() {
        factors.clear();
        covered.fill(false);

        for i in alive_messages(&forward_messages, key) {
            mark_leaves(&forward_messages, i, &mut covered, &mut stack)?;
            factors.push(&forward_messages[i].factor);
        }
        for i in alive_messages(&backward_messages, n_keys - 1 - key) {
            mark_leaves(&backward_messages, i, &mut covered, &mut stack)?;
            factors.push(&backward_messages[i].factor);
        }
        for (i, &covered_i) in covered.iter().enumerate() {
            if covered_i {
                continue;
            }
            factors.push(
                factor_graph
                    .factor(i)
                    .with_context(|| format!("factor {i} not found in factor graph"))?,
            );
        }

        rem_keys.clear();
        for f in &factors {
            rem_keys.extend(f.keys().iter().copied().filter(|&k| k != key));
        }
        rem_keys.sort_unstable();
        rem_keys.dedup();

        let factor = eliminate(factors.iter().copied(), &[key], &rem_keys, size_of)
            .with_context(|| format!("failed to eliminate around key {key}"))?;
        marginals.add_from(key, &factor)?;
    }
    Ok(marginals)
}

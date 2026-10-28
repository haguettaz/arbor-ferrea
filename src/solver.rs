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
// pub fn new(factor_graph: FactorGraph<T>) -> Self {
//     let arena = Arena::new();
//     Smoother {
//         factor_graph,
//         arena,
//     }
// }
//     /// A full pass over the factor graph in the specified elimination order
//     /// Messages are allocated into the provided arena
//     /// The returned messages are references to the corresponding messages
//     /// All messages are kept in memory.
//     pub fn full_pass<'a>(&'a self, order: &[usize]) -> Result<Vec<Vec<&'a Factor<T>>>> {
//         let mut messages = vec![vec![]; order.len()];
//
//         // all are sorted
//         // let mut active_keys: Vec<usize> = Vec::new();
//         // let mut coupled_keys: Vec<usize> = Vec::new();
//         // let mut cleared_keys: Vec<usize> = Vec::new();
//
//         let mut active_keys: HashSet<usize> = HashSet::new();
//         let mut coupled_keys: HashSet<usize> = HashSet::new();
//         let mut cleared_keys: HashSet<usize> = HashSet::new();
//
//         // let mut active_priors: Vec<&'a P> = self.factor_graph.priors().iter().collect();
//         // let mut active_relations: Vec<&'a R> = self.factor_graph.relations().iter().collect();
//         let mut active_factors: Vec<&'a Factor<T>> = self.factor_graph.factors().iter().collect();
//
//         // let mut eliminated_priors: Vec<&'a P> = Vec::new();
//         // let mut eliminated_relations: Vec<&'a R> = Vec::new();
//         let mut cleared_factors: Vec<&'a Factor<T>> = Vec::new();
//
//         for key in order {
//             active_keys.insert(*key);
//
//             coupled_keys.clear();
//             coupled_keys.extend(
//                 // active_relations
//                 active_factors
//                     .iter()
//                     .filter(|f| f.keys().iter().any(|k| !(active_keys.contains(k))))
//                     .flat_map(|f| f.keys())
//                     .copied(),
//             );
//             cleared_keys.clear();
//             cleared_keys.extend(active_keys.iter());
//             cleared_keys.retain(|k| !(k == key || coupled_keys.contains(k)));
//             active_keys.retain(|k| !cleared_keys.contains(k));
//
//             let mut i = 0;
//             while i < active_factors.len() {
//                 if active_factors[i]
//                     .keys()
//                     .iter()
//                     .any(|k| cleared_keys.contains(k))
//                 {
//                     cleared_factors.push(active_factors.swap_remove(i));
//                 } else {
//                     i += 1;
//                 }
//             }
//
//             let kept_keys: Vec<usize> = active_keys
//                 .iter()
//                 .filter(|&k| cleared_factors.iter().any(|f| f.has_key(k)))
//                 .copied()
//                 .collect();
//
//             if !cleared_keys.is_empty() {
//                 // Close the box using the drained eliminated factors (priors + relations)
//                 let kept_keys_vec: Vec<usize> = active_keys
//                     .iter()
//                     .filter(|&k| cleared_factors.iter().any(|f| f.has_key(k)))
//                     .copied()
//                     .collect();
//                 let cleared_keys_vec: Vec<usize> = cleared_keys.iter().cloned().collect();
//                 let msg = Likelihood::from_closure(
//                     cleared_factors,
//                     &kept_keys_vec,
//                     &cleared_keys_vec,
//                     |k| self.factor_graph.size(k).unwrap_or(0),
//                 )
//                 .with_context(|| format!("Failed to compute message for key {}", key))?;
//                 active_factors.push(self.arena.alloc(Factor::Likelihood(msg)));
//             }
//
//             // active_keys.insert(*key);
//             messages[*key].extend(active_factors.iter().filter(|r| r.is_message()).copied());
//         }
//
//         Ok(messages)
//     }
// }

// pub struct Filter<P: Prior, R: Relation> {
//     factor_graph: FactorGraph<P, R>,
// }

// pub struct Solver<'a, P: Prior + 'a, R: Relation + 'a> {
//     factor_graph: &'a FactorGraph<P, R>,
//     // nodes: Vec<Node<F>>,
// }

// impl<'a, P, R> Solver<'a, P, R>
// where
//     P: Prior + Debug + Default + 'a,
//     R: Relation + Debug + Default + 'a,
// {
//     pub fn new(factor_graph: &'a FactorGraph<P, R>) -> Result<Self> {
//         // let mut nodes = Vec::with_capacity(factor_graph.n_keys());
//         // for key in 0..factor_graph.n_keys() {
//         //     reduced_factor_graphs.push(Node::new(key));
//         // }
//
//         // Forward pass
//         let mut active_keys: HashSet<usize> = HashSet::new();
//         let mut coupled_keys: HashSet<usize> = HashSet::new();
//         let mut cleared_keys: HashSet<usize> = HashSet::new();
//
//         let mut active_priors: Vec<&P> = factor_graph.priors().iter().collect();
//         let mut active_relations: Vec<&R> = factor_graph.relations().iter().collect();
//
//         let mut eliminated_priors: Vec<&P> = Vec::new();
//         let mut eliminated_relations: Vec<&R> = Vec::new();
//
//         for key in factor_graph.keys() {
//             coupled_keys.clear();
//             coupled_keys.extend(
//                 active_relations
//                     .iter()
//                     .filter(|relation| {
//                         let min_k = relation.keys().iter().min().unwrap_or(&key);
//                         let max_k = relation.keys().iter().max().unwrap_or(&key);
//                         min_k < &key && max_k > &key
//                     })
//                     .flat_map(|relation| relation.keys())
//                     .copied(),
//             );
//
//             cleared_keys.clear();
//             cleared_keys.extend(active_keys.difference(&coupled_keys).copied());
//
//             // Note: no need to clear eliminated_priors since drain(..) empties it
//             let mut i = 0;
//             while i < active_priors.len() {
//                 if cleared_keys.contains(&active_priors[i].key()) {
//                     eliminated_priors.push(active_priors.remove(i));
//                 } else {
//                     i += 1;
//                 }
//             }
//
//             let mut i = 0;
//             while i < active_relations.len() {
//                 if active_relations[i]
//                     .keys()
//                     .iter()
//                     .any(|k| cleared_keys.contains(k))
//                 {
//                     eliminated_relations.push(active_relations.remove(i));
//                 } else {
//                     i += 1;
//                 }
//             }
//
//             // Close the box using the drained eliminated factors (priors + relations)
//             let message = R::from(
//                 eliminated_priors.drain(..),
//                 eliminated_relations.drain(..),
//                 cleared_keys.iter().copied(),
//                 |key| factor_graph.size(key).unwrap_or(0),
//             )
//             .with_context(|| format!("Failed to compute the forward messages for key {}", key))?;
//
//             if !message.is_empty() {
//                 active_factors.push(AnyFactor::OwnedMessage(message));
//             }
//
//             // Iterate by reference so we don't consume the active_factors pool!
//             for factor in &active_factors {
//                 if let AnyFactor::OwnedMessage(msg) = factor {
//                     graph.forward_messages.push(msg.clone());
//                 }
//             }
//
//             // Update tracking keys without allocating a new HashSet
//             active_keys.retain(|k| !cleared_keys.contains(k));
//             active_keys.insert(key);
//         }
//
//         // Backward pass
//         let mut active_keys: HashSet<usize> = HashSet::new();
//         let mut active_factors: Vec<AnyFactor<'a, F>> = factor_graph
//             .factors()
//             .iter()
//             .map(AnyFactor::BorrowedFactor)
//             .collect();
//
//         let mut coupled_keys: HashSet<usize> = HashSet::new();
//         let mut cleared_keys: HashSet<usize> = HashSet::new();
//         let mut cleared_factors: Vec<AnyFactor<'a, F>> = Vec::new();
//
//         for graph in reduced_factor_graphs.iter_mut().rev() {
//             let key = graph.key;
//
//             coupled_keys.clear();
//             coupled_keys.extend(
//                 active_factors
//                     .iter()
//                     .filter(|factor| {
//                         let min_k = factor.keys().iter().min().unwrap_or(&key);
//                         let max_k = factor.keys().iter().max().unwrap_or(&key);
//                         min_k < &key && max_k > &key
//                     })
//                     .flat_map(|factor| factor.keys())
//                     .copied(),
//             );
//
//             cleared_keys.clear();
//             cleared_keys.extend(active_keys.difference(&coupled_keys).copied());
//
//             // Note: no need to clear cleared_factors since drain(..) empties it
//             let mut i = 0;
//             while i < active_factors.len() {
//                 if active_factors[i]
//                     .keys()
//                     .iter()
//                     .any(|k| cleared_keys.contains(k))
//                 {
//                     cleared_factors.push(active_factors.remove(i));
//                 } else {
//                     i += 1;
//                 }
//             }
//
//             // Close the box using the drained eliminated factors
//             let message = <<F as Factor>::TargetMessage as Message>::close(
//                 cleared_factors.drain(..),
//                 cleared_keys.iter().copied(),
//                 |key| factor_graph.size(key).unwrap_or(0),
//             )
//             .with_context(|| format!("Failed to compute the backward messages for key {}", key))?;
//
//             if !message.is_empty() {
//                 active_factors.push(AnyFactor::OwnedMessage(message));
//             }
//
//             // Iterate by reference so we don't consume the active_factors pool!
//             for factor in &active_factors {
//                 if let AnyFactor::OwnedMessage(msg) = factor {
//                     graph.backward_messages.push(msg.clone());
//                 }
//             }
//
//             // Update tracking keys without allocating a new HashSet
//             active_keys.retain(|k| !cleared_keys.contains(k));
//             active_keys.insert(key);
//         }
//
//         let solver = Solver {
//             factor_graph,
//             reduced_factor_graphs,
//         };
//         Ok(solver)
// }

//     pub fn optimize(&mut self) -> Result<()> {
//         for reduced_factor_graph in self.reduced_factor_graphs.iter_mut() {
//             let key = reduced_factor_graph.key;
//             let closed_factors: HashSet<usize> = reduced_factor_graph
//                 .forward_messages
//                 .iter()
//                 .chain(reduced_factor_graph.backward_messages.iter())
//                 .flat_map(|msg| msg.closed_factors())
//                 .copied()
//                 .collect();
//
//             // Map remaining factors to the BorrowedFactor variant
//             let rem_factors_iter = self
//                 .factor_graph
//                 .factors()
//                 .iter()
//                 .filter(|factor| !closed_factors.contains(&factor.id()))
//                 .map(AnyFactor::BorrowedFactor);
//
//             // Map messages to the OwnedMessage variant.
//             let messages_iter = reduced_factor_graph
//                 .forward_messages
//                 .iter()
//                 .chain(reduced_factor_graph.backward_messages.iter())
//                 .cloned()
//                 .map(AnyFactor::OwnedMessage);
//
//             // Collect them into a unified Vec so we can iterate over them multiple times
//             let all_factors: Vec<AnyFactor<'a, F>> =
//                 rem_factors_iter.chain(messages_iter).collect();
//
//             // Extract all keys to eliminate
//             let mut all_keys: HashSet<usize> = all_factors
//                 .iter()
//                 .flat_map(|node| node.keys().iter().copied())
//                 .collect();
//             all_keys.remove(&reduced_factor_graph.key);
//
//             // Close the box using the unified Vec and the keys
//             let message =
//                 <<F as Factor>::TargetMessage as Message>::close(all_factors, all_keys, |key| {
//                     self.factor_graph.size(key).unwrap_or(0)
//                 })
//                 .with_context(|| format!("Failed to compute the marginalize for key {}", key))?;
//             reduced_factor_graph.marginal = Some(message);
//         }
//
//         Ok(())
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    //
    //     #[derive(Clone, Debug, Default)]
    //     pub struct MyPrior {
    //         pub id: usize,
    //         pub key: usize,
    //     }
    //
    //     impl Prior for MyPrior {
    //         fn id(&self) -> usize {
    //             self.id
    //         }
    //
    //         fn key(&self) -> usize {
    //             self.key
    //         }
    //     }
    //
    //     impl MyPrior {
    //         pub fn new(id: usize, key: usize) -> Self {
    //             Self { id, key }
    //         }
    //     }
    //
    //     #[derive(Debug, PartialEq, Default)]
    //     pub struct MyRelation {
    //         pub ids: Vec<usize>, // Added so Factor::id() has a value to return
    //         pub keys: Vec<usize>,
    //     }
    //
    //     impl MyRelation {
    //         pub fn new(ids: Vec<usize>, keys: Vec<usize>) -> Self {
    //             Self { ids, keys }
    //         }
    //     }
    //
    //     // Base implementation for Factor
    //     impl Relation for MyRelation {
    //         fn keys(&self) -> &[usize] {
    //             &self.keys
    //         }
    //
    //         fn ids(&self) -> &[usize] {
    //             &self.ids
    //         }
    //
    //         fn from<'a, P, R>(
    //             priors: impl IntoIterator<Item = &'a P>,
    //             relations: impl IntoIterator<Item = &'a R>,
    //             keep_keys: impl IntoIterator<Item = usize>,
    //             _clear_keys: impl IntoIterator<Item = usize>,
    //             _ctx: impl Fn(usize) -> usize,
    //         ) -> Result<Self>
    //         where
    //             P: Prior + 'a,
    //             R: Relation + 'a,
    //         {
    //             let mut ids: Vec<usize> = priors
    //                 .into_iter()
    //                 .map(|p| p.id())
    //                 .chain(relations.into_iter().flat_map(|r| r.ids()).copied())
    //                 .collect();
    //             ids.sort_unstable();
    //
    //             let mut keys: Vec<usize> = keep_keys.into_iter().collect();
    //             keys.sort_unstable();
    //
    //             let msg = MyRelation { ids, keys };
    //             Ok(msg)
    //         }
    //     }
    //
    const X0: Symbol = Symbol('x', 0);
    const X1: Symbol = Symbol('x', 1);
    const X2: Symbol = Symbol('x', 2);
    const X3: Symbol = Symbol('x', 3);
    const X4: Symbol = Symbol('x', 4);
    const X5: Symbol = Symbol('x', 5);

    #[test]
    fn test_forward_backward_in_line() {
        let mut factor_graph = FactorGraph::new();

        let x0 = factor_graph.add_variable(X0, 2).unwrap();
        let x1 = factor_graph.add_variable(X1, 2).unwrap();
        let x2 = factor_graph.add_variable(X2, 2).unwrap();
        let x3 = factor_graph.add_variable(X3, 2).unwrap();
        let x4 = factor_graph.add_variable(X4, 2).unwrap();

        factor_graph
            .add_prior(x0, DVector::zeros(2), DMatrix::identity(2, 2))
            .expect("error while adding prior on x0");
        factor_graph
            .add_observation(
                vec![x0, x1],
                DMatrix::from_element(1, 4, 1.0),
                DVector::zeros(1),
                DMatrix::identity(1, 1),
            )
            .expect("error while adding observation of x0 and x1");
        factor_graph
            .add_observation(
                vec![x1, x2],
                DMatrix::from_element(1, 4, 1.0),
                DVector::zeros(1),
                DMatrix::identity(1, 1),
            )
            .expect("error while adding observation of x1 and x2");
        factor_graph
            .add_observation(
                vec![x2, x3],
                DMatrix::from_element(1, 4, 1.0),
                DVector::zeros(1),
                DMatrix::identity(1, 1),
            )
            .expect("error while adding observation of x2 and x3");
        factor_graph
            .add_observation(
                vec![x3, x4],
                DMatrix::from_element(1, 4, 1.0),
                DVector::zeros(1),
                DMatrix::identity(1, 1),
            )
            .expect("error while adding observation of x3 and x4");

        let smoother = Smoother { factor_graph };

        let marginals = smoother.run();
        // assert!(false);
    }

    //     #[test]
    //     fn test_solver_loops() {
    //         let mut factor_graph = FactorGraph::new();
    //
    //         let x0 = factor_graph.add_variable(X0, 1).unwrap();
    //         let x1 = factor_graph.add_variable(X1, 1).unwrap();
    //         let x2 = factor_graph.add_variable(X2, 1).unwrap();
    //         let x3 = factor_graph.add_variable(X3, 1).unwrap();
    //         let x4 = factor_graph.add_variable(X4, 1).unwrap();
    //         let x5 = factor_graph.add_variable(X5, 1).unwrap();
    //
    //         factor_graph
    //             .add_prior(|id| Ok(MyPrior::new(id, x0)))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x0, x1])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x2, x3])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x3, x4])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x4, x5])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x2, x5])))
    //             .unwrap();
    //
    //         let smoother = Smoother::new(factor_graph);
    //
    //         let forward_messages = smoother
    //             .full_pass(&[x0, x1, x2, x3, x4, x5])
    //             .expect("forward pass should work");
    //
    //         let backward_messages = smoother
    //             .full_pass(&[x5, x4, x3, x2, x1, x0])
    //             .expect("backward pass should work");
    //
    //         assert!(forward_messages[x0].is_empty());
    //         assert_eq!(forward_messages[x1].len(), 1);
    //         assert_eq!(forward_messages[x2].len(), 1);
    //         assert_eq!(forward_messages[x3].len(), 1);
    //         assert_eq!(forward_messages[x4].len(), 2);
    //         assert_eq!(forward_messages[x4][0].keys, vec![x2]);
    //         assert_eq!(forward_messages[x4][1].keys, vec![x2, x4]);
    //         assert_eq!(forward_messages[x5].len(), 1);
    //         assert_eq!(forward_messages[x5][0].keys, vec![x5]);
    //
    //         assert!(backward_messages[x5].is_empty());
    //         assert!(backward_messages[x4].is_empty());
    //         assert_eq!(backward_messages[x3].len(), 1);
    //         assert_eq!(backward_messages[x3][0].keys, vec![x3, x5]);
    //         assert_eq!(backward_messages[x2].len(), 1);
    //         assert_eq!(backward_messages[x2][0].keys, vec![x2]);
    //         assert_eq!(backward_messages[x1].len(), 1);
    //         assert_eq!(backward_messages[x0].len(), 1);
    //         assert_eq!(backward_messages[x0][0].keys, vec![x0]);
    //     }
    //
    //     #[test]
    //     fn test_forward_backward_with_nested_loops() {
    //         let mut factor_graph = FactorGraph::new();
    //
    //         let x0 = factor_graph.add_variable(X0, 1).unwrap();
    //         let x1 = factor_graph.add_variable(X1, 1).unwrap();
    //         let x2 = factor_graph.add_variable(X2, 1).unwrap();
    //         let x3 = factor_graph.add_variable(X3, 1).unwrap();
    //         let x4 = factor_graph.add_variable(X4, 1).unwrap();
    //
    //         factor_graph
    //             .add_prior(|id| Ok(MyPrior::new(id, x0)))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x0, x1])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x2, x3])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x3, x4])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x0, x3, x4])))
    //             .unwrap();
    //         factor_graph
    //             .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x4])))
    //             .unwrap();
    //
    //         let smoother = Smoother::new(factor_graph);
    //
    //         let forward_messages = smoother
    //             .full_pass(&[x0, x1, x2, x3, x4])
    //             .expect("forward pass should work");
    //
    //         let backward_messages = smoother
    //             .full_pass(&[x4, x3, x2, x1, x0])
    //             .expect("backward pass should work");
    //
    //         assert!(forward_messages[x0].is_empty());
    //         assert!(forward_messages[x1].is_empty());
    //         assert!(forward_messages[x2].is_empty());
    //         assert_eq!(forward_messages[x3].len(), 1);
    //         assert_eq!(forward_messages[x4].len(), 1);
    //         assert_eq!(forward_messages[x4][0].keys, vec![x4]);
    //
    //         assert!(backward_messages[x4].is_empty());
    //         assert!(backward_messages[x3].is_empty());
    //         assert!(backward_messages[x2].is_empty());
    //         assert_eq!(backward_messages[x1].len(), 1);
    //         assert_eq!(backward_messages[x1][0].keys, vec![x1, x3]);
    //         assert_eq!(backward_messages[x0].len(), 1);
    //         assert_eq!(backward_messages[x0][0].keys, vec![x0]);
    //
    //         // let key = x1;
    //         // println!(
    //         //     "key {} has forward messages {:?} and backward messages {:?}",
    //         //     key, forward_messages[key], backward_messages[key]
    //         // );
    //
    //         // for &key in &[x0, x1, x2, x3, x4] {
    //         //     println!(
    //         //         "key {} has forward messages {:?} and backward messages {:?}",
    //         //         key, forward_messages[key], backward_messages[key]
    //         //     );
    //         // }
    //         // assert!(false);
    //     }
}

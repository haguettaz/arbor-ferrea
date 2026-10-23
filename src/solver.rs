use std::collections::HashSet;
use std::fmt::Debug;

use anyhow::{Context, Result};
use typed_arena::Arena;

use crate::core::*;

pub struct Smoother<P: Prior, R: Relation> {
    pub factor_graph: FactorGraph<P, R>,
    arena: Arena<R>,
}

impl<P: Prior, R: Relation> Smoother<P, R> {
    pub fn new(factor_graph: FactorGraph<P, R>) -> Self {
        let arena = Arena::new();
        Smoother {
            factor_graph,
            arena,
        }
    }
    /// A full pass over the factor graph in the specified elimination order
    /// Messages are allocated into the provided arena
    /// The returned messages are references to the corresponding messages
    /// All messages are kept in memory.
    pub fn full_pass<'a>(&'a self, order: &[usize]) -> Result<Vec<Vec<&'a R>>>
    where
        P: Prior + Default,
        R: Relation + Default,
    {
        let mut messages = vec![vec![]; order.len()];

        let mut active_keys: HashSet<usize> = HashSet::new();
        let mut coupled_keys: HashSet<usize> = HashSet::new();
        let mut eliminated_keys: HashSet<usize> = HashSet::new();

        let mut active_priors: Vec<&'a P> = self.factor_graph.priors().iter().collect();
        let mut active_relations: Vec<&'a R> = self.factor_graph.relations().iter().collect();

        let mut eliminated_priors: Vec<&'a P> = Vec::new();
        let mut eliminated_relations: Vec<&'a R> = Vec::new();

        for key in order {
            active_keys.insert(*key);

            coupled_keys.clear();
            coupled_keys.extend(
                active_relations
                    .iter()
                    .filter(|relation| relation.keys().iter().any(|k| !(active_keys.contains(k))))
                    .flat_map(|relation| relation.keys())
                    .copied(),
            );
            eliminated_keys.clear();
            eliminated_keys.extend(active_keys.iter());
            eliminated_keys.retain(|k| !(k == key || coupled_keys.contains(k)));
            active_keys.retain(|k| !eliminated_keys.contains(k));

            // Note: no need to clear eliminated_priors since drain(..) empties it
            let mut i = 0;
            while i < active_priors.len() {
                if eliminated_keys.contains(&active_priors[i].key()) {
                    eliminated_priors.push(active_priors.remove(i));
                } else {
                    i += 1;
                }
            }

            let mut i = 0;
            while i < active_relations.len() {
                if active_relations[i]
                    .keys()
                    .iter()
                    .any(|k| eliminated_keys.contains(k))
                {
                    eliminated_relations.push(active_relations.remove(i));
                } else {
                    i += 1;
                }
            }

            if !eliminated_keys.is_empty() {
                // Close the box using the drained eliminated factors (priors + relations)
                let remaining_keys: Vec<usize> = active_keys
                    .iter()
                    .filter(|&k| eliminated_relations.iter().any(|r| r.depends_on(*k)))
                    .copied()
                    .collect();
                active_relations.push(
                    self.arena.alloc(
                        R::from(
                            eliminated_priors.drain(..),
                            eliminated_relations.drain(..),
                            remaining_keys,
                            eliminated_keys.iter().copied(),
                            |k| self.factor_graph.size(k).unwrap_or(0),
                        )
                        .with_context(|| format!("Failed to compute message for key {}", key))?,
                    ),
                );
            }

            // active_keys.insert(*key);
            messages[*key].extend(active_relations.iter().filter(|r| r.is_message()).copied());
        }

        Ok(messages)
    }
}

pub struct Filter<P: Prior, R: Relation> {
    factor_graph: FactorGraph<P, R>,
}

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
//         let mut eliminated_keys: HashSet<usize> = HashSet::new();
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
//             eliminated_keys.clear();
//             eliminated_keys.extend(active_keys.difference(&coupled_keys).copied());
//
//             // Note: no need to clear eliminated_priors since drain(..) empties it
//             let mut i = 0;
//             while i < active_priors.len() {
//                 if eliminated_keys.contains(&active_priors[i].key()) {
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
//                     .any(|k| eliminated_keys.contains(k))
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
//                 eliminated_keys.iter().copied(),
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
//             active_keys.retain(|k| !eliminated_keys.contains(k));
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
//         let mut eliminated_keys: HashSet<usize> = HashSet::new();
//         let mut eliminated_factors: Vec<AnyFactor<'a, F>> = Vec::new();
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
//             eliminated_keys.clear();
//             eliminated_keys.extend(active_keys.difference(&coupled_keys).copied());
//
//             // Note: no need to clear eliminated_factors since drain(..) empties it
//             let mut i = 0;
//             while i < active_factors.len() {
//                 if active_factors[i]
//                     .keys()
//                     .iter()
//                     .any(|k| eliminated_keys.contains(k))
//                 {
//                     eliminated_factors.push(active_factors.remove(i));
//                 } else {
//                     i += 1;
//                 }
//             }
//
//             // Close the box using the drained eliminated factors
//             let message = <<F as Factor>::TargetMessage as Message>::close(
//                 eliminated_factors.drain(..),
//                 eliminated_keys.iter().copied(),
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
//             active_keys.retain(|k| !eliminated_keys.contains(k));
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

    #[derive(Clone, Debug, Default)]
    pub struct MyPrior {
        pub id: usize,
        pub key: usize,
    }

    impl Prior for MyPrior {
        fn id(&self) -> usize {
            self.id
        }

        fn key(&self) -> usize {
            self.key
        }
    }

    impl MyPrior {
        pub fn new(id: usize, key: usize) -> Self {
            Self { id, key }
        }
    }

    #[derive(Debug, PartialEq, Default)]
    pub struct MyRelation {
        pub ids: Vec<usize>, // Added so Factor::id() has a value to return
        pub keys: Vec<usize>,
    }

    impl MyRelation {
        pub fn new(ids: Vec<usize>, keys: Vec<usize>) -> Self {
            Self { ids, keys }
        }
    }

    // Base implementation for Factor
    impl Relation for MyRelation {
        fn keys(&self) -> &[usize] {
            &self.keys
        }

        fn ids(&self) -> &[usize] {
            &self.ids
        }

        fn from<'a, P, R>(
            priors: impl IntoIterator<Item = &'a P>,
            relations: impl IntoIterator<Item = &'a R>,
            keep_keys: impl IntoIterator<Item = usize>,
            _clear_keys: impl IntoIterator<Item = usize>,
            _ctx: impl Fn(usize) -> usize,
        ) -> Result<Self>
        where
            P: Prior + 'a,
            R: Relation + 'a,
        {
            let mut ids: Vec<usize> = priors
                .into_iter()
                .map(|p| p.id())
                .chain(relations.into_iter().flat_map(|r| r.ids()).copied())
                .collect();
            ids.sort_unstable();

            let mut keys: Vec<usize> = keep_keys.into_iter().collect();
            keys.sort_unstable();

            let msg = MyRelation { ids, keys };
            Ok(msg)
        }
    }

    const X0: Symbol = Symbol('x', 0);
    const X1: Symbol = Symbol('x', 1);
    const X2: Symbol = Symbol('x', 2);
    const X3: Symbol = Symbol('x', 3);
    const X4: Symbol = Symbol('x', 4);
    const X5: Symbol = Symbol('x', 5);

    #[test]
    fn test_forward_backward_in_line() {
        let mut factor_graph = FactorGraph::new();

        let x0 = factor_graph.add_variable(X0, 1).unwrap();
        let x1 = factor_graph.add_variable(X1, 1).unwrap();
        let x2 = factor_graph.add_variable(X2, 1).unwrap();
        let x3 = factor_graph.add_variable(X3, 1).unwrap();
        let x4 = factor_graph.add_variable(X4, 1).unwrap();

        factor_graph
            .add_prior(|id| Ok(MyPrior::new(id, x0)))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x0, x1])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x2, x3])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x3, x4])))
            .unwrap();

        let smoother = Smoother::new(factor_graph);

        let forward_messages = smoother
            .full_pass(&[x0, x1, x2, x3, x4])
            .expect("forward pass should work");

        let backward_messages = smoother
            .full_pass(&[x4, x3, x2, x1, x0])
            .expect("backward pass should work");

        assert!(forward_messages[x0].is_empty());
        assert_eq!(forward_messages[x1].len(), 1);
        assert_eq!(forward_messages[x2].len(), 1);
        assert_eq!(forward_messages[x2][0].keys, vec![x2]);
        assert_eq!(forward_messages[x3].len(), 1);
        assert_eq!(forward_messages[x4].len(), 1);
        assert_eq!(forward_messages[x4][0].keys, vec![x4]);

        assert!(backward_messages[x4].is_empty());
        assert!(backward_messages[x3].is_empty());
        assert_eq!(backward_messages[x2].len(), 1);
        assert_eq!(backward_messages[x2][0].keys, vec![x2]);
        assert_eq!(backward_messages[x1].len(), 1);
        assert_eq!(backward_messages[x0].len(), 1);
        assert_eq!(backward_messages[x0][0].keys, vec![x0]);
    }

    #[test]
    fn test_solver_loops() {
        let mut factor_graph = FactorGraph::new();

        let x0 = factor_graph.add_variable(X0, 1).unwrap();
        let x1 = factor_graph.add_variable(X1, 1).unwrap();
        let x2 = factor_graph.add_variable(X2, 1).unwrap();
        let x3 = factor_graph.add_variable(X3, 1).unwrap();
        let x4 = factor_graph.add_variable(X4, 1).unwrap();
        let x5 = factor_graph.add_variable(X5, 1).unwrap();

        factor_graph
            .add_prior(|id| Ok(MyPrior::new(id, x0)))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x0, x1])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x2, x3])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x3, x4])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x4, x5])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x2, x5])))
            .unwrap();

        let smoother = Smoother::new(factor_graph);

        let forward_messages = smoother
            .full_pass(&[x0, x1, x2, x3, x4, x5])
            .expect("forward pass should work");

        let backward_messages = smoother
            .full_pass(&[x5, x4, x3, x2, x1, x0])
            .expect("backward pass should work");

        assert!(forward_messages[x0].is_empty());
        assert_eq!(forward_messages[x1].len(), 1);
        assert_eq!(forward_messages[x2].len(), 1);
        assert_eq!(forward_messages[x3].len(), 1);
        assert_eq!(forward_messages[x4].len(), 2);
        assert_eq!(forward_messages[x4][0].keys, vec![x2]);
        assert_eq!(forward_messages[x4][1].keys, vec![x2, x4]);
        assert_eq!(forward_messages[x5].len(), 1);
        assert_eq!(forward_messages[x5][0].keys, vec![x5]);

        assert!(backward_messages[x5].is_empty());
        assert!(backward_messages[x4].is_empty());
        assert_eq!(backward_messages[x3].len(), 1);
        assert_eq!(backward_messages[x3][0].keys, vec![x3, x5]);
        assert_eq!(backward_messages[x2].len(), 1);
        assert_eq!(backward_messages[x2][0].keys, vec![x2]);
        assert_eq!(backward_messages[x1].len(), 1);
        assert_eq!(backward_messages[x0].len(), 1);
        assert_eq!(backward_messages[x0][0].keys, vec![x0]);
    }

    #[test]
    fn test_forward_backward_with_nested_loops() {
        let mut factor_graph = FactorGraph::new();

        let x0 = factor_graph.add_variable(X0, 1).unwrap();
        let x1 = factor_graph.add_variable(X1, 1).unwrap();
        let x2 = factor_graph.add_variable(X2, 1).unwrap();
        let x3 = factor_graph.add_variable(X3, 1).unwrap();
        let x4 = factor_graph.add_variable(X4, 1).unwrap();

        factor_graph
            .add_prior(|id| Ok(MyPrior::new(id, x0)))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x0, x1])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x2, x3])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x3, x4])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x0, x3, x4])))
            .unwrap();
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x4])))
            .unwrap();

        let smoother = Smoother::new(factor_graph);

        let forward_messages = smoother
            .full_pass(&[x0, x1, x2, x3, x4])
            .expect("forward pass should work");

        let backward_messages = smoother
            .full_pass(&[x4, x3, x2, x1, x0])
            .expect("backward pass should work");

        assert!(forward_messages[x0].is_empty());
        assert!(forward_messages[x1].is_empty());
        assert!(forward_messages[x2].is_empty());
        assert_eq!(forward_messages[x3].len(), 1);
        assert_eq!(forward_messages[x4].len(), 1);
        assert_eq!(forward_messages[x4][0].keys, vec![x4]);

        assert!(backward_messages[x4].is_empty());
        assert!(backward_messages[x3].is_empty());
        assert!(backward_messages[x2].is_empty());
        assert_eq!(backward_messages[x1].len(), 1);
        assert_eq!(backward_messages[x1][0].keys, vec![x1, x3]);
        assert_eq!(backward_messages[x0].len(), 1);
        assert_eq!(backward_messages[x0][0].keys, vec![x0]);

        // let key = x1;
        // println!(
        //     "key {} has forward messages {:?} and backward messages {:?}",
        //     key, forward_messages[key], backward_messages[key]
        // );

        // for &key in &[x0, x1, x2, x3, x4] {
        //     println!(
        //         "key {} has forward messages {:?} and backward messages {:?}",
        //         key, forward_messages[key], backward_messages[key]
        //     );
        // }
        // assert!(false);
    }
}

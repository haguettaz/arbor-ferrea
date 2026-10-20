use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fmt::Debug;

use crate::factor_graph::*;

#[derive(Debug)]
pub struct ReducedFactorGraph<F: Factor> {
    key: usize,
    forward_messages: Vec<F::TargetMessage>,
    backward_messages: Vec<F::TargetMessage>,
    marginal: Option<F::TargetMessage>,
}

impl<F: Factor> ReducedFactorGraph<F> {
    pub fn new(key: usize) -> Self {
        ReducedFactorGraph {
            key,
            forward_messages: Vec::new(),
            backward_messages: Vec::new(),
            marginal: None,
        }
    }
}

pub struct Solver<'a, F: Factor + 'a> {
    factor_graph: &'a FactorGraph<F>,
    reduced_factor_graphs: Vec<ReducedFactorGraph<F>>,
}

impl<'a, F> Solver<'a, F>
where
    F: Factor + Debug + Default + 'a,
    F::TargetMessage: Message<TargetFactor = F>,
{
    pub fn new(factor_graph: &'a FactorGraph<F>) -> Result<Self> {
        let mut reduced_factor_graphs = Vec::with_capacity(factor_graph.n_keys());
        for key in 0..factor_graph.n_keys() {
            reduced_factor_graphs.push(ReducedFactorGraph::new(key));
        }

        // Forward pass
        let mut active_keys: HashSet<usize> = HashSet::new();
        let mut active_factors: Vec<AnyFactor<'a, F>> = factor_graph
            .factors()
            .iter()
            .map(AnyFactor::BorrowedFactor)
            .collect();

        let mut coupled_keys: HashSet<usize> = HashSet::new();
        let mut eliminated_keys: HashSet<usize> = HashSet::new();
        let mut eliminated_factors: Vec<AnyFactor<'a, F>> = Vec::new();

        for graph in reduced_factor_graphs.iter_mut() {
            let key = graph.key;

            coupled_keys.clear();
            coupled_keys.extend(
                active_factors
                    .iter()
                    .filter(|factor| {
                        let min_k = factor.keys().iter().min().unwrap_or(&key);
                        let max_k = factor.keys().iter().max().unwrap_or(&key);
                        min_k < &key && max_k > &key
                    })
                    .flat_map(|factor| factor.keys())
                    .copied(),
            );

            eliminated_keys.clear();
            eliminated_keys.extend(active_keys.difference(&coupled_keys).copied());

            // Note: no need to clear eliminated_factors since drain(..) empties it
            let mut i = 0;
            while i < active_factors.len() {
                if active_factors[i]
                    .keys()
                    .iter()
                    .any(|k| eliminated_keys.contains(k))
                {
                    eliminated_factors.push(active_factors.remove(i));
                } else {
                    i += 1;
                }
            }

            // Close the box using the drained eliminated factors
            let message = <<F as Factor>::TargetMessage as Message>::close(
                eliminated_factors.drain(..),
                eliminated_keys.iter().copied(),
            )
            .with_context(|| format!("Failed to compute the forward messages for key {}", key))?;

            if !message.is_empty() {
                active_factors.push(AnyFactor::OwnedMessage(message));
            }

            // Iterate by reference so we don't consume the active_factors pool!
            for factor in &active_factors {
                if let AnyFactor::OwnedMessage(msg) = factor {
                    graph.forward_messages.push(msg.clone());
                }
            }

            // Update tracking keys without allocating a new HashSet
            active_keys.retain(|k| !eliminated_keys.contains(k));
            active_keys.insert(key);
        }

        // Backward pass
        let mut active_keys: HashSet<usize> = HashSet::new();
        let mut active_factors: Vec<AnyFactor<'a, F>> = factor_graph
            .factors()
            .iter()
            .map(AnyFactor::BorrowedFactor)
            .collect();

        let mut coupled_keys: HashSet<usize> = HashSet::new();
        let mut eliminated_keys: HashSet<usize> = HashSet::new();
        let mut eliminated_factors: Vec<AnyFactor<'a, F>> = Vec::new();

        for graph in reduced_factor_graphs.iter_mut().rev() {
            let key = graph.key;

            coupled_keys.clear();
            coupled_keys.extend(
                active_factors
                    .iter()
                    .filter(|factor| {
                        let min_k = factor.keys().iter().min().unwrap_or(&key);
                        let max_k = factor.keys().iter().max().unwrap_or(&key);
                        min_k < &key && max_k > &key
                    })
                    .flat_map(|factor| factor.keys())
                    .copied(),
            );

            eliminated_keys.clear();
            eliminated_keys.extend(active_keys.difference(&coupled_keys).copied());

            // Note: no need to clear eliminated_factors since drain(..) empties it
            let mut i = 0;
            while i < active_factors.len() {
                if active_factors[i]
                    .keys()
                    .iter()
                    .any(|k| eliminated_keys.contains(k))
                {
                    eliminated_factors.push(active_factors.remove(i));
                } else {
                    i += 1;
                }
            }

            // Close the box using the drained eliminated factors
            let message = <<F as Factor>::TargetMessage as Message>::close(
                eliminated_factors.drain(..),
                eliminated_keys.iter().copied(),
            )
            .with_context(|| format!("Failed to compute the backward messages for key {}", key))?;

            if !message.is_empty() {
                active_factors.push(AnyFactor::OwnedMessage(message));
            }

            // Iterate by reference so we don't consume the active_factors pool!
            for factor in &active_factors {
                if let AnyFactor::OwnedMessage(msg) = factor {
                    graph.backward_messages.push(msg.clone());
                }
            }

            // Update tracking keys without allocating a new HashSet
            active_keys.retain(|k| !eliminated_keys.contains(k));
            active_keys.insert(key);
        }

        let solver = Solver {
            factor_graph,
            reduced_factor_graphs,
        };
        Ok(solver)
    }

    pub fn optimize(&mut self) -> Result<()> {
        for reduced_factor_graph in self.reduced_factor_graphs.iter_mut() {
            let key = reduced_factor_graph.key;
            let closed_factors: HashSet<usize> = reduced_factor_graph
                .forward_messages
                .iter()
                .chain(reduced_factor_graph.backward_messages.iter())
                .flat_map(|msg| msg.closed_factors())
                .copied()
                .collect();

            // Map remaining factors to the BorrowedFactor variant
            let rem_factors_iter = self
                .factor_graph
                .factors()
                .iter()
                .filter(|factor| !closed_factors.contains(&factor.id()))
                .map(AnyFactor::BorrowedFactor);

            // Map messages to the OwnedMessage variant.
            let messages_iter = reduced_factor_graph
                .forward_messages
                .iter()
                .chain(reduced_factor_graph.backward_messages.iter())
                .cloned()
                .map(AnyFactor::OwnedMessage);

            // Collect them into a unified Vec so we can iterate over them multiple times
            let all_factors: Vec<AnyFactor<'a, F>> =
                rem_factors_iter.chain(messages_iter).collect();

            // Extract all keys to eliminate
            let mut all_keys: HashSet<usize> = all_factors
                .iter()
                .flat_map(|node| node.keys().iter().copied())
                .collect();
            all_keys.remove(&reduced_factor_graph.key);

            // Close the box using the unified Vec and the keys
            let message = <<F as Factor>::TargetMessage as Message>::close(all_factors, all_keys)
                .with_context(|| {
                format!("Failed to compute the marginalize for key {}", key)
            })?;
            reduced_factor_graph.marginal = Some(message);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Default)]
    pub struct MyFactor {
        pub id: usize, // Added so Factor::id() has a value to return
        pub keys: Vec<usize>,
    }

    impl MyFactor {
        pub fn new(id: usize, keys: Vec<usize>) -> Result<MyFactor> {
            Ok(MyFactor { id, keys })
        }
    }

    #[derive(Clone, Debug)]
    pub struct MyMessage {
        pub keys: Vec<usize>,
        pub closed_factors: Vec<usize>,
    }

    // Base implementation for Factor
    impl FactorBase for MyFactor {
        fn keys(&self) -> &[usize] {
            &self.keys
        }
    }

    impl Factor for MyFactor {
        type TargetMessage = MyMessage;

        fn id(&self) -> usize {
            self.id
        }
    }

    // Base implementation for Message
    impl FactorBase for MyMessage {
        fn keys(&self) -> &[usize] {
            &self.keys
        }
    }

    impl Message for MyMessage {
        type TargetFactor = MyFactor;

        fn closed_factors(&self) -> &[usize] {
            &self.closed_factors
        }

        fn close<'a, I, K>(factors: I, on_keys: K) -> Result<Self>
        where
            I: IntoIterator<Item = AnyFactor<'a, Self::TargetFactor>>,
            K: IntoIterator<Item = usize>,
            Self::TargetFactor: 'a,
        {
            let mut closed_factors = Vec::new();
            let mut all_keys = HashSet::new();

            // Aggregate all keys and track which factors/messages are being closed
            for factor in factors {
                for &key in factor.keys() {
                    all_keys.insert(key);
                }

                match factor {
                    AnyFactor::BorrowedFactor(f) => {
                        closed_factors.push(f.id());
                    }
                    AnyFactor::OwnedMessage(m) => {
                        closed_factors.extend_from_slice(m.closed_factors());
                    }
                }
            }

            // Eliminate the specified keys
            for key in on_keys {
                all_keys.remove(&key);
            }

            // Construct and return the new Message
            let mut final_keys: Vec<usize> = all_keys.into_iter().collect();
            final_keys.sort_unstable(); // Optional: keeps keys deterministic
            closed_factors.sort_unstable(); // Optional: keeps keys deterministic

            Ok(Self {
                keys: final_keys,
                closed_factors,
            })
        }
    }

    const X1: Symbol = Symbol('x', 1);
    const X2: Symbol = Symbol('x', 2);
    const X3: Symbol = Symbol('x', 3);
    const X4: Symbol = Symbol('x', 4);
    const X5: Symbol = Symbol('x', 5);
    const X6: Symbol = Symbol('x', 6);

    #[test]
    fn test_solver_nested_loops() {
        let mut factor_graph = FactorGraph::new();

        let x1 = factor_graph.add_variable(X1, 1).unwrap();
        let x2 = factor_graph.add_variable(X2, 1).unwrap();
        let x3 = factor_graph.add_variable(X3, 1).unwrap();
        let x4 = factor_graph.add_variable(X4, 1).unwrap();
        let x5 = factor_graph.add_variable(X5, 1).unwrap();

        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x1, x2]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x2, x3]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x3, x4]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x4, x5]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x1, x4, x5]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x2, x5]))
            .unwrap();

        let mut solver =
            Solver::new(&factor_graph).expect("new solver from factor graph should succeed");
        assert!(solver.reduced_factor_graphs[0].forward_messages.is_empty());
        assert!(solver.reduced_factor_graphs[3].forward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[3].forward_messages[0].keys,
            [1, 3]
        );
        assert_eq!(
            solver.reduced_factor_graphs[3].forward_messages[0].closed_factors,
            [1, 2]
        );
        assert!(solver.reduced_factor_graphs[4].forward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[0].keys,
            [4]
        );
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[0].closed_factors,
            [0, 1, 2, 3, 4, 5]
        );

        assert!(solver.reduced_factor_graphs[0].backward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[0].backward_messages[0].keys,
            [0]
        );
        assert_eq!(
            solver.reduced_factor_graphs[0].backward_messages[0].closed_factors,
            [0, 1, 2, 3, 4, 5]
        );
        assert!(solver.reduced_factor_graphs[1].backward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[1].backward_messages[0].keys,
            [1, 3]
        );
        assert_eq!(
            solver.reduced_factor_graphs[1].backward_messages[0].closed_factors,
            [1, 2]
        );
        assert!(solver.reduced_factor_graphs[4].backward_messages.is_empty());

        solver.optimize().expect("Optimization should work");
        let mut factor_ids = factor_graph.factor_ids();
        factor_ids.sort_unstable();
        for reduced_factor_graph in &solver.reduced_factor_graphs {
            let closed_factors = &reduced_factor_graph
                .marginal
                .as_ref()
                .unwrap()
                .closed_factors;
            assert_eq!(closed_factors, &factor_ids);
        }
    }

    #[test]
    fn test_solver_line() {
        let mut factor_graph = FactorGraph::new();

        let x1 = factor_graph.add_variable(X1, 1).unwrap();
        let x2 = factor_graph.add_variable(X2, 1).unwrap();
        let x3 = factor_graph.add_variable(X3, 1).unwrap();
        let x4 = factor_graph.add_variable(X4, 1).unwrap();
        let x5 = factor_graph.add_variable(X5, 1).unwrap();

        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x1, x2]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x2, x3]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x3, x4]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x4, x5]))
            .unwrap();

        let mut solver =
            Solver::new(&factor_graph).expect("new solver from factor graph should succeed");

        assert!(solver.reduced_factor_graphs[0].forward_messages.is_empty());
        assert!(solver.reduced_factor_graphs[3].forward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[3].forward_messages[0].keys,
            [3]
        );
        assert_eq!(
            solver.reduced_factor_graphs[3].forward_messages[0].closed_factors,
            [0, 1, 2]
        );
        assert!(solver.reduced_factor_graphs[4].forward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[0].keys,
            [4]
        );
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[0].closed_factors,
            [0, 1, 2, 3]
        );

        assert!(solver.reduced_factor_graphs[0].backward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[0].backward_messages[0].keys,
            [0]
        );
        assert_eq!(
            solver.reduced_factor_graphs[0].backward_messages[0].closed_factors,
            [0, 1, 2, 3]
        );
        assert!(solver.reduced_factor_graphs[1].backward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[1].backward_messages[0].keys,
            [1]
        );
        assert_eq!(
            solver.reduced_factor_graphs[1].backward_messages[0].closed_factors,
            [1, 2, 3]
        );
        assert!(solver.reduced_factor_graphs[4].backward_messages.is_empty());

        solver.optimize().expect("Optimization should work");
        let mut factor_ids = factor_graph.factor_ids();
        factor_ids.sort_unstable();
        for reduced_factor_graph in &solver.reduced_factor_graphs {
            let closed_factors = &reduced_factor_graph
                .marginal
                .as_ref()
                .unwrap()
                .closed_factors;
            assert_eq!(closed_factors, &factor_ids);
        }
    }

    #[test]
    fn test_solver_loops() {
        let mut factor_graph = FactorGraph::new();

        let x1 = factor_graph.add_variable(X1, 1).unwrap();
        let x2 = factor_graph.add_variable(X2, 1).unwrap();
        let x3 = factor_graph.add_variable(X3, 1).unwrap();
        let x4 = factor_graph.add_variable(X4, 1).unwrap();
        let x5 = factor_graph.add_variable(X5, 1).unwrap();
        let x6 = factor_graph.add_variable(X6, 1).unwrap();

        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x1, x2]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x2, x3]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x3, x4]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x4, x5]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x5, x6]))
            .unwrap();
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x3, x6]))
            .unwrap();

        let mut solver =
            Solver::new(&factor_graph).expect("new solver from factor graph should succeed");

        assert!(solver.reduced_factor_graphs[0].forward_messages.is_empty());
        assert!(solver.reduced_factor_graphs[4].forward_messages.len() == 2);
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[0].keys,
            [2]
        );
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[0].closed_factors,
            [0, 1]
        );
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[1].keys,
            [2, 4]
        );
        assert_eq!(
            solver.reduced_factor_graphs[4].forward_messages[1].closed_factors,
            [2, 3]
        );

        assert!(solver.reduced_factor_graphs[2].backward_messages.len() == 1);
        assert_eq!(
            solver.reduced_factor_graphs[2].backward_messages[0].keys,
            [2]
        );
        assert_eq!(
            solver.reduced_factor_graphs[2].backward_messages[0].closed_factors,
            [2, 3, 4, 5]
        );
        assert!(solver.reduced_factor_graphs[4].backward_messages.is_empty());
        assert!(solver.reduced_factor_graphs[5].backward_messages.is_empty());

        solver.optimize().expect("Optimization should work");
        let mut factor_ids = factor_graph.factor_ids();
        factor_ids.sort_unstable();
        for reduced_factor_graph in &solver.reduced_factor_graphs {
            let closed_factors = &reduced_factor_graph
                .marginal
                .as_ref()
                .unwrap()
                .closed_factors;
            assert_eq!(closed_factors, &factor_ids);
        }
    }
}

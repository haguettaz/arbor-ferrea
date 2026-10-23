// use anyhow::Result;
// use std::collections::HashSet;
//
// use arbor_ferrea::core::*;
//
// #[derive(Debug, PartialEq, Default)]
// pub struct MyFactor {
//     pub id: usize, // Added so Factor::id() has a value to return
//     pub keys: Vec<usize>,
// }
//
// impl MyFactor {
//     pub fn new(id: usize, keys: Vec<usize>) -> Result<MyFactor> {
//         Ok(MyFactor { id, keys })
//     }
// }
//
// #[derive(Clone, Debug)]
// pub struct MyMessage {
//     pub keys: Vec<usize>,
//     pub closed_factors: Vec<usize>,
// }
//
// // Base implementation for Factor
// impl FactorBase for MyFactor {
//     fn keys(&self) -> &[usize] {
//         &self.keys
//     }
// }
//
// impl Factor for MyFactor {
//     type TargetMessage = MyMessage;
//
//     fn id(&self) -> usize {
//         self.id
//     }
// }
//
// // Base implementation for Message
// impl FactorBase for MyMessage {
//     fn keys(&self) -> &[usize] {
//         &self.keys
//     }
// }
//
// impl Message for MyMessage {
//     type TargetFactor = MyFactor;
//
//     fn closed_factors(&self) -> &[usize] {
//         &self.closed_factors
//     }
//
//     fn close<'a, I, K, C>(factors: I, on_keys: K, _ctx: C) -> Result<Self>
//     where
//         I: IntoIterator<Item = AnyFactor<'a, Self::TargetFactor>>,
//         K: IntoIterator<Item = usize>,
//         C: Fn(usize) -> usize,
//         Self::TargetFactor: 'a,
//     {
//         let mut closed_factors = Vec::new();
//         let mut all_keys = HashSet::new();
//
//         // Aggregate all keys and track which factors/messages are being closed
//         for factor in factors {
//             for &key in factor.keys() {
//                 all_keys.insert(key);
//             }
//
//             match factor {
//                 AnyFactor::BorrowedFactor(f) => {
//                     closed_factors.push(f.id());
//                 }
//                 AnyFactor::OwnedMessage(m) => {
//                     closed_factors.extend_from_slice(m.closed_factors());
//                 }
//             }
//         }
//
//         // Eliminate the specified keys
//         for key in on_keys {
//             all_keys.remove(&key);
//         }
//
//         // Construct and return the new Message
//         let mut final_keys: Vec<usize> = all_keys.into_iter().collect();
//         final_keys.sort_unstable(); // Optional: keeps keys deterministic
//         closed_factors.sort_unstable(); // Optional: keeps keys deterministic
//
//         Ok(Self {
//             keys: final_keys,
//             closed_factors,
//         })
//     }
// }
//
// const X1: Symbol = Symbol('x', 1);
// const X2: Symbol = Symbol('x', 2);
// const X3: Symbol = Symbol('x', 3);
// const X4: Symbol = Symbol('x', 4);
// const X5: Symbol = Symbol('x', 5);
//
// #[test]
// fn test_factor_graph() {
//     let mut factor_graph = FactorGraph::new();
//
//     let x1 = factor_graph
//         .add_variable(X1, 1)
//         .expect("Adding x1 should work");
//     let x2 = factor_graph
//         .add_variable(X2, 1)
//         .expect("Adding x2 should work");
//     let x3 = factor_graph
//         .add_variable(X3, 1)
//         .expect("Adding x3 should work");
//     let x4 = factor_graph
//         .add_variable(X4, 10000000)
//         .expect("Adding x4 should work");
//     let x5 = factor_graph
//         .add_variable(X5, 1)
//         .expect("Adding x5 should work");
//     let _ = factor_graph
//         .add_variable(X3, 2)
//         .expect_err("Adding x3 again should fail");
//     let x6 = x5 + 1;
//     assert_eq!(factor_graph.n_keys(), 5);
//
//     factor_graph
//         .add_factor(|id| MyFactor::new(id, vec![x1, x2]))
//         .expect("Adding factor between x1 and x2 should work");
//     factor_graph
//         .add_factor(|id| MyFactor::new(id, vec![x2, x3, x1, x5]))
//         .expect("Adding factor between x1, x2, x3, and x5 should work");
//     factor_graph
//         .add_factor(|id| MyFactor::new(id, vec![x4, x6]))
//         .expect_err("Adding factor between x4 and x6 should fail");
//     factor_graph
//         .add_factor(|id| MyFactor::new(id, vec![]))
//         .expect("Adding empty factor should work");
//     assert_eq!(factor_graph.n_factors(), 3);
// }

// use anyhow::Result;
//
// use std::collections::{HashMap, HashSet};
//
// use arbor_ferrea::gaussian::GaussianFactor;
//
// pub struct GaussianMessage {
//     /// The keys of the variables it depends on
//     keys: Vec<usize>,
//     /// The keys of the factors it summarizes
//     factor_ids: Vec<usize>,
// }
//
// pub struct ComputationNode {
//     key: usize, // The key of the target variable
//     msgf: Option<Box<GaussianMessage>>,
//     msgb: Option<Box<GaussianMessage>>,
// }
//
// impl ComputationNode {
//     pub fn new(key: usize) -> Self {
//         ComputationNode {
//             key,
//             msgf: None,
//             msgb: None,
//         }
//     }
//
//     pub fn step<'a>(
//         &self,
//         factors: &[&'a GaussianFactor],
//         m_keys: &HashSet<usize>,
//     ) -> Result<(Box<GaussianMessage>, Vec<&'a GaussianFactor>)> {
//         // Only allocate a vector for the remaining factors
//         let rem_factors: Vec<_> = factors
//             .iter()
//             .copied()
//             .filter(|f| f.keys().iter().any(|k| !m_keys.contains(k)))
//             .collect();
//
//         // Pass a lazy iterator directly to combine
//         let active_iter = factors
//             .iter()
//             .copied()
//             .filter(|f| f.keys().iter().all(|k| m_keys.contains(k)));
//
//         // let msg = combine_factors(active_iter)?;
//
//         // Ok((Box::new(msg), rem_factors))
//     }
// }
//
// // pub fn combine_factors<'a, I>(active_factors: I) -> Result<GaussianFactor>
// // where
// //     I: IntoIterator<Item = &'a GaussianFactor>,
// // {
// //     let iter = active_factors.into_iter();
// //     // Implementation...
// //     todo!();
// // }
//
// pub fn extract_forward_backward_coupling_factors<'a>(
//     factors: &[&'a GaussianFactor],
//     key: usize,
// ) -> Vec<&'a GaussianFactor> {
//     factors
//         .iter()
//         .filter(|f| f.keys().iter().any(|&k| k < key) && f.keys().iter().any(|&k| k > key))
//         .cloned()
//         .collect()
// }
//
// pub struct ComputationLine {
//     nodes: Vec<ComputationNode>,
//     key_to_node: HashMap<usize, usize>,
// }
//
// impl ComputationLine {
//     // It is the caller responsibility to check that order has no duplicate.
//     pub fn new(order: &[usize]) -> Self {
//         let nodes = order.iter().map(|&key| ComputationNode::new(key)).collect();
//         let key_to_node = order.iter().enumerate().map(|(i, &k)| (k, i)).collect();
//         ComputationLine { nodes, key_to_node }
//     }
//
//     /// Returns the computation node in the line corresponding to the given key.
//     pub fn node(&self, key: usize) -> Option<&ComputationNode> {
//         self.key_to_node.get(&key).and_then(|&i| self.nodes.get(i))
//     }
//
//     /// Forward-filtering pass over the computation line.
//     pub fn forward<'a>(&mut self, factors: &[&'a GaussianFactor]) -> Result<()> {
//         let mut m_keys = HashSet::with_capacity(self.nodes.len());
//
//         let mut active_factors: Vec<&'a GaussianFactor> = factors.to_vec();
//
//         for node in self.nodes.iter_mut() {
//             m_keys.insert(node.key);
//             let (msgf, rem_factors) = node.step(&active_factors, &m_keys)?;
//
//             node.msgf = Some(msgf);
//             active_factors = rem_factors;
//         }
//         Ok(())
//     }
//
//     /// Backward-filtering pass over the computation line.
//     pub fn backward<'a>(&mut self, factors: &[&'a GaussianFactor]) -> Result<()> {
//         let mut m_keys = HashSet::with_capacity(self.nodes.len());
//
//         let mut active_factors: Vec<&'a GaussianFactor> = factors.to_vec();
//
//         for node in self.nodes.iter_mut().rev() {
//             m_keys.insert(node.key);
//             let (msgb, rem_factors) = node.step(&active_factors, &m_keys)?;
//             node.msgb = Some(msgb);
//             active_factors = rem_factors;
//         }
//         Ok(())
//     }
//
//     // pub fn solve(&self) -> Result<()> {
//     //     for node in self.nodes.iter() {
//     //         (node.msgf, node.msgb)
//     //     }
//     // }
// }

// fn main() -> Result<()> {
//     todo!();
// }

fn main() {}

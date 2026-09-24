use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use std::hash::Hash;

use anyhow::{Context, Result, bail};

use crate::gmp::{AXBSGaussBlock, AXGaussBlock};
use crate::vars::VarInfo;

type VarId = usize;

enum PoolBlock<'a, T> {
    Ref(&'a AXGaussBlock<T>),
    Owned(Box<AXGaussBlock<T>>),
}

impl<'a, T> std::ops::Deref for PoolBlock<'a, T> {
    type Target = AXGaussBlock<T>;
    fn deref(&self) -> &Self::Target {
        match self {
            PoolBlock::Ref(r) => *r,
            PoolBlock::Owned(o) => o.as_ref(),
        }
    }
}

/// Represents a Gaussian factor graph with a tree structure.
pub struct GaussTree<T> {
    roots: Vec<VarId>,
    parents: Vec<Option<VarId>>,   // sorted by elimination order
    children: Vec<Vec<VarId>>,     // sorted by elimination order
    nodes: Vec<AXBSGaussBlock<T>>, // sorted by elimination order
}

impl<T: Clone + Display + PartialEq + Eq + Hash> GaussTree<T> {
    pub fn new(
        roots: Vec<VarId>,
        parents: Vec<Option<VarId>>,
        children: Vec<Vec<VarId>>,
        nodes: Vec<AXBSGaussBlock<T>>,
    ) -> Self {
        Self {
            roots,
            parents,
            children,
            nodes,
        }
    }

    /// Explores the tree from roots to leaves using an iterative Depth-First Search.
    /// Applies the provided closure to each visited node.
    pub fn traverse_down<F>(&self, mut visitor: F)
    where
        F: FnMut(VarId, &AXBSGaussBlock<T>),
    {
        // Use an explicit stack to prevent stack overflows on very deep trees.
        // We seed it with all the root nodes.
        let mut stack: Vec<VarId> = self.roots.clone();

        while let Some(node_id) = stack.pop() {
            // 1. Apply the closure to the current node
            visitor(node_id, &self.nodes[node_id]);

            // 2. Push children onto the stack.
            // We reverse the iterator so the first child is popped/visited first.
            for &child_id in self.children[node_id].iter().rev() {
                stack.push(child_id);
            }
        }
    }

    /// Upward pass from the leaves to the root(s)
    /// Variables are sorted by their elimination order.
    /// Can be interpreted as max-product message passing in the original factor graph
    pub fn build<'a>(
        vars: &[VarInfo<T>],
        gauss_blocks: impl IntoIterator<Item = &'a AXGaussBlock<T>>,
    ) -> Result<Self>
    where
        T: 'a,
    {
        // Wrap original references in the PoolBlock
        let mut other_blocks: Vec<PoolBlock<'a, T>> =
            gauss_blocks.into_iter().map(PoolBlock::Ref).collect();

        // Check that there is no duplicate variable
        let seen_in_vars = vars.iter().collect::<HashSet<&VarInfo<T>>>();
        if seen_in_vars.len() != vars.len() {
            bail!("Duplicate variable found in variables");
        }
        let n_vars = vars.len();

        // Check that all variables are present in the blocks
        let seen_in_blocks = other_blocks
            .iter()
            .flat_map(|block| block.info_x.iter().map(|(var, _)| var))
            .collect::<HashSet<&VarInfo<T>>>();
        if seen_in_blocks != seen_in_vars {
            bail!("The variables in the variables do not match the variables in the blocks");
        }

        // Map each variable to its elimination order index: O(1) lookups
        let ranks: HashMap<&VarInfo<T>, usize> =
            vars.iter().enumerate().map(|(r, var)| (var, r)).collect();

        let mut nodes = Vec::with_capacity(n_vars);
        let mut roots = Vec::new();
        let mut parents = vec![None; n_vars];
        let mut children = vec![Vec::new(); n_vars];

        for (i, var) in vars.iter().enumerate() {
            // Partition shuffles the PoolBlocks
            let (var_blocks, remaining_blocks): (Vec<PoolBlock<'a, T>>, Vec<PoolBlock<'a, T>>) =
                other_blocks
                    .into_iter()
                    .partition(|block| block.info_x.iter().any(|(v, _)| v == var));

            // Extract pure references for the constructor
            let var_blocks_refs: Vec<&AXGaussBlock<T>> = var_blocks.iter().map(|b| &**b).collect();

            // Needs to be mutable so we can call eliminate_x
            let mut new_node = AXBSGaussBlock::from(var, var_blocks_refs);

            let new_block = new_node
                .eliminate_x()
                .with_context(|| format!("Problem when eliminating {}", var))?;

            let parent = new_node
                .info_s
                .iter()
                .filter_map(|(v, _)| ranks.get(v))
                .min();

            match parent {
                None => roots.push(i),
                Some(&p) => {
                    parents[i] = Some(p);
                    children[p].push(i);
                }
            }

            other_blocks = remaining_blocks;
            other_blocks.push(PoolBlock::Owned(Box::new(new_block)));
            nodes.push(new_node);
        }

        Ok(Self::new(roots, parents, children, nodes))
    }

    //     /// Downward pass from the root(s) to the leaves = solve
    //     pub fn solve(&self, out: &mut Vec<DVector<f64>>) {
    //         for r in self.roots.iter() {
    //             self.down(*r, out)
    //         }
    //     }
    //
    //     /// Decide variable of node i and propagate to its children
    //     fn down(&self, i: VarId, out: &mut Vec<DVector<f64>>) {
    //         let ms = concat_vectors(self.separators[i].iter().map(|&e| &out[e]));
    //         out[i] = self.nodes[i].decide(&ms);
    //         for j in self.children[i].iter() {
    //             self.down(*j, out);
    //         }
    //     }
}

impl<T: Display> Display for GaussTree<T> {
    fn iter_child(&self, i: VarId) -> Option<VarId> {
        self.children[i].iter().next().copied()
    }

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GaussTree:")?;
        let mut indent = 0;
        for r in self.roots.iter() {
            writeln!(f, "{:<indent$} {}", indent, self.nodes[*r].info_x)?;
            indent += 2;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::prelude::*;

    #[test]
    fn test_axbs_gauss_block_from() {
        let x1 = VarInfo::new("x1", 3);
        let x2 = VarInfo::new("x2", 2);
        let x3 = VarInfo::new("x3", 3);
        let x4 = VarInfo::new("x4", 4);

        let order = vec![x1.clone(), x2.clone(), x3.clone(), x4.clone()];

        let gauss_blocks = vec![
            AXGaussBlock::new(
                vec![(x3.clone(), 0), (x1.clone(), 3), (x2.clone(), 6)],
                8,
                3,
                mat![
                    [1.31, 1.32, 1.33, 1.11, 1.12, 1.13, 1.21, 1.22],
                    [2.31, 2.32, 2.33, 2.11, 2.12, 2.13, 2.21, 2.22],
                    [3.31, 3.32, 3.33, 3.11, 3.12, 3.13, 3.21, 3.22]
                ],
                col![1.0, 2.0, 3.0],
                mat![[11.0, 0.0, 0.0], [0.0, 22.0, 0.0], [0.0, 0.0, 33.0]],
            ),
            AXGaussBlock::new(
                vec![(x4.clone(), 0), (x1.clone(), 4)],
                7,
                1,
                mat![[4.41, 4.42, 4.43, 4.44, 4.11, 4.12, 4.13]],
                col![4.0],
                mat![[44.0]],
            ),
            AXGaussBlock::new(
                vec![(x1.clone(), 0)],
                3,
                2,
                mat![[5.11, 5.12, 5.13], [6.11, 6.12, 6.13]],
                col![5.0, 6.0],
                mat![[55.0, 56.0], [56.0, 66.0]],
            ),
            AXGaussBlock::new(
                vec![(x2.clone(), 0), (x3.clone(), 4)],
                7,
                1,
                mat![[4.41, 4.42, 4.43, 4.44, 4.11, 4.12, 4.13]],
                col![4.0],
                mat![[44.0]],
            ),
        ];

        let gauss_tree = GaussTree::build(&order, &gauss_blocks);
        if let Ok(tree) = gauss_tree {
            println!("{:?}", tree);
        }
        assert!(gauss_tree.is_ok());
    }
}

use crate::tree::numeric::NumericFactor;
use crate::tree::symbolic::{SymbolicNode, SymbolicTree};

use anyhow::{Context, Result, bail};

use faer_traits::RealField;

pub type VarId = usize;
pub type FactorId = usize;

pub struct GaussianNode<V: RealField> {
    pub factor: Box<dyn NumericFactor<V>>,
    pub children: Vec<GaussianNode<V>>,
}

pub struct GaussianTree<V: RealField> {
    pub roots: Vec<GaussianNode<V>>,
}

impl<V: RealField> GaussianTree<V> {
    /// Builds the numerical tree by following the symbolic blueprint.
    pub fn from_symbolic<F, N>(
        symbolic_tree: &SymbolicTree,
        numerical_factors: &[N], // The actual heavy Faer matrices
        make_node_factor: &F,
    ) -> Result<Self>
    where
        F: Fn(VarId, Vec<&N>, Vec<Box<dyn NumericFactor<V>>>) -> Box<dyn NumericFactor<V>>,
    {
        let roots = symbolic_tree
            .roots
            .iter()
            .map(|sym_root| Self::build_numeric_node(sym_root, numerical_factors, make_node_factor))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { roots })
    }

    fn build_numeric_node<F, N>(
        sym_node: &SymbolicNode,
        numerical_factors: &[N],
        make_node_factor: &F,
    ) -> Result<(GaussianNode<V>, Box<dyn NumericFactor<V>>)>
    where
        F: Fn(VarId, Vec<&N>, Vec<Box<dyn NumericFactor<V>>>) -> Box<dyn NumericFactor<V>>,
    {
        let mut num_children = Vec::with_capacity(sym_node.children.len());
        let mut incoming_messages = Vec::with_capacity(sym_node.children.len());

        // 1. Recurse down to children (This is easily parallelized with Rayon)
        for sym_child in &sym_node.children {
            let (child_node, child_msg) =
                Self::build_numeric_node(sym_child, numerical_factors, make_node_factor)?;
            num_children.push(child_node);
            incoming_messages.push(child_msg);
        }

        // 2. Fetch the assigned original numerical factors using the blueprint's indices
        let assigned_orig_factors: Vec<&N> = sym_node
            .original_factors
            .iter()
            .map(|&fid| &numerical_factors[fid])
            .collect();

        // 3. Create the node factor combining original factors and incoming child messages
        let mut factor =
            make_node_factor(sym_node.main_var, assigned_orig_factors, incoming_messages);

        // 4. Eliminate mathematically
        let outgoing_message = factor
            .eliminate()
            .with_context(|| format!("Failed to eliminate variable {}", sym_node.main_var))?;

        let node = GaussianNode {
            factor,
            children: num_children,
        };

        Ok((node, outgoing_message))
    }

    // solve_par() remains identical to the previous lock-free recursive implementation
}

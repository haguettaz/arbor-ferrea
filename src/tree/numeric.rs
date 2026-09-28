use anyhow::Result;
use rayon::prelude::*;
use std::ptr;

use crate::tree::symbolic::{SymbolicNode, SymbolicTree};

pub type VarId = usize;
pub type FactorId = usize;

/// A lock-free wrapper allowing concurrent disjoint writes to a shared slice.
#[derive(Copy, Clone)]
pub struct ConcurrentStateBuffer<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: Disjoint concurrent access is mathematically guaranteed by the Tree structure.
unsafe impl<T> Send for ConcurrentStateBuffer<T> {}
unsafe impl<T> Sync for ConcurrentStateBuffer<T> {}

impl<T: Copy> ConcurrentStateBuffer<T> {
    pub fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Writes a slice of values to a specific offset.
    /// SAFETY: The caller must ensure no other thread writes to this specific range concurrently.
    #[inline]
    pub unsafe fn write_slice(&self, offset: usize, data: &[T]) {
        debug_assert!(offset + data.len() <= self.len);
        ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
    }

    /// Reads a slice of values from a specific offset.
    /// SAFETY: The caller must ensure this range has already been written to by a parent thread.
    #[inline]
    pub unsafe fn read_slice(&self, offset: usize, len: usize) -> &[T] {
        debug_assert!(offset + len <= self.len);
        std::slice::from_raw_parts(self.ptr.add(offset), len)
    }
}

// ==========================================
// THE UNIVERSAL NUMERIC TREE TRAIT
// ==========================================

pub trait NumericTree: Sized + Send + Sync {
    /// The original raw numerical factor.
    type Factor: Sync;
    /// The concrete node type.
    type Node: Send + Sync;
    /// The mathematical message passed upward during elimination.
    type Message: Send + Sync;
    /// User-provided context passed down during the solve (e.g., VariableDictionary for offsets).
    type SolveContext: Sync;
    /// The numerical type used for solving.
    type Value: Copy;

    // ==========================================
    // REQUIRED METHODS (Math & Accessors)
    // ==========================================

    fn from_roots(roots: Vec<Self::Node>) -> Self;
    fn roots(&self) -> &[Self::Node];
    fn node_children(node: &Self::Node) -> &[Self::Node];

    /// Upward pass math: Combines factors/messages, eliminates the variable,
    /// and returns the constructed node + outgoing (upward) message.
    fn eliminate_node(
        var: VarId,
        assigned_factors: Vec<&Self::Factor>,
        children: Vec<Self::Node>,
        incoming_messages: Vec<Self::Message>,
    ) -> Result<(Self::Node, Self::Message)>;

    /// Downward pass math: Reads ancestors from buffer, computes this node's state,
    /// and writes the result back to the buffer.
    fn decide_node(
        node: &Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::SolveContext,
    ) -> Result<()>;

    // ==========================================
    // PROVIDED METHODS (The Parallel Algorithms)
    // ==========================================

    /// Builds the numerical tree in parallel by following the symbolic blueprint.
    fn from_symbolic(
        symbolic_tree: &SymbolicTree,
        numerical_factors: &[Self::Factor],
    ) -> Result<Self> {
        // Process all disjoint roots in parallel
        let root_results: Result<Vec<(Self::Node, Self::Message)>> = symbolic_tree
            .roots
            .par_iter()
            .map(|sym_root| Self::build_node_par(sym_root, numerical_factors))
            .collect();

        // Unzip the roots and discard the final outgoing messages (roots have no parents)
        let root_nodes = root_results?.into_iter().map(|(node, _msg)| node).collect();

        Ok(Self::from_roots(root_nodes))
    }

    /// Recursively builds a subtree bottom-up.
    fn build_node_par(
        sym_node: &SymbolicNode,
        numerical_factors: &[Self::Factor],
    ) -> Result<(Self::Node, Self::Message)> {
        // 1. Recurse down to children in parallel. (Rayon waits for them implicitly)
        let child_results: Result<Vec<(Self::Node, Self::Message)>> = sym_node
            .children
            .par_iter()
            .map(|sym_child| Self::build_node_par(sym_child, numerical_factors))
            .collect();

        let mut children = Vec::with_capacity(sym_node.children.len());
        let mut messages = Vec::with_capacity(sym_node.children.len());

        for (child_node, child_msg) in child_results? {
            children.push(child_node);
            messages.push(child_msg);
        }

        // 2. Fetch the original numerical factors via symbolic indices
        let assigned_factors: Vec<&Self::Factor> = sym_node
            .original_factors
            .iter()
            .map(|&fid| &numerical_factors[fid])
            .collect();

        // 3. Perform elimination
        Self::eliminate_node(sym_node.var, assigned_factors, children, messages)
    }

    /// Solves the tree top-down in parallel.
    /// `values` is modified in place with the final state update.
    fn solve_par(&self, values: &mut [Self::Value], ctx: &Self::SolveContext) -> Result<()> {
        let buffer = ConcurrentStateBuffer::new(values);

        self.roots()
            .par_iter()
            .try_for_each(|root| Self::solve_node_par(root, &buffer, ctx))
    }

    /// Recursively solves a subtree top-down.
    fn solve_node_par(
        node: &Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::SolveContext,
    ) -> Result<()> {
        // 1. Compute state and write it directly to the global buffer
        Self::decide_node(node, buffer, ctx)?;

        // 2. Parallelize children instantly (they read from the buffer we just wrote to)
        Self::node_children(node)
            .par_iter()
            .try_for_each(|child| Self::solve_node_par(child, buffer, ctx))
    }
}

#[cfg(test)]
mod tests {}

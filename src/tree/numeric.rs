use anyhow::{Context, Result};
use rayon::prelude::*;

use super::symbolic::{SymbolicNode, SymbolicTree};
use crate::variable::VarId;

use std::ptr;

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
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
        }
    }

    /// Reads a slice of values from a specific offset.
    /// SAFETY: The caller must ensure this range has already been written to by a parent thread.
    #[inline]
    pub unsafe fn read_slice(&self, offset: usize, len: usize) -> &[T] {
        debug_assert!(offset + len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) }
    }

    /// Returns a mutable slice to a specific region of the buffer.
    /// SAFETY: The caller must ensure no other thread accesses this specific range concurrently.
    #[inline]
    pub unsafe fn get_mut_slice<'a>(&self, offset: usize, len: usize) -> &'a mut [T] {
        debug_assert!(offset + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }
}

// ==========================================
// THE UNIVERSAL NUMERIC TREE TRAIT
// ==========================================

pub trait NumericTree: Sized + Send + Sync {
    /// The concrete node type.
    type Node: Send + Sync;
    /// The original raw numerical factor.
    type Factor: Sync;
    /// The mathematical message passed upward during elimination.
    type Message: Send + Sync;
    /// The type of variable values
    type Value: Copy + Send + Sync;
    /// User-provided context passed up during the build and the solve.
    type Context: Send + Sync;

    // ==========================================
    // REQUIRED METHODS (Math & Accessors)
    // ==========================================

    fn from_roots(roots: Vec<Self::Node>) -> Self;

    fn roots(&self) -> &[Self::Node];
    fn roots_mut(&mut self) -> &mut [Self::Node];

    fn node_children(node: &Self::Node) -> &[Self::Node];
    fn node_children_mut(node: &mut Self::Node) -> &mut [Self::Node];

    /// Combines factors/messages to construct a node (var, separator)
    fn build_node(
        var: VarId,
        separator: Vec<VarId>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self::Node>,
        ctx: &Self::Context,
    ) -> Result<Self::Node>;

    /// Eliminate the variable from the node, returning the outgoing (upward) message.
    fn build_out_message(node: &mut Self::Node, ctx: &Self::Context) -> Result<Self::Message>;

    /// Downward pass math: Reads ancestors from buffer, computes this node's state,
    /// and writes the result back to the buffer.
    fn decide_node(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()>;

    // ==========================================
    // PROVIDED METHODS (The Parallel Algorithms)
    // ==========================================

    /// Builds the numerical tree in parallel by following the symbolic blueprint.
    fn from_symbolic(
        symbolic_tree: &SymbolicTree,
        factors: &[Self::Factor],
        context: &Self::Context,
    ) -> Result<Self> {
        // Process all disjoint roots in parallel
        let root_results: Result<Vec<(Self::Node, Self::Message)>> = symbolic_tree
            .roots
            .par_iter()
            .map(|sym_root| Self::build_node_par(sym_root, factors, context))
            .collect();

        // Unzip the roots and discard the final outgoing messages (roots have no parents)
        let root_nodes = root_results?.into_iter().map(|(node, _msg)| node).collect();

        Ok(Self::from_roots(root_nodes))
    }

    /// Recursively builds a subtree bottom-up.
    fn build_node_par(
        sym_node: &SymbolicNode,
        factors: &[Self::Factor],
        context: &Self::Context,
    ) -> Result<(Self::Node, Self::Message)> {
        // 1. Recurse down to children in parallel. (Rayon waits for them implicitly)
        let var = sym_node.var;
        let separator = sym_node.separator.clone();

        let child_results: Result<Vec<(Self::Node, Self::Message)>> = sym_node
            .children
            .par_iter()
            .map(|sym_child| Self::build_node_par(sym_child, factors, context))
            .collect();

        let mut children = Vec::with_capacity(sym_node.children.len());
        let mut messages = Vec::with_capacity(sym_node.children.len());

        for (child_node, child_msg) in child_results? {
            children.push(child_node);
            messages.push(child_msg);
        }

        // Fetch the original numerical factors via symbolic indices
        let assigned_factors: Vec<&Self::Factor> =
            sym_node.factors.iter().map(|&fid| &factors[fid]).collect();

        // Assemble the node
        let mut node = Self::build_node(
            var,
            separator,
            assigned_factors,
            messages,
            children,
            context,
        )
        .with_context(|| format!("Failed to build node for variable {var}"))?;

        // Eliminate the node to produce the message for its parent
        let out_msg = Self::build_out_message(&mut node, context)
            .with_context(|| format!("Failed to build outgoing message for variable {var}"))?;

        Ok((node, out_msg))
    }

    /// Solves the tree top-down in parallel.
    /// `values` is modified in place with the final state update.
    fn solve_par(&mut self, values: &mut [Self::Value], ctx: &Self::Context) -> Result<()> {
        let buffer = ConcurrentStateBuffer::new(values);

        // Use par_iter_mut() to safely split mutable access across disjoint roots
        self.roots_mut()
            .par_iter_mut()
            .try_for_each(|root| Self::solve_node_par(root, &buffer, ctx))
    }

    /// Recursively solves a subtree top-down.
    fn solve_node_par(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()> {
        // 1. Compute state and write it directly to the global buffer
        Self::decide_node(node, buffer, ctx)?;

        // 2. Parallelize children instantly
        // Use par_iter_mut() to safely split mutable access across disjoint child subtrees
        Self::node_children_mut(node)
            .par_iter_mut()
            .try_for_each(|child| Self::solve_node_par(child, buffer, ctx))
    }
}

#[cfg(test)]
mod tests {}

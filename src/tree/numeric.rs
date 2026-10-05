//! Abstract implementation of a parallel tree-based numerical solver.
//!
//! Numerical factorizations proceed bottom-up over the symbolic tree, spawning
//! parent tasks as child dependencies resolve. A subsequent top-down pass solves
//! independent branches concurrently, writing results into a provably race-free
//! shared buffer.

use anyhow::{Context, Result};
use rayon::prelude::*;

use super::symbolic::{SymbolicNode, SymbolicTree};

use std::ptr;

/// A lock-free wrapper allowing concurrent disjoint writes to a shared slice.
#[derive(Copy, Clone)]
pub struct ConcurrentStateBuffer<T> {
    /// Raw pointer to the start of the backing buffer.
    ptr: *mut T,
    /// Total number of elements in the backing buffer.
    len: usize,
}

// SAFETY: Disjoint concurrent access is mathematically guaranteed by the tree structure.
unsafe impl<T> Send for ConcurrentStateBuffer<T> {}
unsafe impl<T> Sync for ConcurrentStateBuffer<T> {}

impl<T: Copy> ConcurrentStateBuffer<T> {
    /// Wraps an existing mutable slice into a concurrent buffer view.
    pub fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Verifies that the buffer contains at least `required` elements.
    pub fn ensure_len(&self, required: usize) -> Result<()> {
        anyhow::ensure!(
            self.len >= required,
            "Values buffer length ({}) is smaller than the registered layout ({})",
            self.len,
            required
        );
        Ok(())
    }

    /// Copies a slice of data into the buffer starting at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee no other thread concurrently accesses the subslice
    /// `[offset..offset + data.len()]`.
    #[inline]
    pub unsafe fn write_slice(&self, offset: usize, data: &[T]) {
        debug_assert!(offset + data.len() <= self.len);
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
        }
    }

    /// Reads an immutable subslice of length `len` starting at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `[offset..offset + len]` contains valid, initialized
    /// data and is not being modified concurrently by another thread.
    #[inline]
    pub unsafe fn read_slice(&self, offset: usize, len: usize) -> &[T] {
        debug_assert!(offset + len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) }
    }

    /// Obtains a mutable subslice of length `len` starting at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that no other thread accesses `[offset..offset + len]`
    /// while this borrow is active.
    #[inline]
    pub unsafe fn get_mut_slice<'a>(&self, offset: usize, len: usize) -> &'a mut [T] {
        debug_assert!(offset + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }
}

// ==========================================
// THE UNIVERSAL NUMERIC TREE TRAIT
// ==========================================

/// Numerical elimination tree enabling parallel assembly and solving.
pub trait NumericTree: Sized + Send + Sync {
    /// Tree node encapsulating the elimination clique and its scratchpads.
    type Node: Send + Sync;

    /// Input factor constraining variables within nodes.
    type Factor: Sync;

    /// Upward marginal message passed from children to parents during elimination.
    type Message: Send + Sync;

    /// Scalar element type for variable vectors and numerical operations.
    type Value: Copy + Send + Sync;

    /// Shared contextual metadata threaded through the build and solve phases.
    type Context: Send + Sync;

    // ==========================================
    // REQUIRED METHODS (Math & Accessors)
    // ==========================================

    /// Instantiates a numerical tree from a collection of root nodes.
    fn from_roots(roots: Vec<Self::Node>) -> Self;

    /// Returns a shared slice of the tree roots.
    fn roots(&self) -> &[Self::Node];

    /// Returns a mutable slice of the tree roots.
    fn roots_mut(&mut self) -> &mut [Self::Node];

    /// Returns a shared slice of the children belonging to `node`.
    fn node_children(node: &Self::Node) -> &[Self::Node];

    /// Returns a mutable slice of the children belonging to `node`.
    fn node_children_mut(node: &mut Self::Node) -> &mut [Self::Node];

    /// Constructs a numerical node by merging local factors with incoming child messages.
    fn build_node(
        main: usize,
        separator: Vec<usize>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self::Node>,
        ctx: &Self::Context,
    ) -> Result<Self::Node>;

    /// Eliminates the node's main variable, producing an upward marginal message.
    fn build_out_message(node: &mut Self::Node, ctx: &Self::Context) -> Result<Self::Message>;

    /// Computes the node's variable assignment conditioned on separator values and writes it to `buffer`.
    fn decide_node(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()>;

    // ==========================================
    // PROVIDED METHODS (The Parallel Algorithms)
    // ==========================================

    /// Instantiates and factorizes the tree in parallel according to a symbolic blueprint.
    fn from_symbolic(
        symbolic_tree: &SymbolicTree,
        factors: &[Self::Factor],
        context: &Self::Context,
    ) -> Result<Self> {
        let root_results: Result<Vec<(Self::Node, Self::Message)>> = symbolic_tree
            .roots
            .par_iter()
            .map(|sym_root| Self::build_node_par(sym_root, factors, context))
            .collect();

        let root_nodes = root_results?.into_iter().map(|(node, _msg)| node).collect();

        Ok(Self::from_roots(root_nodes))
    }

    /// Recursively factorizes a symbolic subtree bottom-up using Rayon tasks.
    fn build_node_par(
        sym_node: &SymbolicNode,
        factors: &[Self::Factor],
        context: &Self::Context,
    ) -> Result<(Self::Node, Self::Message)> {
        let id = sym_node.main;
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

        let assigned_factors: Vec<&Self::Factor> =
            sym_node.factors.iter().map(|&fid| &factors[fid]).collect();

        let mut node =
            Self::build_node(id, separator, assigned_factors, messages, children, context)
                .with_context(|| format!("Failed to build node for variable {id}"))?;

        let out_msg = Self::build_out_message(&mut node, context)
            .with_context(|| format!("Failed to build outgoing message for variable {id}"))?;

        Ok((node, out_msg))
    }

    /// Executes the top-down back-substitution pass in parallel, updating `values` in place.
    fn solve_par(&mut self, values: &mut [Self::Value], ctx: &Self::Context) -> Result<()> {
        let buffer = ConcurrentStateBuffer::new(values);

        self.roots_mut()
            .par_iter_mut()
            .try_for_each(|root| Self::solve_node_par(root, &buffer, ctx))
    }

    /// Recursively solves a subtree top-down across Rayon threads.
    fn solve_node_par(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()> {
        Self::decide_node(node, buffer, ctx)?;

        Self::node_children_mut(node)
            .par_iter_mut()
            .try_for_each(|child| Self::solve_node_par(child, buffer, ctx))
    }
}

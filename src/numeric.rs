//! Abstract implementation of a tree-based numerical solver.
//!
//! A [`NumericTree`] mirrors a [`SymbolicTree`], with one [`NumericClique`] per symbolic clique.
//! It is used in two passes:
//!
//! 1. **Build** (bottom-up): each clique gathers its assigned factors and the messages of its
//!    children, eliminates its frontal variable, and sends a message over its separator variables
//!    to its parent.
//! 2. **Solve** (top-down): each clique computes its frontal variable from the values of its
//!    separator variables, which belong to its ancestors and are thus already solved.
//!
//! Both passes come in a sequential (`_seq`) and a parallel (`_par`, using [Rayon]) variant.
//! In the solve pass, results are written to a single shared buffer through a
//! [`ConcurrentStateBuffer`]: each clique writes only its frontal variable and reads only
//! variables of its ancestors, so concurrent accesses are disjoint.
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::symbolic::SymbolicFactor;

use super::symbolic::{SymbolicClique, SymbolicTree};

use std::ptr;

/// Shared data threaded through the build and solve passes, mapping variable keys to their
/// location in the state buffer.
///
/// The memory layouts of distinct keys must not overlap and must fit in the state buffer, since
/// the solve pass relies on them for disjoint concurrent accesses.
pub trait NumericContext: Send + Sync {
    /// Returns the `(offset, size)` of `key` in the state buffer, or `None` if unknown.
    fn memory_layout(&self, key: usize) -> Option<(usize, usize)>;
}

/// A lock-free view on a slice, allowing concurrent accesses to disjoint subslices.
///
/// It does not borrow the slice it is created from: the caller must keep the slice alive, and
/// otherwise unused, for as long as the view is used.
#[derive(Copy, Clone)]
pub struct ConcurrentStateBuffer<T> {
    /// Raw pointer to the start of the backing buffer.
    ptr: *mut T,
    /// Total number of elements in the backing buffer.
    len: usize,
}

// SAFETY: All accesses go through `unsafe` methods, whose callers guarantee that concurrent
// accesses are disjoint, as ensured by the tree structure in the solve pass.
unsafe impl<T> Send for ConcurrentStateBuffer<T> {}
unsafe impl<T> Sync for ConcurrentStateBuffer<T> {}

impl<T: Copy> ConcurrentStateBuffer<T> {
    /// Creates a view on `slice`.
    pub fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Checks that the buffer holds at least `required` elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is shorter than `required`.
    pub fn contains(&self, start: usize, end: usize) -> bool {
        (start < end) && (end < self.len)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Copies `data` into the buffer, starting at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `offset + data.len()` does not exceed the buffer length
    /// (checked only in debug builds), and that no other thread concurrently accesses
    /// `[offset..offset + data.len()]`.
    #[inline]
    pub unsafe fn write_slice(&self, offset: usize, data: &[T]) {
        debug_assert!(offset + data.len() <= self.len);
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
        }
    }

    /// Returns the subslice of length `len` starting at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `offset + len` does not exceed the buffer length (checked
    /// only in debug builds), and that `[offset..offset + len]` is not modified while the returned
    /// slice is alive.
    #[inline]
    pub unsafe fn read_slice(&self, offset: usize, len: usize) -> &[T] {
        debug_assert!(offset + len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) }
    }

    /// Returns the mutable subslice of length `len` starting at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `offset + len` does not exceed the buffer length (checked
    /// only in debug builds), and that `[offset..offset + len]` is not accessed by anything else
    /// while the returned slice is alive. The lifetime `'a` is unbounded and must not outlive
    /// the underlying slice.
    #[inline]
    pub unsafe fn get_mut_slice<'a>(&self, offset: usize, len: usize) -> &'a mut [T] {
        debug_assert!(offset + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }
}

// ==========================================
// THE UNIVERSAL NUMERIC TREE TRAIT
// ==========================================

/// A forest of numerical cliques, built and solved by the provided methods.
///
/// Implementors only provide access to the roots and children; the build and solve passes are
/// provided.
pub trait NumericTree: Sized + Send + Sync {
    /// Clique type, which defines the factors, messages, scalars, and context.
    type Clique: NumericClique;

    // ==========================================
    // REQUIRED METHODS
    // ==========================================

    /// Creates a forest from its root cliques.
    fn from_roots(roots: Vec<Self::Clique>) -> Self;

    /// Returns the root cliques.
    fn roots(&self) -> &[Self::Clique];

    /// Returns the root cliques mutably.
    fn roots_mut(&mut self) -> &mut [Self::Clique];

    /// Returns the children of `clique`.
    fn children(clique: &Self::Clique) -> &[Self::Clique];

    /// Returns the children of `clique` mutably.
    fn children_mut(clique: &mut Self::Clique) -> &mut [Self::Clique];

    // ==========================================
    // PROVIDED METHODS
    // ==========================================

    /// Builds the tree from `symbolic_tree` sequentially, eliminating every clique bottom-up.
    ///
    /// The factor indices of the symbolic cliques refer to `factors`. The messages of the roots
    /// are discarded.
    ///
    /// # Errors
    ///
    /// Returns an error if any clique fails to build, see [`NumericClique::build_and_close`].
    fn from_symbolic_seq<C: NumericContext>(
        symbolic_tree: &SymbolicTree,
        factors: &[<Self::Clique as NumericClique>::Factor],
        context: &C,
    ) -> Result<Self> {
        let root_results: Result<Vec<(Self::Clique, <Self::Clique as NumericClique>::Message)>> =
            symbolic_tree
                .roots
                .iter()
                .map(|sym_root| Self::build_and_close_clique_seq(sym_root, factors, context))
                .collect();

        let root_cliques = root_results?
            .into_iter()
            .map(|(clique, _msg)| clique)
            .collect();

        Ok(Self::from_roots(root_cliques))
    }

    /// Parallel variant of [`from_symbolic_seq`](Self::from_symbolic_seq), building independent
    /// subtrees concurrently with [Rayon](https://docs.rs/rayon/latest/rayon/).
    ///
    /// # Errors
    ///
    /// Returns an error if any clique fails to build, see [`NumericClique::build_and_close`].
    fn from_symbolic_par<C: NumericContext>(
        symbolic_tree: &SymbolicTree,
        factors: &[<Self::Clique as NumericClique>::Factor],
        context: &C,
    ) -> Result<Self> {
        let root_results: Result<Vec<(Self::Clique, <Self::Clique as NumericClique>::Message)>> =
            symbolic_tree
                .roots
                .par_iter()
                .map(|sym_root| Self::build_and_close_clique_par(sym_root, factors, context))
                .collect();

        let root_cliques = root_results?
            .into_iter()
            .map(|(clique, _msg)| clique)
            .collect();

        Ok(Self::from_roots(root_cliques))
    }

    /// Builds the subtree rooted at `sym_clique` sequentially, returning its root clique and the
    /// message to its parent.
    ///
    /// The children are built first, and their messages are passed, together with the factors
    /// assigned to `sym_clique`, to [`NumericClique::build_and_close`].
    ///
    /// # Errors
    ///
    /// Returns an error if any clique of the subtree fails to build or if a symbolic clique refers
    /// to a factor index out of bounds.
    fn build_and_close_clique_seq<C: NumericContext>(
        sym_clique: &SymbolicClique,
        factors: &[<Self::Clique as NumericClique>::Factor],
        context: &C,
    ) -> Result<(Self::Clique, <Self::Clique as NumericClique>::Message)> {
        let front_key = sym_clique.front_key;
        let sep_keys = sym_clique.sep_keys.clone();

        let child_results: Result<Vec<(Self::Clique, <Self::Clique as NumericClique>::Message)>> =
            sym_clique
                .children
                .iter()
                .map(|sym_child| Self::build_and_close_clique_seq(sym_child, factors, context))
                .collect();

        let mut children = Vec::with_capacity(sym_clique.children.len());
        let mut messages = Vec::with_capacity(sym_clique.children.len());

        for (child_clique, child_msg) in child_results? {
            children.push(child_clique);
            messages.push(child_msg);
        }

        let assigned_factors: Vec<&<Self::Clique as NumericClique>::Factor> = sym_clique
                    .factors
                    .iter()
                    .map(|&fid| {
                        factors.get(fid).with_context(|| {
                            format!(
                                "Factor index {fid} is out of bounds (total factors: {}) while building clique for variable {front_key}",
                                factors.len()
                            )
                        })
                    })
                    .collect::<Result<_>>()?;

        let (clique, out_msg) = Self::Clique::build_and_close(
            front_key,
            sep_keys,
            assigned_factors,
            messages,
            children,
            context,
        )
        .with_context(|| {
            format!("Failed to build and close the clique for variable {front_key}")
        })?;

        Ok((clique, out_msg))
    }

    /// Parallel variant of [`build_and_close_clique_seq`](Self::build_and_close_clique_seq),
    /// building the children concurrently with [Rayon](https://docs.rs/rayon/latest/rayon/).
    ///
    /// # Errors
    ///
    /// Returns an error if any clique of the subtree fails to build or if a symbolic clique
    /// refers to a factor index out of bounds.
    fn build_and_close_clique_par<C: NumericContext>(
        sym_clique: &SymbolicClique,
        factors: &[<Self::Clique as NumericClique>::Factor],
        context: &C,
    ) -> Result<(Self::Clique, <Self::Clique as NumericClique>::Message)> {
        let front_key = sym_clique.front_key;
        let sep_keys = sym_clique.sep_keys.clone();

        let child_results: Result<Vec<(Self::Clique, <Self::Clique as NumericClique>::Message)>> =
            sym_clique
                .children
                .par_iter()
                .map(|sym_child| Self::build_and_close_clique_par(sym_child, factors, context))
                .collect();

        let mut children = Vec::with_capacity(sym_clique.children.len());
        let mut messages = Vec::with_capacity(sym_clique.children.len());

        for (child_clique, child_msg) in child_results? {
            children.push(child_clique);
            messages.push(child_msg);
        }

        let assigned_factors: Vec<&<Self::Clique as NumericClique>::Factor> = sym_clique
                    .factors
                    .iter()
                    .map(|&fid| {
                        factors.get(fid).with_context(|| {
                            format!(
                                "Factor index {fid} is out of bounds (total factors: {}) while building clique for variable {front_key}",
                                factors.len()
                            )
                        })
                    })
                    .collect::<Result<_>>()?;

        let (clique, out_msg) = Self::Clique::build_and_close(
            front_key,
            sep_keys,
            assigned_factors,
            messages,
            children,
            context,
        )
        .with_context(|| {
            format!("Failed to build and close the clique for variable {front_key}")
        })?;

        Ok((clique, out_msg))
    }

    /// Solves all variables top-down, writing each into `buffer` at its location in `ctx`.
    ///
    /// The roots are processed sequentially; see [`solve_clique_seq`](Self::solve_clique_seq).
    ///
    /// # Errors
    ///
    /// Returns an error if a variable is missing from `ctx`, or if any clique fails to decide.
    fn solve_seq<C: NumericContext>(
        &mut self,
        buffer: &mut [<Self::Clique as NumericClique>::Scalar],
        ctx: &C,
    ) -> Result<()> {
        let concurrent_buffer = ConcurrentStateBuffer::new(buffer);

        self.roots_mut()
            .iter_mut()
            .try_for_each(|root| Self::solve_clique_seq(root, &concurrent_buffer, ctx))
    }

    /// Solves the frontal variable of `clique` from its separator variables in `buffer`, then
    /// solves its children.
    ///
    /// The separator variables must already be solved, i.e., `clique` is a root or its parent was
    /// solved before.
    ///
    /// # Errors
    ///
    /// Returns an error if the memory layouts are invalid or if any clique fails to decide.
    fn solve_clique_seq<C: NumericContext>(
        clique: &mut Self::Clique,
        buffer: &ConcurrentStateBuffer<<Self::Clique as NumericClique>::Scalar>,
        ctx: &C,
    ) -> Result<()> {
        // Resolve mutable frontal slice
        let front_key = clique.front_key();
        let (f_start, f_size) = ctx
            .memory_layout(front_key)
            .with_context(|| "Memory layout for frontal variable not found")?;
        let f_end = f_start + f_size;

        // Enforce valid memory layouts
        anyhow::ensure!(
            buffer.contains(f_start, f_end),
            "Invalid memory layout for the provided buffer, expected range between 0 and {}, got {} and {}",
            buffer.len(),
            f_start,
            f_end
        );

        // Resolve immutable separator slices
        let sep_keys = clique.sep_keys();
        let mut sep_in = Vec::with_capacity(sep_keys.len());

        for &key in sep_keys {
            let (s_start, s_size) = ctx
                .memory_layout(key)
                .with_context(|| "Memory layout for separator variable not found")?;
            let s_end = s_start + s_size;

            // Enforce valid memory layouts
            anyhow::ensure!(
                buffer.contains(s_start, s_end),
                "Invalid memory layout for the provided buffer, expected range between 0 and {}, got {} and {}",
                buffer.len(),
                s_start,
                s_end
            );

            // Enforce disjoint memory layouts to guarantee safety
            anyhow::ensure!(
                f_end <= s_start || s_end <= f_start,
                "Memory layout overlap detected between frontal variable {} and separator variable {}",
                front_key,
                key
            );

            // SAFETY: The check above guarantees this region does not overlap with the frontal region.
            // The tree execution schedule mathematically guarantees no other threads are writing to this region.
            sep_in.push(unsafe { buffer.read_slice(s_start, s_size) });
        }

        // SAFETY: The bounds check above guarantees exclusive access relative to the separator reads.
        let front_out = unsafe { buffer.get_mut_slice(f_start, f_size) };

        // Delegate to the mathematically safe domain logic
        clique.decide(front_out, &sep_in)?;

        // Recurse to children
        clique
            .children_mut()
            .iter_mut()
            .try_for_each(|child| Self::solve_clique_seq(child, buffer, ctx))
    }

    /// Parallel variant of [`solve_seq`](Self::solve_seq), solving independent subtrees
    /// concurrently with [Rayon](https://docs.rs/rayon/latest/rayon/).
    ///
    /// # Errors
    ///
    /// Returns an error if a variable is missing from `ctx`, or if any clique fails to decide.
    fn solve_par<C: NumericContext>(
        &mut self,
        buffer: &mut [<Self::Clique as NumericClique>::Scalar],
        ctx: &C,
    ) -> Result<()> {
        let concurrent_buffer = ConcurrentStateBuffer::new(buffer);

        self.roots_mut()
            .par_iter_mut()
            .try_for_each(|root| Self::solve_clique_par(root, &concurrent_buffer, ctx))
    }

    /// Parallel variant of [`solve_clique_seq`](Self::solve_clique_seq), solving the children
    /// concurrently.
    ///
    /// # Errors
    ///
    /// Returns an error if the memory layouts are invalid or if any clique fails to decide.
    fn solve_clique_par<C: NumericContext>(
        clique: &mut Self::Clique,
        buffer: &ConcurrentStateBuffer<<Self::Clique as NumericClique>::Scalar>,
        ctx: &C,
    ) -> Result<()> {
        // Resolve mutable frontal slice
        let front_key = clique.front_key();
        let (f_start, f_size) = ctx
            .memory_layout(front_key)
            .with_context(|| "Memory layout for frontal variable not found")?;
        let f_end = f_start + f_size;

        // Enforce valid memory layouts
        anyhow::ensure!(
            buffer.contains(f_start, f_end),
            "Invalid memory layout for the provided buffer, expected range between 0 and {}, got {} and {}",
            buffer.len(),
            f_start,
            f_end
        );

        // Resolve immutable separator slices
        let sep_keys = clique.sep_keys();
        let mut sep_in = Vec::with_capacity(sep_keys.len());

        for &key in sep_keys {
            let (s_start, s_size) = ctx
                .memory_layout(key)
                .with_context(|| "Memory layout for separator variable not found")?;
            let s_end = s_start + s_size;

            // Enforce valid memory layouts
            anyhow::ensure!(
                buffer.contains(s_start, s_end),
                "Invalid memory layout for the provided buffer, expected range between 0 and {}, got {} and {}",
                buffer.len(),
                s_start,
                s_end
            );

            // Enforce disjoint memory layouts to guarantee safety
            anyhow::ensure!(
                f_end <= s_start || s_end <= f_start,
                "Memory layout overlap detected between frontal variable {} and separator variable {}",
                front_key,
                key
            );

            // SAFETY: The check above guarantees this region does not overlap with the frontal region.
            // The tree execution schedule mathematically guarantees no other threads are writing to this region.
            sep_in.push(unsafe { buffer.read_slice(s_start, s_size) });
        }

        // SAFETY: The bounds check above guarantees exclusive access relative to the separator reads.
        let front_out = unsafe { buffer.get_mut_slice(f_start, f_size) };

        // Delegate to the mathematically safe domain logic
        clique.decide(front_out, &sep_in)?;

        // Recurse to children
        clique
            .children_mut()
            .iter_mut()
            .try_for_each(|child| Self::solve_clique_par(child, buffer, ctx))
    }
}

// ==========================================
// THE UNIVERSAL NUMERIC CLIQUE TRAIT
// ==========================================

/// A clique eliminating one frontal variable given its separator variables.
pub trait NumericClique: Sized + Send + Sync {
    /// Factor type assigned to cliques by the symbolic tree.
    type Factor: Sync + SymbolicFactor;

    /// Message type sent by a clique to its parent, over the separator variables.
    type Message: Send + Sync;

    /// Scalar type of the state buffer.
    type Scalar: Copy + Send + Sync;

    // /// Context shared by all cliques in the build and solve passes.
    // type Context: NumericContext;

    /// Returns the child cliques.
    fn children(&self) -> &[Self];

    /// Returns the child cliques mutably.
    fn children_mut(&mut self) -> &mut [Self];

    /// Returns the key of the frontal variable.
    fn front_key(&self) -> usize;

    /// Returns the keys of the separator variables, in the order expected by
    /// [`decide`](Self::decide).
    fn sep_keys(&self) -> &[usize];

    /// Builds the clique of `front_key` and eliminates it, returning the clique and the message
    /// over `sep_keys` to its parent.
    ///
    /// `assigned_factors` are the factors assigned to the clique by the symbolic tree, and
    /// `incoming_messages` the messages of `children`, in the same order. The tree guarantees that
    /// `sep_keys` has no duplicates and does not contain `front_key`, and that all factors and
    /// messages involve only variables in `{front_key} ∪ sep_keys`.
    ///
    /// # Errors
    ///
    /// Returns an error if the elimination fails, e.g., if a variable is missing from `ctx`.
    fn build_and_close<C: NumericContext>(
        front_key: usize,
        sep_keys: Vec<usize>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self>,
        ctx: &C,
    ) -> Result<(Self, Self::Message)>;

    /// Writes the value of the frontal variable into `out`, given the values `sep_vals` of the
    /// separator variables, aligned with [`sep_keys`](Self::sep_keys).
    ///
    /// The tree provides `out` and `sep_vals` with the lengths given by the context's memory
    /// layout, so implementations need no `unsafe` code.
    ///
    /// # Errors
    ///
    /// Returns an error if the frontal variable cannot be computed.
    fn decide(&mut self, out: &mut [Self::Scalar], sep_vals: &[&[Self::Scalar]]) -> Result<()>;
}

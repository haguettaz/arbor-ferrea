use anyhow::{Context, Result};
use dyn_stack::MemBuffer;
use dyn_stack::MemStack;
use faer::linalg::cholesky::llt;
use faer::linalg::matmul::matmul;
use faer::prelude::*;
use faer::{Accum, ColMut, ColRef, Par, Spec};
use faer_traits::RealField;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use super::numeric::{ConcurrentStateBuffer, NumericTree};
use super::symbolic::SymbolicFactor;
use crate::factor::gaussian::GaussianFactor;
use crate::variable::{VarId, dictionary::VarDict};

/// The main variable X and separator variable S are associated with the matrices A and B, respectively.
/// For convenience, the virtual variables Y := A X + Z and Z := B S are introduced.
pub struct GaussianNode<T> {
    pub var: VarId,                     // main variable (x)
    pub separator: Vec<VarId>,          // separator variables (s)
    pub children: Vec<GaussianNode<T>>, // Gaussian node children in the tree

    // Dimensions
    pub dim_x: usize, // main variable dimension
    pub dim_s: usize, // separator dimension
    pub dim_y: usize, // observation dimension

    // ---- Matrices and Vectors ----
    a: Mat<T>,                         // main variable matrix -- shape (dim_y, dim_x)
    b: Mat<T>,                         // separator variables matrix -- shape (dim_y, dim_s)
    b_ctx: Vec<(VarId, usize, usize)>, // memory layout of the separator variables in b
    xiby: Col<T>,                      // backward weighted mean vector for y -- shape (dim_y)
    wby: Mat<T>,                       // backward precision matrix for y -- shape (dim_y, dim_y)
    xitx: Col<T>,                      // dual weighted mean vectors for x -- shape (dim_x)
    rtx: Mat<T>, // dual square-root precision matrix for x -- shape (dim_x, dim_x)

    // ---- Workspaces ----
    wby_a: Mat<T>, // (dim_y, dim_x)
    s: Col<T>,     // (dim_s)
    b_s: Col<T>,   // (dim_y)
    mem_buf: MemBuffer,
}

impl<T: RealField + Copy + Send + Sync + Debug + Display> GaussianNode<T> {
    pub fn get_memory_layout(&self, var: VarId) -> Option<(usize, usize)> {
        self.b_ctx
            .iter()
            .find(|(v, _, _)| *v == var)
            .map(|(_, offset, size)| (*offset, *size))
    }

    pub fn try_new(
        var: VarId,
        separator: Vec<VarId>,
        children: Vec<Self>,
        dim_x: usize,
        dim_y: usize,
        dim_s: usize,
        a: Mat<T>,
        b: Mat<T>,
        b_ctx: Vec<(VarId, usize, usize)>,
        xiby: Col<T>,
        wby: Mat<T>,
    ) -> Result<Self> {
        let wby_a = &wby * &a;

        let xitx = Col::zeros(dim_x);
        let rtx = Mat::zeros(dim_x, dim_x);

        // let h_xitx = Col::zeros(dim_x);
        // let h_at_wby = Mat::zeros(dim_x, dim_y);
        let s = Col::zeros(dim_s);
        let b_s = Col::zeros(dim_y);

        // Compute the size and alignment of the required scratch space for Cholesky decomposition and associated solver
        let cholesky_memory =
            llt::factor::cholesky_in_place_scratch::<T>(dim_x, Par::Seq, Spec::default());
        let solve_memory = llt::solve::solve_in_place_scratch::<T>(dim_x, dim_y, Par::Seq);
        let mem_buf = MemBuffer::new(cholesky_memory.or(solve_memory));

        let node = Self {
            var,
            separator,
            children,
            dim_x,
            dim_y,
            dim_s,
            a,
            b,
            b_ctx,
            xiby,
            wby,
            wby_a,
            xitx,
            rtx,
            // h_xitx,
            // h_at_wby,
            s,
            b_s,
            mem_buf,
        };

        Ok(node)
    }

    /// Updates the node's state by computing `xitx`, `rtx`, `h_xitx`, and `h_at_wby`.
    pub fn update(&mut self) -> Result<()> {
        let mut stack = MemStack::new(&mut self.mem_buf);

        // Compute xitx = a.t xiby, which is also used in solve_x
        matmul(
            self.xitx.rb_mut(),
            Accum::Replace,
            self.a.transpose().rb(),
            self.xiby.rb(),
            T::one(),
            Par::Seq,
        );

        // Compute wtx = (a.t wby a) and its Cholesky decomposition rtx, the later being also used in solve_x
        matmul(
            self.rtx.rb_mut(),
            Accum::Replace,
            self.a.transpose().rb(),
            self.wby_a.rb(),
            T::one(),
            Par::Seq,
        );
        llt::factor::cholesky_in_place(
            self.rtx.rb_mut(),
            llt::factor::LltRegularization::default(), // no regularization
            Par::Seq,
            &mut stack,
            default(),
        )
        .with_context(|| format!("Cholesky factorization failed for variable {}: precision matrix is singular or not positive definite. Ensure observation precision matrices are positive definite and the measurement Jacobian A has full column rank.", self.var))?;

        Ok(())
    }

    /// Create a [`GaussianFactor`] by eliminating the [`GaussianNode`]'s main variable.
    /// Warning: `a` must have full column rank (its columns must be linearly independent).
    fn eliminate_main(&mut self) -> Result<GaussianFactor<T>> {
        let vars = self.separator.clone();

        let dim_x = self.dim_s;
        let dim_y = self.dim_y;

        let a = self.b.clone();
        let a_ctx = self.b_ctx.clone();

        // Allocate memory for the stack from the node's memory buffer
        let mut stack = MemStack::new(&mut self.mem_buf);

        // Compute xitx = a.t xiby, which is also used in solve_x
        matmul(
            self.xitx.rb_mut(),
            Accum::Replace,
            self.a.transpose().rb(),
            self.xiby.rb(),
            T::one(),
            Par::Seq,
        );

        // Compute wtx = (a.t wby a) and its Cholesky decomposition rtx, the later being also used in solve_x
        matmul(
            self.rtx.rb_mut(),
            Accum::Replace,
            self.a.transpose().rb(),
            self.wby_a.rb(),
            T::one(),
            Par::Seq,
        );
        llt::factor::cholesky_in_place(
            self.rtx.rb_mut(),
            llt::factor::LltRegularization::default(),
            Par::Seq,
            &mut stack,
            default(),
        )
        .with_context(|| format!("Cholesky factorization failed for variable {}: precision matrix is singular or not positive definite. Ensure observation precision matrices are positive definite and the measurement Jacobian A has full column rank.", self.var))?;

        // Compute xibz = xiby - wby a h a.t xiby = xiby - wby_a h_xitx
        // where h_xitx = h a.t xiby with hinv = a.t wby a
        let mut h_xitx = self.xitx.clone();
        llt::solve::solve_in_place(self.rtx.as_ref(), h_xitx.as_mat_mut(), Par::Seq, &mut stack);

        // Compute wbz = wby - wby a h a.t wby
        let mut h_at_wby = self.wby_a.transpose().to_owned();
        llt::solve::solve_in_place(self.rtx.as_ref(), h_at_wby.as_mut(), Par::Seq, &mut stack);

        let mut xiby = self.xiby.clone();
        matmul(
            xiby.rb_mut(),
            Accum::Add,
            self.wby_a.rb(),
            h_xitx.rb(),
            T::one().neg(),
            Par::Seq,
        );

        let mut wby = self.wby.clone();
        matmul(
            wby.rb_mut(),
            Accum::Add,
            self.wby_a.rb(),
            h_at_wby.rb(),
            T::one().neg(),
            Par::Seq,
        );

        let msg = GaussianFactor {
            vars,
            dim_x,
            dim_y,
            a,
            a_ctx,
            xiby,
            wby,
        };

        Ok(msg)
    }

    /// Decides the value of the node's variables using the current state of the tree.
    /// Warning:
    /// - A few fields must be up-to-date: `b`, `s`, `rtx`, `wby_a`.
    /// - `x` must be a mutable slice of the same length as the node's x dimension.
    pub fn decide(&mut self, x: &mut [T]) {
        let mut stack = MemStack::new(&mut self.mem_buf);

        let mut x_col = ColMut::from_slice_mut(x);

        matmul(
            self.b_s.as_mut(),
            Accum::Replace,
            self.b.as_ref(),
            &self.s,
            T::one(),
            Par::Seq,
        );

        x_col.copy_from(&self.xitx);
        matmul(
            x_col.rb_mut(),
            Accum::Add,
            self.wby_a.transpose(),
            &self.b_s,
            T::one().neg(),
            Par::Seq,
        );

        llt::solve::solve_in_place(self.rtx.as_ref(), x_col.as_mat_mut(), Par::Seq, &mut stack);
    }

    /// Helper method to recursively format the tree with ASCII branches
    fn fmt_recursive(&self, f: &mut Formatter<'_>, prefix: &str, is_last: bool) -> FmtResult {
        let branch_marker = if is_last { "└── " } else { "├── " };

        writeln!(
            f,
            "{}{}[Var: {}, Separator: {:?}, A: {:?}, B: {:?}, Wby: {:?}, xiby: {:?}]",
            prefix, branch_marker, self.var, self.separator, self.a, self.b, self.wby, self.xiby
        )?;

        let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
        let num_children = self.children.len();

        for (i, child) in self.children.iter().enumerate() {
            child.fmt_recursive(f, &child_prefix, i == num_children - 1)?;
        }

        Ok(())
    }
}

pub struct GaussianTree<T> {
    pub roots: Vec<GaussianNode<T>>,
}

impl<T> SymbolicFactor for GaussianFactor<T> {
    fn vars(&self) -> &[VarId] {
        &self.vars
    }
}

impl<T: Send + Sync + Copy + RealField + Display + Debug> NumericTree for GaussianTree<T> {
    /// The concrete node type.
    type Node = GaussianNode<T>;
    /// The original raw numerical factor.
    type Factor = GaussianFactor<T>;
    /// The mathematical message passed upward during elimination.
    type Message = GaussianFactor<T>;
    /// The type of variable values
    type Value = T;
    /// User-provided context passed up during the build and the solve.
    type Context = VarDict;

    // ==========================================
    // REQUIRED METHODS (Math & Accessors)
    // ==========================================

    fn from_roots(roots: Vec<Self::Node>) -> Self {
        Self { roots }
    }

    fn roots(&self) -> &[Self::Node] {
        &self.roots
    }

    fn roots_mut(&mut self) -> &mut [Self::Node] {
        &mut self.roots
    }

    fn node_children(node: &Self::Node) -> &[Self::Node] {
        &node.children
    }

    fn node_children_mut(node: &mut Self::Node) -> &mut [Self::Node] {
        &mut node.children
    }

    fn build_node(
        var: VarId,
        separator: Vec<VarId>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self::Node>,
        ctx: &Self::Context,
    ) -> Result<Self::Node> {
        let dim_x = ctx
            .get_size(var)
            .with_context(|| format!("Missing size for variable with id: {}", var))?;

        // Create the context for the separator variables
        let mut b_ctx = Vec::with_capacity(separator.len());
        let mut start = 0;
        for &sep in &separator {
            let dim = ctx
                .get_size(sep)
                .with_context(|| format!("Missing size for variable with id: {}", sep))?;
            b_ctx.push((sep, start, dim));
            start += dim;
        }
        let dim_s = start; // the total dimension of the separator variables

        // Get the total observation dimension (from incoming messages and assigned factors)
        let dim_y = assigned_factors.iter().map(|f| f.dim_y).sum::<usize>()
            + incoming_messages.iter().map(|m| m.dim_y).sum::<usize>();

        // Create matrices/vectors by block rows
        let mut a = Mat::zeros(dim_y, dim_x);
        let mut b = Mat::zeros(dim_y, dim_s);
        let mut xiby = Col::zeros(dim_y);
        let mut wby = Mat::zeros(dim_y, dim_y);

        let mut offset_obs = 0;
        let all_factors = assigned_factors.into_iter().chain(incoming_messages.iter());
        for factor in all_factors {
            let size_obs = factor.dim_y;

            // Populate xiby
            let mut sub_xiby = xiby.as_mut().subrows_mut(offset_obs, size_obs);
            sub_xiby.copy_from(&factor.xiby);

            // Populate wby
            let mut sub_wby = wby
                .as_mut()
                .submatrix_mut(offset_obs, offset_obs, size_obs, size_obs);
            sub_wby.copy_from(&factor.wby);

            // Populate a
            if let Some((offset, size)) = factor.get_memory_layout(var) {
                let mut sub_a = a.as_mut().subrows_mut(offset_obs, size_obs);
                sub_a.copy_from(&factor.a.subcols(offset, size));
            }

            // Populate b
            for &(sep, offset_sep, size_sep) in &b_ctx {
                if let Some((offset, size)) = factor.get_memory_layout(sep) {
                    let mut sub_b = b
                        .as_mut()
                        .submatrix_mut(offset_obs, offset_sep, size_obs, size_sep);
                    sub_b.copy_from(&factor.a.subcols(offset, size));
                }
            }

            offset_obs += size_obs;
        }

        let mut node = GaussianNode::try_new(
            var, separator, children, dim_x, dim_y, dim_s, a, b, b_ctx, xiby, wby,
        )
        .with_context(|| format!("Failed to build Gaussian node for var: {}", var))?;
        node.update()
            .with_context(|| format!("Failed to update Gaussian node for var: {}", var))?;
        Ok(node)
    }

    fn build_out_message(node: &mut Self::Node, _ctx: &Self::Context) -> Result<Self::Message> {
        node.eliminate_main()
    }

    fn decide_node(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<Self::Value>,
        ctx: &Self::Context,
    ) -> Result<()> {
        // Gather separator values from the global buffer directly into the column
        for &sep in &node.separator {
            let offset_in_buffer = ctx
                .get_offset(sep)
                .with_context(|| format!("Missing offset for variable with id: {}", sep))?;
            let (offset, size) = node.get_memory_layout(sep).unwrap();

            unsafe {
                node.s
                    .as_mut()
                    .subrows_mut(offset, size)
                    .copy_from(ColRef::from_slice(
                        buffer.read_slice(offset_in_buffer, size),
                    ));
            }
        }

        // Obtain a direct mutable view into the global array and solve in-place
        let (offset, size) = ctx
            .get_memory_layout(node.var)
            .with_context(|| format!("Missing memory layout for variable with id: {}", node.var))?;
        unsafe {
            let x_slice = buffer.get_mut_slice(offset, size);
            node.decide(x_slice);
        }

        Ok(())
    }
}

impl<T: Display + Debug + Send + Sync + Copy + RealField> Display for GaussianTree<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if self.roots.is_empty() {
            return write!(f, "Empty SymbolicTree");
        }

        for (i, root) in self.roots.iter().enumerate() {
            // Print the root node
            writeln!(
                f,
                "Root [Var: {}, Separator: {:?}, A: {:?}, B: {:?}, Wby: {:?}, xiby: {:?}]",
                root.var, root.separator, root.a, root.b, root.wby, root.xiby
            )?;

            // Print all children recursively
            let num_children = root.children.len();
            for (j, child) in root.children.iter().enumerate() {
                child.fmt_recursive(f, "", j == num_children - 1)?;
            }

            // Add a blank line between multiple roots
            if i < self.roots.len() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

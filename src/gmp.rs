// use std::collections::HashMap;
// use std::hash::Hash;

// use nalgebra::{Cholesky, Mat, MatView, Col, ColView, ColViewMut, stack};

use dyn_stack::{MemBuffer, MemStack};
use eros::Result;
use faer::linalg::{cholesky::llt, matmul::matmul};
use faer::prelude::*;
use faer::{Accum, Conj, Par, Spec};

// pub struct GaussianFactor<T> {
//     vars: Vec<(T, usize)>, // (label, dim)
//     var_range: HashMap<T, (usize, usize)>,
//     mean: Col<f64>,
//     covariance: Mat<f64>,
// }

// impl<T: Clone + PartialEq + Eq + Hash> GaussianFactor<T> {
//     pub fn new(vars: Vec<(T, usize)>, mean: Col<f64>, covariance: Mat<f64>) -> Self {
//         let var_range = vars
//             .iter()
//             .scan(0, |acc, (name, dim)| {
//                 let start = *acc;
//                 *acc += *dim;
//                 Some((name.clone(), (start, *dim)))
//             })
//             .collect::<HashMap<_, (_, _)>>();
//         Self {
//             vars,
//             var_range,
//             mean,
//             covariance,
//         }
//     }

//     pub fn contains(&self, label: &T) -> bool {
//         self.vars.iter().any(|var| var.0 == *label)
//     }

//     pub fn vars_iter(&self) -> impl Iterator<Item = &(T, usize)> {
//         self.vars.iter()
//     }
// }

// pub struct LinearGaussianFactor<T> {
//     /// A linear Gaussian factor representing || mat * vars - mean ||_covariance^2
//     /// Should we rather use weighted means and precisions?
//     vars: HashMap<T, (usize, usize)>, // (var: (offset, dim))
//     mat: Mat<f64>,
//     // var_range: HashMap<T, (usize, usize)>, // column range for each variable (first, length)
//     mean: Col<f64>,
//     covariance: Mat<f64>,
// }

// impl<T: Clone + PartialEq + Eq + Hash> LinearGaussianFactor<T> {
//     pub fn new(
//         vars: HashMap<T, usize>, // (name: dim)
//         mat: Mat<f64>,
//         mean: Col<f64>,
//         covariance: Mat<f64>,
//     ) -> Self {
//         let vars = vars
//             .into_iter()
//             .scan(0, |acc, (name, dim)| {
//                 let start = *acc;
//                 *acc += dim;
//                 Some((name, (start, dim)))
//             })
//             .collect::<HashMap<_, (_, _)>>();
//         Self {
//             vars,
//             mat,
//             mean,
//             covariance,
//         }
//     }

//     pub fn combine(&self, other: &Self) -> Self {
//         // Collect all unique variables and determine unique column layout.
//         let mut vars: HashMap<T, (usize, usize)> = HashMap::new();
//         let mut total_cols = 0;
//         for (var, &(_, dim)) in &self.vars {
//             if !vars.contains_key(var) {
//                 vars.insert(var.clone(), (total_cols, dim));
//                 total_cols += dim;
//             }
//         }
//         for (var, &(_, dim)) in &other.vars {
//             if !vars.contains_key(var) {
//                 vars.insert(var.clone(), (total_cols, dim));
//                 total_cols += dim;
//             }
//         }

//         // Allocate matrix filled with zeros
//         let self_rows = self.mat.nrows();
//         let other_rows = other.mat.nrows();
//         let total_rows = self_rows + other_rows;
//         let mut mat = Mat::<f64>::zeros(total_rows, total_cols);

//         // Copy blocks from self.mat into the top block row: [0..self_rows]
//         for (var, &(self_col_start, dim)) in &self.vars {
//             let (col_start, _) = vars.get(var).unwrap();
//             let self_block = self
//                 .mat
//                 .view_range(.., self_col_start..self_col_start + dim);
//             let mut target_slice = mat.view_range_mut(0..self_rows, *col_start..*col_start + dim);
//             target_slice.copy_from(&self_block);
//         }

//         // Copy blocks from other.mat into the top block row: [self_rows..total_rows]
//         for (var, &(other_col_start, dim)) in &other.vars {
//             let (col_start, _) = vars.get(var).unwrap();
//             let other_block = other
//                 .mat
//                 .view_range(.., other_col_start..other_col_start + dim);
//             let mut target_slice =
//                 mat.view_range_mut(self_rows..total_rows, *col_start..*col_start + dim);
//             target_slice.copy_from(&other_block);
//         }

//         Self {
//             vars,
//             mat,
//             mean,
//             covariance,
//         }
//     }

//     pub fn cols_var(&self, var: &T) -> Result<MatView<f64>, String> {
//         match self.var_range.get(var) {
//             Some((start, dim)) => Ok(self.mat.columns(*start, *dim)),
//             None => Err("Variable not found".to_string()),
//         }
//     }

//     #[inline]
//     pub fn mean(&self) -> &Col<f64> {
//         &self.mean
//     }

//     #[inline]
//     pub fn covariance(&self) -> &Mat<f64> {
//         &self.covariance
//     }

//     pub fn contains(&self, label: &T) -> bool {
//         self.vars.iter().any(|var| var.0 == *label)
//     }

//     pub fn vars_iter(&self) -> impl Iterator<Item = &(T, usize)> {
//         self.vars.iter()
//     }
// }

// pub struct Gaussian {
//     pub mean: Col<f64>,
//     pub cov: Mat<f64>,
// }

// pub struct LocalGaussianFactor<T> {
//     var: (T, usize),       // (label, dim)
//     map_mat: Mat<f64>, // use view instead??
//     prior_msg: Gaussian,
//     workspace: Col<f64>, // holds the decision
//     cond_vars: Vec<(T, usize)>,
//     cond_map_mat: Mat<f64>,
//     cond_msg: Gaussian,
//     cond_workspace: Col<f64>,
// }

// impl<T: Clone + PartialEq + Eq + Hash> LocalGaussianFactor<T> {
//     // pub fn from_lgfs(lgfs: &[LinearGaussianFactor<T>], vars: &[T], cond_vars: &[T]) -> Self {
//     //     let vars: Vec<T> = vars.iter().cloned().collect();
//     //     let cond_vars: Vec<T> = cond_vars.iter().cloned().collect();

//     //     let mut mat = Mat::zeros(0, 0);
//     //     let mut cond_mat = Mat::zeros(0, 0);
//     //     let mut mean = Col::zeros(0);
//     //     let mut covariance = Mat::zeros(0, 0);

//     //     for lgf in lgfs {
//     //         todo!()
//     //     }
//     // }

//     pub fn decide(&mut self) {
//         let a = a;
//         let v_x = &self.prior_msg.cov;
//         let v_s = &self.cond_msg.cov;
//         let a_vx = a * v_x; // Allocate buffer for (A * Vx) once: shape (dim_s, dim_x)

//         // 1. y = mbs - A * mfx
//         // Pre-allocate once into the vector that will hold the solved `z`
//         self.cond_workspace.copy_from(&self.cond_msg.mean);
//         // z = (-1.0) * A * prior_mean + (1.0) * z
//         self.cond_workspace.gemv(-1.0, a, &self.prior_msg.mean, 1.0);

//         // Allocate P by cloning Vs once, then accumulate A * Vx * A^T into it:
//         // P = (1.0) * (a_vx) * A^T + (1.0) * P
//         let mut p = v_s.clone(); // no need to allocate a new matrix here???
//         p.gemm_tr(1.0, &a_vx, a, 1.0);

//         // 2. Solve P * z = tmp in place
//         // Cholesky consumes `p` in place, then `solve_mut` overwrites `z`
//         let chol_p = p.cholesky().expect("Matrix P is not positive-definite");
//         chol_p.solve_mut(&mut self.cond_workspace); // `z` now holds the solved value

//         // 4. Result = mfx + Vfx * A^T * z
//         self.workspace.copy_from(&self.prior_msg.mean);
//         self.workspace
//             .gemv_tr(1.0, &a_vx, &self.cond_workspace, 1.0);
//     }

//     pub fn cond_vars_contain(&self, label: &T) -> bool {
//         self.cond_vars.iter().any(|var| var.0 == *label)
//     }

//     pub fn cond_vars_iter(&self) -> impl Iterator<Item = &(T, usize)> {
//         self.cond_vars.iter()
//     }
// }

// /// An observation block in a Gaussian factor graph
// // Gaussian messages are parameterized by mean vector (m) and covariance matrix (V)
// /// Represents y = A x + B z where x is the variable of interest, z the separator, and y the observation
// pub struct ObsBlock {
//     a: Mat<f64>,
//     b: Mat<f64>,
//     mby: Col<f64>,
//     vby: Mat<f64>,
// }

// impl ObsBlock {
//     pub fn new(a: Mat<f64>, b: Mat<f64>, mby: Col<f64>, vby: Mat<f64>) -> Self {
//         Self { a, b, mby, vby }
//     }

//     pub fn forward(&self) {
//         todo!()
//     }

//     /// Forward pass through the observation block.
//     /// The result (mean) is written to `out`, which must be pre-allocated.
//     pub fn forward_m_only(
//         &self,
//         mfx: &Col<f64>,
//         vfx: &Mat<f64>,
//         mz: &Col<f64>, // separator decisions
//     ) -> Col<f64> {
//         let a_vx = &self.a * vfx; // Allocate buffer for (A * Vx) once: shape (dim_s, dim_x)

//         // y = my - A * mfx
//         let mut wsy = &self.mby - &self.b * mz;
//         wsy.gemv(-1.0, &self.a, &mfx, 1.0);

//         // Allocate P by cloning vy once, then accumulate A * Vx * A^T into it:
//         // g_inv = A Vx A.t + Vy
//         let mut g_inv = self.vby.clone();
//         g_inv.gemm_tr(1.0, &a_vx, &self.a, 1.0);

//         // Solve P z = y in place
//         // Cholesky consumes `p` in place, then `solve_mut` overwrites `z`
//         let chol = g_inv.cholesky().expect("Matrix G is not positive-definite");
//         chol.solve_mut(&mut wsy); // `z` now holds the solved value

//         // Result = mfx + Vfx * A^T * z
//         let mut mx = mfx.clone();
//         mx.gemv_tr(1.0, &a_vx, &wsy, 1.0);
//         mx
//     }
// }

/// A constraint block in a Gaussian factor graph with matrices A and B.
/// X is the main variable and S the separator variable.
/// Y = A X + Z and Z = B S are virtual variables.
pub struct ConstraintBlock {
    // ---- Logical Dimensions ----
    pub dim_x: usize,
    pub dim_s: usize,
    pub dim_y: usize,

    // ---- Matrices and Vectors ----
    a: Mat<f64>,     // with shape (dim_y, dim_x)
    b: Mat<f64>,     // with shape (dim_y, dim_s)
    xiby: Col<f64>,  // y backward weighted mean vector with shape (dim_y)
    wby: Mat<f64>,   // y backward precision matrix with shape (dim_y, dim_y)
    wby_a: Mat<f64>, // with shape (dim_y, dim_x)
    xifx: Col<f64>,  // x forward weighted mean vector with shape (dim_x)
    wfx: Mat<f64>,   // x forward precision matrix with shape (dim_x, dim_x)
    xitx: Col<f64>,  // x dual weighted mean vectors with shape (dim_x)
    rtx: Mat<f64>,   // x dual square-root precision matrix with shape (dim_x, dim_x)
    xibz: Col<f64>,  // z backward weighted mean vectors with shape (dim_y)
    wbz: Mat<f64>,   // z backward precision matrix with shape (dim_y, dim_y)

    // ---- Workspaces ----
    h_xitx: Col<f64>,   // (dim_x)
    h_at_wby: Mat<f64>, // (dim_x, dim_y)
    b_s: Col<f64>,      // (dim_y)
    mem_buf: MemBuffer,
}

impl ConstraintBlock {
    pub fn new(
        a: Mat<f64>,
        b: Mat<f64>,
        xiby: Col<f64>,
        wby: Mat<f64>,
        xifx: Col<f64>,
        wfx: Mat<f64>,
    ) -> Self {
        // 1. Extract logical dimensions
        let dim_y = a.nrows();
        let dim_x = a.ncols();
        let dim_s = b.ncols();

        // 2. Validate all inputs against logical dimensions
        // This guarantees the block is perfectly sized for its entire lifetime.
        assert_eq!(b.nrows(), dim_y);
        assert_eq!(xiby.nrows(), dim_y);
        assert_eq!(wby.nrows(), dim_y);
        assert_eq!(wby.ncols(), dim_y);
        assert_eq!(xifx.nrows(), dim_x);
        assert_eq!(wfx.nrows(), dim_x);
        assert_eq!(wfx.ncols(), dim_x);

        // 3. Allocate internal matrices and workspaces based on exact dimensions
        let wby_a = &wby * &a;

        let xitx = Col::zeros(dim_x);
        let rtx = Mat::zeros(dim_x, dim_x);
        let xibz = Col::zeros(dim_y);
        let wbz = Mat::zeros(dim_y, dim_y);

        let h_xitx = Col::zeros(dim_x);
        let h_at_wby = Mat::zeros(dim_x, dim_y);
        let b_s = Col::zeros(dim_y);

        // Compute the size and alignment of the required scratch space for Cholesky decompositions and associated solver
        let cholesky_memory =
            llt::factor::cholesky_in_place_scratch::<f64>(dim_x, Par::Seq, Spec::default());
        let solve_memory = llt::solve::solve_in_place_scratch::<f64>(dim_x, dim_y, Par::Seq);

        // Allocate the scratch space
        let mem_buf = MemBuffer::new(cholesky_memory.or(solve_memory));

        Self {
            dim_x,
            dim_y,
            dim_s,
            a,
            b,
            xiby,
            wby,
            wby_a,
            xifx,
            wfx,
            xitx,
            rtx,
            xibz,
            wbz,
            h_xitx,
            h_at_wby,
            b_s,
            mem_buf,
        }
    }

    pub fn eliminate_x(&mut self) -> Result<()> {
        let mut stack = MemStack::new(&mut self.mem_buf);

        // Compute xitx = xifx + a.t xiby, which is also used in solve_x
        self.xitx.copy_from(&self.xifx);
        matmul(
            &mut self.xitx,
            Accum::Add,
            self.a.transpose(),
            &self.xiby,
            1.0,
            Par::Seq,
        );

        // Compute wtx = (wfx + a.t wby a) and its Cholesky decomposition rtx, the later being also used in solve_x
        self.rtx.copy_from(&self.wfx);
        matmul(
            &mut self.rtx,
            Accum::Add,
            self.a.transpose(),
            &self.wby_a,
            1.0,
            Par::Seq,
        );

        // compute the decomposition
        llt::factor::cholesky_in_place(
            self.rtx.as_mut(),
            llt::factor::LltRegularization::default(),
            Par::Seq,
            &mut stack,
            default(),
        )?;

        // Compute xibz = xiby - wby a h (xifx + a.t xiby) = xiby - wby_a h_xitx
        // where h_xitx = h (xifx + a.t xiby) with hinv = (wfx + a.t wby a)
        self.h_xitx.copy_from(&self.xitx);
        llt::solve::solve_in_place_with_conj(
            self.rtx.as_ref(),
            Conj::No,
            self.h_xitx.as_mat_mut(),
            Par::Seq,
            &mut stack,
        );
        self.xibz.copy_from(&self.xiby);
        matmul(
            &mut self.xibz,
            Accum::Add,
            &self.wby_a,
            &self.h_xitx,
            -1.0,
            Par::Seq,
        );

        // Compute wbz = wby - wby a h a.t wby
        self.h_at_wby.copy_from(&self.wby_a.transpose());
        llt::solve::solve_in_place_with_conj(
            self.rtx.as_ref(),
            Conj::No,
            self.h_at_wby.as_mut(),
            Par::Seq,
            &mut stack,
        );
        self.wbz.copy_from(&self.wby);
        matmul(
            &mut self.wbz,
            Accum::Add,
            &self.wby_a,
            &self.h_at_wby,
            -1.0,
            Par::Seq,
        );

        Ok(())
    }

    pub fn solve_x(&mut self, mut x: ColMut<f64>, s: ColRef<f64>) -> Result<()> {
        let mut stack = MemStack::new(&mut self.mem_buf);

        matmul(&mut self.b_s, Accum::Replace, &self.b, s, 1.0, Par::Seq);

        x.copy_from(&self.xitx);
        matmul(
            x.rb_mut(),
            Accum::Add,
            self.wby_a.transpose(),
            &self.b_s,
            -1.0,
            Par::Seq,
        );

        llt::solve::solve_in_place_with_conj(
            self.rtx.as_ref(),
            Conj::No,
            x.as_mat_mut(),
            Par::Seq,
            &mut stack,
        );
        Ok(())
    }
}

//     pub fn forward_mx_vx(&self) {
//         todo!()
//     }

//     /// Forward pass through the observation block.
//     /// The result (mean) is written to `out`, which must be pre-allocated.
//     pub fn forward_mx(
//         &self,
//         mx: &Col<f64>,
//         vx: &Mat<f64>,
//         z: &Col<f64>, // separator decisions
//     ) -> Col<f64> {
//         let a_vx = &self.a * vx; // Allocate buffer for (A * Vx) once: shape (dim_s, dim_x)

//         // y = my - A * mx
//         let mut wsy = &self.mby - &self.b * z;
//         wsy.gemv(-1.0, &self.a, &mx, 1.0);

//         // g_inv = A Vx A.t + Vy
//         let mut g_inv = self.vby.clone();
//         g_inv.gemm_tr(1.0, &a_vx, &self.a, 1.0);

//         // Solve P z = y in place
//         // Cholesky consumes `p` in place, then `solve_mut` overwrites `z`
//         let chol = g_inv.cholesky().expect("Matrix G is not positive-definite");
//         chol.solve_mut(&mut wsy); // `z` now holds the solved value

//         // Result = mx + Vx * A^T * z
//         let mut mx = mx.clone();
//         mx.gemv_tr(1.0, &a_vx, &wsy, 1.0);
//         mx
//     }
// }

// impl<T> GaussMsg<T> {
//     pub fn new(
//         vars: Vec<T>,
//         a: Mat<f64>,
//         mean: Col<f64>,
//         covariance: Mat<f64>,
//     ) -> Result<Self, String> {
//         let n = vars.len();
//         let m = mean.len();

//         if a.nrows() != m || a.ncols() != n {
//             return Err(format!(
//                 "Matrix A dimension mismatch: expected ({m}, {n}), got ({}, {})",
//                 a.nrows(),
//                 a.ncols()
//             ));
//         }
//         if covariance.nrows() != m || covariance.ncols() != m {
//             return Err(format!(
//                 "Covariance dimension mismatch: expected ({m}, {m}), got ({}, {})",
//                 covariance.nrows(),
//                 covariance.ncols()
//             ));
//         }

//         Ok(Self {
//             vars,
//             a,
//             mean,
//             covariance,
//         })
//     }

//     #[inline]
//     pub fn n(&self) -> usize {
//         self.vars.len()
//     }

//     #[inline]
//     pub fn m(&self) -> usize {
//         self.mean.len()
//     }
// }

// /// A Gaussian factor that depends on a single variable (potentially multivariate)
// pub struct SingleGaussMsg<T> {
//     var: T,
//     params: f64, // linear transformation, mean, covariance...
// }

// /// A Gaussian factor that depends on multiple variables (potentially multivariate)
// pub struct MultiGaussMsg<T> {
//     pub vars: Vec<T>,
//     pub params: f64, // linear transformation, mean, covariance...
// }

// /// A Gaussian factor that depends on a single variable (potentially multivariate) and multiple conditioning variables (potentially multivariate)
// pub struct DownGaussMsg<T> {
//     var: T,
//     cond_vars: Vec<T>,
//     cond_params: Vec<f64>, // linear transformation, mean, covariance...
// }

// impl<T> CondGaussMsg<T> {
//     pub fn new(var: T) -> Self {
//         Self {
//             var,
//             cond_vars: Vec::new(),
//             cond_params: Vec::new(),
//         }
//     }

//     pub fn combine(&mut self, other: &GaussFactor<T>) {
//         todo!()
//     }

//     pub fn cond_vars_iter(&self) -> impl Iterator<Item = &T> {
//         self.cond_vars.iter()
//     }

//     pub fn reduce(&self) -> Option<GaussFactor<T>> {
//         todo!()
//     }

//     // pub fn new(var: T, cond_vars: Vec<T>, cond_params: f64) -> Self {
//     //     Self {
//     //         var,
//     //         cond_vars,
//     //         cond_params,
//     //     }
//     // }
// }

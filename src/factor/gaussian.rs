use anyhow::{Result, bail};

use faer::linalg::matmul::matmul;
use faer::prelude::*;
use faer::{Accum, ColRef, Par};
use faer_traits::RealField;
use log::debug;

use super::anchor::AnchorFactor;
use crate::variable::VarId;

/// A Gaussian factor on the observation Y := A X, with weighted mean `xiby` and precision `wby`.
pub struct GaussianFactor<T> {
    pub vars: Vec<VarId>,
    pub dim_x: usize,
    pub dim_y: usize,

    // ---- Matrices and Vectors ----
    pub a: Mat<T>, // the matrix A with dimensions (dim_y, dim_x)
    pub a_ctx: Vec<(VarId, usize, usize)>, // the context of the variables in the A matrix (var, offset, size)
    pub xiby: Col<T>,                      // y backward weighted mean vector with shape (dim_y)
    pub wby: Mat<T>,                       // y backward precision matrix with shape (dim_y, dim_y)
}

impl<T: RealField> GaussianFactor<T> {
    pub fn new(
        vars: Vec<VarId>,
        a: Mat<T>,
        a_ctx: Vec<(VarId, usize, usize)>,
        xiby: Col<T>,
        wby: Mat<T>,
    ) -> Self {
        Self {
            vars,
            dim_x: a.ncols(),
            dim_y: a.nrows(),
            a,
            a_ctx,
            xiby,
            wby,
        }
    }

    pub fn get_memory_layout(&self, var: VarId) -> Option<(usize, usize)> {
        self.a_ctx
            .iter()
            .find(|(v, _, _)| *v == var)
            .map(|(_, offset, size)| (*offset, *size))
    }

    /// Eliminates a fixed variable from the factor by updating `a`, `a_ctx`, `vars`, and `xiby` (`wby` is unchanged).
    pub fn eliminate_fixed(&mut self, var: VarId, val: &[T]) -> Result<()> {
        match self.get_memory_layout(var) {
            None => {
                debug!("Variable {} not found in a_ctx.", var);
            }
            Some((offset, size)) => {
                if val.len() != size {
                    bail!(
                        "Provided value length ({}) does not match variable size ({}) for variable {}.",
                        val.len(),
                        size,
                        var
                    );
                }

                // Update xiby = xiby - wby * c_val
                let c = self.a.subcols(offset, size);
                let val_col = ColRef::from_slice(val);
                let mut c_val = Col::<T>::zeros(self.dim_y);
                matmul(
                    c_val.rb_mut(),
                    Accum::Replace,
                    c,
                    val_col.rb(),
                    T::one(),
                    Par::Seq,
                );
                matmul(
                    self.xiby.as_mut().as_mat_mut(),
                    Accum::Add,
                    self.wby.rb(),
                    c_val.as_mat(),
                    T::one().neg(),
                    Par::Seq,
                );

                // Remove columns from matrix A in-place (zero heap allocations)
                let remaining_cols_after = self.dim_x - (offset + size);
                if remaining_cols_after > 0 {
                    let col_stride = self.a.col_stride();
                    unsafe {
                        let dst = self.a.as_ptr_mut().offset(offset as isize * col_stride);
                        let src = self
                            .a
                            .as_ptr()
                            .offset((offset + size) as isize * col_stride);
                        let num_elements = remaining_cols_after * (col_stride as usize);
                        std::ptr::copy(src, dst, num_elements);
                    }
                }
                self.dim_x -= size;
                self.a.truncate(self.dim_y, self.dim_x);

                // Update vars
                self.vars.retain(|&v| v != var);

                // Remove variable from a_ctx and shift subsequent offsets down
                self.a_ctx.retain_mut(|(v, o, _)| {
                    if *v == var {
                        false
                    } else {
                        if *o > offset {
                            *o -= size;
                        }
                        true
                    }
                });
            }
        }
        Ok(())
    }
}

pub fn preprocess_factors<T: RealField>(
    gaussian_factors: &mut [GaussianFactor<T>],
    anchor_factors: &[AnchorFactor<T>],
) -> Result<()> {
    for anchor in anchor_factors {
        gaussian_factors
            .iter_mut()
            .try_for_each(|f| f.eliminate_fixed(anchor.var, &anchor.val))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_eliminate_fixed() {
        let mut factor = GaussianFactor::<f64>::new(
            vec![],
            Mat::zeros(0, 0),
            vec![],
            Col::zeros(0),
            Mat::zeros(0, 0),
        );

        factor
            .eliminate_fixed(42, &[3.0, 3.0])
            .expect("Error while eliminating missing fixed variable: should do nothing.");
        assert_eq!(factor.vars, vec![]);

        let mut factor = GaussianFactor::<f64>::new(
            vec![0, 1],
            mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            vec![(0, 0, 2), (1, 2, 1)],
            col![0.0, 4.0],
            mat![[1.0, 0.2], [0.2, 0.8]],
        );

        factor
            .eliminate_fixed(42, &[1.0, -2.0])
            .expect("Error while eliminating irrelevant fixed variable: should do nothing.");
        assert_eq!(factor.vars, vec![0, 1]);
        assert_eq!(factor.a, mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(factor.a_ctx, vec![(0, 0, 2), (1, 2, 1)]);
        assert_eq!(factor.xiby, col![0.0, 4.0]);
        assert_eq!(factor.wby, mat![[1.0, 0.2], [0.2, 0.8]]);

        factor
            .eliminate_fixed(0, &[1.0, -2.0])
            .expect("Error while eliminating relevant fixed variable.");
        assert_eq!(factor.vars, vec![1]);
        assert_eq!(factor.a, mat![[3.0], [6.0]]);
        assert_eq!(factor.a_ctx, vec![(1, 0, 1)]);
        assert_eq!(factor.xiby, col![4.2, 9.4]);
        assert_eq!(factor.wby, mat![[1.0, 0.2], [0.2, 0.8]]);
    }

    #[test]
    fn test_preprocess_factors() {
        let mut gaussian_factors = vec![
            GaussianFactor::<f64>::new(
                vec![0, 1],
                mat![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                vec![(0, 0, 2), (1, 2, 1)],
                col![0.0, 0.0],
                mat![[1.0, 0.0], [0.0, 0.8]],
            ),
            GaussianFactor::<f64>::new(
                vec![1, 2],
                mat![[1.0, 2.0], [1.0, 2.0]],
                vec![(1, 0, 1), (2, 1, 1)],
                col![0.0, 0.0],
                mat![[1.0, 0.0], [0.0, 0.8]],
            ),
            GaussianFactor::<f64>::new(
                vec![2, 3, 4],
                mat![[2.0, 3.0, 4.0]],
                vec![(2, 0, 1), (3, 1, 1), (4, 2, 1)],
                col![0.0],
                mat![[1.0]],
            ),
        ];

        let anchor_factors = vec![
            AnchorFactor::new(0, vec![3.0, 3.0]),
            AnchorFactor::new(4, vec![-1.0]),
        ];

        preprocess_factors(&mut gaussian_factors, &anchor_factors)
            .expect("Error while preprocessing factors.");
        assert_eq!(gaussian_factors[0].vars, vec![1]);
        assert_eq!(gaussian_factors[0].a, mat![[1.0], [1.0]]);
        assert_eq!(gaussian_factors[0].a_ctx, vec![(1, 0, 1)]);
        assert_eq!(gaussian_factors[1].vars, vec![1, 2]);
        assert_eq!(gaussian_factors[1].a, mat![[1.0, 2.0], [1.0, 2.0]]);
        assert_eq!(gaussian_factors[1].a_ctx, vec![(1, 0, 1), (2, 1, 1)]);
        assert_eq!(gaussian_factors[2].vars, vec![2, 3]);
        assert_eq!(gaussian_factors[2].a, mat![[2.0, 3.0]]);
        assert_eq!(gaussian_factors[2].a_ctx, vec![(2, 0, 1), (3, 1, 1)]);
    }
}

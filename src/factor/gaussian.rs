/*!
A module for Gaussian factors.

For notation convention, we refer to
H.-A. Loeliger, L. Bruderer, H. Malmberg, F. Wadehn, and N. Zalmai,
"On sparsity by NUV-EM, Gaussian message passing, and Kalman smoothing,"
2016 Information Theory & Applications Workshop (ITA), La Jolla, CA, Jan. 31 - Feb. 5, 2016.
*/

use anyhow::Result;
use log::debug;

use faer::linalg::matmul::matmul;
use faer::prelude::*;
use faer::{Accum, ColRef, Par};
use faer_traits::RealField;

use super::anchor::AnchorFactor;
use crate::variable::VarId;
use crate::variable::gaussian::VarBlock;

/// A Gaussian factor on the observation Y := sum_i Ai Xi, parameterized by
/// weighted mean and precision.
pub struct GaussianFactor<T> {
    pub vars: Vec<VarId>,
    pub blk_x: Vec<VarBlock<T>>,
    pub dim_y: usize,
    pub xiby: Col<T>, // y backward weighted mean vector with shape (dim_y)
    pub wby: Mat<T>,  // y backward precision matrix with shape (dim_y, dim_y)
}

impl<T: RealField> GaussianFactor<T> {
    /// Creates a new Gaussian factor.
    /// It is the caller responsibility to ensure that all dimensions are compatible.
    pub fn new(blk_x: Vec<VarBlock<T>>, xiby: Col<T>, wby: Mat<T>) -> Self {
        let dim_y = xiby.nrows();

        let vars = blk_x.iter().map(|blk| blk.id).collect();
        Self {
            vars,
            blk_x,
            dim_y,
            xiby,
            wby,
        }
    }

    pub fn get_blk(&self, id: VarId) -> Option<&VarBlock<T>> {
        self.blk_x.iter().find(|blk| blk.id == id)
    }

    /// Eliminates a fixed variable from the factor.
    /// The following fields are modified in place: `blk_x`, `vars`, `xiby`.
    /// Note that eliminating a variable does not impact wby.
    pub fn eliminate_fixed(&mut self, id: VarId, value: &[T], tmp: &mut [T]) -> Result<()> {
        match self.blk_x.iter().enumerate().find(|&(_, blk)| blk.id == id) {
            None => debug!("Variable {} not found in the Gaussian factor.", id),
            Some((i, blk)) => {
                let tmp_y = &mut tmp[..self.dim_y];
                // Update xiby = xiby - wby * ai ui
                matmul(
                    ColMut::from_slice_mut(tmp_y).as_mat_mut(), // contains ai * ui
                    Accum::Replace,
                    blk.a.as_ref(),
                    ColRef::from_slice(value).as_mat(),
                    T::one(),
                    Par::Seq,
                );
                matmul(
                    self.xiby.as_mat_mut(),
                    Accum::Add,
                    self.wby.as_ref(),
                    ColRef::from_slice(tmp_y).as_mat(),
                    T::one().neg(),
                    Par::Seq,
                );

                // Remove elements
                self.vars.swap_remove(i);
                self.blk_x.swap_remove(i);
            }
        }
        Ok(())
    }
}

/// Preprocess Gaussian factors in place by eliminating the anchored (fixed) variables.
/// See eliminate_fixed method in [`GaussianFactor`] for more details.
pub fn preprocess_factors<T: RealField>(
    gaussian_factors: &mut [GaussianFactor<T>],
    anchor_factors: &[AnchorFactor<T>],
) -> Result<()> {
    // Allocate once a sufficiently large workspace to work with all factors.
    let max_dim_y = gaussian_factors.iter().map(|f| f.dim_y).max().unwrap_or(0);
    println!("max dim y {}", max_dim_y);
    let mut tmp = vec![T::zero(); max_dim_y];

    for anchor in anchor_factors {
        gaussian_factors
            .iter_mut()
            .try_for_each(|f| f.eliminate_fixed(anchor.id, &anchor.value, &mut tmp))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_eliminate_fixed() {
        let mut factor = GaussianFactor::<f64>::new(vec![], Col::zeros(0), Mat::zeros(0, 0));
        let mut tmp = vec![0.0; factor.dim_y]; // the workspace needs to be larger than the y_dim

        let res = factor.eliminate_fixed(42, &[3.0, 3.0], &mut tmp);
        assert!(res.is_ok());
        assert_eq!(factor.blk_x, vec![]);

        let mut factor = GaussianFactor::<f64>::new(
            vec![
                VarBlock {
                    id: 0,
                    a: mat![[1.0, 2.0], [4.0, 5.0]],
                },
                VarBlock {
                    id: 1,
                    a: mat![[3.0], [6.0]],
                },
            ],
            col![0.0, 4.0],
            mat![[1.0, 0.2], [0.2, 0.8]],
        );
        let mut tmp = vec![0.0; factor.dim_y];

        let res = factor.eliminate_fixed(42, &[3.0, 3.0], &mut tmp);
        assert!(res.is_ok());
        assert_eq!(factor.blk_x.len(), 2);
        assert_eq!(factor.blk_x[0].id, 0);
        assert_eq!(factor.blk_x[0].a, mat![[1.0, 2.0], [4.0, 5.0]]);
        assert_eq!(factor.blk_x[1].id, 1);
        assert_eq!(factor.blk_x[1].a, mat![[3.0], [6.0]]);
        assert_eq!(factor.xiby, col![0.0, 4.0]);
        assert_eq!(factor.wby, mat![[1.0, 0.2], [0.2, 0.8]]);

        let res = factor.eliminate_fixed(0, &[1.0, -2.0], &mut tmp);
        assert!(res.is_ok());
        assert_eq!(factor.blk_x.len(), 1);
        assert_eq!(factor.blk_x[0].id, 1);
        assert_eq!(factor.blk_x[0].a, mat![[3.0], [6.0]]);
        assert_eq!(factor.xiby, col![4.2, 9.4]);
        assert_eq!(factor.wby, mat![[1.0, 0.2], [0.2, 0.8]]);
    }

    #[test]
    fn test_preprocess_factors() {
        let mut gaussian_factors = vec![
            GaussianFactor::<f64>::new(
                vec![
                    VarBlock {
                        id: 0,
                        a: mat![[0.0, 0.0], [0.0, 0.0]],
                    },
                    VarBlock {
                        id: 1,
                        a: mat![[1.0], [1.0]],
                    },
                ],
                col![0.0, 0.0],
                mat![[1.0, 0.0], [0.0, 0.8]],
            ),
            GaussianFactor::<f64>::new(
                vec![
                    VarBlock {
                        id: 1,
                        a: mat![[1.0], [1.0]],
                    },
                    VarBlock {
                        id: 2,
                        a: mat![[2.0], [2.0]],
                    },
                ],
                col![0.0, 0.0],
                mat![[1.0, 0.0], [0.0, 0.8]],
            ),
            GaussianFactor::<f64>::new(
                vec![
                    VarBlock {
                        id: 2,
                        a: mat![[2.0]],
                    },
                    VarBlock {
                        id: 3,
                        a: mat![[3.0]],
                    },
                    VarBlock {
                        id: 4,
                        a: mat![[4.0]],
                    },
                ],
                col![0.0],
                mat![[1.0]],
            ),
        ];

        let anchor_factors = vec![
            AnchorFactor::new(0, vec![3.0, 3.0]),
            AnchorFactor::new(4, vec![-1.0]),
        ];

        let res = preprocess_factors(&mut gaussian_factors, &anchor_factors);
        assert!(res.is_ok());
        assert_eq!(gaussian_factors[0].vars.len(), 1);
        assert_eq!(gaussian_factors[0].vars[0], 1);

        assert_eq!(gaussian_factors[1].vars.len(), 2);
        assert_eq!(gaussian_factors[1].vars[0], 1);
        assert_eq!(gaussian_factors[1].vars[1], 2);

        assert_eq!(gaussian_factors[2].vars.len(), 2);
        assert_eq!(gaussian_factors[2].vars[0], 2);
        assert_eq!(gaussian_factors[2].vars[1], 3);
    }
}

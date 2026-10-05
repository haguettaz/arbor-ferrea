//! Gaussian factors over linear observation models.

use anyhow::{Result, ensure};
use log::debug;

use faer::linalg::matmul::matmul;
use faer::prelude::*;
use faer::{Accum, ColRef, Par};
use faer_traits::RealField;

use super::pin::PinFactor;
use super::symbolic::SymbolicFactor;

/// A variable paired with its linear transformation matrix.
#[derive(Debug, PartialEq, Clone)]
pub struct LinVar<T> {
    /// Variable identifier.
    pub id: usize,
    /// Transformation matrix.
    pub mat: Mat<T>,
}

/// A Gaussian factor parameterized by an observation precision matrix and weighted mean.
pub struct GaussianFactor<T> {
    /// Linear variable blocks coupling each variable to the observation.
    pub lin_vars: Vec<LinVar<T>>,

    /// Identifiers of involved variables, aligned with `lin_vars`.
    vars: Vec<usize>,

    /// Observation dimension.
    pub dim_obs: usize,

    /// Observation weighted mean vector of shape `(dim_obs)`.
    pub xib_obs: Col<T>,

    /// Observation precision matrix of shape `(dim_obs, dim_obs)`.
    pub wb_obs: Mat<T>,
}

impl<T: RealField> GaussianFactor<T> {
    /// Creates a factor without validating parameters.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `wb_obs` is square with dimension matching
    /// `xib_obs.nrows()`, that all `lin_vars` matrices have `dim_obs` rows,
    /// and that variable IDs are unique.
    /// Use [`try_new`](Self::try_new) for checked construction.
    pub fn new(lin_vars: Vec<LinVar<T>>, xib_obs: Col<T>, wb_obs: Mat<T>) -> Self {
        let vars = lin_vars.iter().map(|lin_var| lin_var.id).collect();

        let dim_obs = xib_obs.nrows();

        Self {
            lin_vars,
            vars,
            dim_obs,
            xib_obs,
            wb_obs,
        }
    }

    /// Creates a factor after verifying matrix dimensions and variable uniqueness.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `xi_obs` does not have shape `(dim_obs,)`.
    /// - `wb_obs` does not have shape `(dim_obs, dim_obs)`.
    /// - Any transformation matrix in `lin_vars` does not have `dim_obs` rows.
    /// - `lin_vars` contains duplicate variable IDs.
    pub fn try_new(lin_vars: Vec<LinVar<T>>, xib_obs: Col<T>, wb_obs: Mat<T>) -> Result<Self> {
        // Check observation dimension
        ensure!(
            (xib_obs.nrows() == wb_obs.nrows()) && (xib_obs.nrows() == wb_obs.ncols()),
            format!(
                "Incompatible dimension found for xib_obs ({} rows) and wb_obs ({} rows and {} cols).",
                xib_obs.nrows(),
                wb_obs.nrows(),
                wb_obs.ncols()
            )
        );
        let dim_obs = xib_obs.nrows();

        let mut seen = std::collections::HashSet::with_capacity(lin_vars.len());
        let mut vars = Vec::with_capacity(lin_vars.len());
        for lin_var in &lin_vars {
            ensure!(
                seen.insert(lin_var.id),
                format!("Duplicate variable {} in Gaussian factor.", lin_var.id)
            );
            ensure!(
                lin_var.mat.nrows() == dim_obs,
                format!(
                    "Incompatible dimension found for variable {}. The current observation dimension is {} but the variable matrix has {} rows.",
                    lin_var.id,
                    dim_obs,
                    lin_var.mat.nrows()
                )
            );
            vars.push(lin_var.id);
        }

        let factor = Self {
            lin_vars,
            vars,
            dim_obs,
            xib_obs,
            wb_obs,
        };
        Ok(factor)
    }

    /// Returns a reference to the [`LinVar`] associated with `id`, or `None` if absent.
    pub fn lin_var(&self, id: usize) -> Option<&LinVar<T>> {
        self.lin_vars.iter().find(|lin_var| lin_var.id == id)
    }

    /// Returns `true` if this factor constrains the variable `id`.
    /// See also [`lin_var`](Self::lin_var).
    pub fn contains(&self, id: usize) -> bool {
        self.vars.contains(&id)
    }

    /// Eliminates a pinned variable in place, updating `xib_obs` using `tmp` workspace.
    pub fn eliminate_fixed(&mut self, id: usize, value: &[T], tmp: &mut [T]) -> Result<()> {
        match self
            .lin_vars
            .iter()
            .enumerate()
            .find(|&(_, lin_var)| lin_var.id == id)
        {
            None => debug!("Variable {} not found in the Gaussian factor.", id),
            Some((i, lin_var)) => {
                anyhow::ensure!(
                    value.len() == lin_var.mat.ncols(),
                    "Pin value length ({}) does not match variable {} size ({})",
                    value.len(),
                    id,
                    lin_var.mat.ncols()
                );
                let tmp_obs = &mut tmp[..self.dim_obs];
                // Update xib_obs <- xib_obs - wb_obs * lin_var.mat lin_var.value
                matmul(
                    ColMut::from_slice_mut(tmp_obs).as_mat_mut(),
                    Accum::Replace,
                    lin_var.mat.as_ref(),
                    ColRef::from_slice(value).as_mat(),
                    T::one(),
                    Par::Seq,
                ); // tmp_obs = lin_var.mat lin_var.value
                matmul(
                    self.xib_obs.as_mat_mut(),
                    Accum::Add,
                    self.wb_obs.as_ref(),
                    ColRef::from_slice(tmp_obs).as_mat(),
                    T::one().neg(),
                    Par::Seq,
                ); // xib_obs = xib_obs - wb_obs tmp_obs

                // Remove elements
                self.vars.swap_remove(i);
                self.lin_vars.swap_remove(i);
            }
        }
        Ok(())
    }
}

impl<T> SymbolicFactor for GaussianFactor<T> {
    fn vars(&self) -> &[usize] {
        &self.vars
    }
}

/// Eliminates all pinned variables from the given factors in place.
pub fn preprocess_factors<T: RealField>(
    gaussian_factors: &mut [GaussianFactor<T>],
    pin_factors: &[PinFactor<T>],
) -> Result<()> {
    // Allocate once a sufficiently large workspace to work with all factors.
    let max_dim_obs = gaussian_factors
        .iter()
        .map(|f| f.dim_obs)
        .max()
        .unwrap_or(0);
    let mut tmp = vec![T::zero(); max_dim_obs];

    for pin_factor in pin_factors {
        gaussian_factors
            .iter_mut()
            .try_for_each(|f| f.eliminate_fixed(pin_factor.id, &pin_factor.value, &mut tmp))?;
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
        let mut tmp = vec![0.0; factor.dim_obs]; // the workspace needs to be larger than the y_dim

        let res = factor.eliminate_fixed(42, &[3.0, 3.0], &mut tmp);
        assert!(res.is_ok());
        assert_eq!(factor.lin_vars, vec![]);

        let mut factor = GaussianFactor::<f64>::new(
            vec![
                LinVar {
                    id: 0,
                    mat: mat![[1.0, 2.0], [4.0, 5.0]],
                },
                LinVar {
                    id: 1,
                    mat: mat![[3.0], [6.0]],
                },
            ],
            col![0.0, 4.0],
            mat![[1.0, 0.2], [0.2, 0.8]],
        );
        let mut tmp = vec![0.0; factor.dim_obs];

        let res = factor.eliminate_fixed(42, &[3.0, 3.0], &mut tmp);
        assert!(res.is_ok());
        assert_eq!(factor.lin_vars.len(), 2);
        assert_eq!(factor.lin_vars[0].id, 0);
        assert_eq!(factor.lin_vars[0].mat, mat![[1.0, 2.0], [4.0, 5.0]]);
        assert_eq!(factor.lin_vars[1].id, 1);
        assert_eq!(factor.lin_vars[1].mat, mat![[3.0], [6.0]]);
        assert_eq!(factor.xib_obs, col![0.0, 4.0]);
        assert_eq!(factor.wb_obs, mat![[1.0, 0.2], [0.2, 0.8]]);

        let res = factor.eliminate_fixed(0, &[1.0, -2.0], &mut tmp);
        assert!(res.is_ok());
        assert_eq!(factor.lin_vars.len(), 1);
        assert_eq!(factor.lin_vars[0].id, 1);
        assert_eq!(factor.lin_vars[0].mat, mat![[3.0], [6.0]]);
        assert_eq!(factor.xib_obs, col![4.2, 9.4]);
        assert_eq!(factor.wb_obs, mat![[1.0, 0.2], [0.2, 0.8]]);
    }

    #[test]
    fn test_preprocess_factors() {
        let mut gaussian_factors = vec![
            GaussianFactor::<f64>::new(
                vec![
                    LinVar {
                        id: 0,
                        mat: mat![[0.0, 0.0], [0.0, 0.0]],
                    },
                    LinVar {
                        id: 1,
                        mat: mat![[1.0], [1.0]],
                    },
                ],
                col![0.0, 0.0],
                mat![[1.0, 0.0], [0.0, 0.8]],
            ),
            GaussianFactor::<f64>::new(
                vec![
                    LinVar {
                        id: 1,
                        mat: mat![[1.0], [1.0]],
                    },
                    LinVar {
                        id: 2,
                        mat: mat![[2.0], [2.0]],
                    },
                ],
                col![0.0, 0.0],
                mat![[1.0, 0.0], [0.0, 0.8]],
            ),
            GaussianFactor::<f64>::new(
                vec![
                    LinVar {
                        id: 2,
                        mat: mat![[2.0]],
                    },
                    LinVar {
                        id: 3,
                        mat: mat![[3.0]],
                    },
                    LinVar {
                        id: 4,
                        mat: mat![[4.0]],
                    },
                ],
                col![0.0],
                mat![[1.0]],
            ),
        ];

        let pin_factors = vec![
            PinFactor::new(0, vec![3.0, 3.0]),
            PinFactor::new(4, vec![-1.0]),
        ];

        let res = preprocess_factors(&mut gaussian_factors, &pin_factors);
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

// use std::collections::HashMap;
// use std::fmt::Debug;
//
// use nalgebra::linalg::Cholesky;
// use nalgebra::{DMatrix, DMatrixView, DVector, RealField};
//
// use anyhow::{Context, Result, bail, ensure};
//
//
//
//
// #[derive(Debug)]
// pub struct Marginal<T: RealField> {
//     pub key: usize,
//     pub mean: DVector<T>,
//     pub covariance: DMatrix<T>,
// }
//
// // impl<T: RealField> Marginal<T> {
// //     pub fn from(
// //         key: usize,
// //         factors: I,
// //     ) -> Result<Self>
// //     where
// //         I: {
// //         let dim_x = likelihood.dim_x();
// //
// //         let (mut xi, mut w) = match prior_or_none {
// //             None => (DVector::zeros(dim_x), DMatrix::zeros(dim_x, dim_x)),
// //             Some(prior) => {
// //                 ensure!(
// //                     likelihood.keys() == prior.keys(),
// //                     "prior and likelihood keys do not coincide"
// //                 );
// //                 ensure!(
// //                     likelihood.dim_x() == prior.dim_x(),
// //                     "prior and likelihood dimensions do not coincide"
// //                 );
// //                 (prior.xifx.clone(), prior.wfx.clone())
// //             }
// //         };
// //         xi.gemv_tr(T::one(), &likelihood.a, &likelihood.xiby, T::one());
// //         w.quadform(T::one(), &likelihood.wby, &likelihood.a, T::one());
// //
// //         let chol_w =
// //             Cholesky::new(w)
// //                 .with_context(||format!("Cholesky failed while computing the marginal. Consider adding dumb priors for variable with {key}"))?;
// //
// //         let mean = chol_w.solve(&xi);
// //         let cov = chol_w.inverse();
// //
// //         Ok(Marginal { key, mean, cov })
// //     }
// // }
//
// // #[derive(Debug)]
// // pub struct Prior<T: RealField> {
// //     key: usize,
// //     xifx: DVector<T>,
// //     wfx: DMatrix<T>,
// // }
//
// // impl<T: RealField> Prior<T> {
// //     pub fn keys(&self) -> &[usize] {
// //         std::slice::from_ref(&self.key)
// //     }
// //
// //     pub fn has_key(&self, key: usize) -> bool {
// //         self.key == key
// //     }
// //
// //     pub fn dim_x(&self) -> usize {
// //         self.xifx.nrows()
// //     }
// // }
//
//
//
//
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use nalgebra::{dmatrix, dvector};
//
//     const X1: Symbol = Symbol('x', 1);
//     const X2: Symbol = Symbol('x', 2);
//     const X3: Symbol = Symbol('x', 3);
//     const X42: Symbol = Symbol('x', 42);
//
//     #[test]
//     fn test_factor_graph() {
//         let mut factor_graph = FactorGraph::new();
//
//         let x1 = factor_graph
//             .add_variable(X1, 1)
//             .expect("Adding a new variable must work.");
//         let x2 = factor_graph
//             .add_variable(X2, 3)
//             .expect("Adding a new 3-dimensional variable must also work.");
//         let x3 = factor_graph
//             .add_variable(X3, 1_000_000)
//             .expect("Adding a new, very high-dimension variable must also work.");
//         let _ = factor_graph
//             .add_variable(X2, 3)
//             .expect_err("Adding an already registered variable must fail.");
//         let not_registered_x42 = 42; // a not registered variable
//
//         assert_eq!(
//             factor_graph.n_keys(),
//             3,
//             "The factor graph has three registered variables: x1, x2, and x3"
//         );
//
//         let symbolx1 = factor_graph.symbol(x1);
//         assert_eq!(
//             symbolx1,
//             Some(&X1),
//             "A registered variable has an accessible symbol."
//         );
//         let not_registered_symbol = factor_graph.symbol(not_registered_x42);
//         assert_eq!(
//             not_registered_symbol, None,
//             "A non-registered variable has no symbol."
//         );
//
//         let x2_from_its_symbol = factor_graph.key(&X2);
//         assert_eq!(
//             x2_from_its_symbol,
//             Some(x2),
//             "A registered variable can be accessed via its symbol."
//         );
//         let x42_from_its_symbol = factor_graph.key(&X42);
//         assert_eq!(
//             x42_from_its_symbol, None,
//             "A non-registered variable cannot be accessed via its symbol."
//         );
//
//         let sizex3 = factor_graph.size(x3);
//         assert_eq!(
//             sizex3,
//             Some(1_000_000),
//             "The size of a registered variable is accessible from its key."
//         );
//         let size_not_registered_x42 = factor_graph.size(not_registered_x42);
//         assert_eq!(
//             size_not_registered_x42, None,
//             "The size of a non-registered variable is not accessible."
//         );
//
//         factor_graph
//             .add_prior(x1, DVector::zeros(1), DMatrix::identity(1, 1))
//             .expect("Adding a prior on a registered variable must work.");
//         factor_graph
//             .add_prior(x2, DVector::zeros(2), DMatrix::identity(3, 3))
//             .expect_err("Adding a prior with incorrect dimensions must fail.");
//         factor_graph
//             .add_prior(x2, DVector::zeros(3), DMatrix::identity(3, 2))
//             .expect_err("Adding a prior with incorrect dimensions must fail.");
//         factor_graph
//             .add_prior(
//                 not_registered_x42,
//                 DVector::zeros(2),
//                 DMatrix::identity(2, 2),
//             )
//             .expect_err("Adding a prior on a non-registered variable must fail.");
//
//         factor_graph
//             .add_observation(
//                 vec![x1, x2],
//                 DMatrix::from_element(2, 4, 1.0),
//                 DVector::zeros(2),
//                 DMatrix::identity(2, 2),
//             )
//             .expect("Adding a relation between registered variables must work.");
//         factor_graph
//             .add_observation(
//                 vec![x2, x1],
//                 DMatrix::from_element(2, 4, 1.0),
//                 DVector::zeros(2),
//                 DMatrix::identity(2, 2),
//             )
//             .expect_err("Adding an observation with unsorted keys must fail.");
//         factor_graph
//             .add_observation(
//                 vec![x1, x2, not_registered_x42],
//                 DMatrix::from_element(3, 5, 1.0),
//                 DVector::zeros(3),
//                 DMatrix::identity(3, 3),
//             )
//             .expect_err("Adding an observation with unregistered variables must fail.");
//
//         assert_eq!(
//             factor_graph.n_factors(),
//             2,
//             "There should be 2 factors in total: 1 prior and 1 likelihood."
//         );
//     }
//
//     #[test]
//     fn test_likelihood_eliminate() {
//         let sizes = vec![2, 2, 3, 2, 1];
//         let size_of = |k| sizes[k];
//
//         let f0 = Factor {
//             layout: Layout::new(vec![2], size_of),
//             a: dmatrix![1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, 1.0],
//             xiby: dvector![0.6, 0.6, 0.6],
//             wby: dmatrix![1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, 1.0],
//         };
//
//         let f1 = Factor {
//             layout: Layout::new(vec![0, 1, 2], size_of),
//             a: dmatrix![
//                 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.1;
//                 0.0, 1.0, 0.0, 0.0, 0.0, 1.2, -1.0;
//                 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0
//             ],
//             wby: dmatrix![1.0, 0.2, 0.0; 0.2, 1.0, 0.0; 0.0, 0.0, 1.0],
//             xiby: dvector![0.4, 1.04, -0.5],
//         };
//
//         let f2 = Factor {
//             layout: Layout::new(vec![1, 3, 4], size_of),
//             a: dmatrix![
//                 0.0, 0.2, 0.0, 0.0, 0.2;
//                 0.8, 0.0, 0.0, 1.0, 0.8
//             ],
//             wby: dmatrix![0.8, 0.1; 0.1, 1.0],
//             xiby: dvector![0.05, 0.5],
//         };
//
//         let new_factor =
//             Factor::eliminate(vec![&f0, &f1, &f2], &[0, 1, 3], &[2, 4], size_of).unwrap();
//
//         assert_eq!(new_factor.keys(), [0, 1, 3]);
//         assert!(new_factor.dim_y() <= new_factor.dim_x());
//
//         println!("new_factor.xiby[0]: {}", new_factor.xiby[0]);
//         println!("new_factor.xiby[1]: {}", new_factor.xiby[1]);
//         println!("new_factor.xiby[2]: {}", new_factor.xiby[2]);
//         println!("new_factor.xiby[3]: {}", new_factor.xiby[3]);
//         println!("new_factor.xiby[4]: {}", new_factor.xiby[4]);
//
//         println!("new_factor.wby[0]: {}", new_factor.wby[0]);
//         println!("new_factor.wby[2]: {}", new_factor.wby[2]);
//         println!("new_factor.wby[7]: {}", new_factor.wby[7]);
//         println!("new_factor.wby[13]: {}", new_factor.wby[13]);
//         println!("new_factor.wby[19]: {}", new_factor.wby[19]);
//         println!("new_factor.wby[24]: {}", new_factor.wby[24]);
//
//         assert!((new_factor.xiby[0] - -0.18566446904015022f64).abs() < 1e-9);
//         assert!((new_factor.xiby[1] - 0.2564704789814862f64).abs() < 1e-9);
//         assert!((new_factor.xiby[2] - -0.5849271856215628f64).abs() < 1e-9);
//         assert!((new_factor.xiby[3] - -0.08977272727272727f64).abs() < 1e-9);
//         assert!((new_factor.xiby[4] - 0.02244318181818189f64).abs() < 1e-9);
//
//         assert!((new_factor.wby[0] - 0.49401045386071685f64).abs() < 1e-9);
//         assert!((new_factor.wby[2] - -0.005219289506583182).abs() < 1e-9);
//         assert!((new_factor.wby[7] - -0.0347294345005616).abs() < 1e-9);
//         assert!((new_factor.wby[13] - 0f64).abs() < 1e-9);
//         assert!((new_factor.wby[19] - -0.17954545454545454f64).abs() < 1e-9);
//         assert!((new_factor.wby[24] - 0.04488636363636378f64).abs() < 1e-9);
//     }
//
//     #[test]
//     fn test_likelihood_eliminate_loop() {
//         let sizes = vec![1, 1, 1];
//         let size_of = |k| sizes[k];
//
//         let f0 = Factor {
//             layout: Layout::new(vec![1], size_of),
//             a: dmatrix![1.0],
//             xiby: dvector![0.2],
//             wby: dmatrix![1.0],
//         };
//         let f1 = Factor {
//             layout: Layout::new(vec![0, 1], size_of),
//             a: dmatrix![1.0, 1.0],
//             wby: dmatrix![0.6],
//             xiby: dvector![0.06],
//         };
//         let f2 = Factor {
//             layout: Layout::new(vec![0, 2], size_of),
//             a: dmatrix![1.0, 1.0],
//             wby: dmatrix![1.0],
//             xiby: dvector![0.8],
//         };
//         let f3 = Factor {
//             layout: Layout::new(vec![1, 2], size_of),
//             a: dmatrix![1.0, 1.0],
//             wby: dmatrix![0.5],
//             xiby: dvector![0.5],
//         };
//         //
//         //         let factors = vec![
//         //             FactorView::Prior(&f0),
//         //             FactorView::Likelihood(&f1),
//         //             FactorView::Likelihood(&f2),
//         //             FactorView::Likelihood(&f3),
//         //         ];
//         //         let kept_keys = vec![0];
//         //         let cleared_keys = vec![1, 2];
//
//         // the total observation dimension is larger than the total dimension of the kept variables
//         // the factor should close a
//         let new_factor = Factor::eliminate([&f0, &f1, &f2, &f3], &[0], &[1, 2], size_of).unwrap();
//
//         assert!(new_factor.dim_y() <= new_factor.dim_x());
//         assert_eq!(new_factor.keys(), [0]);
//         assert!((new_factor.a[0] - 1f64).abs() < 1e-9);
//         assert!((new_factor.xiby[0] - -0.051724137931034475f64).abs() < 1e-9);
//         assert!((new_factor.wby[0] - 0.8965517241379309f64).abs() < 1e-9);
//     }
//
//     // further test:
//     // Branch equivalence: run the same inputs through eliminate_to_z and eliminate_to_x, then compare Aᵀξ and AᵀWA. Use blocks of different sizes (for example 2 and 3), since equal sizes would hide the transposition bug.
//     // Scalar sanity case: one likelihood y = x + u with W = w, and a prior on u with precision p. The expected result is Λ'_xx = wp/(w+p).
//     // Edge cases: no cleared keys, a prior on a kept key plus a likelihood on the same key, and a prior on a cleared key.
// }

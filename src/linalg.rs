// use nalgebra::{DMatrix, DVector};

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::prelude::*;
use faer::{Accum, Conj, Par, Spec};

use eros::Result;

// pub fn concat_with_views(vectors: &[DVector<f64>]) -> DVector<f64> {
//     let total_len: usize = vectors.iter().map(|v| v.len()).sum();
//     let mut result = DVector::zeros(total_len);

//     let mut offset = 0;
//     for v in vectors {
//         result.rows_range_mut(offset..offset + v.len()).copy_from(v);
//         offset += v.len();
//     }

//     result
// }

// pub fn concat_vectors<'a>(vectors: impl IntoIterator<Item = &'a DVector<f64>>) -> DVector<f64> {
//     let flat_vec: Vec<f64> = vectors
//         .into_iter()
//         .flat_map(|v| v.as_slice())
//         .copied()
//         .collect();

//     DVector::from_vec(flat_vec)
// }

// /// The operation x = a y where a is a matrix and x and y are vectors
// pub fn gemv<T, LhsT, RhsT, M, N, K>(x: impl AsMatMut<T = T, Rows = M, Cols = N>, a: MatRef<f64>, y: ColRef<f64>)
// where
//     T: ComplexField,
//     LhsT: Conjugate<Canonical = T>,
//     RhsT: Conjugate<Canonical = T>,
//     M: Shape,
//     N: Shape,
//     K: Shape,
// {

//     dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
//     beta: Accum,
//     lhs: impl AsMatRef<T = T, Rows = M, Cols = K>,
//     conj_lhs: Conj,
//     rhs: impl AsMatRef<T = T, Rows = K, Cols = N>,
//     conj_rhs: Conj,
//     alpha: T,
//     par: Par,

//     matmul(&mut x, Accum::Replace, &a, &y, T.one(), Par::Seq);
// }

// /// The operation x = a.t y where a is a matrix and x and y are vectors
// pub fn gemv_tr(x: ColMut<f64>, a: MatRef<f64>, y: ColRef<f64>) {
//     matmul_with_conj(&mut x, Accum::Replace, &a, Conj::Yes, &y, Conj::No, 1.0, Par::Seq);
// }

// /// The operation x = x + a y where a is a matrix and x and y are vectors
// pub fn cgemv(x: ColMut<f64>, a: MatRef<f64>, y: ColRef<f64>) {
//     matmul(&mut x, Accum::Add, &a, &y, 1.0, Par::Seq);
// }

// /// The operation x = x + a y where a is a matrix and x and y are vectors
// pub fn cgemv_tr(x: ColMut<f64>, a: MatRef<f64>, y: ColRef<f64>) {
//     matmul_with_conj(&mut x, Accum::Add, &a, Conj::Yes, &y, Conj::No, 1.0, Par::Seq);
// }

// pub fn gemm(&mut out, ) {
//     todo!();
// }

// pub fn gemm_tr(&mut out, ) {
//     todo!();
// }

// pub fn gemm_tr(&mut out, ) {
//     todo!();
// }

// pub fn cholesky(&mut out, ) {
//     todo!();
// }

// pub fn solve_inplace(&mut out, &r) {
//     todo!();
// }

// pub fn rsolve_inplace(&mut out, ) {
//     todo!();
// }
//
//
pub fn cholesky_inplace(a: MatMut<f64>, a_dim: usize, b_dim: usize) -> Result<()> {
    // compute the size and alignment of the required scratch space
    let cholesky_memory =
        llt::factor::cholesky_in_place_scratch::<f64>(a_dim, Par::Seq, Spec::default());
    let solve_memory = llt::solve::solve_in_place_scratch::<f64>(a_dim, b_dim, Par::Seq);

    // allocate the scratch space
    let mut memory = MemBuffer::new(cholesky_memory.or(solve_memory));
    let stack = MemStack::new(&mut memory);

    // compute the decomposition
    llt::factor::cholesky_in_place(
        a,
        // no regularization
        llt::factor::LltRegularization::default(),
        // no multithreading
        Par::Seq,
        // scratch space
        stack,
        // default settings
        default(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky_inplace() {
        let mut b = mat![[10.0, 2.0], [2.0, 10.0]];
        let _ = cholesky_inplace(b.rb_mut(), 2, 1).unwrap();

        let a = mat![[3.162277, 0.0], [0.6324555, 3.0983866]]; // ground truth

        println!("cholesky decomposition is {:?}", a);
        zip!(&a, &b).for_each(|unzip!(a, b)| assert!((a - b).abs() < 1e-6));
    }
}

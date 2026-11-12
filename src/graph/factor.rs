use std::fmt::Debug;

use nalgebra::{DMatrix, DMatrixView, DVector, RealField};

use super::layout::Layout;

#[derive(Debug)]
pub struct Factor<T: RealField> {
    layout: Layout,
    a: DMatrix<T>,
    xiby: DVector<T>,
    wby: DMatrix<T>,
}

impl<T: RealField> Factor<T> {
    pub fn new(layout: Layout, a: DMatrix<T>, xiby: DVector<T>, wby: DMatrix<T>) -> Self {
        Self {
            layout,
            a,
            xiby,
            wby,
        }
    }

    pub fn keys(&self) -> &[usize] {
        &self.layout.keys()
    }

    pub fn xiby(&self) -> &DVector<T> {
        &self.xiby
    }

    pub fn wby(&self) -> &DMatrix<T> {
        &self.wby
    }

    pub fn wbx(&self) -> DMatrix<T> {
        let mut wbx = DMatrix::zeros(self.dim_x(), self.dim_x());
        wbx.quadform(T::one(), &self.wby, &self.a, T::zero());
        wbx
    }

    pub fn xibx(&self) -> DVector<T> {
        let mut xibx = DVector::zeros(self.dim_x());
        xibx.gemv_tr(T::one(), &self.a, &self.xiby, T::zero());
        xibx
    }

    pub fn blocks(&self) -> impl Iterator<Item = (usize, DMatrixView<'_, T>)> {
        self.layout.keys().iter().enumerate().map(move |(i, &key)| {
            let (icol, ncols) = (
                self.layout.offsets[i],
                self.layout.offsets[i + 1] - self.layout.offsets[i],
            );
            (key, self.a.columns(icol, ncols))
        })
    }

    pub fn dim_x(&self) -> usize {
        self.a.ncols()
    }

    pub fn dim_y(&self) -> usize {
        self.a.nrows()
    }
}

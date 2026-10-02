/*!
A helper module for Gaussian variables.
*/

use faer::prelude::*;

use super::VarId;

#[derive(Debug, PartialEq, Clone)]
pub struct VarBlock<T> {
    pub id: VarId,
    pub a: Mat<T>,
}

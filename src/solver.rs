//! Solver (filter and smoother) for exact inference on Gaussian factor graphs.
//!
//! Implements Gaussian message-passing over factors graphs.
//!
//! For theoretical background, refer to the
//! [lecture notes](https://doi.org/10.3929/ethz-c-000785981)
//! of the class *Model-Based Estimation and Signal Analysis*
//! (Prof. H.-A. Loeliger, ETH Zürich, 2025).

use anyhow::{Context, Result, ensure};
use nalgebra::RealField;

use super::graph::FactorGraph;

pub(crate) mod eliminate;
pub mod estimate;
pub(crate) mod filter;
pub(crate) mod pass;
pub(crate) mod select;
pub(crate) mod smoother;

use self::estimate::{Marginal, Marginals};
use self::smoother::smooth;

impl<T: RealField> FactorGraph<T> {
    pub fn smooth(&self) -> Result<Marginals<T>> {
        smooth(&self)
    }

    pub fn filter(&self) -> Result<Marginal<T>> {
        todo!();
    }
}

//
// pub enum Target {
//     All,
//     LastPose,
//     LastNPose(usize),
//     AllPose,
//     LastLandmark,
//     LastNLandmark(usize),
//     AllLandmark,
// }
//
// pub enum Record {
//     Yes,
//     No,
// }
//
// pub struct EstimatorConfig {
//     pub window: usize,
//     pub record: Record,
// }
//
// pub struct RealTimeEstimator {}
//
// impl<T: RealField> RealTimeEstimator<T> {
//     pub fn new(config: EstimatorConfig) -> Self;
//     pub fn add_odometry(&mut self, ...) -> Result<()>; // odometry measurement between consecutive pose
//     pub fn add_observation(&mut self, ...) -> Result<()>; // observation of landmarks; after data-association so that we know if the landmark has already been observed or not.
//     pub fn map(&mut self, t: Target<'_>) -> Result<MapEstimates<T>>; // the map estimate of all targeted states
//     pub fn marginals(&mut self, t: Target<'_>) -> Result<Marginals<T>>; // the marginals of each targeted states
// }

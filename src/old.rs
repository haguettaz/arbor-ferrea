// use crate::lie::{Point2, Point3, Pose2, Pose3};
// use nalgebra::{DMatrix, DVector};

// // Lightweight key to identify variables
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub struct Key(pub u64);

// // Heterogeneous state values (Pose2, Pose3, Point2, Point3, etc.)
// #[derive(Debug, Clone, PartialEq)]
// pub enum Value {
//     Pose2(Pose2),
//     Pose3(Pose3),
//     Point2(Point2),
//     Point3(Point3),
// }

// // Map from variable keys to current linearization values
// use std::collections::HashMap;
// pub type Variables = HashMap<Key, Value>;

// pub trait Factor: Send + Sync {
//     /// Returns the fixed list of keys connected by this factor.
//     fn keys(&self) -> &[Key];

//     /// Computes the residual vector: r = h(x) - z
//     fn residual(&self, values: &Variables) -> DVector<f64>;

//     /// Linearizes the factor: returns the Jacobians w.r.t. each key and the RHS vector.
//     /// Jacobians order must match the order in `keys()`.
//     fn linearize(&self, values: &Variables) -> (Vec<DMatrix<f64>>, DVector<f64>);
// }

// /// A general n-ary factor holding an arbitrary, immutable number of keys
// pub struct NaryFactor<M> {
//     keys: Box<[Key]>, // Immutable once allocated, no extra capacity overhead
//     measurement: M,
//     noise_model: NoiseModel,
// }

// impl<M: Send + Sync> NaryFactor<M> {
//     pub fn new(keys: impl Into<Box<[Key]>>, measurement: M, noise_model: NoiseModel) -> Self {
//         Self {
//             keys: keys.into(),
//             measurement,
//             noise_model,
//         }
//     }
// }

// impl<M: Send + Sync> Factor for NaryFactor<M> {
//     fn keys(&self) -> &[Key] {
//         &self.keys
//     }

//     fn residual(&self, _variables: &Variables) -> DVector<f64> {
//         // Evaluate non-linear measurement function over all variables in self.keys
//         todo!()
//     }

//     fn linearize(&self, _variables: &Variables) -> (Vec<DMatrix<f64>>, DVector<f64>) {
//         // Return one Jacobian block per key in self.keys
//         todo!()
//     }
// }

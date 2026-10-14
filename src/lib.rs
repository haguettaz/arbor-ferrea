//! A solver for sparse least squares problems using tree-structured
//! Gaussian factor graphs.
//!
//! This crate provides safe, concurrent tree orchestration and fast linear algebra
//! to execute exact inference via sequential or parallel variable elimination. Typical
//! applications include:
//!
//! * **Robotics:** State estimation, trajectory optimization, and SLAM (Simultaneous Localization and Mapping).
//! * **Computer Vision:** Bundle adjustment and 3D camera reconstruction.
//! * **Geodesy & Engineering:** Geodetic network adjustment and structural analysis.

pub mod gaussian;
pub mod numeric;
pub mod symbolic;
pub mod ui;

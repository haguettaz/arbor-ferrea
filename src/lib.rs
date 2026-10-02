//! A high-performance Rust library for statistical estimation in robotics, built around
//! tree-structured Gaussian factor graphs.
//!
//! Engineered from the ground up for safety and speed, it leverages Rust’s type system
//! alongside high-performance linear algebra routines powered by [faer](https://faer.veganb.tw).
//! The architecture draws heavy inspiration from [gtsam](https://gtsam.org).

pub mod factor;
pub mod helper;
pub mod tree;
pub mod variable;

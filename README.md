# 🦀 arbor-ferrea

A high-performance Rust library for statistical estimation in robotics, built around tree-structured Gaussian factor graphs.

Engineered from the ground up for safety and speed, it leverages Rust’s type system alongside high-performance linear algebra routines powered by [faer](https://faer.veganb.tw). 
The architecture draws heavy inspiration from [gtsam](https://gtsam.org).

*Note: If you are looking for a more mature, general-purpose factor graph solver in the Rust ecosystem, check out [fact.rs](https://docs.rs/factrs/latest/factrs/index.html).*

---

## Crate Structure Overview

A modular approach will keep the crate clean and scalable for embedded targets.

`manifold` *(math foundation)*

*	Trait `Manifold`: Defines `dim()`, `retract(delta)` for state updates, and `local(other)` for differences.
*	Lie Groups: Implementations of `SO3`, `SE3`, `SO2`, `SE2`, and standard `Rn` vectors.

`autodiff` *(differentiation engine)*

*	Implementation of dual numbers (e.g., `a + b eps`$ where `\epsilon^2 = 0`) for forward-mode automatic differentiation.
*	Integration with Lie groups so that a function mapping `SE3` to `Rn` can automatically yield analytical Jacobians with respect to the tangent space.

`core` *(graph elements)*
*	Key: A unique identifier for variables (e.g., using a u64 or a typed struct like Symbol('x', 1)).
*	Values: A heterogeneous map (or typemap) storing the current estimates of all variables, indexed by Key.
*	FactorGraph: A collection of Factor traits.

`factors` *(observation models)*
*	Trait `Factor`: Defines `keys() -> Vec`, `error(values) -> Vector`, and `linearize(values) -> (Matrix, Vector)` (returns Jacobians and residuals).
*	Implementations: `PriorFactor`, `BetweenFactor` (odometry), and `ProjectionFactor` (camera/landmarks).

`solver` *(linearization & optimization)*

* Symbolic Tree Construction: A sequential initial step that analyzes the graph structure to build a symbolic elimination tree.
* Parallel Numeric Assembly: Relying on the symbolic tree, this step linearizes the observation models in parallel. Instead of assembling a monolithic sparse matrix, it populates a tree-structured Gaussian factor graph, utilizing `faer` for efficient local dense matrix operations within the tree nodes.
* Tree-Based Resolution: Solves for the tangent-space step (`\Delta x`) by performing backward substitution from the root of the tree down to the leaves.

## 🗺️ Development Roadmap

### Phase 1: Lie Groups & Manifolds

- [ ] Define a `Manifold` trait with `retract(self, delta: Vector) -> Self` and `local(self, other: Self) -> Vector`.
- [ ] Implement vector spaces (`R2`, `R3`).
- [ ] Implement `SO3` (quaternions or rotation matrices) and `SE3` (rigid transformations) using the exponential and logarithmic maps for retraction and local coordinates.
- [ ] Test: Verify that `x.retract(x.local(y)) == y`.

### Phase 2: Autodiff Engine

Writing manual Jacobians for every factor is error-prone. Forward-mode autodiff via dual numbers is highly efficient for the small dimensionalities typical in robotics (e.g., 6D for SE(3)).

- [ ] Implement a `Dual` number struct.
- [ ] Extend basic math operations `(std::ops)` to support Dual types.
- [ ] Write a wrapper that evaluates a residual function `f(x)` by injecting dual numbers to extract the Jacobian matrix evaluated at the tangent space.

### Phase 3: Core Types & Values

- [ ]	Implement a `Key` system. A simple `u64` works, but a macro generating GTSAM-style keys (e.g., X(1) for poses, L(1) for landmarks) is highly ergonomic.
- [ ]	Build the `Values` struct. Since Rust is strongly typed, use a `TypeMap` or a trait-object-based dictionary (e.g., `HashMap`) to store mixed types like `SE3` and `R3` safely.
- [x]	Define the `Factor` trait and create a basic `Tree`.

### Phase 4: Factors & Linearization

- [x]	Implement a `PinFactor` (pin the first pose).
- [ ] Implement a `BetweenFactor` (relative pose measurements).
- [ ]	Implement the linearize method for these factors, utilizing your `Autodiff` engine to automatically compute the Jacobians and the residual error.

### Phase 5: Sparse Linear Algebra & Solvers

- [x] Implement a sequential algorithm that analyzes the factor graph topology (variables and connectivity) to build a symbolic elimination tree.
- [x] Relying on the symbolic tree, linearize the observation models in parallel. Use `faer` for efficient local dense matrix operations during variable elimination to populate the tree-structured Gaussian factor graph.
- [x] Solve for the tangent-space step `dx` by propagating the solution from the root of the elimination tree down to the leaves.
- [ ] Implement the non-linear optimization loop: evaluate error -> construct symbolic/numeric trees -> solve `dx` via backward substitution -> update `Values` using the `retract` operator -> repeat until convergence.

### Phase 6: Embedded Polish & Ergonomics (v1.0)

- [ ]	Refine the API. Aim for a builder pattern.
- [ ] Audit for `#![no_std]` compliance if running on bare-metal ARM Cortex-M or RISC-V, ensuring you use `alloc` instead of `std` for vectors/maps.

### Future Releases

- [ ] Incremental updates
- [ ] Fluid relinearization (iSAM2)
- [ ] IMU pre-integration
- [ ] NUP integration

## 📚 References

1. **Factor Graphs**   
   H.-A. Loeliger, J. Dauwels, J. Hu, S. Korl, Li Ping, and F. Kschischang, "The factor graph approach to model-based signal processing," *Proceedings of the IEEE*, vol. 95, no. 6, pp. 1295-1322, June 2007.
2. **Square Root SAM**   
   F. Dellaert and M. Kaess, "Square root SAM: simultaneous localization and mapping via square root information smoothing," *The International Journal of Robotics Research (IJRR)*, 2006.
3. **iSAM**   
   M. Kaess, A. Ranganathan, and F. Dellaert, "iSAM: incremental smoothing and mapping," *IEEE Transactions on Robotics (T-RO)*, 2008.
4. **iSAM2**  
   M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert, "iSAM2: incremental smoothing and mapping using the Bayes tree," *The International Journal of Robotics Research (IJRR)*, vol. 31, no. 2, pp. 216–235, 2012.
5. **Lie Theory**   
   J. Solà, J. Deray, D. Atchuthan, “A micro Lie theory for state estimation in robotics,” arxiv:1812.01537, Dec. 2021.

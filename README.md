# 🦀 sam-rs

A high-performance Rust library for non-linear factor graph optimization, state estimation, and Simultaneous Localization and Mapping (SLAM).

Heavily inspired by [GTSAM](https://gtsam.org), `sam-rs` is designed from the ground up to leverage Rust’s type safety, memory guarantees, and high-performance linear algebra routines powered by the [faer](https://faer.veganb.tw) crate.

---

## 🗺️ Roadmap & Features

### Bayes Tree & Solver Core
- [ ] Multifrontal variable elimination via QR and Cholesky factorizations
- [ ] Clique amalgamation and chordal Bayes tree assembly
- [ ] Incremental updates with fluid relinearization (iSAM2)
- [ ] Variable reordering heuristics (COLAMD / Constrained COLAMD)
- [ ] Parallel solver execution (on parallel branches)

### Geometry & Lie Groups
- [ ] $SO(2)$ / $SE(2)$ manifold representations (2D SLAM)
- [ ] $SO(3)$ / $SE(3)$ manifold representations with retract/local coordinates (3D SLAM)
- [ ] Point representations ($\mathbb{R}^2$, $\mathbb{R}^3$)

### Robust Estimation & Factors
- [ ] Common SLAM constraints
- [ ] Non-Gaussian / NUP (Normal with Unknown Parameters) integration

---

## 📚 Theoretical References

1. **Square Root SAM**  
   F. Dellaert and M. Kaess, *"Square Root SAM: Simultaneous Localization and Mapping via Square Root Information Smoothing,"* *The International Journal of Robotics Research (IJRR)*, 2006.
2. **iSAM**  
   M. Kaess, A. Ranganathan, and F. Dellaert, *"iSAM: Incremental Smoothing and Mapping,"* *IEEE Transactions on Robotics (T-RO)*, 2008.
3. **iSAM2**  
   M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert, *"iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree,"* *The International Journal of Robotics Research (IJRR)*, 31(2), pp. 216–235, 2012.

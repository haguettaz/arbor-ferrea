# 🦀 sam-rs

A high-performance Rust library for non-linear factor graph optimization, with a focus on state estimation for robotics.

Heavily inspired by [GTSAM](https://gtsam.org), `sam-rs` is designed from the ground up to leverage Rust’s type safety, memory guarantees, and high-performance linear algebra routines powered by the [faer](https://faer.veganb.tw) crate.

---

## 🗺️ Roadmap & Features

### Gaussian Tree & Solver Core
- [ ] Upward-filtering (Gaussian tree assembly) downward-deciding (solve)
- [ ] Incremental updates
- [ ] Fluid relinearization (iSAM2)
- [ ] Variable reordering heuristics (COLAMD / Constrained COLAMD)
- [ ] Parallel solver execution (on parallel branches)

### Geometry & Lie Groups
- [ ] $SO(2)$ / $SE(2)$ manifold representations (2D SLAM)
- [ ] $SO(3)$ / $SE(3)$ manifold representations with retract/local coordinates (3D SLAM)
- [ ] Point representations ($\mathbb{R}^2$, $\mathbb{R}^3$)

### Robust Estimation & Factors
- [ ] Common SLAM constraints
- [ ] NUP (Normal with Unknown Parameters) integration

---

## 📚 References

1. **Factor Graphs**
   H.-A. Loeliger, J. Dauwels, J. Hu, S. Korl, Li Ping, and F. Kschischang, "The factor graph approach to model-based signal processing," *Proceedings of the IEEE*, vol. 95, no. 6, pp. 1295-1322, June 2007.
2. **Square Root SAM**  
   F. Dellaert and M. Kaess, "Square Root SAM: Simultaneous localization and mapping via square root information smoothing," *The International Journal of Robotics Research (IJRR)*, 2006.
3. **iSAM**  
   M. Kaess, A. Ranganathan, and F. Dellaert, "iSAM: Incremental smoothing and mapping," *IEEE Transactions on Robotics (T-RO)*, 2008.
4. **iSAM2**  
   M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert, "iSAM2: Incremental smoothing and mapping using the Bayes tree," *The International Journal of Robotics Research (IJRR)*, vol. 31, no. 2, pp. 216–235, 2012.

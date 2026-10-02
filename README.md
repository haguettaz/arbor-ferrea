# 🦀 arbor-ferrea

A high-performance Rust library for statistical estimation in robotics, built around tree-structured Gaussian factor graphs.

Engineered from the ground up for safety and speed, it leverages Rust’s type system alongside high-performance linear algebra routines powered by [faer](https://faer.veganb.tw). 
The architecture draws heavy inspiration from [gtsam](https://gtsam.org).

*Note: If you are looking for a more mature, general-purpose factor graph solver in the Rust ecosystem, check out [fact.rs](https://docs.rs/factrs/latest/factrs/index.html).*

---

## 🗺️ Roadmap & Features

### Gaussian Tree & Solver Core
- [x] Gaussian tree assembly (variable elimination = max-product Gaussian message-passing)
- [x] Solving (max-product deciding)
- [x] Parallel build and solve
- [ ] Autodifferentiation
- [ ] Incremental updates
- [ ] Fluid relinearization (iSAM2)
- [ ] Variable elimination heuristics (COLAMD / Constrained COLAMD)

### Geometry & Lie Groups
- [ ] $SO(2)$ / $SE(2)$ manifold representations (2D SLAM)
- [ ] $SO(3)$ / $SE(3)$ manifold representations with retract/local coordinates (3D SLAM)
- [ ] Point representations ($\mathbb{R}^2$, $\mathbb{R}^3$)

### Robust Estimation & Factors
- [x] Handle fixed variables via anchoring factors
- [ ] Implement common SLAM constraints
- [ ] NUP integration

---

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

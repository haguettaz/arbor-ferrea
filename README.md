# 🦀 arbor-ferrea

A Rust solver for sparse linear least-squares problems, built around tree-structured Gaussian factor graphs.

The crate implements elimination on junction trees, using [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/index.html) for dense frontal factorizations with support for both sequential and parallel ([`rayon`](https://docs.rs/rayon/latest/rayon/)) execution. It is intended primarily for research and experimental online estimation workflows.

***Note:** This project is under active development and the API is evolving. For a mature, general-purpose factor-graph optimization library in Rust, check out [`fact.rs`](https://docs.rs/factrs/latest/factrs/index.html).*

## 🗺️ Roadmap & Features

### Symbolic and Numeric Trees
- [x] Sequential build and solve
- [x] Parallel build and solve
- [ ] Variable elimination heuristics (COLAMD / Constrained COLAMD)
- [ ] Efficient updates (same tree structure but new data)
- [ ] Incremental updates (extend tree structure)
- [ ] Tree cut

### Gaussian Tree
- [x] Sequential build and solve
- [x] Parallel build and solve
- [ ] Variable elimination heuristics (COLAMD / Constrained COLAMD)
- [ ] Efficient updates (same tree structure but new data)
- [ ] Incremental updates (extend tree structure)
- [ ] Tree cut
 
### Robust Estimation & Factors
- [ ] NUP integration, e.g., Huber kernel

### Helper Functions
- [x] Variable dictionary
- [ ] Factor dictionary

## 📚 References

- H.-A. Loeliger, J. Dauwels, J. Hu, S. Korl, Li Ping, and F. Kschischang, 
**The factor graph approach to model-based signal processing,** 
*Proceedings of the IEEE, vol. 95, no. 6, pp. 1295-1322, June 2007.*
- M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert,
**iSAM2: incremental smoothing and mapping using the Bayes tree,** 
*The International Journal of Robotics Research (IJRR), vol. 31, no. 2, pp. 216–235, 2012.*

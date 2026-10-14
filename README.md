# 🦀 arbor-ferrea

A high-performance Rust solver for sparse least-square, built around tree-structured Gaussian factor graphs.

Engineered from the ground up for safety and speed, it leverages Rust’s type system alongside high-performance linear algebra routines. 

*Note: If you are looking for a more mature, general-purpose factor graph solver in the Rust ecosystem, check out [fact.rs](https://docs.rs/factrs/latest/factrs/index.html).*

---

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

---

## 📚 References

- H.-A. Loeliger, J. Dauwels, J. Hu, S. Korl, Li Ping, and F. Kschischang, 
**The factor graph approach to model-based signal processing,** 
*Proceedings of the IEEE, vol. 95, no. 6, pp. 1295-1322, June 2007.*
- M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert,
**iSAM2: incremental smoothing and mapping using the Bayes tree,** 
*The International Journal of Robotics Research (IJRR), vol. 31, no. 2, pp. 216–235, 2012.*

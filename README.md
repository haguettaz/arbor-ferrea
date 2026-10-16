# 🦀 arbor-ferrea

A Rust solver for sparse linear least-squares problems, built around tree-structured Gaussian factor graphs.

The crate resolves cyclic graphs by sequentially eliminating variables (max-product message passing) to form a rooted computation tree. 
This implements the upward-filtering / downward-deciding framework detailed in the *Model-Based Estimation and Signal Analysis* [lecture notes](https://doi.org/10.3929/ethz-c-000785981) (Prof. Loeliger, ETH Zürich).

Dense matrix operations rely on [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/index.html), with support for sequential and parallel ([`rayon`](https://docs.rs/rayon/latest/rayon/)) tree passes.

> This project is under active research and development; the API is subject to change. If you need a mature, general-purpose factor-graph optimization library in Rust today, check out [`factrs`](https://docs.rs/factrs/latest/factrs/index.html).

## ⚡ Quickstart

Add the crate to your `Cargo.toml`:

```bash
cargo add arbor-ferrea
```

```rust
use anyhow::Result;
use arbor_ferrea::ui::*;

fn main() -> Result<()> {
    // initialize an empty factor graph
    let mut factor_graph = GaussianFactorGraph::new();
    
    // register variables: (symbol, dimension)
    let x1 = factor_graph.add_variable(VarSymbol('x', 1), 1)?;
    let x2 = factor_graph.add_variable(VarSymbol('x', 2), 1)?;
    
    // prior factor on x1: x1 ~ N(0.0, 1.0)
    factor_graph.add_factor(
        vec![x1],
        vec![dmatrix![1.0]],       // Measurement matrix A
        dvector![0.0],             // Mean / measurement
        dmatrix![1.0],             // Information / weight matrix W
    )?;
    
    // odometry factor between x1 and x2: x2 - x1 ~ N(2.0, 1.0)
    factor_graph.add_factor(
        vec![x1, x2],
        vec![dmatrix![-1.0], dmatrix![1.0]], // [-I, I]
        dvector![2.0],
        dmatrix![1.0],
    )?;
    
    // build symbolic computation tree along elimination order
    let order = [x1, x2];
    let symb_tree = factor_graph.build_symbolic_tree(&order)?;
    
    // sequential solve
    let solution = factor_graph.solve_seq(&symb_tree)?;
    // or in parallel: factor_graph.solve_par(&symb_tree)?;
    
    println!("x1: {:?}", factor_graph.value(x1).unwrap());
    println!("x2: {:?}", factor_graph.value(x2).unwrap());
    
    Ok(())
}
```


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

### User Interface
- [x] Safe Gaussian factor graph construction

## 📚 References

- H.-A. Loeliger, J. Dauwels, J. Hu, S. Korl, Li Ping, and F. Kschischang, 
**The factor graph approach to model-based signal processing,** 
*Proceedings of the IEEE, vol. 95, no. 6, pp. 1295-1322, June 2007.*
- M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert,
**iSAM2: incremental smoothing and mapping using the Bayes tree,** 
*The International Journal of Robotics Research (IJRR), vol. 31, no. 2, pp. 216–235, 2012.*

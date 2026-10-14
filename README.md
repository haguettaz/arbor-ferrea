# 🦀 arbor-ferrea

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
use arbor_ferrea::prelude::*;

fn main() -> Result<()> {
    // Build the dictionary and register domain variables
    let mut var_dict = VarDict::new();
    let x1 = var_dict.register(VarSymbol('x', 1), 3);
    let x2 = var_dict.register(VarSymbol('x', 2), 3);
    let x3 = var_dict.register(VarSymbol('x', 3), 3);
    let l1 = var_dict.register(VarSymbol('l', 1), 2);
    let l2 = var_dict.register(VarSymbol('l', 2), 2);

    // Setup the global state buffer
    let mut buffer = vec![f64::NAN; dict.total_size()];

    // Setup the factors between the variables
    let factors: Vec<GaussianFactor> = ...;
    
    // Build the symbolic tree
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let symb_tree = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order)?;

    // Sequential build pass (for parallel execution, use _par instead)
    let gauss_tree = GaussianTree::from_symbolic_seq(&symb_tree, &factors, &var_dict)?;

    // Sequential solve pass (for parallel execution, use _par instead)
    gauss_tree.solve_seq(&mut buffer, &dict)?;

    Ok(())
}


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

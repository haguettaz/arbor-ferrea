# 🦀 arbor-ferrea

A Rust solver for sparse linear least-squares problems, built around Gaussian factor graphs.

Dense matrix operations rely on [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/index.html).

> This project is under active research and development; the API is subject to change. If you need a mature, general-purpose factor-graph optimization library in Rust today, check out [`factrs`](https://docs.rs/factrs/latest/factrs/index.html).

## ⚡ Quickstart

Add the crate to your `Cargo.toml`:

```bash
cargo add arbor-ferrea
```

```rust
use anyhow::Result;

use arbor_ferrea::gaussian::*;
use arbor_ferrea::solver::*;

const X1: Symbol = Symbol('x', 1);
const X2: Symbol = Symbol('x', 2);
const X3: Symbol = Symbol('x', 3);
const L1: Symbol = Symbol('l', 1);
const L2: Symbol = Symbol('l', 2);

fn main() -> Result<()> {

    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let l1 = factor_graph.add_variable(L1, 2).unwrap();
    let l2 = factor_graph.add_variable(L2, 2).unwrap();

    factor_graph
        .add_prior(
            x1,
            dvector![0.6, 0.6, 0.6],
            dmatrix![
                1.0, 0.0, 0.0;
                0.0, 1.0, 0.0;
                0.0, 0.0, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_observation(
            vec![x1, x2],
            dmatrix![
                1.0, 0.0, 0.0, -1.0, 0.0, 0.0;
                0.0, 1.0, 0.0, 0.0, -1.0, 0.0;
                0.0, 0.0, 1.0, 0.0, 0.0, -1.0
            ],
            dvector![0.2, 1.0, -0.5],
            dmatrix![
                1.0, 0.2, 0.0;
                0.2, 1.0, 0.0;
                0.0, 0.0, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_observation(
            vec![x2, x3],
            dmatrix![
                1.0, 0.0, 0.0, -1.0, 0.0, 0.0;
                0.0, 1.0, 0.0, 0.0, -1.0, 0.0;
                0.0, 0.0, 1.0, 0.0, 0.0, -1.0
            ],
            dvector![-0.1, 0.8, 0.1],
            dmatrix![
                1.0, 0.0, 0.3;
                0.0, 1.0, 0.1;
                0.3, 0.1, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_observation(
            vec![x1, l1],
            dmatrix![
                0.5, 0.5, 0.0, -1.0, 0.0;
                0.0, 0.7, 0.2, 0.0, -1.0
            ],
            dvector![0.0, 0.5],
            dmatrix![
                1.0, 0.2;
                0.2, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_observation(
            vec![x2, l1],
            dmatrix![
                    0.6, 0.4, 0.0, -0.5, -0.5;
                    0.1, 0.8, 0., 0.0, -0.8
            ],
            dvector![0.1, 0.2],
            dmatrix![
                0.8, 0.1;
                0.1, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_observation(
            vec![x3, l2],
            dmatrix![
                    0.6, 0.4, 0.0, -0.5, -0.5;
                    0.1, 0.8, 0.0, 0.0, -0.8
            ],
            dvector![0.8, 0.0],
            dmatrix![
                2.0, 0.2;
                0.2, 2.0
            ],
        )
        .unwrap();

    let smoother = Smoother::new(factor_graph);
    let marginals = smoother.run()?;
}
```

## 📚 References

- H.-A. Loeliger, J. Dauwels, J. Hu, S. Korl, Li Ping, and F. Kschischang, 
**The factor graph approach to model-based signal processing,** 
*Proceedings of the IEEE, vol. 95, no. 6, pp. 1295-1322, June 2007.*
- M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert,
**iSAM2: incremental smoothing and mapping using the Bayes tree,** 
*The International Journal of Robotics Research (IJRR), vol. 31, no. 2, pp. 216–235, 2012.*

# 🦀 arbor-ferrea

<!--[![crates.io](https://img.shields.io/crates/v/arbor-ferrea.svg)](https://crates.io/crates/arbor-ferrea)
[![docs.rs](https://docs.rs/arbor-ferrea/badge.svg)](https://docs.rs/arbor-ferrea)-->

A Rust solver for sparse **linear least-squares** problems and linear-Gaussian
state estimation, based on Gaussian message passing over Gaussian factor graphs.

> **Status:** this project is under active research and development, and the
> API is subject to change. If you need a mature, general-purpose
> factor-graph optimization library in Rust today, see
> [`factrs`](https://docs.rs/factrs/latest/factrs/index.html).

Describe your problem as a collection of Gaussian factors connecting
variables. The solver runs message passing on the resulting factor graph and
returns the estimates. The solution is **exact**, including on graphs with
loops; the method is detailed in the [report](report/report.pdf).

Only linear-Gaussian problems are supported. Nonlinear problems are out of
scope (linearize them first).

## ⚡ Quickstart

A prior on `x1` and one observation linking `x1` and `x2`:

```rust
use anyhow::Result;
use nalgebra::{dmatrix, dvector};

use arbor_ferrea::gaussian::*;
use arbor_ferrea::solver::*;

const X1: Symbol = Symbol('x', 1);
const X2: Symbol = Symbol('x', 2);

fn main() -> Result<()> {
    let mut factor_graph = FactorGraph::new();

    // Variables: a symbol and a dimension.
    let x1 = factor_graph.add_variable(X1, 2)?;
    let x2 = factor_graph.add_variable(X2, 2)?;

    // Prior: x1 ~ N(mean, covariance)
    factor_graph.add_prior(
        x1,
        dvector![0.0, 0.0],
        dmatrix![
            1.0, 0.0;
            0.0, 1.0
        ],
    )?;

    // Observation: z = A [x1; x2] + noise, noise ~ N(0, covariance)
    // Arguments: variables, A, z, covariance.
    factor_graph.add_observation(
        vec![x1, x2],
        dmatrix![
            1.0, 0.0, -1.0, 0.0;
            0.0, 1.0, 0.0, -1.0
        ],
        dvector![0.2, 1.0],
        dmatrix![
            1.0, 0.2;
            0.2, 1.0
        ],
    )?;

    // Estimate all variables.
    let smoother = Smoother::new(factor_graph);
    let marginals = smoother.run()?;
    println!("{marginals:?}");

    Ok(())
}
```

A larger example, with five variables and a loop in the graph, is in
[`examples/`](examples/).

## 🧭 Choosing what to compute

The solver is organized along two independent axes.

#### What to estimate

| Mode         | Estimates                      | Status      |
|--------------|--------------------------------|-------------|
| **Smoother** | All states                     | Available   |
| **Filter**   | Only the last state            | Coming soon |

#### With or without uncertainties

| Algorithm       | Output                                                      | Status      |
|-----------------|-------------------------------------------------------------|-------------|
| **Sum-product** | Marginals -- estimates **with** uncertainties                 | Available   |
| **Max-product** | MAP -- estimates **without** uncertainties                    | Coming soon |

## 🗺️ Roadmap

- Filter mode (last variable only).
- Estimation without uncertainties (max-product).

## 🤝 Contributing

The API is still moving, so please open an issue to discuss a change before sending a pull request.

## 📄 License

See [LICENSE](LICENSE).

## 📚 References

H.-A. Loeliger, J. Dauwels, J. Hu, S. Korl, Li Ping, and F. Kschischang,
  **The factor graph approach to model-based signal processing,**
  *Proceedings of the IEEE, vol. 95, no. 6, pp. 1295-1322, June 2007.*
  [doi:10.1109/JPROC.2007.896497](https://doi.org/10.1109/JPROC.2007.896497)
<!--- M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, and F. Dellaert,
  **iSAM2: incremental smoothing and mapping using the Bayes tree,**
  *The International Journal of Robotics Research (IJRR), vol. 31, no. 2, pp. 216–235, 2012.*
  [doi:10.1177/0278364911430419](https://doi.org/10.1177/0278364911430419)-->

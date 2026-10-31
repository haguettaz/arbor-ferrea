//! Five variables (three poses, two landmarks) and six factors.
//! The graph contains loops (x1, x2, l1), and the solution is still exact.

use anyhow::Result;
use nalgebra::{dmatrix, dvector};

use arbor_ferrea::gaussian::*;
use arbor_ferrea::solver::*;

const X1: Symbol = Symbol('x', 1);
const X2: Symbol = Symbol('x', 2);
const X3: Symbol = Symbol('x', 3);
const L1: Symbol = Symbol('l', 1);
const L2: Symbol = Symbol('l', 2);

fn main() -> Result<()> {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3)?;
    let x2 = factor_graph.add_variable(X2, 3)?;
    let x3 = factor_graph.add_variable(X3, 3)?;
    let l1 = factor_graph.add_variable(L1, 2)?;
    let l2 = factor_graph.add_variable(L2, 2)?;

    // Prior on x1.
    factor_graph.add_prior(
        x1,
        dvector![0.6, 0.6, 0.6],
        dmatrix![
            1.0, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ],
    )?;

    // Odometry-like observations between consecutive poses.
    factor_graph.add_observation(
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
    )?;
    factor_graph.add_observation(
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
    )?;

    // Pose-landmark observations.
    factor_graph.add_observation(
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
    )?;
    factor_graph.add_observation(
        vec![x2, l1],
        dmatrix![
            0.6, 0.4, 0.0, -0.5, -0.5;
            0.1, 0.8, 0.0, 0.0, -0.8
        ],
        dvector![0.1, 0.2],
        dmatrix![
            0.8, 0.1;
            0.1, 1.0
        ],
    )?;
    factor_graph.add_observation(
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
    )?;

    let smoother = Smoother::new(factor_graph);
    let marginals = smoother.run()?;
    println!("{marginals:?}");

    Ok(())
}

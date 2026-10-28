// //! Tests for the Gaussian smoother.
// //! The solutions can also be computed by solving
// //! (At Wy A) mx != At xiy for x,
// //! where A, Wy, and xiy represents the whole system.
//
use nalgebra::{dmatrix, dvector};
// use std::iter::zip;
//
// // use arbor_ferrea::numeric::NumericTree;
// // use arbor_ferrea::symbolic::{SymbolicFactor, SymbolicTree};
// use arbor_ferrea::ui::*;
//
use arbor_ferrea::gaussian::*;
use arbor_ferrea::solver::Smoother;

const X1: Symbol = Symbol('x', 1);
const X2: Symbol = Symbol('x', 2);
const X3: Symbol = Symbol('x', 3);
const X4: Symbol = Symbol('x', 4);
const X5: Symbol = Symbol('x', 5);
const L1: Symbol = Symbol('l', 1);
const L2: Symbol = Symbol('l', 2);

#[test]
/// Context: empty factor graphs
fn run_smoother_empty() {
    let factor_graph: FactorGraph<f64> = FactorGraph::new();
    let smoother = Smoother::new(factor_graph);
    let marginals = smoother.run().expect("smoothing failed");
    assert!(marginals.is_empty());
}

///
#[test]
/// Context: incomplete system with no global anchor
fn run_smoother_line_singular() {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let x4 = factor_graph.add_variable(X4, 3).unwrap();
    let x5 = factor_graph.add_variable(X5, 3).unwrap();

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
            vec![x3, x4],
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
            vec![x4, x5],
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

    let smoother = Smoother::new(factor_graph);
    assert!(smoother.run().is_err());
}
//
#[test]
/// Context: valid system with fixed x1 = (0,0,0)
fn run_smoother_line_anchored() {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let x4 = factor_graph.add_variable(X4, 3).unwrap();
    let x5 = factor_graph.add_variable(X5, 3).unwrap();

    factor_graph
        .add_prior(
            x1,
            dvector![0.0, 0.0, 0.0],
            dmatrix![
                1e15, 0.0, 0.0;
                0.0, 1e15, 0.0;
                0.0, 0.0, 1e15
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
            vec![x3, x4],
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
            vec![x4, x5],
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

    let smoother = Smoother::new(factor_graph);
    assert!(smoother.run().is_ok());
}

#[test]
fn run_smoother_line() {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let x4 = factor_graph.add_variable(X4, 3).unwrap();
    let x5 = factor_graph.add_variable(X5, 3).unwrap();

    factor_graph
        .add_prior(
            x1,
            dvector![0.0, 0.0, 0.0],
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
            vec![x3, x4],
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
            vec![x4, x5],
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

    let smoother = Smoother::new(factor_graph);
    assert!(smoother.run().is_ok());
}

///
#[test]
/// Context: incomplete system with no global anchor
fn run_smoother_loopy_singular() {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let l1 = factor_graph.add_variable(L1, 2).unwrap();
    let l2 = factor_graph.add_variable(L2, 2).unwrap();

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
                0.0, 0.75, 0.25, 0.0, -1.0
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
                    0.1, 0.8, 0., 0.0, -0.8
            ],
            dvector![0.8, 0.0],
            dmatrix![
                2.0, 0.2;
                0.2, 2.0
            ],
        )
        .unwrap();

    let smoother = Smoother::new(factor_graph);
    assert!(smoother.run().is_err());
}
//
#[test]
/// Context: valid system with fixed x1 = (0,0,0)
fn run_smoother_loopy_anchored() {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let l1 = factor_graph.add_variable(L1, 2).unwrap();
    let l2 = factor_graph.add_variable(L2, 2).unwrap();

    factor_graph
        .add_prior(
            x1,
            dvector![0.0, 0.0, 0.0],
            dmatrix![
                1e15, 0.0, 0.0;
                0.0, 1e15, 0.0;
                0.0, 0.0, 1e15
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
                0.0, 0.75, 0.25, 0.0, -1.0
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
                    0.1, 0.8, 0., 0.0, -0.8
            ],
            dvector![0.8, 0.0],
            dmatrix![
                2.0, 0.2;
                0.2, 2.0
            ],
        )
        .unwrap();

    let smoother = Smoother::new(factor_graph);
    assert!(smoother.run().is_ok());
}

#[test]
fn run_smoother_loopy() {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let l1 = factor_graph.add_variable(L1, 2).unwrap();
    let l2 = factor_graph.add_variable(L2, 2).unwrap();

    factor_graph
        .add_prior(
            x1,
            dvector![0.0, 0.0, 0.0],
            dmatrix![
                1e15, 0.0, 0.0;
                0.0, 1e15, 0.0;
                0.0, 0.0, 1e15
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
                0.0, 0.75, 0.25, 0.0, -1.0
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
                    0.1, 0.8, 0., 0.0, -0.8
            ],
            dvector![0.8, 0.0],
            dmatrix![
                2.0, 0.2;
                0.2, 2.0
            ],
        )
        .unwrap();

    let smoother = Smoother::new(factor_graph);
    assert!(smoother.run().is_ok());
}

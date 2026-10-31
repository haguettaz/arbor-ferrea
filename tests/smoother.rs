// //! Tests for the Gaussian smoother.
// //! The solutions can also be computed by solving
// //! (At Wy A) mx = At Wy my for mx,
// //! where A, Wy, and xiy represents the whole system.
//
use nalgebra::{RealField, dmatrix, dvector};
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

fn print_result<T: RealField>(marginals: &Vec<Marginal<T>>) {
    for p in marginals {
        println!(
            "key {} has mean {:#} and covariance {:#}",
            p.key, p.mean, p.cov
        );
    }
}

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
            dvector![0.6, 0.6, 0.6],
            dmatrix![
                1e-15, 0.0, 0.0;
                0.0, 1e-15, 0.0;
                0.0, 0.0, 1e-15
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
                1.0, 0.0, 0.0;
                0.0, 2.0, 0.2;
                0.0, 0.2, 2.0
            ],
        )
        .unwrap();

    let smoother = Smoother::new(factor_graph);
    let marginals = smoother.run().unwrap();

    assert!((marginals[x1].mean[0] - 0.6f64).abs() < 1e-9);
    assert!((marginals[x1].mean[1] - 0.6f64).abs() < 1e-9);
    assert!((marginals[x1].mean[2] - 0.6f64).abs() < 1e-9);
    assert!((marginals[x2].mean[0] - 0.4f64).abs() < 1e-9);
    assert!((marginals[x2].mean[1] - -0.4f64).abs() < 1e-9);
    assert!((marginals[x2].mean[2] - 1.1f64).abs() < 1e-9);
    assert!((marginals[x3].mean[0] - 0.5f64).abs() < 1e-9);
    assert!((marginals[x3].mean[1] - -1.2f64).abs() < 1e-9);
    assert!((marginals[x3].mean[2] - 1f64).abs() < 1e-9);
    assert!((marginals[x4].mean[0] - 0.3f64).abs() < 1e-9);
    assert!((marginals[x4].mean[1] - -2.2f64).abs() < 1e-9);
    assert!((marginals[x4].mean[2] - 1.5f64).abs() < 1e-9);
    assert!((marginals[x5].mean[0] - 0.1f64).abs() < 1e-9);
    assert!((marginals[x5].mean[1] - -3.2f64).abs() < 1e-9);
    assert!((marginals[x5].mean[2] - 2f64).abs() < 1e-9);
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
                1.0, 0.0, 0.0;
                0.0, 2.0, 0.2;
                0.0, 0.2, 2.0
            ],
        )
        .unwrap();

    let smoother = Smoother::new(factor_graph);
    let marginals = smoother.run().unwrap();

    assert!((marginals[x1].mean[0] - 0f64).abs() < 1e-9);
    assert!((marginals[x1].mean[1] - 0f64).abs() < 1e-9);
    assert!((marginals[x1].mean[2] - 0f64).abs() < 1e-9);
    assert!((marginals[x2].mean[0] - -0.2f64).abs() < 1e-9);
    assert!((marginals[x2].mean[1] - -1f64).abs() < 1e-9);
    assert!((marginals[x2].mean[2] - 0.5f64).abs() < 1e-9);
    assert!((marginals[x3].mean[0] - -0.1f64).abs() < 1e-9);
    assert!((marginals[x3].mean[1] - -1.8f64).abs() < 1e-9);
    assert!((marginals[x3].mean[2] - 0.4f64).abs() < 1e-9);
    assert!((marginals[x4].mean[0] - -0.3f64).abs() < 1e-9);
    assert!((marginals[x4].mean[1] - -2.8f64).abs() < 1e-9);
    assert!((marginals[x4].mean[2] - 0.9f64).abs() < 1e-9);
    assert!((marginals[x5].mean[0] - -0.5f64).abs() < 1e-9);
    assert!((marginals[x5].mean[1] - -3.8f64).abs() < 1e-9);
    assert!((marginals[x5].mean[2] - 1.4f64).abs() < 1e-9);
}

#[test]
fn run_smoother_loop() {
    let mut factor_graph = FactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 1).unwrap();
    let x2 = factor_graph.add_variable(X2, 1).unwrap();
    let x3 = factor_graph.add_variable(X3, 1).unwrap();

    factor_graph
        .add_prior(x2, dvector![0.2], dmatrix![1.0])
        .unwrap();
    factor_graph
        .add_observation(
            vec![x1, x2],
            dmatrix![1.0, 1.0],
            dvector![0.1],
            dmatrix![0.6],
        )
        .unwrap();
    factor_graph
        .add_observation(
            vec![x1, x3],
            dmatrix![1.0, 1.0],
            dvector![0.8],
            dmatrix![1.0],
        )
        .unwrap();
    factor_graph
        .add_observation(
            vec![x2, x3],
            dmatrix![1.0, 1.0],
            dvector![1.0],
            dmatrix![0.5],
        )
        .unwrap();

    let smoother = Smoother::new(factor_graph);
    let marginals = smoother.run().unwrap();

    assert!((marginals[x1].mean[0] - -0.05737704918032777f64).abs() < 1e-9);
    assert!((marginals[x2].mean[0] - 0.16721311475409825).abs() < 1e-9);
    assert!((marginals[x3].mean[0] - 0.8409836065573769f64).abs() < 1e-9);
}

#[test]
/// Context: incomplete system with no global anchor
fn run_smoother_slam_singular() {
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
                    0.1, 0.8, 0.0, 0.0, -0.8
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
    assert!(smoother.run().is_err());
}

#[test]
/// Context: valid system with fixed x1 = (0,0,0)
fn run_smoother_slam_anchored() {
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
                1e-15, 0.0, 0.0;
                0.0, 1e-15, 0.0;
                0.0, 0.0, 1e-15
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
    let marginals = smoother.run().unwrap();

    assert!((marginals[x1].mean[0] - 0f64).abs() < 1e-9);
    assert!((marginals[x1].mean[1] - 0f64).abs() < 1e-9);
    assert!((marginals[x1].mean[2] - 0f64).abs() < 1e-9);
    assert!((marginals[x2].mean[0] - -0.09905619973311207f64).abs() < 1e-9);
    assert!((marginals[x2].mean[1] - -0.7729435658383524f64).abs() < 1e-9);
    assert!((marginals[x2].mean[2] - 0.5f64).abs() < 1e-9);
    assert!((marginals[x3].mean[0] - 0.0009438002668878695f64).abs() < 1e-9);
    assert!((marginals[x3].mean[1] - -1.5729435658383526f64).abs() < 1e-9);
    assert!((marginals[x3].mean[2] - 0.4f64).abs() < 1e-9);
    assert!((marginals[l1].mean[0] - -0.07232187939553504f64).abs() < 1e-9);
    assert!((marginals[l1].mean[1] - -0.7267273325639271f64).abs() < 1e-9);
    assert!((marginals[l2].mean[0] - -1.2843967015454254f64).abs() < 1e-9);
    assert!((marginals[l2].mean[1] - -1.5728255908049915f64).abs() < 1e-9);
}

#[test]
fn run_smoother_slam() {
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
    let marginals = smoother.run().unwrap();

    assert!((marginals[x1].mean[0] - 0.6414681199221776f64).abs() < 1e-9);
    assert!((marginals[x1].mean[1] - 0.6305950058258917f64).abs() < 1e-9);
    assert!((marginals[x1].mean[2] - 0.5641386059266061f64).abs() < 1e-9);
    assert!((marginals[x1].cov[2] - 0.011845650541686714f64).abs() < 1e-9);
    assert!((marginals[x1].cov[5] - 0.010674829779356918f64).abs() < 1e-9);
    assert!((marginals[x1].cov[6] - 0.011845650541686714f64).abs() < 1e-9);

    assert!((marginals[x3].mean[0] - 0.6340414328187938f64).abs() < 1e-9);
    assert!((marginals[x3].mean[1] - -0.985118269149018f64).abs() < 1e-9);
    assert!((marginals[x3].mean[2] - 0.9641386059266073f64).abs() < 1e-9);

    assert!((marginals[l1].mean[0] - 0.567031475537499f64).abs() < 1e-9);
    assert!((marginals[l1].mean[1] - -0.13169048375615228f64).abs() < 1e-9);

    assert!((marginals[l2].mean[0] - -0.7213818058899935f64).abs() < 1e-9);
    assert!((marginals[l2].mean[1] - -0.9058630900466684f64).abs() < 1e-9);
    assert!((marginals[l2].cov[0] - 13.069821055981416f64).abs() < 1e-9);
    assert!((marginals[l2].cov[1] - -2.669764109763409f64).abs() < 1e-9);
}

//! Tests for the Gaussian-tree based solver.
//! The solutions can also be computed by solving
//! (At Wy A) mx != At xiy for x,
//! where A, Wy, and xiy represents the whole system.

use nalgebra::{dmatrix, dvector};
use std::iter::zip;

// use arbor_ferrea::gaussian::*;
// use arbor_ferrea::numeric::NumericTree;
// use arbor_ferrea::symbolic::{SymbolicFactor, SymbolicTree};
use arbor_ferrea::ui::*;

const X1: VarSymbol = VarSymbol('x', 1);
const X2: VarSymbol = VarSymbol('x', 2);
const X3: VarSymbol = VarSymbol('x', 3);
const L1: VarSymbol = VarSymbol('l', 1);
const L2: VarSymbol = VarSymbol('l', 2);

#[test]
/// Context: empty factor graphs
fn run_solver_empty() {
    let mut factor_graph = GaussianFactorGraph::new();

    let res = factor_graph.build_symbolic_tree(&[]);
    assert!(res.is_ok());
    let tree = res.unwrap();
    assert!(tree.roots.is_empty());

    // sequential execution
    let res = factor_graph.solve_seq(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));

    // parallel execution
    let res = factor_graph.solve_par(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
/// Context: incomplete system with no global anchor
fn run_solver_singular() {
    let mut factor_graph = GaussianFactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let l1 = factor_graph.add_variable(L1, 2).unwrap();
    let l2 = factor_graph.add_variable(L2, 2).unwrap();

    factor_graph
        .add_factor(
            vec![x1, x2],
            vec![
                dmatrix![
                    1.0, 0.0, 0.0;
                    0.0, 1.0, 0.0;
                    0.0, 0.0, 1.0
                ],
                dmatrix![
                    -1.0, 0.0, 0.0;
                    0.0, -1.0, 0.0;
                    0.0, 0.0, -1.0
                ],
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
        .add_factor(
            vec![x2, x3],
            vec![
                dmatrix![
                    1.0, 0.0, 0.0;
                    0.0, 1.0, 0.0;
                    0.0, 0.0, 1.0
                ],
                dmatrix![
                    -1.0, 0.0, 0.0;
                    0.0, -1.0, 0.0;
                    0.0, 0.0, -1.0
                ],
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
        .add_factor(
            vec![x1, l1],
            vec![
                dmatrix![
                    0.5, 0.5, 0.0;
                    0.0, 0.75, 0.25
                ],
                dmatrix![
                    -1.0, 0.0;
                    0.0, -1.0
                ],
            ],
            dvector![0.0, 0.5],
            dmatrix![
                1.0, 0.2;
                0.2, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x2, l1],
            vec![
                dmatrix![
                    0.6, 0.4, 0.0;
                    0.1, 0.8, 0.1
                ],
                dmatrix![
                    -0.5, -0.5;
                    0.0, -0.8
                ],
            ],
            dvector![0.1, 0.2],
            dmatrix![
                0.8, 0.1;
                0.1, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x3, l2],
            vec![
                dmatrix![
                    0.6, 0.4, 0.0;
                    0.1, 0.8, 0.1
                ],
                dmatrix![
                    -0.5, -0.5;
                    0.0, -0.8
                ],
            ],
            dvector![0.8, 0.0],
            dmatrix![
                2.0, 0.2;
                0.2, 2.0
            ],
        )
        .unwrap();

    // build symbolic tree from elimination order
    let res = factor_graph.build_symbolic_tree(&[l1, l2, x1, x2, x3]);
    assert!(res.is_ok());
    let tree = res.unwrap();
    assert!(tree.roots.len() == 1);

    // sequential execution
    let res = factor_graph.solve_seq(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));

    // parallel execution
    let res = factor_graph.solve_par(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}
//
#[test]
/// Context: valid system with fixed x1 = (0,0,0)
fn run_solver_anchored() {
    let mut factor_graph = GaussianFactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let l1 = factor_graph.add_variable(L1, 2).unwrap();
    let l2 = factor_graph.add_variable(L2, 2).unwrap();

    factor_graph
        .add_factor(
            vec![x1],
            vec![dmatrix![
                1.0, 0.0, 0.0;
                0.0, 1.0, 0.0;
                0.0, 0.0, 1.0
            ]],
            dvector![0.0, 0.0, 0.0],
            dmatrix![
                1e12, 0.0, 0.0;
                0.0, 1e12, 0.0;
                0.0, 0.0, 1e12
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x1, x2],
            vec![
                dmatrix![
                    1.0, 0.0, 0.0;
                    0.0, 1.0, 0.0;
                    0.0, 0.0, 1.0
                ],
                dmatrix![
                    -1.0, 0.0, 0.0;
                    0.0, -1.0, 0.0;
                    0.0, 0.0, -1.0
                ],
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
        .add_factor(
            vec![x2, x3],
            vec![
                dmatrix![
                    1.0, 0.0, 0.0;
                    0.0, 1.0, 0.0;
                    0.0, 0.0, 1.0
                ],
                dmatrix![
                    -1.0, 0.0, 0.0;
                    0.0, -1.0, 0.0;
                    0.0, 0.0, -1.0
                ],
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
        .add_factor(
            vec![x1, l1],
            vec![
                dmatrix![
                    0.5, 0.5, 0.0;
                    0.0, 0.75, 0.25
                ],
                dmatrix![
                    -1.0, 0.0;
                    0.0, -1.0
                ],
            ],
            dvector![0.0, 0.5],
            dmatrix![
                1.0, 0.2;
                0.2, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x2, l1],
            vec![
                dmatrix![
                    0.6, 0.4, 0.0;
                    0.1, 0.8, 0.1
                ],
                dmatrix![
                    -0.5, -0.5;
                    0.0, -0.8
                ],
            ],
            dvector![0.1, 0.2],
            dmatrix![
                0.8, 0.1;
                0.1, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x3, l2],
            vec![
                dmatrix![
                    0.6, 0.4, 0.0;
                    0.1, 0.8, 0.1
                ],
                dmatrix![
                    -0.5, -0.5;
                    0.0, -0.8
                ],
            ],
            dvector![0.8, 0.0],
            dmatrix![
                2.0, 0.2;
                0.2, 2.0
            ],
        )
        .unwrap();

    // build symbolic tree from elimination order
    let res = factor_graph.build_symbolic_tree(&[l1, l2, x1, x2, x3]);
    assert!(res.is_ok());
    let tree = res.unwrap();
    assert!(tree.roots.len() == 1);

    let sol = [
        0.0,
        0.0,
        0.0,
        -0.14678087372748258,
        -0.7867415150745383,
        0.5218166038192673,
        -0.046780873727480105,
        -1.5867415150745345,
        0.421816603819266,
        -0.015065534911002251,
        -0.7332315731040928,
        -1.3856682117195396,
        -1.5398620488130612,
    ];

    // sequential execution
    let res = factor_graph.solve_seq(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(zip(out.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));
    let x1_value = factor_graph.value(x1).unwrap_or_default();
    assert!(x1_value.iter().all(|&v| v.abs() < 1e-9));

    // parallel execution
    let res = factor_graph.solve_par(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(zip(out.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));
    let x1_value = factor_graph.value(x1).unwrap_or_default();
    assert!(x1_value.iter().all(|&v| v.abs() < 1e-9));
}

#[test]
fn run_solver() {
    let mut factor_graph = GaussianFactorGraph::new();

    let x1 = factor_graph.add_variable(X1, 3).unwrap();
    let x2 = factor_graph.add_variable(X2, 3).unwrap();
    let x3 = factor_graph.add_variable(X3, 3).unwrap();
    let l1 = factor_graph.add_variable(L1, 2).unwrap();
    let l2 = factor_graph.add_variable(L2, 2).unwrap();

    factor_graph
        .add_factor(
            vec![x1],
            vec![dmatrix![
                1.0, 0.0, 0.0;
                0.0, 1.0, 0.0;
                0.0, 0.0, 1.0
            ]],
            dvector![0.0, 0.0, 0.0],
            dmatrix![
                1.0, 0.0, 0.0;
                0.0, 1.0, 0.0;
                0.0, 0.0, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x1, x2],
            vec![
                dmatrix![
                    1.0, 0.0, 0.0;
                    0.0, 1.0, 0.0;
                    0.0, 0.0, 1.0
                ],
                dmatrix![
                    -1.0, 0.0, 0.0;
                    0.0, -1.0, 0.0;
                    0.0, 0.0, -1.0
                ],
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
        .add_factor(
            vec![x2, x3],
            vec![
                dmatrix![
                    1.0, 0.0, 0.0;
                    0.0, 1.0, 0.0;
                    0.0, 0.0, 1.0
                ],
                dmatrix![
                    -1.0, 0.0, 0.0;
                    0.0, -1.0, 0.0;
                    0.0, 0.0, -1.0
                ],
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
        .add_factor(
            vec![x1, l1],
            vec![
                dmatrix![
                    0.5, 0.5, 0.0;
                    0.0, 0.75, 0.25
                ],
                dmatrix![
                    -1.0, 0.0;
                    0.0, -1.0
                ],
            ],
            dvector![0.0, 0.5],
            dmatrix![
                1.0, 0.2;
                0.2, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x2, l1],
            vec![
                dmatrix![
                    0.6, 0.4, 0.0;
                    0.1, 0.8, 0.1
                ],
                dmatrix![
                    -0.5, -0.5;
                    0.0, -0.8
                ],
            ],
            dvector![0.1, 0.2],
            dmatrix![
                0.8, 0.1;
                0.1, 1.0
            ],
        )
        .unwrap();
    factor_graph
        .add_factor(
            vec![x3, l2],
            vec![
                dmatrix![
                    0.6, 0.4, 0.0;
                    0.1, 0.8, 0.1
                ],
                dmatrix![
                    -0.5, -0.5;
                    0.0, -0.8
                ],
            ],
            dvector![0.8, 0.0],
            dmatrix![
                2.0, 0.2;
                0.2, 2.0
            ],
        )
        .unwrap();

    // build symbolic tree from elimination order
    let res = factor_graph.build_symbolic_tree(&[l1, l2, x1, x2, x3]);
    assert!(res.is_ok());
    let tree = res.unwrap();
    assert!(tree.roots.len() == 1);

    let sol = [
        0.061317172417491496,
        0.01728312604474211,
        -0.035691200279322946,
        -0.09039976561722562,
        -0.7751798098155087,
        0.48576334881212835,
        0.009600234382776648,
        -1.5751798098155054,
        0.38576334881212704,
        0.027602227090634472,
        -0.7222038685912746,
        -1.3228642046769274,
        -1.5257593619161423,
    ];

    // sequential execution
    let res = factor_graph.solve_seq(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(zip(out.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));

    // parallel execution
    let res = factor_graph.solve_par(&tree);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(zip(out.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));
}

//! Tests for the Gaussian-tree based solver.
//! The solutions can also be computed by solving
//! (At Wy A) mx != At xiy for x,
//! where A, Wy, and xiy represents the whole system.

use nalgebra::{dmatrix, dvector};
use std::iter::zip;

use arbor_ferrea::gaussian::*;
use arbor_ferrea::helper::{VarDict, VarSymbol};
use arbor_ferrea::numeric::NumericTree;
use arbor_ferrea::symbolic::{SymbolicFactor, SymbolicTree};

#[test]
/// Context: singular system with no constraint
fn run_solver_empty() {
    // Build the dictionary and register domain variables
    let mut dict = VarDict::new();
    let x1 = dict.register(VarSymbol('x', 1), 3);
    let x2 = dict.register(VarSymbol('x', 2), 3);
    let x3 = dict.register(VarSymbol('x', 3), 3);
    let l1 = dict.register(VarSymbol('l', 1), 2);
    let l2 = dict.register(VarSymbol('l', 2), 2);

    let factors: Vec<GaussianFactor> = vec![];
    let mut buffer = vec![f64::NAN; dict.total_size()];

    // Build the symbolic tree
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let res = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert!(symb_tree.roots.is_empty());

    // SEQUENTIAL
    // Build pass
    let res = GaussianTree::from_symbolic_seq(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert!(gauss_tree.roots.is_empty());
    // Solve pass
    let res = gauss_tree.solve_seq(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(buffer.iter().all(|v| v.is_nan()));

    // PARALLEL
    // Build pass
    let res = GaussianTree::from_symbolic_par(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert!(gauss_tree.roots.is_empty());
    // Solve pass
    let res = gauss_tree.solve_par(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(buffer.iter().all(|v| v.is_nan()));
}

#[test]
/// Context: incomplete system with no global anchor
fn run_solver_singular() {
    // Build the dictionary and register domain variables
    let mut dict = VarDict::new();
    let x1 = dict.register(VarSymbol('x', 1), 3);
    let x2 = dict.register(VarSymbol('x', 2), 3);
    let x3 = dict.register(VarSymbol('x', 3), 3);
    let l1 = dict.register(VarSymbol('l', 1), 2);
    let l2 = dict.register(VarSymbol('l', 2), 2);

    let f1 = GaussianFactor::try_new(
        vec![(x1, 3), (x2, 3)],
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
    let f2 = GaussianFactor::try_new(
        vec![(x2, 3), (x3, 3)],
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
    let f3 = GaussianFactor::try_new(
        vec![(x1, 3), (l1, 2)],
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
    let f4 = GaussianFactor::try_new(
        vec![(x2, 3), (l1, 2)],
        dmatrix![
            0.6, 0.4, 0.0, -0.5, -0.5;
            0.1, 0.8, 0.1, 0.0, -0.8
        ],
        dvector![0.1, 0.2],
        dmatrix![
            0.8, 0.1;
            0.1, 1.0
        ],
    )
    .unwrap();
    let f5 = GaussianFactor::try_new(
        vec![(x3, 3), (l2, 2)],
        dmatrix![
            0.6, 0.4, 0.0, -0.5, -0.5;
            0.1, 0.8, 0.1, 0.0, -0.8
        ],
        dvector![0.8, 0.0],
        dmatrix![
            2.0, 0.2;
            0.2, 2.0
        ],
    )
    .unwrap();

    let factors = vec![f1, f2, f3, f4, f5];
    let mut buffer = vec![f64::NAN; dict.total_size()];

    // Build the symbolic tree
    // let order = [x1, x2, x3, l1, l2]; // user-defined elimination order
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let res = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert_eq!(symb_tree.roots.len(), 1);

    // SEQUENTIAL
    // Build pass
    let res = GaussianTree::from_symbolic_seq(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert_eq!(gauss_tree.roots.len(), 1);
    // Solve pass
    let res = gauss_tree.solve_seq(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(buffer.iter().all(|v| v.is_finite()));

    // PARALLEL
    // Build pass
    let res = GaussianTree::from_symbolic_par(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert_eq!(gauss_tree.roots.len(), 1);
    // Solve pass
    let res = gauss_tree.solve_par(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(buffer.iter().all(|v| v.is_finite()));
}

#[test]
/// Context: valid system with fixed x1 = (0,0,0)
fn run_solver_anchored() {
    // Build the dictionary and register domain variables
    let mut dict = VarDict::new();
    let x1 = dict.register(VarSymbol('x', 1), 3);
    let x2 = dict.register(VarSymbol('x', 2), 3);
    let x3 = dict.register(VarSymbol('x', 3), 3);
    let l1 = dict.register(VarSymbol('l', 1), 2);
    let l2 = dict.register(VarSymbol('l', 2), 2);

    let f0 = GaussianFactor::try_new(
        vec![(x1, 3)],
        dmatrix![
            1.0, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ],
        dvector![0.0, 0.0, 0.0],
        dmatrix![
            1e12, 0.0, 0.0;
            0.0, 1e12, 0.0;
            0.0, 0.0, 1e12
        ],
    )
    .unwrap();
    let f1 = GaussianFactor::try_new(
        vec![(x1, 3), (x2, 3)],
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
    let f2 = GaussianFactor::try_new(
        vec![(x2, 3), (x3, 3)],
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
    let f3 = GaussianFactor::try_new(
        vec![(x1, 3), (l1, 2)],
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
    let f4 = GaussianFactor::try_new(
        vec![(x2, 3), (l1, 2)],
        dmatrix![
            0.6, 0.4, 0.0, -0.5, -0.5;
            0.1, 0.8, 0.1, 0.0, -0.8
        ],
        dvector![0.1, 0.2],
        dmatrix![
            0.8, 0.1;
            0.1, 1.0
        ],
    )
    .unwrap();
    let f5 = GaussianFactor::try_new(
        vec![(x3, 3), (l2, 2)],
        dmatrix![
            0.6, 0.4, 0.0, -0.5, -0.5;
            0.1, 0.8, 0.1, 0.0, -0.8
        ],
        dvector![0.8, 0.0],
        dmatrix![
            2.0, 0.2;
            0.2, 2.0
        ],
    )
    .unwrap();
    let factors = vec![f0, f1, f2, f3, f4, f5];

    let mut buffer = vec![f64::NAN; dict.total_size()];

    // Ground truth computed beforehand by solving the whole system
    let sol = [
        0.0,
        0.0,
        0.0,
        0.03601984870829765,
        -0.8070126430347709,
        0.5205154245108021,
        0.15268651537496425,
        -1.6014570874792147,
        0.4649598689552463,
        0.10078828017507614,
        -0.7293659652796087,
        -0.4322764206813341,
        -1.4737462389328886,
    ];

    // Build the symbolic tree
    // let order = [x1, x2, x3, l1, l2]; // user-defined elimination order
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let res = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert_eq!(symb_tree.roots.len(), 1);

    // SEQUENTIAL
    // Build pass
    let res = GaussianTree::from_symbolic_seq(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert_eq!(gauss_tree.roots.len(), 1);
    // Solve pass
    let res = gauss_tree.solve_seq(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(zip(buffer.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));

    // PARALLEL
    // Build pass
    let res = GaussianTree::from_symbolic_par(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert_eq!(gauss_tree.roots.len(), 1);
    // Solve pass
    let res = gauss_tree.solve_par(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(zip(buffer.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));
}

#[test]
fn run_solver() {
    // Build the dictionary and register domain variables
    let mut dict = VarDict::new();
    let x1 = dict.register(VarSymbol('x', 1), 3);
    let x2 = dict.register(VarSymbol('x', 2), 3);
    let x3 = dict.register(VarSymbol('x', 3), 3);
    let l1 = dict.register(VarSymbol('l', 1), 2);
    let l2 = dict.register(VarSymbol('l', 2), 2);

    let f0 = GaussianFactor::try_new(
        vec![(x1, 3)],
        dmatrix![
            1.0, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ],
        dvector![0.0, 0.0, 0.0],
        dmatrix![
            1.0, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ],
    )
    .unwrap();
    let f1 = GaussianFactor::try_new(
        vec![(x1, 3), (x2, 3)],
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
    let f2 = GaussianFactor::try_new(
        vec![(x2, 3), (x3, 3)],
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
    let f3 = GaussianFactor::try_new(
        vec![(x1, 3), (l1, 2)],
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
    let f4 = GaussianFactor::try_new(
        vec![(x2, 3), (l1, 2)],
        dmatrix![
            0.6, 0.4, 0.0, -0.5, -0.5;
            0.1, 0.8, 0.1, 0.0, -0.8
        ],
        dvector![0.1, 0.2],
        dmatrix![
            0.8, 0.1;
            0.1, 1.0
        ],
    )
    .unwrap();
    let f5 = GaussianFactor::try_new(
        vec![(x3, 3), (l2, 2)],
        dmatrix![
            0.6, 0.4, 0.0, -0.5, -0.5;
            0.1, 0.8, 0.1, 0.0, -0.8
        ],
        dvector![0.8, 0.0],
        dmatrix![
            2.0, 0.2;
            0.2, 2.0
        ],
    )
    .unwrap();
    let factors = vec![f0, f1, f2, f3, f4, f5];

    let mut buffer = vec![f64::NAN; dict.total_size()];

    // Ground truth computed beforehand by solving the whole system
    let sol = vec![
        0.04932654118398348,
        0.021495778955759128,
        -0.030558548462094673,
        0.08192631723556142,
        -0.7906037691495,
        0.48957333737672887,
        0.19859298390222846,
        -1.5850482135939445,
        0.4340177818211732,
        0.13837542772367273,
        -0.7148723936331374,
        -0.3823409808998208,
        -1.4554668173734686,
    ];

    // Build the symbolic tree
    // let order = [x1, x2, x3, l1, l2]; // user-defined elimination order
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let res = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert_eq!(symb_tree.roots.len(), 1);

    // SEQUENTIAL
    // Build pass
    let res = GaussianTree::from_symbolic_seq(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert_eq!(gauss_tree.roots.len(), 1);
    // Solve pass
    let res = gauss_tree.solve_seq(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(zip(buffer.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));

    // PARALLEL
    // Build pass
    let res = GaussianTree::from_symbolic_par(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert_eq!(gauss_tree.roots.len(), 1);
    // Solve pass
    let res = gauss_tree.solve_par(&mut buffer, &dict);
    assert!(res.is_ok());
    assert!(zip(buffer.iter(), sol.iter()).all(|(b, s)| (b - s).abs() < 1e-9));
}

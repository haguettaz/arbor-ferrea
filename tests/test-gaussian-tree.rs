//! Tests for the Gaussian-tree based solver.
//! The solutions can also be computed by solving
//! (At W A) x != At xi for x,
//! where A, W, and xi represents the whole system.

use faer::prelude::*;

use arbor_ferrea::factor::anchor::AnchorFactor;
use arbor_ferrea::factor::gaussian::{GaussianFactor, preprocess_factors};
use arbor_ferrea::helper::build_buffer;
use arbor_ferrea::tree::gaussian::GaussianTree;
use arbor_ferrea::tree::numeric::NumericTree;
use arbor_ferrea::tree::symbolic::{SymbolicFactor, SymbolicTree};
use arbor_ferrea::variable::dictionary::{VarDict, VarSymbol};

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

    let mut factors: Vec<GaussianFactor<f64>> = vec![];

    // 0. Pre-process factors (left unchanged) and init state buffer
    let res = preprocess_factors(&mut factors, &[]);
    assert!(res.is_ok());
    assert!(factors.is_empty());
    let res = build_buffer(&[], &dict, f64::NAN);
    assert!(res.is_ok());
    let mut out = res.unwrap();
    assert!(out.iter().all(|v| v.is_nan()));

    // 1. Build the symbolic tree
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let res = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert!(symb_tree.roots.is_empty());

    // 2. Perform the parallel build pass (bottom-up)
    let res = GaussianTree::from_symbolic(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert!(gauss_tree.roots.is_empty());

    // 3. Perform parallel solve (Top-Down)
    let res = gauss_tree.solve_par(&mut out, &dict);
    assert!(res.is_ok());
    assert!(out.iter().all(|v| v.is_nan()));
}

#[test]
/// Context: incomplete/singular system with no global anchor
fn run_solver_singular() {
    // Build the dictionary and register domain variables
    let mut dict = VarDict::new();
    let x1 = dict.register(VarSymbol('x', 1), 3);
    let x2 = dict.register(VarSymbol('x', 2), 3);
    let x3 = dict.register(VarSymbol('x', 3), 3);
    let l1 = dict.register(VarSymbol('l', 1), 2);
    let l2 = dict.register(VarSymbol('l', 2), 2);

    let mut factors = vec![
        GaussianFactor::new(
            vec![x1],
            mat![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            vec![(x1, 0, 3)],
            col![0.0, 0.0, 0.0],
            mat![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ),
        GaussianFactor::new(
            vec![x1, x2],
            mat![
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
            ],
            vec![(x1, 0, 3), (x2, 3, 3)],
            col![0.2, 1.0, -0.5],
            mat![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        GaussianFactor::new(
            vec![x2, x3],
            mat![
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
            ],
            vec![(x2, 0, 3), (x3, 3, 3)],
            col![-0.1, 0.8, 0.1],
            mat![[1.0, 0.0, 0.3], [0.0, 1.0, 0.1], [0.3, 0.1, 1.0]],
        ),
        GaussianFactor::new(
            vec![x1, l1],
            mat![[0.5, 0.5, 0.0, -1.0, 0.0], [0.0, 0.75, 0.25, 0.0, -1.0]],
            vec![(x1, 0, 3), (l1, 3, 2)],
            col![0.0, 0.5],
            mat![[1.0, 0.2], [0.2, 1.0]],
        ),
        GaussianFactor::new(
            vec![x2, l1],
            mat![[0.6, 0.4, 0.0, -0.5, -0.5], [0.1, 0.8, 0.1, 0.0, -0.8]],
            vec![(x2, 0, 3), (l1, 3, 2)],
            col![0.1, 0.2],
            mat![[0.8, 0.1], [0.1, 1.0]],
        ),
        GaussianFactor::new(
            vec![x3, l2],
            mat![[0.6, 0.4, 0.0, -0.5, -0.5], [0.1, 0.8, 0.1, 0.0, -0.8]],
            vec![(x3, 0, 3), (l2, 3, 2)],
            col![0.8, 0.0],
            mat![[2.0, 0.2], [0.2, 2.0]],
        ),
    ];

    // 0. Pre-process factors (do nothing) and init state buffer
    let res = preprocess_factors(&mut factors, &[]);
    assert!(res.is_ok());
    assert_eq!(factors.len(), 6);
    let res = build_buffer(&[], &dict, f64::NAN);
    assert!(res.is_ok());
    let out = res.unwrap();
    assert!(out.iter().all(|v| v.is_nan()));

    // 1. Build the symbolic tree
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let res = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert_eq!(symb_tree.roots.len(), 1);

    // 2. Perform the parallel build pass (bottom-up): should fail!
    let res = GaussianTree::from_symbolic(&symb_tree, &factors, &dict);
    assert!(res.is_err());

    // var=2, rtx=[
    // [0.05269289355567835, -0.03290560543586823, -0.01905444671986705],
    // [-0.0329056054358684, 0.05534416149468514, 0.0034353797251171175],
    // [-0.019054446719867127, 0.0034353797251171158, 0.008949082977065542],
    // ]
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

    // Build the Gaussian and anchoring factors
    let anchors = vec![AnchorFactor::new(x1, vec![0.0, 0.0, 0.0])];
    let mut factors = vec![
        GaussianFactor::new(
            vec![x1, x2],
            mat![
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
            ],
            vec![(x1, 0, 3), (x2, 3, 3)],
            col![0.2, 1.0, -0.5],
            mat![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        GaussianFactor::new(
            vec![x2, x3],
            mat![
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
            ],
            vec![(x2, 0, 3), (x3, 3, 3)],
            col![-0.1, 0.8, 0.1],
            mat![[1.0, 0.0, 0.3], [0.0, 1.0, 0.1], [0.3, 0.1, 1.0]],
        ),
        GaussianFactor::new(
            vec![x1, l1],
            mat![[0.5, 0.5, 0.0, -1.0, 0.0], [0.0, 0.75, 0.25, 0.0, -1.0]],
            vec![(x1, 0, 3), (l1, 3, 2)],
            col![0.0, 0.5],
            mat![[1.0, 0.2], [0.2, 1.0]],
        ),
        GaussianFactor::new(
            vec![x2, l1],
            mat![[0.6, 0.4, 0.0, -0.5, -0.5], [0.1, 0.8, 0.1, 0.0, -0.8]],
            vec![(x2, 0, 3), (l1, 3, 2)],
            col![0.1, 0.2],
            mat![[0.8, 0.1], [0.1, 1.0]],
        ),
        GaussianFactor::new(
            vec![x3, l2],
            mat![[0.6, 0.4, 0.0, -0.5, -0.5], [0.1, 0.8, 0.1, 0.0, -0.8]],
            vec![(x3, 0, 3), (l2, 3, 2)],
            col![0.8, 0.0],
            mat![[2.0, 0.2], [0.2, 2.0]],
        ),
    ];

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

    // 0. Preprocess factors: eliminate anchored variables and init buffer properly
    let res = preprocess_factors(&mut factors, &anchors);
    assert!(res.is_ok());
    assert!(factors.iter().all(|f| !f.vars.contains(&x1)));
    let res = build_buffer(&anchors, &dict, f64::NAN);
    assert!(res.is_ok());
    let mut out = res.unwrap();
    assert_eq!(&out[..3], &[0.0, 0.0, 0.0]);
    assert!(&out[3..].iter().all(|v| v.is_nan()));

    // 1. Build the symbolic tree
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let res = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert_eq!(symb_tree.roots.len(), 1);

    // 2. Perform the parallel build pass (bottom-up)
    let res = GaussianTree::from_symbolic(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut gauss_tree = res.unwrap();
    assert_eq!(gauss_tree.roots.len(), 1);

    // 3. Perform parallel solve (Top-Down)
    let res = gauss_tree.solve_par(&mut out, &dict);
    assert!(res.is_ok());
    assert!(
        out.iter()
            .zip(sol.iter())
            .all(|(b, s)| (b - s).abs() < 1e-9)
    );
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

    // Build the Gaussian factors
    let mut factors = vec![
        GaussianFactor::new(
            vec![x1],
            mat![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            vec![(x1, 0, 3)],
            col![0.0, 0.0, 0.0],
            mat![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        GaussianFactor::new(
            vec![x1, x2],
            mat![
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
            ],
            vec![(x1, 0, 3), (x2, 3, 3)],
            col![0.2, 1.0, -0.5],
            mat![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        GaussianFactor::new(
            vec![x2, x3],
            mat![
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
            ],
            vec![(x2, 0, 3), (x3, 3, 3)],
            col![-0.1, 0.8, 0.1],
            mat![[1.0, 0.0, 0.3], [0.0, 1.0, 0.1], [0.3, 0.1, 1.0]],
        ),
        GaussianFactor::new(
            vec![x1, l1],
            mat![[0.5, 0.5, 0.0, -1.0, 0.0], [0.0, 0.75, 0.25, 0.0, -1.0]],
            vec![(x1, 0, 3), (l1, 3, 2)],
            col![0.0, 0.5],
            mat![[1.0, 0.2], [0.2, 1.0]],
        ),
        GaussianFactor::new(
            vec![x2, l1],
            mat![[0.6, 0.4, 0.0, -0.5, -0.5], [0.1, 0.8, 0.1, 0.0, -0.8]],
            vec![(x2, 0, 3), (l1, 3, 2)],
            col![0.1, 0.2],
            mat![[0.8, 0.1], [0.1, 1.0]],
        ),
        GaussianFactor::new(
            vec![x3, l2],
            mat![[0.6, 0.4, 0.0, -0.5, -0.5], [0.1, 0.8, 0.1, 0.0, -0.8]],
            vec![(x3, 0, 3), (l2, 3, 2)],
            col![0.8, 0.0],
            mat![[2.0, 0.2], [0.2, 2.0]],
        ),
    ];

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

    // Solve starts here!
    // 0. Preprocess factors: eliminate anchored variables and init buffer properly
    let res = preprocess_factors(&mut factors, &[]);
    assert!(res.is_ok());
    let res = build_buffer(&[], &dict, f64::NAN);
    assert!(res.is_ok());
    let mut out = res.unwrap();
    assert!(out.iter().all(|v| v.is_nan()));

    // 1. Build the symbolic tree
    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let symb_tree = SymbolicTree::build(factors.iter().map(|f| f as &dyn SymbolicFactor), &order)
        .expect("Error building the symbolic tree");
    assert_eq!(symb_tree.roots.len(), 1);

    // 2. Perform the parallel build pass (bottom-up)
    let mut gauss_tree: GaussianTree<f64> =
        GaussianTree::from_symbolic(&symb_tree, &factors, &dict)
            .expect("Error building the Gaussian tree");
    assert_eq!(gauss_tree.roots.len(), 1);

    // 3. Perform parallel solve (Top-Down)
    let res = gauss_tree.solve_par(&mut out, &dict);
    assert!(res.is_ok());
    assert!(
        out.iter()
            .zip(sol.iter())
            .all(|(b, s)| (b - s).abs() < 1e-9)
    );
}

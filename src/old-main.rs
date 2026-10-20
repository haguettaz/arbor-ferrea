use anyhow::Result;
use nalgebra::{dmatrix, dvector};

use arbor_ferrea::ui::*;

const X1: VarSymbol = VarSymbol('x', 1);
const X2: VarSymbol = VarSymbol('x', 2);
const X3: VarSymbol = VarSymbol('x', 3);
const L1: VarSymbol = VarSymbol('l', 1);
const L2: VarSymbol = VarSymbol('l', 2);

fn main() -> Result<()> {
    let mut factor_graph = GaussianFactorGraph::new();
    let x1 = factor_graph.add_variable(X1, 3)?;
    let x2 = factor_graph.add_variable(X2, 3)?;
    let x3 = factor_graph.add_variable(X3, 3)?;
    let l1 = factor_graph.add_variable(L1, 2)?;
    let l2 = factor_graph.add_variable(L2, 2)?;

    factor_graph.add_factor(
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
    )?;
    factor_graph.add_factor(
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
    )?;
    factor_graph.add_factor(
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
    )?;
    factor_graph.add_factor(
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
    )?;
    factor_graph.add_factor(
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
    )?;
    factor_graph.add_factor(
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
    )?;

    let order = [l1, l2, x1, x2, x3]; // user-defined elimination order
    let tree = factor_graph.build_symbolic_tree(&order)?;
    let sol = factor_graph.solve_seq(&tree)?;

    println!("solution: {:?}", sol);

    Ok(())
}

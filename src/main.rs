use nalgebra::{DMatrixView, DVector, DVectorView, Matrix2, RealField, matrix, vector};

fn main() {
    let mut x = vec![0, 1, 2, 3, 4];
    println!("x before remove: {:?}", x);
    println!("length before remove: {}", x.len());
    x.swap_remove(2);
    println!("x after remove: {:?}", x);
    println!("length after remove: {}", x.len());

    //     let wfx = matrix![1.0, 0.0; 0.0, 1e-15];
    //     let xifx = vector![1.0, 0.0];
    //     let bt = matrix![1.0; 0.0];
    //     let wby = matrix![1.0];
    //     let xiby = matrix![2.0];
    //
    //     let mut wfx_plus_bt_wby_b = wfx.clone();
    //     wfx_plus_bt_wby_b.quadform_tr(1.0, &bt, &wby, 1.0);
    //     println!("wfx_plus_bt_wby_b: {:?}", wfx_plus_bt_wby_b);
    //
    //     let mut xifx_plus_bt_xiby = xifx.clone();
    //     xifx_plus_bt_xiby.gemv(1.0, &bt, &xiby, 1.0);
    //     println!("xifx_plus_bt_xiby: {:?}", xifx_plus_bt_xiby);
    //
    //     let bt_wby = bt * wby;
    //     println!("bt_wby: {:?}", bt_wby);
    //
    //     let chol_wfx_plus_bt_wby_b = wfx_plus_bt_wby_b.cholesky().unwrap();
    //     let l = chol_wfx_plus_bt_wby_b.l_dirty();
    //     let g = l.solve_lower_triangular(&bt_wby);
    //     let h = l.solve_lower_triangular(&xifx_plus_bt_xiby);
    //     println!("g: {:?}", g);
    //     println!("h: {:?}", h);

    // let lblt_wfx_plus_bt_wby_b = wfx_plus_bt_wby_b.lblt();
    // println!("lblt: {:?}", lblt_wfx_plus_bt_wby_b);
    // let h = lblt_wfx_plus_bt_wby_b.solve(&bt_wby);
    // println!("with lblt: {:?}", h);
}

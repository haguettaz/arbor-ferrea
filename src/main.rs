use faer::linalg::solvers::Svd;
use faer::prelude::*;

fn main() {
    let a = mat![[1.0, 0.0], [0.0, 0.0]];
    let b = col![1.0, 1.0];

    let svd = Svd::new(a.as_ref()).unwrap();
    println!("{:?}", svd);

    let ainv = svd.pseudoinverse();
    println!("{:?}", ainv);

    let x = svd.solve_lstsq(&b);
    println!("{:?}", x);
}

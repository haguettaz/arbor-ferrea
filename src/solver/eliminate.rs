use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DMatrixView, DVector, RealField};

use anyhow::{Context, Result, bail, ensure};

use crate::graph::{factor::Factor, layout::Layout};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Kept,
    Cleared,
}

/// One block column of a factor's A, already located in the kept or cleared layout.
struct Blk<'a, T: RealField> {
    side: Side,
    icol: usize,
    ncols: usize,
    mat: DMatrixView<'a, T>,
}

/// Classify every block of `f` once. The result is sorted by key, hence by `pos` within each side.
fn classify<'a, T: RealField>(
    f: &'a Factor<T>,
    kept: &Layout,
    cleared: &Layout,
    out: &mut Vec<Blk<'a, T>>,
) -> Result<()> {
    out.clear();
    for (key, mat) in f.blocks() {
        let (side, (icol, ncols)) = if let Some((icol, ncols)) = kept.find(key) {
            (Side::Kept, (icol, ncols))
        } else if let Some((icol, ncols)) = cleared.find(key) {
            (Side::Cleared, (icol, ncols))
        } else {
            bail!("likelihood key {key} is neither kept nor cleared");
        };
        out.push(Blk {
            side,
            icol,
            ncols,
            mat,
        });
    }
    Ok(())
}

/// Create a new factor by aggregating observation models and prior models and eliminating some variables.
/// It is the caller responsability to check that the kept and cleared keys are compatible and valid with factors.
/// This is equivalent to max-product and sum-product message passing for closing box.
/// The likelihood factor is expressed with minimal observation dimension: if the total dimension of all input factors is larger than the total dimensions of the kept variables, the observation is expressed after projecting it in the kept variable space
pub fn eliminate<'a, I, T: RealField>(
    factors: I,
    kept_keys: &[usize],
    cleared_keys: &[usize],
    size_of: impl Fn(usize) -> usize,
) -> Result<Factor<T>>
where
    I: IntoIterator<Item = &'a Factor<T>>,
    I::IntoIter: Clone,
{
    debug_assert!(cleared_keys.is_sorted(), "unsorted cleared keys");
    debug_assert!(kept_keys.is_sorted(), "unsorted kept keys");

    let factors = factors.into_iter();

    for f in factors.clone() {
        for key in f.keys() {
            if let (Err(_), Err(_)) = (
                kept_keys.binary_search(&key),
                cleared_keys.binary_search(&key),
            ) {
                bail!("key {key} is ambiguous: neither kept nor cleared");
            }
        }
    }

    // init layouts for both kept and cleared variables
    let kept = Layout::new(kept_keys.to_vec(), &size_of);
    let cleared = Layout::new(cleared_keys.to_vec(), &size_of);

    let dim_y = factors.clone().map(|f| f.dim_y()).sum();

    // branch on the minimal observation model
    if dim_y <= kept.dim() {
        eliminate_to_z(factors, kept, cleared, dim_y)
    } else {
        let max_dim_y = factors.clone().map(|f| f.dim_y()).max().unwrap_or(0);
        eliminate_to_x(factors, kept, cleared, max_dim_y)
    }
}

fn eliminate_to_z<'a, T: RealField>(
    factors: impl Iterator<Item = &'a Factor<T>>,
    kept: Layout,
    cleared: Layout,
    dim_y: usize,
) -> Result<Factor<T>> {
    let (dim_x, dim_u) = (kept.dim(), cleared.dim());

    let mut a = DMatrix::zeros(dim_y, dim_x);
    let mut xiby = DVector::zeros(dim_y);
    let mut wby = DMatrix::zeros(dim_y, dim_y);

    let mut bt_xiby = DVector::zeros(dim_u);
    let mut bt_wby = DMatrix::zeros(dim_u, dim_y);
    let mut bt_wby_b = DMatrix::zeros(dim_u, dim_u);

    let mut blks = Vec::new();

    let mut irow = 0;
    for f in factors {
        let nrows = f.dim_y();
        classify(f, &kept, &cleared, &mut blks)?;

        xiby.rows_mut(irow, nrows).copy_from(&f.xiby());
        wby.view_mut((irow, irow), (nrows, nrows))
            .copy_from(&f.wby());

        for blk in blks.iter().filter(|b| b.side == Side::Kept) {
            // update a
            a.view_mut((irow, blk.icol), (nrows, blk.ncols))
                .copy_from(&blk.mat);
        }

        for blk in blks.iter().filter(|b| b.side == Side::Cleared) {
            // update bt_xiby
            bt_xiby
                .rows_mut(blk.icol, blk.ncols)
                .gemv_tr(T::one(), &blk.mat, &f.xiby(), T::one());

            // update bt_wby
            bt_wby
                .view_mut((blk.icol, irow), (blk.ncols, nrows))
                .gemm_tr(T::one(), &blk.mat, &f.wby(), T::one());

            // update bt_wby_b (block diagonal)
            bt_wby_b
                .view_mut((blk.icol, blk.icol), (blk.ncols, blk.ncols))
                .gemm(
                    T::one(),
                    &bt_wby.view((blk.icol, irow), (blk.ncols, nrows)),
                    &blk.mat,
                    T::one(),
                );
            for other_blk in blks
                .iter()
                .filter(|other_blk| other_blk.side == Side::Cleared && other_blk.icol < blk.icol)
            {
                // update bt_wby_b (lower block triangle)
                bt_wby_b
                    .view_mut((blk.icol, other_blk.icol), (blk.ncols, other_blk.ncols))
                    .gemm(
                        T::one(),
                        &bt_wby.view((blk.icol, irow), (blk.ncols, nrows)),
                        &other_blk.mat,
                        T::one(),
                    );
            }
        }

        irow += nrows;
    }

    let chol = Cholesky::new(bt_wby_b).with_context(|| {
        format!(
            "cholesky failed: cleared variables {:?} are under-constrained",
            cleared.keys()
        )
    })?;
    let l = chol.l_dirty();
    let (mut g, mut h) = (bt_wby, bt_xiby);
    ensure!(
        l.solve_lower_triangular_mut(&mut g) && l.solve_lower_triangular_mut(&mut h),
        "singular factor in elimination"
    );

    xiby.gemv_tr(-T::one(), &g, &h, T::one());
    wby.gemm_tr(-T::one(), &g, &g, T::one());

    Ok(Factor::new(kept, a, xiby, wby))
}

fn eliminate_to_x<'a, T: RealField>(
    factors: impl Iterator<Item = &'a Factor<T>>,
    kept: Layout,
    cleared: Layout,
    max_dim_y: usize,
) -> Result<Factor<T>> {
    let (dim_x, dim_u) = (kept.dim(), cleared.dim());

    let mut at_xiby = DVector::zeros(dim_x);
    let mut bt_xiby = DVector::zeros(dim_u);

    let mut at_wby_a = DMatrix::zeros(dim_x, dim_x);
    let mut bt_wby_a = DMatrix::zeros(dim_u, dim_x);
    let mut bt_wby_b = DMatrix::zeros(dim_u, dim_u); // lower triangle only until the end

    let mut scratch = DMatrix::zeros(kept.max_block().max(cleared.max_block()), max_dim_y);
    let mut blks = Vec::new();

    for f in factors {
        classify(f, &kept, &cleared, &mut blks)?;
        let nrows = f.dim_y();

        for blk in blks.iter().filter(|b| b.side == Side::Kept) {
            // update at_xiby
            at_xiby
                .rows_mut(blk.icol, blk.ncols)
                .gemv_tr(T::one(), &blk.mat, &f.xiby(), T::one());

            // init at_wby
            let mut at_wby = scratch.view_mut((0, 0), (blk.ncols, nrows));
            at_wby.gemm_tr(T::one(), &blk.mat, &f.wby(), T::zero());

            // update at_wby_a (diagonal and lower triangle)
            at_wby_a
                .view_mut((blk.icol, blk.icol), (blk.ncols, blk.ncols))
                .gemm(T::one(), &at_wby, &blk.mat, T::one());
            for other_blk in blks
                .iter()
                .filter(|other_blk| other_blk.side == Side::Kept && other_blk.icol < blk.icol)
            {
                at_wby_a
                    .view_mut((blk.icol, other_blk.icol), (blk.ncols, other_blk.ncols))
                    .gemm(T::one(), &at_wby, &other_blk.mat, T::one());
            }
        }

        for blk in blks.iter().filter(|b| b.side == Side::Cleared) {
            // update bt_xiby
            bt_xiby
                .rows_mut(blk.icol, blk.ncols)
                .gemv_tr(T::one(), &blk.mat, &f.xiby(), T::one());

            // init bt_wby
            let mut bt_wby = scratch.view_mut((0, 0), (blk.ncols, nrows));
            bt_wby.gemm_tr(T::one(), &blk.mat, &f.wby(), T::zero());

            for other_blk in blks.iter().filter(|other_blk| other_blk.side == Side::Kept) {
                // update bt_wby_a
                bt_wby_a
                    .view_mut((blk.icol, other_blk.icol), (blk.ncols, other_blk.ncols))
                    .gemm(T::one(), &bt_wby, &other_blk.mat, T::one());
            }

            // update bt_wby_b (diagonal and lower triangle)
            bt_wby_b
                .view_mut((blk.icol, blk.icol), (blk.ncols, blk.ncols))
                .gemm(T::one(), &bt_wby, &blk.mat, T::one());
            for other_blk in blks
                .iter()
                .filter(|other_blk| other_blk.side == Side::Cleared && other_blk.icol < blk.icol)
            {
                bt_wby_b
                    .view_mut((blk.icol, other_blk.icol), (blk.ncols, other_blk.ncols))
                    .gemm(T::one(), &bt_wby, &other_blk.mat, T::one());
            }
        }
    }

    let chol = Cholesky::new(bt_wby_b).with_context(|| {
        format!(
            "cholesky failed: cleared variables {:?} are under-constrained",
            cleared.keys()
        )
    })?;
    let l = chol.l_dirty();
    let (mut g, mut h) = (bt_wby_a, bt_xiby);
    ensure!(
        l.solve_lower_triangular_mut(&mut g) && l.solve_lower_triangular_mut(&mut h),
        "singular factor in elimination"
    );

    at_xiby.gemv_tr(-T::one(), &g, &h, T::one());
    at_wby_a.fill_upper_triangle_with_lower_triangle();
    at_wby_a.gemm_tr(-T::one(), &g, &g, T::one());

    Ok(Factor::new(
        kept,
        DMatrix::identity(dim_x, dim_x),
        at_xiby,
        at_wby_a,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_likelihood_eliminate() {
        let sizes = vec![2, 2, 3, 2, 1];
        let size_of = |k| sizes[k];

        let f0 = Factor::new(
            Layout::new(vec![2], size_of),
            dmatrix![1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, 1.0],
            dvector![0.6, 0.6, 0.6],
            dmatrix![1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, 1.0],
        );

        let f1 = Factor::new(
            Layout::new(vec![0, 1, 2], size_of),
            dmatrix![
                1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.1;
                0.0, 1.0, 0.0, 0.0, 0.0, 1.2, -1.0;
                0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0
            ],
            dvector![0.4, 1.04, -0.5],
            dmatrix![1.0, 0.2, 0.0; 0.2, 1.0, 0.0; 0.0, 0.0, 1.0],
        );

        let f2 = Factor::new(
            Layout::new(vec![1, 3, 4], size_of),
            dmatrix![
                0.0, 0.2, 0.0, 0.0, 0.2;
                0.8, 0.0, 0.0, 1.0, 0.8
            ],
            dvector![0.05, 0.5],
            dmatrix![0.8, 0.1; 0.1, 1.0],
        );

        let new_factor = eliminate(vec![&f0, &f1, &f2], &[0, 1, 3], &[2, 4], size_of).unwrap();

        assert_eq!(new_factor.keys(), [0, 1, 3]);
        assert!(new_factor.dim_y() <= new_factor.dim_x());
        assert!((new_factor.xiby()[0] - -0.18566446904015022f64).abs() < 1e-9);
        assert!((new_factor.xiby()[1] - 0.2564704789814862f64).abs() < 1e-9);
        assert!((new_factor.xiby()[2] - -0.5669726401670172f64).abs() < 1e-9);
        assert!((new_factor.xiby()[3] - 0.16770992358560477f64).abs() < 1e-9);
        assert!((new_factor.xiby()[4] - 0f64).abs() < 1e-9);

        assert!((new_factor.wby()[0] - 0.49401045386071685f64).abs() < 1e-9);
        assert!((new_factor.wby()[2] - -0.005219289506583182f64).abs() < 1e-9);
        assert!((new_factor.wby()[7] - 0.29230607371306017f64).abs() < 1e-9);
        assert!((new_factor.wby()[13] - -0.0347294345005616f64).abs() < 1e-9);
        assert!((new_factor.wby()[19] - -0.0439290200137418f64).abs() < 1e-9);
        assert!((new_factor.wby()[24] - 0f64).abs() < 1e-9);
    }

    #[test]
    fn test_likelihood_eliminate_loop() {
        let sizes = vec![1, 1, 1];
        let size_of = |k| sizes[k];

        let f0 = Factor::new(
            Layout::new(vec![1], size_of),
            dmatrix![1.0],
            dvector![0.2],
            dmatrix![1.0],
        );
        let f1 = Factor::new(
            Layout::new(vec![0, 1], size_of),
            dmatrix![1.0, 1.0],
            dvector![0.06],
            dmatrix![0.6],
        );
        let f2 = Factor::new(
            Layout::new(vec![0, 2], size_of),
            dmatrix![1.0, 1.0],
            dvector![0.8],
            dmatrix![1.0],
        );
        let f3 = Factor::new(
            Layout::new(vec![1, 2], size_of),
            dmatrix![1.0, 1.0],
            dvector![0.5],
            dmatrix![0.5],
        );

        // the total observation dimension is larger than the total dimension of the kept variables
        // the factor should close a
        let new_factor = eliminate([&f0, &f1, &f2, &f3], &[0], &[1, 2], size_of).unwrap();

        assert!(new_factor.dim_y() <= new_factor.dim_x());
        assert_eq!(new_factor.keys(), [0]);
        assert!((new_factor.xiby()[0] - -0.051724137931034475f64).abs() < 1e-9);
        assert!((new_factor.wby()[0] - 0.8965517241379309f64).abs() < 1e-9);
    }

    // further test:
    // Branch equivalence: run the same inputs through eliminate_to_z and eliminate_to_x, then compare Aᵀξ and AᵀWA. Use blocks of different sizes (for example 2 and 3), since equal sizes would hide the transposition bug.
    // Scalar sanity case: one likelihood y = x + u with W = w, and a prior on u with precision p. The expected result is Λ'_xx = wp/(w+p).
    // Edge cases: no cleared keys, a prior on a kept key plus a likelihood on the same key, and a prior on a cleared key.
}

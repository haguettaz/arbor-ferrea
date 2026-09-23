use std::fmt::Display;
use std::hash::Hash;

use anyhow::{Context, Result, bail};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::{cholesky::llt, matmul::matmul};
use faer::prelude::*;
use faer::{Accum, Par, Spec};
use std::collections::HashSet;

use crate::vars::VarInfo;

/// A linear block Y = A X in a Gaussian factor graph.
/// X is the main variable constrained through Y by A.
/// Note: in a future version, we might generalize A to a real-valued bilinear operator acting on real-valued vectors and symmetric matrices.
pub struct AXGaussBlock<T> {
    // ---- Labels and Sizes ----
    pub info_x: Vec<(VarInfo<T>, usize)>, // the informations about the x variables and their positions in the A matrix

    // ---- Logical Dimensions ----
    pub dim_x: usize, // note: should be equal to the total size of the x variables
    pub dim_y: usize,

    // ---- Matrices and Vectors ----
    pub a: Mat<f64>,    // the matrix A with dimensions (dim_y, dim_x)
    pub xiby: Col<f64>, // y backward weighted mean vector with shape (dim_y)
    pub wby: Mat<f64>,  // y backward precision matrix with shape (dim_y, dim_y)
}

impl<T: PartialEq + Clone> AXGaussBlock<T> {
    pub fn new(
        info_x: Vec<(VarInfo<T>, usize)>,
        dim_x: usize,
        dim_y: usize,
        a: Mat<f64>,
        xiby: Col<f64>,
        wby: Mat<f64>,
    ) -> Self {
        Self {
            info_x,
            dim_x,
            dim_y,
            a,
            xiby,
            wby,
        }
    }
}

/// An elimination block in a Gaussian factor graph.
/// The main variable X and separator variable S are associated with the matrices A and B, respectively.
/// For convenience, the virtual variables Y := A X + Z and Z := B S are introduced.
pub struct AXBSGaussBlock<T> {
    // ---- Labels and Sizes ----
    pub info_x: VarInfo<T>, // the informations about the x variable
    pub info_s: Vec<(VarInfo<T>, usize)>, // the informations about the s variables and their positions in the B matrix

    // ---- Logical Dimensions ----
    pub dim_x: usize,
    pub dim_s: usize,
    pub dim_y: usize,

    // ---- Matrices and Vectors ----
    a: Mat<f64>,     // with shape (dim_y, dim_x)
    b: Mat<f64>,     // with shape (dim_y, dim_s)
    xifx: Col<f64>,  // x forward weighted mean vector with shape (dim_x)
    wfx: Mat<f64>,   // x forward precision matrix with shape (dim_x, dim_x)
    xiby: Col<f64>,  // y backward weighted mean vector with shape (dim_y)
    wby: Mat<f64>,   // y backward precision matrix with shape (dim_y, dim_y)
    wby_a: Mat<f64>, // with shape (dim_y, dim_x)
    xitx: Col<f64>,  // x dual weighted mean vectors with shape (dim_x)
    rtx: Mat<f64>,   // x dual square-root precision matrix with shape (dim_x, dim_x)
    xibz: Col<f64>,  // z backward weighted mean vectors with shape (dim_y)
    wbz: Mat<f64>,   // z backward precision matrix with shape (dim_y, dim_y)

    // ---- Workspaces ----
    h_xitx: Col<f64>,   // (dim_x)
    h_at_wby: Mat<f64>, // (dim_x, dim_y)
    b_s: Col<f64>,      // (dim_y)
    mem_buf: MemBuffer,
}

impl<T: PartialEq + Clone + Display + Eq + Hash> AXBSGaussBlock<T> {
    pub fn new(
        info_x: VarInfo<T>,               // the informations about the x variable
        info_s: Vec<(VarInfo<T>, usize)>, // the informations about the s variables and their positions in the B matrix
        dim_x: usize,
        dim_s: usize,
        dim_y: usize,
        a: Mat<f64>,
        b: Mat<f64>,
        xifx: Col<f64>,
        wfx: Mat<f64>,
        xiby: Col<f64>,
        wby: Mat<f64>,
    ) -> Self {
        let wby_a = &wby * &a;

        let xitx = Col::zeros(dim_x);
        let rtx = Mat::zeros(dim_x, dim_x);
        let xibz = Col::zeros(dim_y);
        let wbz = Mat::zeros(dim_y, dim_y);

        let h_xitx = Col::zeros(dim_x);
        let h_at_wby = Mat::zeros(dim_x, dim_y);
        let b_s = Col::zeros(dim_y);

        // Compute the size and alignment of the required scratch space for Cholesky decompositions and associated solver
        let cholesky_memory =
            llt::factor::cholesky_in_place_scratch::<f64>(dim_x, Par::Seq, Spec::default());
        let solve_memory = llt::solve::solve_in_place_scratch::<f64>(dim_x, dim_y, Par::Seq);

        // Allocate the scratch space
        let mem_buf = MemBuffer::new(cholesky_memory.or(solve_memory));

        Self {
            info_x,
            info_s,
            dim_x,
            dim_s,
            dim_y,
            a,
            b,
            xifx,
            wfx,
            xiby,
            wby,
            wby_a,
            xitx,
            rtx,
            xibz,
            wbz,
            h_xitx,
            h_at_wby,
            b_s,
            mem_buf,
        }
    }

    /// Build an [`AXBSGaussBlock`] from a [`VarInfo`] and a slice of [`AXGaussBlock`]s.
    /// Warning: the x variable must be present in all blocks.
    pub fn from(info_x: VarInfo<T>, gauss_blocks: &[AXGaussBlock<T>]) -> Self {
        // Partition Gaussian blocks into X and X+S blocks
        let (gauss_blocks_xs, gauss_blocks_x): (Vec<&AXGaussBlock<T>>, Vec<&AXGaussBlock<T>>) =
            gauss_blocks
                .iter()
                .partition(|block| block.info_x.len() > 1);

        // Build the separator variable information
        let info_s = Self::build_info_s(&info_x, &gauss_blocks_xs);

        // Compute the logical dimensions
        let dim_x = info_x.size;
        let dim_s = info_s.iter().map(|(info_u, _)| info_u.size).sum::<usize>();
        let dim_y = gauss_blocks_xs.iter().map(|b| b.dim_y).sum::<usize>();

        // Create block from Gaussian blocks on X only
        let xifx = Self::build_xifx(dim_x, &gauss_blocks_x);
        let wfx = Self::build_wfx(dim_x, &gauss_blocks_x);

        // Create block from Gaussian blocks on X and S variables
        let a = Self::build_a(dim_x, dim_y, &info_x, &gauss_blocks_xs);
        let b = Self::build_b(dim_s, dim_y, &info_s, &gauss_blocks_xs);
        let xiby = Self::build_xiby(dim_y, &gauss_blocks_xs);
        let wby = Self::build_wby(dim_y, &gauss_blocks_xs);

        Self::new(
            info_x, info_s, dim_x, dim_s, dim_y, a, b, xifx, wfx, xiby, wby,
        )
    }

    fn build_info_s(
        info_x: &VarInfo<T>,
        gauss_blocks_xs: &[&AXGaussBlock<T>],
    ) -> Vec<(VarInfo<T>, usize)> {
        let mut info_s: Vec<(VarInfo<T>, usize)> = Vec::new();
        let mut seen_labels: HashSet<T> = HashSet::new();

        let mut offset = 0;
        for block in gauss_blocks_xs {
            for (info_u, _) in &block.info_x {
                if info_u.label != info_x.label && !seen_labels.contains(&info_u.label) {
                    seen_labels.insert(info_u.label.clone());
                    info_s.push((info_u.clone(), offset));
                    offset += info_u.size;
                }
            }
        }
        info_s
    }

    fn build_xifx(dim_x: usize, ax_gauss_blocks: &[&AXGaussBlock<T>]) -> Col<f64> {
        // Compute xifx
        let mut xifx = Col::zeros(dim_x);

        for block in ax_gauss_blocks {
            matmul(
                xifx.rb_mut(),
                Accum::Add,
                block.a.transpose(),
                &block.xiby,
                1.0,
                Par::Seq,
            );
        }

        xifx
    }

    fn build_wfx(dim_x: usize, ax_gauss_blocks: &[&AXGaussBlock<T>]) -> Mat<f64> {
        // Allocate temporary storage matrix for matmul operations
        let max_dim_y = ax_gauss_blocks
            .iter()
            .map(|block| block.dim_y)
            .max()
            .unwrap_or(0);
        let mut tmp_storage = Mat::zeros(dim_x, max_dim_y);

        let mut wfx = Mat::zeros(dim_x, dim_x);
        for block in ax_gauss_blocks {
            // Slice the columns up to the current block's dim_y
            let mut tmp = tmp_storage.as_mut().subcols_mut(0, block.dim_y);
            matmul(
                &mut tmp,
                Accum::Replace,
                &block.a.transpose(),
                &block.wby,
                1.0,
                Par::Seq,
            );
            matmul(wfx.rb_mut(), Accum::Add, tmp, &block.a, 1.0, Par::Seq);
        }

        wfx
    }

    fn build_a(
        dim_x: usize,
        dim_y: usize,
        info_x: &VarInfo<T>,
        abxs_gauss_blocks: &[&AXGaussBlock<T>],
    ) -> Mat<f64> {
        // Create A matrix by block rows
        let mut a = Mat::zeros(dim_y, dim_x);
        let mut row = 0;
        for block in abxs_gauss_blocks {
            // Extract the block rows corresponding to this Gaussian block
            let mut a_rows = a.as_mut().subrows_mut(row, block.dim_y);

            // Extract the block column corresponding to the main variable x
            if let Some((block_info_x, block_offset_x)) = block
                .info_x
                .iter()
                .filter(|(block_info_var, _)| {
                    (block_info_var.label == info_x.label) && (block_info_var.size == info_x.size)
                })
                .next()
            {
                a_rows.copy_from(&block.a.subcols(*block_offset_x, block_info_x.size));
            }

            row += block.dim_y;
        }

        a
    }

    fn build_b(
        dim_s: usize,
        dim_y: usize,
        info_s: &[(VarInfo<T>, usize)],
        abxs_gauss_blocks: &[&AXGaussBlock<T>],
    ) -> Mat<f64> {
        // Create B matrix by block rows
        let mut b = Mat::zeros(dim_y, dim_s);
        let mut row = 0;
        for block in abxs_gauss_blocks {
            // Extract the block rows corresponding to this Gaussian block
            let mut b_rows = b.as_mut().subrows_mut(row, block.dim_y);

            // Create block row one variable (u) at a time
            for (info_u, offset_u) in info_s {
                if let Some((block_info_u, block_offset_u)) = block
                    .info_x
                    .iter()
                    .filter(|(block_info_var, _)| {
                        (block_info_var.label == info_u.label)
                            && (block_info_var.size == info_u.size)
                    })
                    .next()
                {
                    // Extract the block column corresponding to this variable (u)
                    let mut b_block = b_rows.as_mut().subcols_mut(*offset_u, info_u.size);
                    b_block.copy_from(&block.a.subcols(*block_offset_u, block_info_u.size));
                }
            }

            row += block.dim_y;
        }
        b
    }

    fn build_xiby(dim_y: usize, abxs_gauss_blocks: &[&AXGaussBlock<T>]) -> Col<f64> {
        let mut xiby = Col::zeros(dim_y);
        let mut row = 0;
        for block in abxs_gauss_blocks {
            // Extract the block rows corresponding to this Gaussian block
            let mut xiby_rows = xiby.as_mut().subrows_mut(row, block.dim_y);
            xiby_rows.copy_from(&block.xiby);
            row += block.dim_y;
        }
        xiby
    }

    fn build_wby(dim_y: usize, abxs_gauss_blocks: &[&AXGaussBlock<T>]) -> Mat<f64> {
        let mut wby = Mat::zeros(dim_y, dim_y);
        let mut diag = 0;
        for block in abxs_gauss_blocks {
            // Extract the block rows corresponding to this Gaussian block
            let mut wby_block = wby
                .as_mut()
                .submatrix_mut(diag, diag, block.dim_y, block.dim_y);
            wby_block.copy_from(&block.wby);
            diag += block.dim_y;
        }
        wby
    }

    pub fn eliminate_x(&mut self) -> Result<()> {
        // should rather return a CompactLinearGaussBlock
        let mut stack = MemStack::new(&mut self.mem_buf);

        // Compute xitx = xifx + a.t xiby, which is also used in solve_x
        self.xitx.copy_from(&self.xifx);
        matmul(
            &mut self.xitx,
            Accum::Add,
            self.a.transpose(),
            &self.xiby,
            1.0,
            Par::Seq,
        );

        // Compute wtx = (wfx + a.t wby a) and its Cholesky decomposition rtx, the later being also used in solve_x
        self.rtx.copy_from(&self.wfx);
        matmul(
            &mut self.rtx,
            Accum::Add,
            self.a.transpose(),
            &self.wby_a,
            1.0,
            Par::Seq,
        );

        // compute the decomposition
        llt::factor::cholesky_in_place(
            self.rtx.as_mut(),
            llt::factor::LltRegularization::default(),
            Par::Seq,
            &mut stack,
            default(),
        )?;

        // Compute xibz = xiby - wby a h (xifx + a.t xiby) = xiby - wby_a h_xitx
        // where h_xitx = h (xifx + a.t xiby) with hinv = (wfx + a.t wby a)
        self.h_xitx.copy_from(&self.xitx);
        llt::solve::solve_in_place(
            self.rtx.as_ref(),
            self.h_xitx.as_mat_mut(),
            Par::Seq,
            &mut stack,
        );
        self.xibz.copy_from(&self.xiby);
        matmul(
            &mut self.xibz,
            Accum::Add,
            &self.wby_a,
            &self.h_xitx,
            -1.0,
            Par::Seq,
        );

        // Compute wbz = wby - wby a h a.t wby
        self.h_at_wby.copy_from(&self.wby_a.transpose());
        llt::solve::solve_in_place(
            self.rtx.as_ref(),
            self.h_at_wby.as_mut(),
            Par::Seq,
            &mut stack,
        );
        self.wbz.copy_from(&self.wby);
        matmul(
            &mut self.wbz,
            Accum::Add,
            &self.wby_a,
            &self.h_at_wby,
            -1.0,
            Par::Seq,
        );

        Ok(())
    }

    pub fn solve_x(&mut self, mut x: ColMut<f64>, s: ColRef<f64>) -> Result<()> {
        let mut stack = MemStack::new(&mut self.mem_buf);

        matmul(&mut self.b_s, Accum::Replace, &self.b, s, 1.0, Par::Seq);

        x.copy_from(&self.xitx);
        matmul(
            x.rb_mut(),
            Accum::Add,
            self.wby_a.transpose(),
            &self.b_s,
            -1.0,
            Par::Seq,
        );

        llt::solve::solve_in_place(self.rtx.as_ref(), x.as_mat_mut(), Par::Seq, &mut stack);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axbs_gauss_block_from() {
        let info_x = VarInfo::new("x1", 3);

        let gauss_blocks = vec![
            AXGaussBlock::new(
                vec![
                    (VarInfo::new("x3", 3), 0),
                    (VarInfo::new("x1", 3), 3),
                    (VarInfo::new("x2", 2), 6),
                ],
                8,
                3,
                mat![
                    [1.31, 1.32, 1.33, 1.11, 1.12, 1.13, 1.21, 1.22],
                    [2.31, 2.32, 2.33, 2.11, 2.12, 2.13, 2.21, 2.22],
                    [3.31, 3.32, 3.33, 3.11, 3.12, 3.13, 3.21, 3.22]
                ],
                col![1.0, 2.0, 3.0],
                mat![[11.0, 0.0, 0.0], [0.0, 22.0, 0.0], [0.0, 0.0, 33.0]],
            ),
            AXGaussBlock::new(
                vec![(VarInfo::new("x4", 4), 0), (VarInfo::new("x1", 3), 4)],
                7,
                1,
                mat![[4.41, 4.42, 4.43, 4.44, 4.11, 4.12, 4.13]],
                col![4.0],
                mat![[44.0]],
            ),
            AXGaussBlock::new(
                vec![(VarInfo::new("x1", 3), 0)],
                3,
                2,
                mat![[5.11, 5.12, 5.13], [6.11, 6.12, 6.13]],
                col![5.0, 6.0],
                mat![[55.0, 56.0], [56.0, 66.0]],
            ),
        ];

        let axbs_gauss_block = AXBSGaussBlock::from(info_x, &gauss_blocks);

        // check info_x
        assert_eq!(axbs_gauss_block.info_x.label, "x1");
        assert_eq!(axbs_gauss_block.info_x.size, 3);

        // check info_s
        assert_eq!(axbs_gauss_block.info_s.len(), 3);
        assert_eq!(axbs_gauss_block.info_s[0].0.label, "x3");
        assert_eq!(axbs_gauss_block.info_s[0].0.size, 3);
        assert_eq!(axbs_gauss_block.info_s[0].1, 0);
        assert_eq!(axbs_gauss_block.info_s[1].0.label, "x2");
        assert_eq!(axbs_gauss_block.info_s[1].0.size, 2);
        assert_eq!(axbs_gauss_block.info_s[1].1, 3);
        assert_eq!(axbs_gauss_block.info_s[2].0.label, "x4");
        assert_eq!(axbs_gauss_block.info_s[2].0.size, 4);
        assert_eq!(axbs_gauss_block.info_s[2].1, 5);

        // check logical dimensions
        assert_eq!(axbs_gauss_block.dim_x, 3);
        assert_eq!(axbs_gauss_block.dim_s, 9);
        assert_eq!(axbs_gauss_block.dim_y, 4);

        // check x-factors
        let a = mat![[5.11, 5.12, 5.13], [6.11, 6.12, 6.13]];
        let xiby = col![5.0, 6.0];
        let wby = mat![[55.0, 56.0], [56.0, 66.0]];
        let xifx = a.transpose() * xiby;
        zip!(&xifx, &axbs_gauss_block.xifx).for_each(|unzip!(a, b)| assert!((a - b).abs() < 1e-9));
        let wfx = a.transpose() * wby * a;
        zip!(&wfx, &axbs_gauss_block.wfx).for_each(|unzip!(a, b)| assert!((a - b).abs() < 1e-9));

        // check x-s factors
        let a = mat![
            [1.11, 1.12, 1.13],
            [2.11, 2.12, 2.13],
            [3.11, 3.12, 3.13],
            [4.11, 4.12, 4.13]
        ];
        zip!(&a, &axbs_gauss_block.a).for_each(|unzip!(a, b)| assert!((a - b).abs() < 1e-9));
        let b = mat![
            [1.31, 1.32, 1.33, 1.21, 1.22, 0.00, 0.00, 0.00, 0.00],
            [2.31, 2.32, 2.33, 2.21, 2.22, 0.00, 0.00, 0.00, 0.00],
            [3.31, 3.32, 3.33, 3.21, 3.22, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 4.41, 4.42, 4.43, 4.44]
        ];
        zip!(&b, &axbs_gauss_block.b).for_each(|unzip!(a, b)| assert!((a - b).abs() < 1e-9));
        let xiby = col![1.0, 2.0, 3.0, 4.0];
        zip!(&xiby, &axbs_gauss_block.xiby).for_each(|unzip!(a, b)| assert!((a - b).abs() < 1e-9));
        let wby = mat![
            [11.0, 0.0, 0.0, 0.0],
            [0.0, 22.0, 0.0, 0.0],
            [0.0, 0.0, 33.0, 0.0],
            [0.0, 0.0, 0.0, 44.0]
        ];
        zip!(&wby, &axbs_gauss_block.wby).for_each(|unzip!(a, b)| assert!((a - b).abs() < 1e-9));
    }
}

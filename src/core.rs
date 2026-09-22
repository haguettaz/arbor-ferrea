use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use nalgebra::{DMatrix, DVector};

// use crate::gmp::{DownGaussMsg, MultiGaussMsg, SingleGaussMultiCondMsg};
use crate::gmp::ObsBlock;
use crate::linalg::*;

type VarId = usize;

/// Represents a Gaussian factor graph with a tree structure.
pub struct GaussTree<T> {
    // index: HashMap<T, VarId>,
    // inverse: HashMap<VarId, T>,
    roots: Vec<VarId>,
    parents: Vec<VarId>,
    children: Vec<Vec<VarId>>,
    separators: Vec<Vec<VarId>>,
    nodes: Vec<Node>,
}

impl GaussTree {
    /// Upward pass from the leaves to the root(s)
    /// Variables are indexed by their elimination order.
    /// Can be interpreted as max-product message passing in the original factor graph
    pub fn build(vars: &[VarId], gfs: &[GaussFactor]) -> Self {
        // // We assume all variables in the factors are present in the order!!

        // // Map each variable to its elimination order index: O(1) lookups
        // let order_rank: HashMap<&T, usize> = order
        //     .iter()
        //     .enumerate()
        //     .map(|(idx, var)| (var, idx))
        //     .collect();

        // // Pool of active factors represented as sets of variable indices
        // let mut gfs_pool: Vec<HashSet<usize>> = gfs
        //     .iter()
        //     .map(|gf| {
        //         gf.vars
        //             .iter()
        //             .filter_map(|v| order_rank.get(v).copied())
        //             .collect::<HashSet<usize>>()
        //     })
        //     .collect();

        // Map each variable to its parent variable's index
        let mut parents: Vec<VarId> = Vec::with_capacity(vars.len());
        let mut children: Vec<Vec<VarId>> = Vec::with_capacity(vars.len());
        let mut separators: Vec<Vec<VarId>> = Vec::with_capacity(vars.len());
        let mut nodes: Vec<Node> = Vec::with_capacity(vars.len());

        let mut gfs_pool: Vec<GaussFactor> = gfs.to_vec();

        // Eliminate variables sequentially
        for x in vars.iter() {
            // Store the Gaussian message for the current variable conditioned on its parents
            // Create the conditional Gaussian message (downward, from its parents)

            // Partition factors: those containing current variable vs remaining factors
            let (x_gfs, other_gfs) = gfs.iter().partition(|f| f.contains(*x));
            // Partition factors: those with multiple variables vs single-variable factors
            let (xs_gfs, xx_gfs) = x_gfs.iter().partition(|f| f.nvars() > 1);

            // Compute gaussian prior from xx_gfs
            // let mfx = ...;
            // let vfx = ...;

            // Compute observation block from xs_gfs
            //
            // let obs_block = ObsBlock::new();

            let node = Node::new(mfx, vfx, obs_block);

            //
            let mut rem_gfs = Vec::new();
            for gf in tmp_gfs {
                if gf.contains(var) {
                    // Collect all other uneliminated variables connected to the variable
                    // for v in gf.vars_iter() {
                    //     if order_rank[&v] > i {
                    //         sep_set.insert(v);
                    //     }
                    // }
                    down_gmsg.combine(&gf);
                } else {
                    rem_gfs.push(gf);
                }
            }
            tmp_gfs = rem_gfs;

            // The parent is the variable in the separator with the lowest order rank
            if let Some(&parent) = down_gmsg.cond_vars_iter().min_by_key(|v| order_rank[v]) {
                parents.insert(*var, *parent);
                if let Some(new_gf) = down_gmsg.get_reduction() {
                    tmp_gfs.push(new_gf);
                }
            }

            // Add fill-in edge (Schur complement / message) back to active factors
            if !separator_set.is_empty() {
                // Note: we should also propagate the factor of the separator set (with variable `i` eliminated)
                factor_pool.push(separator_set);
                // push factor corresponding to dmsg..
            }
        }

        // Convert internal indices back into the original generic type T
        let mut children: HashMap<T, Vec<T>> = HashMap::new();
        let mut parents: HashMap<T, T> = HashMap::new();
        let mut roots = Vec::new();

        for (idx, var) in order.iter().enumerate() {
            children.entry(var.clone()).or_default();
            if let Some(&p_idx) = parents_idx.get(&idx) {
                let parent_var = &order[p_idx];
                children
                    .entry(parent_var.clone())
                    .or_default()
                    .push(var.clone());
                parents.insert(var.clone(), parent_var.clone());
            } else {
                roots.push(var.clone());
            }
        }

        Self {
            nodes: HashMap::new(),
            roots,
            children,
            parents,
        }
    }

    /// Downward pass from the root(s) to the leaves = solve
    pub fn solve(&self, out: &mut Vec<DVector<f64>>) {
        for r in self.roots.iter() {
            self.down(*r, out)
        }
    }

    /// Decide variable of node i and propagate to its children
    fn down(&self, i: VarId, out: &mut Vec<DVector<f64>>) {
        let ms = concat_vectors(self.separators[i].iter().map(|&e| &out[e]));
        out[i] = self.nodes[i].decide(&ms);
        for j in self.children[i].iter() {
            self.down(*j, out);
        }
    }
}

pub struct Node {
    // dim: usize,
    mfx: DVector<f64>,
    vfx: DMatrix<f64>,
    obs_block: ObsBlock,
}

impl Node {
    pub fn new(mfx: DVector<f64>, vfx: DMatrix<f64>, obs_block: ObsBlock) -> Self {
        Self {
            mfx,
            vfx,
            obs_block,
        }
    }

    pub fn decide(&self, ms: &DVector<f64>) -> DVector<f64> {
        self.obs_block.forward_m_only(&self.mfx, &self.vfx, ms)
    }
}

pub struct GaussFactor {
    vars: Vec<(VarId, usize)>, // variables and their dimensions
    a: DMatrix<f64>,
    mby: DVector<f64>,
    vby: DMatrix<f64>,
}

impl GaussFactor {
    pub fn contains(&self, var: VarId) -> bool {
        self.vars.iter().any(|(v, _)| *v == var)
    }

    pub fn nvars(&self) -> usize {
        self.vars.len()
    }
}

impl GaussFactor {
    pub fn combine(&self, other: &Self) -> Self {
        // Collect all unique variables and determine unique column layout.
        let mut vars: HashMap<T, (usize, usize)> = HashMap::new();
        let mut total_cols = 0;
        for (var, &(_, dim)) in &self.vars {
            if !vars.contains_key(var) {
                vars.insert(var.clone(), (total_cols, dim));
                total_cols += dim;
            }
        }
        for (var, &(_, dim)) in &other.vars {
            if !vars.contains_key(var) {
                vars.insert(var.clone(), (total_cols, dim));
                total_cols += dim;
            }
        }

        // Allocate matrix filled with zeros
        let self_rows = self.mat.nrows();
        let other_rows = other.mat.nrows();
        let total_rows = self_rows + other_rows;
        let mut mat = DMatrix::<f64>::zeros(total_rows, total_cols);

        // Copy blocks from self.mat into the top block row: [0..self_rows]
        for (var, &(self_col_start, dim)) in &self.vars {
            let (col_start, _) = vars.get(var).unwrap();
            let self_block = self
                .mat
                .view_range(.., self_col_start..self_col_start + dim);
            let mut target_slice = mat.view_range_mut(0..self_rows, *col_start..*col_start + dim);
            target_slice.copy_from(&self_block);
        }

        // Copy blocks from other.mat into the top block row: [self_rows..total_rows]
        for (var, &(other_col_start, dim)) in &other.vars {
            let (col_start, _) = vars.get(var).unwrap();
            let other_block = other
                .mat
                .view_range(.., other_col_start..other_col_start + dim);
            let mut target_slice =
                mat.view_range_mut(self_rows..total_rows, *col_start..*col_start + dim);
            target_slice.copy_from(&other_block);
        }

        Self {
            vars,
            mat,
            mean,
            covariance,
        }
    }
}

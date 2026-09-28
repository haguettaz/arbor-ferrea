use faer_traits::RealField;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter, Result as FmtResult};

use anyhow::{Context, Result, bail};

type VarId = usize;

pub trait NodeFactor<V: RealField> {
    /// The primary variable eliminated/solved by this node.
    fn main_var(&self) -> VarId;

    /// The value of the primary variable.
    fn main_val(&self) -> V;

    /// The variables connecting this node to its parent in the Bayes tree.
    fn sep_vars(&self) -> &[VarId];

    /// Upward pass (max-product).
    /// Marginalizes out `main` and returns a newly formed factor involving only the `separators`.
    fn eliminate(&mut self) -> Result<Box<dyn Factor>>;

    /// Downward pass (deciding).
    /// Computes the optimal estimate for `main`, given known values for the `separators`.
    fn decide(&mut self, sep_vals: &[V]) -> Result<()>;
}

pub trait Factor {
    /// Variables involved in this factor constraint
    fn vars(&self) -> &[VarId];
}

enum AnyFactor<'a> {
    Ref(&'a dyn Factor),
    Owned(Box<dyn Factor + 'a>),
}

impl<'a> std::ops::Deref for AnyFactor<'a> {
    type Target = dyn Factor + 'a;
    fn deref(&self) -> &Self::Target {
        match self {
            AnyFactor::Ref(r) => *r,
            AnyFactor::Owned(o) => o.as_ref(),
        }
    }
}

pub struct TreeNode<V> {
    pub factor: Box<dyn NodeFactor<V>>,
    pub children: Vec<TreeNode<V>>,
}

/// Represents a tree structure
pub struct Tree<V> {
    pub roots: Vec<TreeNode<V>>,
}

impl<V: RealField> Tree<V> {
    pub fn new(roots: Vec<TreeNode<V>>) -> Self {
        Self { roots }
    }

    /// Two-phase build process for a Bayes Tree.
    /// Phase 1: Symbolic build (fast, single-threaded)
    /// Phase 2: Numeric build (heavy-lifting, multi-threaded)
    pub fn build<'a, F>(
        factors: impl IntoIterator<Item = &'a dyn Factor>, // factors constraining the variables
        order: &[VarId],                                   // variable elimination order,
        make_tree_factor: F, // function to create a NodeFactor from a variable and a list of factors constraining it
    ) -> Result<Self>
    where
        F: Fn(VarId, Vec<&dyn Factor>) -> Box<dyn NodeFactor<V>>,
    {
        // Wrap original references in the AnyFactor enum
        let mut all_factors: Vec<AnyFactor<'a>> = factors.into_iter().map(AnyFactor::Ref).collect();

        // Check that there is no duplicate variable
        let seen_in_order = order.iter().collect::<HashSet<&VarId>>();
        if seen_in_order.len() != order.len() {
            bail!("Duplicate variable found in variables");
        }
        let n_vars = order.len();

        // Check that all variables are present in the blocks
        let seen_in_factors = all_factors
            .iter()
            .flat_map(|factor| factor.vars().iter())
            .collect::<HashSet<&VarId>>();
        if seen_in_factors != seen_in_order {
            bail!("The variables in the variables do not match the variables in the blocks");
        }

        // Map each variable to its elimination order index: O(1) lookups
        let ranks: HashMap<VarId, usize> = order.iter().enumerate().map(|(r, i)| (*i, r)).collect();

        let mut nodes = Vec::with_capacity(n_vars);
        let mut roots = Vec::new();
        let mut parents = vec![None; n_vars];
        let mut children = vec![Vec::new(); n_vars];

        for cur_var in order.iter() {
            // Partition shuffles the AnyFactor
            let (cur_factors, rem_factors): (Vec<AnyFactor<'a>>, Vec<AnyFactor<'a>>) = all_factors
                .into_iter()
                .partition(|factor| factor.vars().iter().any(|var| var == cur_var));

            // Extract pure references for the constructor
            let cur_factors_ref: Vec<&dyn Factor> =
                cur_factors.iter().map(|factor| &**factor).collect();

            // Needs to be mutable so we can call eliminate_x
            let mut new_tree_factor = make_tree_factor(*cur_var, cur_factors_ref);

            let new_factor = new_tree_factor
                .eliminate()
                .with_context(|| format!("Problem when eliminating variable {}", cur_var))?;

            let parent = new_factor
                .vars()
                .iter()
                .filter_map(|var| ranks.get(var))
                .min();

            match parent {
                None => roots.push(*cur_var),
                Some(&p_var) => {
                    parents[*cur_var] = Some(p_var);
                    children[p_var].push(*cur_var);
                }
            }

            all_factors = rem_factors;
            all_factors.push(AnyFactor::Owned(new_factor));
            nodes.push(new_tree_factor);
        }

        Ok(Self::new(roots, parents, children, nodes))
    }

    /// Builds a symbolic tree from a list of factors.
    fn symbolic_build() {}

    fn numeric_build() {}

    /// Explores the tree from roots to leaves using an iterative Depth-First Search.
    /// Applies the provided closure to each visited node.
    pub fn traverse_down<F>(&self, mut visitor: F)
    where
        F: FnMut(VarId, &Box<dyn NodeFactor<V>>),
    {
        // Use an explicit stack to prevent stack overflows on very deep trees.
        // We seed it with all the root nodes.
        let mut stack: Vec<VarId> = self.roots.clone();

        while let Some(i) = stack.pop() {
            // Apply the closure to the current node
            visitor(i, &self.nodes[i]);

            // Push children onto the stack.
            // We reverse the iterator so the first child is popped/visited first.
            for &c in self.children[i].iter().rev() {
                stack.push(c);
            }
        }
    }

    /// Parallel downward pass from the root(s) to the leaves
    pub fn solve_par(&mut self) -> Result<()> {
        // Roots have no ancestors, so they start with an empty context
        let initial_context = HashMap::new();

        // Process all roots in parallel
        self.roots
            .par_iter_mut()
            .try_for_each(|root| Self::solve_node_par(root, &initial_context))
    }

    fn solve_node_par(node: &mut TreeNode<V>, ancestor_vals: &HashMap<VarId, V>) -> Result<()> {
        // 1. Gather the exact separator values this factor needs from the ancestors
        let sep_vars = node.factor.sep_vars();
        let mut sep_vals = Vec::with_capacity(sep_vars.len());

        for &sep_var in sep_vars {
            let val = ancestor_vals.get(&sep_var).with_context(|| {
                format!(
                    "Separator {} not evaluated before {}",
                    sep_var,
                    node.factor.main_var()
                )
            })?;
            sep_vals.push(val.clone());
        }

        // 2. Decide the current node's value (this mutates the factor)
        node.factor.decide(&sep_vals)?;

        // 3. If there are no children, we can return early to save clone operations
        if node.children.is_empty() {
            return Ok(());
        }

        // 4. Create a new context for the children by cloning the ancestor context
        // and adding this node's newly decided value.
        // (Cloning the HashMap is relatively cheap for the depth of a typical factor graph,
        // but can be optimized with an immutable persistent data structure like `im::HashMap` if needed).
        let mut child_context = ancestor_vals.clone();
        child_context.insert(node.factor.main_var(), node.factor.main_val());

        // 5. Process all children in parallel
        node.children
            .par_iter_mut()
            .try_for_each(|child| Self::solve_node_par(child, &child_context))
    }
}

impl<V: RealField + Display> Display for Tree<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut result = Ok(());

        self.traverse_down(|var, tree_factor| {
            // Only continue writing if earlier writes succeeded
            if result.is_ok() {
                if let Err(e) = writeln!(
                    f,
                    "variable {} has value {} and children {:?}",
                    var,
                    tree_factor.main_val(),
                    tree_factor.sep_vars()
                ) {
                    result = Err(e);
                }
            }
        });

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleFactor {
        vars: Vec<VarId>,
    }

    impl Factor for SimpleFactor {
        fn vars(&self) -> &[VarId] {
            &self.vars
        }
    }

    struct SimpleNodeFactor {
        main_var: VarId,
        main_val: f64,
        sep_vars: Vec<VarId>,
    }

    impl NodeFactor<f64> for SimpleNodeFactor {
        fn main_var(&self) -> VarId {
            self.main_var
        }

        fn main_val(&self) -> f64 {
            self.main_val
        }

        fn sep_vars(&self) -> &[VarId] {
            &self.sep_vars
        }

        fn eliminate(&mut self) -> Result<Box<dyn Factor>> {
            Ok(Box::new(SimpleFactor {
                vars: self.sep_vars.clone(),
            }))
        }

        fn decide(&mut self, sep_vals: &[f64]) -> Result<()> {
            self.main_val = sep_vals.iter().sum::<f64>();
            Ok(())
        }
    }

    #[test]
    fn test_build_tree() {
        let f0 = SimpleFactor { vars: vec![0] };
        let f1 = SimpleFactor { vars: vec![0, 1] };
        let f2 = SimpleFactor { vars: vec![1, 2] };
        let f3 = SimpleFactor { vars: vec![2, 3] };
        let f4 = SimpleFactor { vars: vec![0, 3] };

        let factors: [&dyn Factor; 5] = [&f0, &f1, &f2, &f3, &f4];

        let tree = Tree::build(factors, &[0, 1, 2, 3], |i, factors| {
            let sep_vars = factors
                .iter()
                .flat_map(|factor| {
                    factor
                        .vars()
                        .iter()
                        .filter_map(|&v| if v != i { Some(v) } else { None })
                })
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            Box::new(SimpleNodeFactor {
                main_var: i,
                main_val: 0.0,
                sep_vars: sep_vars,
            })
        })
        .unwrap();
    }
}

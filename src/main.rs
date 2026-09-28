use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Debug)]
pub struct EliminationTree<T> {
    pub roots: Vec<T>,
    pub children: HashMap<T, Vec<T>>,
    pub parents: HashMap<T, T>,
}

pub fn build_elimination_tree<T>(order: &[T], edges: &[Vec<T>]) -> EliminationTree<T>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
{
    // Map each variable to its elimination order index: O(1) lookups
    let order_rank: HashMap<&T, usize> = order
        .iter()
        .enumerate()
        .map(|(idx, var)| (var, idx))
        .collect();

    println!("Order: {:?}", order);

    // Pool of active factors represented as sets of variable indices
    let mut factor_pool: Vec<HashSet<usize>> = edges
        .iter()
        .map(|edge| {
            edge.iter()
                .filter_map(|v| order_rank.get(v).copied())
                .collect::<HashSet<usize>>()
        })
        .collect();

    // Map each variable to its parent variable's index
    let mut parents_idx: HashMap<usize, usize> = HashMap::new();

    // Eliminate variables sequentially
    for (i, _) in order.iter().enumerate() {
        // Partition factors: those containing variable `i` vs remaining factors
        let mut separator_set: HashSet<usize> = HashSet::new();
        let mut remaining_pool = Vec::new();

        for factor in factor_pool {
            if factor.contains(&i) {
                // Collect all other uneliminated variables connected to `i`
                for &v in &factor {
                    if v > i {
                        // store also the factor
                        separator_set.insert(v);
                    }
                }
            } else {
                remaining_pool.push(factor);
            }
        }
        factor_pool = remaining_pool;

        println!("Variables: {:?}", i);
        println!("Separator: {:?}", separator_set);
        println!("Factor pool 1: {:?}", factor_pool);

        // The parent of `i` is the variable in the separator with the lowest index
        if let Some(&parent_idx) = separator_set.iter().min() {
            parents_idx.insert(i, parent_idx);
        }

        // Add fill-in edge (Schur complement / message) back to active factors
        if !separator_set.is_empty() {
            // Note: we should also propagate the factor of the separator set (with variable `i` eliminated)
            factor_pool.push(separator_set);
        }
        println!("Factor pool 2: {:?}", factor_pool);
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

    EliminationTree {
        roots,
        children,
        parents,
    }
}

fn main() {
    // index: HashMap<T, VarId>,
    // inverse: HashMap<VarId, T>,

    // 1. Elimination Order: l1 -> l2 -> x0 -> x1 -> x2
    let order = vec!["x1", "l1", "x0", "x2", "l2"];
    // let order = vec!["l1", "l2", "x0", "x1", "x2"];

    // 2. Multi-edges (can have 1, 2, 3, or any number of variables)
    let edges = vec![
        vec!["x0"],
        vec!["x0", "x1"],
        vec!["x1", "x2"],
        vec!["l1", "x0"],
        vec!["l1", "x1"],
        vec!["l2", "x2"],
        // Multi-dimensional factor example:
        // vec!["x0", "x1", "x2"],
    ];

    let tree = build_elimination_tree(&order, &edges);

    println!("Root(s): {:?}", tree.roots);
    println!("\nDirected parent-to-child relationships:");
    for (parent, children) in &tree.children {
        if !children.is_empty() {
            println!("  {parent} -> {:?}", children);
        }
    }

    println!("\nParent pointers:");
    for (child, parent) in &tree.parents {
        println!("  parent({child}) = {parent}");
    }
}

// Should we use a BTreeMap??

// A node in the Bayes tree corresponding to a variable (e.g., robot pose or landmark)
// Should support reordering
// Contains conditional dependencies on parent clique (sparse block rows)
// Estimates are stored in the clique

// Objects to be considered:
// 1. Factor represent constraints between variables; for now we only have factor in one (prior) or two variables (odometry, loop closures, landmarks)
// 2. Variable represent the state variables (robot pose, landmark positions)
// 3. BayesTree represent the square root matrix R of the problem. A Bayes tree is made of cliques (frontal ; separator), organized in a tree structure.
//
//
//

// Algorithm: iSAM2_Update(FactorGraph, NewFactors, CurrentEstimate, BayesTree, Thresholds)
// Input:
//     NewFactors       : Set of newly observed factor measurements (odometry, loop closures, landmarks)
//     CurrentEstimate  : Current linearization points / variable estimates Θ
//     BayesTree        : Current Bayes tree T representing the square root matrix R
//     Thresholds       : Relinearization threshold β, check frequency

// Output:
//     Updated BayesTree and CurrentEstimate Θ

// 1.  // Step 1: Add new factors to the nonlinear factor graph
//     FactorGraph.add(NewFactors)
//     Initialize newly observed variables in CurrentEstimate

// 2.  // Step 2: Check for variables needing relinearization
//     MarkedVars = empty_set()
//     for variable j in BayesTree:
//         Δj = ComputeDelta(j, BayesTree)    // Current delta from linearization point
//         if ||Δj|| > Thresholds.β:
//             MarkedVars.insert(j)

// 3.  // Step 3: Identify affected variables and factors
//     // Variables involved in new factors
//     InvolvedVars = Variables(NewFactors) ∪ MarkedVars

//     // Find all clique subtrees affected by these variables up to the roots
//     AffectedCliques = BayesTree.findSubtreesAffecting(InvolvedVars)

// 4.  // Step 4: Convert affected cliques back into a factor graph
//     // Remove the affected cliques from the Bayes tree
//     OrphanSubtrees = BayesTree.removeCliques(AffectedCliques)

//     // Reconstruct linear/nonlinear factors from the removed cliques
//     LinearFactors = FactorGraph.getFactorsFor(InvolvedVars)
//     for clique C in AffectedCliques:
//         LinearFactors.add(C.toFactor())

// 5.  // Step 5: Linearize factors containing marked variables
//     for factor f in LinearFactors:
//         if f.containsAny(MarkedVars) or f in NewFactors:
//             Linearize f around CurrentEstimate

// 6.  // Step 6: Variable Elimination (Reordering & Factorization)
//     // Order affected variables (e.g., using COLAMD constrained by top-of-tree structure)
//     Ordering = ConstrainedColamd(LinearFactors, InvolvedVars)

//     // Eliminate variables to obtain a new set of cliques (chordal Bayes net)
//     NewCliques = MultifrontalElimination(LinearFactors, Ordering)
//     if Sj+1 union xj+1 = Sj, then merge clique Cj and Cj+1

// 7.  // Step 7: Re-integrate into the Bayes Tree
//     // Rebuild the top of the Bayes tree using the new cliques
//     NewSubtree = BuildBayesTree(NewCliques)

//     // Re-attach orphaned subtrees to the new cliques
//     BayesTree.attach(NewSubtree, OrphanSubtrees)

// 8.  // Step 8: Update Estimate (Back-substitution)
//     // Solves only the top of the tree, or full tree if requested
//     Δ = BayesTree.optimize(OnlyAffected=False) // Solves R * Δ = d via back-substitution
//     CurrentEstimate = CurrentEstimate ⊕ Δ

// 9.  return CurrentEstimate, BayesTree

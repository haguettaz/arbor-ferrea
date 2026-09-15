fn main() {
    println!("Hello, world!");
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

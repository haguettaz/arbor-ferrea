# Forward–Backward Filtering Architecture

**Shared State:** a centralized registry of all initial factors, accessible across all computational nodes.

## Computation Node Structure

Each computation node maintains:
* **Target variable:** the primary variable assigned to this node.
* **Elimination set:** the running set of eliminated variables propagated through the tree.
* **Message cache:** the local storage for outgoing messages generated in both directions (forward and backward).

## Message Passing Protocol

When a computation node propagates information to a child node, it transmits:
* Pointers to initial factors that have not yet been marginalized.
* Pointers to active messages originating from previous nodes that involve non-eliminated variables.
* The updated set of all eliminated variables (including the node's own target variable).

## Local Node Processing

Upon receiving inputs, the node executes:
1. Closing: Identifies all factors and incoming messages whose scopes are fully contained within the updated eliminated set and close them.
2. Local Caching: Stores the resulting condensed message locally.
3. Propagation: Forwards the new condensed message alongside all remaining factors (and messages) to the next node.

**Key invariant:** every message must retain an explicit manifest of the original factor IDs it summarizes to ensure strict factor accounting.

## State Marginal Recovery

To compute the posterior marginal distribution at a specific variable, one simply needs to combine all incoming messages arriving at the node with any uneliminated factors from the global pool that are directly incident to it.

⚠️ Verify that every factor contributing to the marginal is accounted for exactly once (no duplicate message contributions and no omitted global factors).

use std::collections::HashSet;

pub type VarId = u64;
pub type CliqueId = u64;

pub struct CliqueNode {
    id: CliqueId,
    front: HashSet<VarId>,
    separator: HashSet<VarId>,
    children: Vec<CliqueId>,
    parent: Option<CliqueId>,
}

pub struct BayesTree {
    nodes: Vec<CliqueNode>,
    pub root: Option<CliqueId>,
    pub var_to_clique: HashMap<VarId, CliqueId>, // Inverted index: maps each frontal variable to the owning clique
}

impl BayesTree {
    pub fn build_from_gauss_graph(
        variables: &[VarId],
        factors: &[Factor],
        ordering: &[VarId],
    ) -> Self {
        todo!()
    }
}

impl CliqueNode {
    pub fn new(front: HashSet<VarId>, separator: HashSet<VarId>) -> Self {
        Self {
            front,
            separator,
            children: None,
            parent: None,
        }
    }

    pub fn front(&self) -> &HashSet<u64> {
        &self.front
    }

    pub fn separator(&self) -> &HashSet<u64> {
        &self.separator
    }

    pub fn merge_with_child(&mut self, child: &mut Clique) {
        // New front = union of fronts
        // New separator = union of separators minus parent front

        // Remove separator variables from the child that will be absorbed into the front
        child.separator.retain(|k| !self.front.contains(k));

        self.front.reserve(child.front.len());
        self.front.extend(child.front.drain());

        self.separator.reserve(child.separator.len());
        self.separator.extend(child.separator.drain());
    }
}

// Agglomerate cliques
// build tree from factor graph
// variable elimination: factor graph, variable ordering
//
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_with_child() {
        let mut parent = Clique::new(HashSet::from([2]), HashSet::from([1]));
        let mut child = Clique::new(HashSet::from([3]), HashSet::from([2]));
        parent.merge_with_child(&mut child);
        assert_eq!(parent.front, HashSet::from([2, 3]));
        assert_eq!(parent.separator, HashSet::from([1]));
    }
}

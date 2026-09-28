use std::collections::HashSet;

use arbor_ferrea::tree::numeric::{ConcurrentStateBuffer, NumericTree};
use arbor_ferrea::tree::symbolic::{SymbolicFactor, SymbolicTree};

type VarId = usize;

struct AdderNode {
    var: VarId,
    children: Vec<AdderNode>,
    separator: Vec<VarId>,
    value: i32,
}

struct AdderTree {
    roots: Vec<AdderNode>,
}

struct AdderFactor {
    vars: Vec<VarId>,
    value: i32,
}

impl SymbolicFactor for AdderFactor {
    fn vars(&self) -> &[VarId] {
        &self.vars
    }
}

struct AdderMessage {
    vars: Vec<VarId>,
    value: i32,
}

struct AdderContext {
    pub get_memory_layout: fn(VarId) -> (usize, usize),
}

impl NumericTree for AdderTree {
    /// The original raw numerical factor.
    type Factor = AdderFactor;
    /// The concrete node type.
    type Node = AdderNode;
    /// The mathematical message passed upward during elimination.
    type Message = AdderMessage;
    /// User-provided context passed down during the solve (e.g., VariableDictionary for offsets).
    type SolveContext = AdderContext;
    /// The numerical type used for solving.
    type Value = i32;

    // ==========================================
    // REQUIRED METHODS (Math & Accessors)
    // ==========================================

    fn from_roots(roots: Vec<Self::Node>) -> Self {
        Self { roots }
    }

    fn roots(&self) -> &[Self::Node] {
        &self.roots
    }

    fn node_children(node: &Self::Node) -> &[Self::Node] {
        &node.children
    }

    fn eliminate_node(
        var: VarId,
        assigned_factors: Vec<&Self::Factor>,
        children: Vec<Self::Node>,
        incoming_messages: Vec<Self::Message>,
    ) -> anyhow::Result<(Self::Node, Self::Message)> {
        let mut value = 0;
        let mut separator = HashSet::new();
        for factor in assigned_factors {
            value += factor.value;
            for fvar in &factor.vars {
                separator.insert(*fvar);
            }
        }
        for message in &incoming_messages {
            value += message.value;
            for mvar in &message.vars {
                separator.insert(*mvar);
            }
        }
        separator.remove(&var);

        let vars: Vec<VarId> = separator.into_iter().collect();
        let node = Self::Node {
            var,
            children,
            separator: vars.clone(),
            value,
        };
        let outgoing_message = Self::Message { vars, value };
        Ok((node, outgoing_message))
    }

    fn decide_node(
        node: &Self::Node,
        buffer: &ConcurrentStateBuffer<i32>,
        ctx: &Self::SolveContext,
    ) -> anyhow::Result<()> {
        // 1. Gather separator values from the global buffer
        let mut sep_vals: Vec<i32> = Vec::new();
        for &sep_var in &node.separator {
            let (offset, dim) = (ctx.get_memory_layout)(sep_var);
            unsafe {
                // Guaranteed safe because parents always run before children
                let parent_data = buffer.read_slice(offset, dim);
                sep_vals.extend_from_slice(parent_data);
            }
        }

        // 2. Perform the actual math
        let value = sep_vals.iter().fold(node.value, |acc, &val| acc.max(val));

        // 3. Write our computed values into our slot in the global buffer
        let (my_offset, _) = (ctx.get_memory_layout)(node.var);
        unsafe {
            // Guaranteed safe because topology guarantees no other thread writes to our slot
            buffer.write_slice(my_offset, &[value]);
        }

        Ok(())
    }
}

#[test]
fn build_and_solve_adder_tree() {
    let factors = vec![
        AdderFactor {
            vars: vec![0],
            value: 1,
        },
        AdderFactor {
            vars: vec![0, 1],
            value: 3,
        },
        AdderFactor {
            vars: vec![1, 2],
            value: 1,
        },
        AdderFactor {
            vars: vec![0, 3],
            value: 5,
        },
        AdderFactor {
            vars: vec![1, 3],
            value: 2,
        },
        AdderFactor {
            vars: vec![2, 4],
            value: 3,
        },
    ];

    let order = [3, 4, 0, 1, 2];

    let symbolic_factors = factors.iter().map(|f| f as &dyn SymbolicFactor);
    let symbolic_tree =
        SymbolicTree::build(symbolic_factors, &order).expect("Error building the symbolic tree");
    assert_eq!(symbolic_tree.roots.len(), 1);

    let adder_tree =
        AdderTree::from_symbolic(&symbolic_tree, &factors).expect("Error building the adder tree");

    assert_eq!(adder_tree.roots.len(), 1);
    assert_eq!(adder_tree.roots[0].value, 15);

    let ctx = AdderContext {
        get_memory_layout: |i| (i, 1),
    };
    let mut values = vec![0; 5];
    adder_tree
        .solve_par(&mut values, &ctx)
        .expect("Error solving the adder tree");
    assert_eq!(values, [15, 15, 15, 15, 15]);
}

use anyhow::{Context, Result};

use arbor_ferrea::factor::symbolic::SymbolicFactor;
use arbor_ferrea::tree::numeric::{ConcurrentStateBuffer, NumericTree};
use arbor_ferrea::tree::symbolic::SymbolicTree;
use arbor_ferrea::variable::dictionary::{VarDict, VarSymbol};

struct AdderNode {
    var: usize,
    separator: Vec<usize>,
    children: Vec<AdderNode>,
    value: i32,
}

struct AdderTree {
    roots: Vec<AdderNode>,
}

struct AdderFactor {
    vars: Vec<usize>,
    value: i32,
}

impl SymbolicFactor for AdderFactor {
    fn vars(&self) -> &[usize] {
        &self.vars
    }
}

struct AdderMessage {
    _vars: Vec<usize>,
    value: i32,
}

impl NumericTree for AdderTree {
    /// The concrete node type.
    type Node = AdderNode;
    /// The numerical type used for solving.
    type Value = i32;
    /// The original raw numerical factor.
    type Factor = AdderFactor;
    /// The mathematical message passed upward during elimination.
    type Message = AdderMessage;
    /// User-provided context passed up during the build and the solve.
    type Context = VarDict;

    // ==========================================
    // REQUIRED METHODS (Math & Accessors)
    // ==========================================

    fn from_roots(roots: Vec<Self::Node>) -> Self {
        Self { roots }
    }

    fn roots(&self) -> &[Self::Node] {
        &self.roots
    }

    fn roots_mut(&mut self) -> &mut [Self::Node] {
        &mut self.roots
    }

    fn node_children(node: &Self::Node) -> &[Self::Node] {
        &node.children
    }

    fn node_children_mut(node: &mut Self::Node) -> &mut [Self::Node] {
        &mut node.children
    }

    fn build_node(
        var: usize,
        separator: Vec<usize>,
        assigned_factors: Vec<&Self::Factor>,
        incoming_messages: Vec<Self::Message>,
        children: Vec<Self::Node>,
        _ctx: &Self::Context,
    ) -> Result<Self::Node> {
        let mut value = assigned_factors.iter().fold(0, |acc, f| acc + f.value);
        value += incoming_messages.iter().fold(0, |acc, msg| acc + msg.value);

        let node = Self::Node {
            var,
            separator,
            children,
            value,
        };

        Ok(node)
    }

    fn build_out_message(node: &mut Self::Node, _ctx: &Self::Context) -> Result<Self::Message> {
        Ok(Self::Message {
            _vars: node.separator.clone(),
            value: node.value,
        })
    }

    fn decide_node(
        node: &mut Self::Node,
        buffer: &ConcurrentStateBuffer<i32>,
        ctx: &Self::Context,
    ) -> Result<()> {
        // 1. Gather separator values from the global buffer
        let mut sep_vals: Vec<i32> = Vec::new();
        for &sep in &node.separator {
            let (offset, dim) = ctx
                .memory_layout(sep)
                .with_context(|| format!("Missing size for variable with id: {}", sep))?;
            unsafe {
                // Guaranteed safe because parents always run before children
                let parent_data = buffer.read_slice(offset, dim);
                sep_vals.extend_from_slice(parent_data);
            }
        }

        // 2. Perform the actual math
        let value = sep_vals.iter().fold(node.value, |acc, &val| acc.max(val));

        // 3. Write our computed values into our slot in the global buffer
        let my_offset = ctx
            .offset(node.var)
            .with_context(|| format!("Missing size for variable with id: {}", node.var))?;
        unsafe {
            // Guaranteed safe because topology guarantees no other thread writes to our slot
            buffer.write_slice(my_offset, &[value]);
        }

        Ok(())
    }
}

#[test]
fn run_solver_empty() {
    // Build the dictionary and register domain variables
    let mut dict = VarDict::new();
    let x1 = dict.register(VarSymbol('x', 1), 1);
    let x2 = dict.register(VarSymbol('x', 2), 1);
    let x3 = dict.register(VarSymbol('x', 3), 1);
    let l1 = dict.register(VarSymbol('l', 1), 1);
    let l2 = dict.register(VarSymbol('l', 2), 1);

    let factors: Vec<AdderFactor> = vec![];

    let order = [l1, l2, x1, x2, x3];

    let symb_factors = factors.iter().map(|f| f as &dyn SymbolicFactor);
    let res = SymbolicTree::build(symb_factors, &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert_eq!(symb_tree.roots.len(), 0);

    let res = AdderTree::from_symbolic(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut adder_tree = res.unwrap();
    assert_eq!(adder_tree.roots.len(), 0);

    let mut buffer = vec![0; 5];
    let res = adder_tree.solve_par(&mut buffer, &dict);
    assert!(res.is_ok());
    assert_eq!(buffer, vec![0; 5]);
}

#[test]
fn run_solver() {
    // Build the dictionary and register domain variables
    let mut dict = VarDict::new();
    let x1 = dict.register(VarSymbol('x', 1), 1);
    let x2 = dict.register(VarSymbol('x', 2), 1);
    let x3 = dict.register(VarSymbol('x', 3), 1);
    let l1 = dict.register(VarSymbol('l', 1), 1);
    let l2 = dict.register(VarSymbol('l', 2), 1);

    // We pass an empty slice of factors here for the mock example
    let factors = vec![
        AdderFactor {
            vars: vec![x1],
            value: 1,
        },
        AdderFactor {
            vars: vec![x1, x2],
            value: 3,
        },
        AdderFactor {
            vars: vec![x2, x3],
            value: 1,
        },
        AdderFactor {
            vars: vec![x1, l1],
            value: 5,
        },
        AdderFactor {
            vars: vec![x2, l1],
            value: 2,
        },
        AdderFactor {
            vars: vec![x2, l2],
            value: 3,
        },
    ];

    let order = [l1, l2, x1, x2, x3];

    let symb_factors = factors.iter().map(|f| f as &dyn SymbolicFactor);
    let res = SymbolicTree::build(symb_factors, &order);
    assert!(res.is_ok());
    let symb_tree = res.unwrap();
    assert_eq!(symb_tree.roots.len(), 1);

    let res = AdderTree::from_symbolic(&symb_tree, &factors, &dict);
    assert!(res.is_ok());
    let mut adder_tree = res.unwrap();
    assert_eq!(adder_tree.roots.len(), 1);

    let mut buffer = vec![0; 5];
    let res = adder_tree.solve_par(&mut buffer, &dict);
    assert!(res.is_ok());
    assert_eq!(buffer, vec![15; 5]);
}

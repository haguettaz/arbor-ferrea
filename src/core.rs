use std::collections::HashMap;
use std::fmt::Debug;

use anyhow::{Result, bail, ensure};

pub trait Prior: Sized {
    fn id(&self) -> usize;
    fn key(&self) -> usize;

    fn depends_on(&self, key: usize) -> bool {
        self.key() == key
    }
}

pub trait Relation: Sized {
    fn keys(&self) -> &[usize];
    fn is_empty(&self) -> bool {
        self.keys().is_empty()
    }
    fn depends_on(&self, key: usize) -> bool {
        self.keys().contains(&key)
    }

    fn ids(&self) -> &[usize];
    fn is_message(&self) -> bool {
        self.ids().len() > 1
    }

    fn from<'a, P, R>(
        priors: impl IntoIterator<Item = &'a P>,
        relations: impl IntoIterator<Item = &'a R>,
        keep_keys: impl IntoIterator<Item = usize>,
        clear_keys: impl IntoIterator<Item = usize>,
        ctx: impl Fn(usize) -> usize,
    ) -> Result<Self>
    where
        P: Prior + 'a,
        R: Relation + 'a;
}

/// Variable symbol
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub char, pub usize);

#[derive(Default)]
pub struct FactorGraph<P: Prior, R: Relation> {
    priors: Vec<P>,
    relations: Vec<R>,
    key_to_symbol: Vec<Symbol>,
    key_to_size: Vec<usize>,
    symbol_to_key: HashMap<Symbol, usize>,
}

impl<P: Prior + Default, R: Relation + Default> FactorGraph<P, R> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_keys(&self) -> usize {
        self.key_to_symbol.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = usize> {
        0..self.key_to_symbol.len()
    }

    pub fn key(&self, symbol: &Symbol) -> Option<usize> {
        self.symbol_to_key.get(symbol).copied()
    }

    pub fn symbol(&self, key: usize) -> Option<&Symbol> {
        self.key_to_symbol.get(key)
    }

    pub fn size(&self, key: usize) -> Option<usize> {
        self.key_to_size.get(key).copied()
    }

    pub fn n_priors(&self) -> usize {
        self.priors.len()
    }

    pub fn priors(&self) -> &[P] {
        &self.priors
    }

    pub fn n_relations(&self) -> usize {
        self.relations.len()
    }

    pub fn relations(&self) -> &[R] {
        &self.relations
    }

    pub fn add_variable(&mut self, symbol: Symbol, size: usize) -> Result<usize> {
        if let Some(_) = self.symbol_to_key.get(&symbol) {
            bail!("Variable {:?} has already been registered.", symbol)
        }

        let id = self.key_to_symbol.len();
        self.symbol_to_key.insert(symbol, id);
        self.key_to_symbol.push(symbol);
        self.key_to_size.push(size);
        Ok(id)
    }

    pub fn add_prior<C>(&mut self, create_prior: C) -> Result<usize>
    where
        C: FnOnce(usize) -> Result<P>,
    {
        let id = self.n_priors() + self.n_relations();
        let prior = create_prior(id)?;

        ensure!(
            prior.key() < self.n_keys(),
            "The provided prior concerns a variable that has not been registered yet.",
        );

        self.priors.push(prior);
        Ok(id)
    }

    pub fn add_relation<C>(&mut self, create_relation: C) -> Result<usize>
    where
        C: FnOnce(usize) -> Result<R>,
    {
        let id = self.n_priors() + self.n_relations();
        let relation = create_relation(id)?;

        ensure!(
            relation.keys().iter().all(|&k| k < self.n_keys()),
            "The provided relation concerns variables that have not been registered yet.",
        );

        self.relations.push(relation);
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Default)]
    pub struct MyPrior {
        pub id: usize,
        pub key: usize,
    }

    impl Prior for MyPrior {
        fn id(&self) -> usize {
            self.id
        }

        fn key(&self) -> usize {
            self.key
        }
    }

    impl MyPrior {
        pub fn new(id: usize, key: usize) -> Self {
            Self { id, key }
        }
    }

    #[derive(Debug, PartialEq, Default)]
    pub struct MyRelation {
        pub ids: Vec<usize>, // Added so Factor::id() has a value to return
        pub keys: Vec<usize>,
    }

    impl MyRelation {
        pub fn new(ids: Vec<usize>, keys: Vec<usize>) -> Self {
            Self { ids, keys }
        }
    }

    // Base implementation for Factor
    impl Relation for MyRelation {
        fn keys(&self) -> &[usize] {
            &self.keys
        }

        fn ids(&self) -> &[usize] {
            &self.ids
        }

        fn from<'a, P, R>(
            priors: impl IntoIterator<Item = &'a P>,
            relations: impl IntoIterator<Item = &'a R>,
            keep_keys: impl IntoIterator<Item = usize>,
            _clear_keys: impl IntoIterator<Item = usize>,
            _ctx: impl Fn(usize) -> usize,
        ) -> Result<Self>
        where
            P: Prior + 'a,
            R: Relation + 'a,
        {
            let mut ids: Vec<usize> = priors
                .into_iter()
                .map(|p| p.id())
                .chain(relations.into_iter().flat_map(|r| r.ids()).copied())
                .collect();
            ids.sort_unstable();

            let mut keys: Vec<usize> = keep_keys.into_iter().collect();
            keys.sort_unstable();

            let msg = MyRelation { ids, keys };
            Ok(msg)
        }
    }

    const X1: Symbol = Symbol('x', 1);
    const X2: Symbol = Symbol('x', 2);
    const X3: Symbol = Symbol('x', 3);
    const X42: Symbol = Symbol('x', 42);

    #[test]
    fn test_factor_graph() {
        let mut factor_graph = FactorGraph::new();

        let x1 = factor_graph
            .add_variable(X1, 1)
            .expect("Adding a new variable should work.");
        let x2 = factor_graph
            .add_variable(X2, 3)
            .expect("Adding a new 3-dimensional variable should also work.");
        let x3 = factor_graph
            .add_variable(X3, 1_000_000)
            .expect("Adding a new, very high-dimension variable should also work.");
        let _ = factor_graph
            .add_variable(X2, 3)
            .expect_err("Adding an already registered variable should fail.");
        let not_registered_x42 = 42; // a not registered variable

        assert_eq!(
            factor_graph.n_keys(),
            3,
            "The factor graph has three registered variables: x1, x2, and x3"
        );

        let symbol_x1 = factor_graph.symbol(x1);
        assert_eq!(
            symbol_x1,
            Some(&X1),
            "A registered variable has an accessible symbol."
        );
        let not_registered_symbol = factor_graph.symbol(not_registered_x42);
        assert_eq!(
            not_registered_symbol, None,
            "A non-registered variable has no symbol."
        );

        let x2_from_its_symbol = factor_graph.key(&X2);
        assert_eq!(
            x2_from_its_symbol,
            Some(x2),
            "A registered variable can be accessed via its symbol."
        );
        let x42_from_its_symbol = factor_graph.key(&X42);
        assert_eq!(
            x42_from_its_symbol, None,
            "A non-registered variable cannot be accessed via its symbol."
        );

        let size_x3 = factor_graph.size(x3);
        assert_eq!(
            size_x3,
            Some(1_000_000),
            "The size of a registered variable is accessible from its key."
        );
        let size_not_registered_x42 = factor_graph.size(not_registered_x42);
        assert_eq!(
            size_not_registered_x42, None,
            "The size of a non-registered variable is not accessible."
        );

        factor_graph
            .add_prior(|id| Ok(MyPrior::new(id, x1)))
            .expect("Adding a prior on a registered variable should work.");
        factor_graph
            .add_prior(|id| Ok(MyPrior::new(id, not_registered_x42)))
            .expect_err("Adding a prior on a non-registered variable should not work.");

        assert_eq!(factor_graph.n_priors(), 1);

        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2])))
            .expect("Adding a relation between registered variables should work.");
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2, x3])))
            .expect("Adding a relation between registered variables should work.");
        factor_graph
            .add_relation(|id| Ok(MyRelation::new(vec![id], vec![x1, x2, x3, not_registered_x42])))
            .expect_err("Adding a relation between registered variables and not registered variable should not work.");

        assert_eq!(factor_graph.n_relations(), 2,);
    }
}

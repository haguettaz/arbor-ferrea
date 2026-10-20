use anyhow::{Result, bail, ensure};
use std::collections::HashMap;
use std::fmt::Debug;

/// Variable symbol
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(pub char, pub usize);

pub trait FactorBase {
    fn keys(&self) -> &[usize];
    fn is_empty(&self) -> bool {
        self.keys().is_empty()
    }
}

pub trait Factor: FactorBase + Sized {
    type TargetMessage: Message;
    fn id(&self) -> usize;
}

pub trait Message: FactorBase + Sized + Clone {
    type TargetFactor: Factor<TargetMessage = Self>;
    fn closed_factors(&self) -> &[usize];
    fn close<'a, I, K>(factors: I, on_keys: K) -> Result<Self>
    where
        I: IntoIterator<Item = AnyFactor<'a, Self::TargetFactor>>,
        K: IntoIterator<Item = usize>,
        Self::TargetFactor: 'a;
}

pub enum AnyFactor<'a, F: Factor> {
    BorrowedFactor(&'a F),
    OwnedMessage(F::TargetMessage),
}

// Forward the base methods through the enum
impl<'a, F: Factor> FactorBase for AnyFactor<'a, F> {
    fn keys(&self) -> &[usize] {
        match self {
            Self::BorrowedFactor(f) => f.keys(),
            Self::OwnedMessage(m) => m.keys(),
        }
    }
}

#[derive(Default)]
pub struct FactorGraph<F: Factor> {
    factors: Vec<F>,
    key_to_symbol: Vec<Symbol>,
    key_to_size: Vec<usize>,
    symbol_to_key: HashMap<Symbol, usize>,
}

impl<F: Factor + Default> FactorGraph<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_keys(&self) -> usize {
        self.key_to_symbol.len()
    }

    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    pub fn factors(&self) -> &[F] {
        &self.factors
    }

    pub fn factor_ids(&self) -> Vec<usize> {
        self.factors.iter().map(|factor| factor.id()).collect()
    }

    pub fn factor(&self, f_id: usize) -> Option<&F> {
        self.factors.get(f_id)
    }

    pub fn symbol(&self, key: usize) -> Option<&Symbol> {
        self.key_to_symbol.get(key)
    }

    pub fn size(&self, key: usize) -> Option<&usize> {
        self.key_to_size.get(key)
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

    pub fn add_factor<C>(&mut self, create_factor: C) -> Result<usize>
    where
        C: FnOnce(usize) -> Result<F>,
    {
        let id = self.factors.len();
        let factor = create_factor(id)?;

        ensure!(
            factor.keys().iter().all(|&k| k < self.n_keys()),
            "The provided factor contains variables that have not been registered yet.",
        );

        self.factors.push(factor);
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[derive(Debug, PartialEq, Default)]
    pub struct MyFactor {
        pub id: usize, // Added so Factor::id() has a value to return
        pub keys: Vec<usize>,
    }

    impl MyFactor {
        pub fn new(id: usize, keys: Vec<usize>) -> Result<MyFactor> {
            Ok(MyFactor { id, keys })
        }
    }

    #[derive(Clone, Debug)]
    pub struct MyMessage {
        pub keys: Vec<usize>,
        pub closed_factors: Vec<usize>,
    }

    // Base implementation for Factor
    impl FactorBase for MyFactor {
        fn keys(&self) -> &[usize] {
            &self.keys
        }
    }

    impl Factor for MyFactor {
        type TargetMessage = MyMessage;

        fn id(&self) -> usize {
            self.id
        }
    }

    // Base implementation for Message
    impl FactorBase for MyMessage {
        fn keys(&self) -> &[usize] {
            &self.keys
        }
    }

    impl Message for MyMessage {
        type TargetFactor = MyFactor;

        fn closed_factors(&self) -> &[usize] {
            &self.closed_factors
        }

        fn close<'a, I, K>(factors: I, on_keys: K) -> Result<Self>
        where
            I: IntoIterator<Item = AnyFactor<'a, Self::TargetFactor>>,
            K: IntoIterator<Item = usize>,
            Self::TargetFactor: 'a,
        {
            let mut closed_factors = Vec::new();
            let mut all_keys = HashSet::new();

            // Aggregate all keys and track which factors/messages are being closed
            for factor in factors {
                for &key in factor.keys() {
                    all_keys.insert(key);
                }

                match factor {
                    AnyFactor::BorrowedFactor(f) => {
                        closed_factors.push(f.id());
                    }
                    AnyFactor::OwnedMessage(m) => {
                        closed_factors.extend_from_slice(m.closed_factors());
                    }
                }
            }

            // Eliminate the specified keys
            for key in on_keys {
                all_keys.remove(&key);
            }

            // Construct and return the new Message
            let mut final_keys: Vec<usize> = all_keys.into_iter().collect();
            final_keys.sort_unstable(); // Optional: keeps keys deterministic
            closed_factors.sort_unstable(); // Optional: keeps keys deterministic

            Ok(Self {
                keys: final_keys,
                closed_factors,
            })
        }
    }

    const X1: Symbol = Symbol('x', 1);
    const X2: Symbol = Symbol('x', 2);
    const X3: Symbol = Symbol('x', 3);
    const X4: Symbol = Symbol('x', 4);
    const X5: Symbol = Symbol('x', 5);

    #[test]
    fn test_factor_graph() {
        let mut factor_graph = FactorGraph::new();

        let x1 = factor_graph
            .add_variable(X1, 1)
            .expect("Adding x1 should work");
        let x2 = factor_graph
            .add_variable(X2, 1)
            .expect("Adding x2 should work");
        let x3 = factor_graph
            .add_variable(X3, 1)
            .expect("Adding x3 should work");
        let x4 = factor_graph
            .add_variable(X4, 10000000)
            .expect("Adding x4 should work");
        let x5 = factor_graph
            .add_variable(X5, 1)
            .expect("Adding x5 should work");
        let _ = factor_graph
            .add_variable(X3, 2)
            .expect_err("Adding x3 again should fail");
        let x6 = x5 + 1;
        assert_eq!(factor_graph.n_keys(), 5);

        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x1, x2]))
            .expect("Adding factor between x1 and x2 should work");
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x2, x3, x1, x5]))
            .expect("Adding factor between x1, x2, x3, and x5 should work");
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![x4, x6]))
            .expect_err("Adding factor between x4 and x6 should fail");
        factor_graph
            .add_factor(|id| MyFactor::new(id, vec![]))
            .expect("Adding empty factor should work");
        assert_eq!(factor_graph.n_factors(), 3);
    }
}

/// Offsets of a sorted key set inside a stacked vector
#[derive(Debug)]
pub(crate) struct Layout {
    pub keys: Vec<usize>,
    pub offsets: Vec<usize>,
}

impl Layout {
    pub fn new(keys: Vec<usize>, size_of: impl Fn(usize) -> usize) -> Self {
        let mut offsets = Vec::with_capacity(keys.len() + 1);
        let mut acc = 0;
        offsets.push(0);
        for &k in &keys {
            acc += size_of(k);
            offsets.push(acc);
        }
        Self { keys, offsets }
    }

    pub fn dim(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }

    pub fn keys(&self) -> &[usize] {
        &self.keys
    }

    /// (icol, ncols) of `key`'s block, if present.
    pub fn find(&self, key: usize) -> Option<(usize, usize)> {
        let i = self.keys.binary_search(&key).ok()?;
        Some((self.offsets[i], self.offsets[i + 1] - self.offsets[i]))
    }

    pub fn max_block(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }
}

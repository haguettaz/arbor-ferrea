use std::fmt::{self, Display, Formatter};

// #[derive(Clone, PartialEq, Eq, Hash)]
// pub struct VarInfo<T> {
//     pub label: T,
//     pub size: usize,
//     pub offset: usize,
// }

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct VarInfo<T> {
    pub label: T,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub chr: u8, // e.g., b'x' for poses, b'l' for landmarks
    pub index: u64,
}

impl<T: Clone + Display + PartialEq> VarInfo<T> {
    pub fn new(label: T, size: usize) -> Self {
        Self { label, size }
    }
}

impl<T: Display> Display for VarInfo<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "var {} (size={})", self.label, self.size)
    }
}

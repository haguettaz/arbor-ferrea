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

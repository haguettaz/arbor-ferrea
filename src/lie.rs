#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LieValue {
    R1(Rn<1>),
    R2(Rn<2>),
    R3(Rn<3>),
    So2(So2),
    Se2(Se2),
    So3(So3),
    Se3(Se3),
}

impl LieValue {
    pub fn dof(&self) -> usize {
        match self {
            Self::R1(_) => 1,
            Self::R2(_) => 2,
            Self::R3(_) => 3,
            Self::So2(_) => 1,
            Self::Se2(_) => 3,
            Self::So3(_) => 3,
            Self::Se3(_) => 6,
        }
    }

    pub fn retract(&self, delta: &[f64]) -> Self {
        match self {
            Self::R1(v) => Self::R1(v.retract(delta)),
            Self::So3(v) => Self::So3(v.retract(delta)),
            Self::Se3(v) => Self::Se3(v.retract(delta)),
            // ... exhaust remaining variants
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rn<const N: usize>(pub [f64; N]);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So2 {
    /// Unit complex number [cos(θ), sin(θ)]
    pub complex: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se2 {
    pub translation: [f64; 2],
    pub rotation: So2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So3 {
    /// Unit quaternion [x, y, z, w]
    pub unit_quaternion: [f64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    pub translation: [f64; 3],
    pub rotation: So3,
}

pub trait LieGroup: Sized + Copy {
    /// Dimension of the tangent space / Lie algebra (DoF).
    const DOF: usize;

    /// Number of parameters used to represent the group element.
    const PARAM_DIM: usize;

    /// The retraction / exp map: X ⊕ delta
    fn retract(&self, delta: &[f64]) -> Self;

    /// The inverse retraction / log map: X ⊖ Y
    fn local(&self, other: &Self) -> Vec<f64>;

    /// Identity element
    fn identity() -> Self;

    /// Group inverse
    fn inverse(&self) -> Self;

    /// Group multiplication / composition: self * other
    fn compose(&self, other: &Self) -> Self;
}

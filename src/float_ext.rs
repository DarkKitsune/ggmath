use num::traits::Float;

/// A trait that provides extra functions for floating point types
pub trait FloatExt: Float {
    fn two() -> Self;
    fn half() -> Self;
    fn pi() -> Self;
    fn tau() -> Self;
    fn squared(self) -> Self;
    fn lerp(self, other: Self, amount: Self) -> Self;
}

impl<T: Float> FloatExt for T {
    fn two() -> Self {
        Self::one() + Self::one()
    }

    fn half() -> Self {
        Self::one() / Self::two()
    }

    fn pi() -> Self {
        Self::from(std::f64::consts::PI).expect("Pi is not representable")
    }

    fn tau() -> Self {
        Self::from(std::f64::consts::TAU).expect("Tau is not representable")
    }

    fn squared(self) -> Self {
        self * self
    }

    fn lerp(self, other: Self, amount: Self) -> Self {
        self + (other - self) * amount
    }
}

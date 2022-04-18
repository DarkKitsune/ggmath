use num_traits::Float;

pub trait FloatExt: Float {
    fn two() -> Self;
    fn half() -> Self;
    fn squared(self) -> Self;
}

impl<T: Float> FloatExt for T {
    fn two() -> Self {
        Self::one() + Self::one()
    }

    fn half() -> Self {
        Self::one() / Self::two()
    }

    fn squared(self) -> Self {
        self * self
    }
}
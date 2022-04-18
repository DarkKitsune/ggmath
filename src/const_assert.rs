pub struct ConstAssert<const CONDITION: bool>;

pub trait IsTrue {}
pub trait IsFalse {}

impl IsTrue for ConstAssert<true> {}
impl IsFalse for ConstAssert<false> {}

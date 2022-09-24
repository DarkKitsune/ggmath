use num_traits::{One, Zero};

use crate::{init_array, prelude::Vector};

/// A linear congruential generator for generating pseudo-random values efficiently.
#[derive(Debug, Clone, PartialEq)]
pub struct Lcg {
    seed: u64,
}

impl Lcg {
    pub fn new(seed: impl ToSeed) -> Self {
        Self {
            seed: seed.to_seed(),
        }
    }

    pub fn next<T: FromRandom>(&mut self) -> T {
        T::from_lcg(self)
    }

    fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(0xB2788569).wrapping_add(0x5A6EC706);
        ((self.seed) & 0xFFFFFFFF) as u32
    }
}

/// A linear congruential generator for generating pseudo-random values efficiently.
/// Values are normally distributed (biased towards 0.5). When generating random points on a circle or sphere,
/// `DIMENSIONS` should equal the number of dimensions of the sphere, but can also be used
/// to adjust the curve, as the higher `DIMENSIONS` is, the more values are biased towards 0.5.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalLcg<const DIMENSIONS: usize> {
    random: Lcg,
    stored: [f64; DIMENSIONS],
    current: usize,
}

impl<const DIMENSIONS: usize> NormalLcg<DIMENSIONS> {
    pub fn new(seed: impl ToSeed) -> Self {
        Self {
            random: Lcg::new(seed),
            stored: [0.0; DIMENSIONS],
            current: DIMENSIONS,
        }
    }

    pub fn next_f64(&mut self) -> f64 {
        if self.current == DIMENSIONS {
            loop {
                let test_group = Vector::new([
                    init_array!([f64; DIMENSIONS], mut |_| -0.5 + f64::from_lcg(&mut self.random)),
                ]);
                if test_group.length_squared() > 0.25 {
                    self.stored = test_group.into();
                    break;
                }
            }
            self.current = 1;
        } else {
            self.current += 1;
        }
        self.stored[self.current - 1]
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    pub fn next_normal_f32(&mut self) -> Vector<f32, DIMENSIONS> {
        Vector::new([init_array!([f32; DIMENSIONS], mut |_| self.next_f32())]).normalized()
    }

    pub fn next_normal_f64(&mut self) -> Vector<f64, DIMENSIONS> {
        Vector::new([init_array!([f64; DIMENSIONS], mut |_| self.next_f64())]).normalized()
    }
}

pub trait FromRandom {
    fn from_lcg(random: &mut Lcg) -> Self;
}

impl FromRandom for u8 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as u8
    }
}

impl FromRandom for i8 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as i8
    }
}

impl FromRandom for u16 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as u16
    }
}

impl FromRandom for i16 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as i16
    }
}

impl FromRandom for u32 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32()
    }
}

impl FromRandom for i32 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as i32
    }
}

impl FromRandom for u64 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as u64 | ((random.next_u32() as u64) << 32)
    }
}

impl FromRandom for i64 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as i64 | ((random.next_u32() as i64) << 32)
    }
}

impl FromRandom for f32 {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() as f32 / 0xFFFFFFFFu32 as f32
    }
}

impl FromRandom for f64 {
    fn from_lcg(random: &mut Lcg) -> Self {
        u64::from_lcg(random) as f64 / 0xFFFFFFFFFFFFFFFFu64 as f64
    }
}

impl FromRandom for bool {
    fn from_lcg(random: &mut Lcg) -> Self {
        random.next_u32() & 1 == 1
    }
}

impl<T: FromRandom + Copy + Zero + One, const N: usize> FromRandom for Vector<T, N> {
    fn from_lcg(random: &mut Lcg) -> Self {
        Vector::new([init_array!([T; N], mut |_| T::from_lcg(random))])
    }
}

impl<const COUNT: usize> FromRandom for [u8; COUNT] {
    fn from_lcg(random: &mut Lcg) -> Self {
        let mut result = [0; COUNT];
        let mut offset = 0;
        while offset + 3 < COUNT {
            unsafe {
                (result.as_ptr() as *mut u32)
                    .add(offset)
                    .write(random.next_u32());
            }
            offset += 4;
        }
        while offset < COUNT {
            result[offset] = u8::from_lcg(random);
            offset += 1;
        }
        result
    }
}

pub trait ToSeed {
    /// Create a seed for a random number generator from this value.
    fn to_seed(&self) -> u64;
    /// Returns a random value of type `R` using `self` as a seed.
    /// A linear congruential generator is used to generate the random value.
    fn into_random<R: FromRandom>(self) -> R
    where
        Self: Sized,
    {
        R::from_lcg(&mut Lcg::new(self))
    }
}

impl ToSeed for u8 {
    fn to_seed(&self) -> u64 {
        *self as u64
    }
}

impl ToSeed for i8 {
    fn to_seed(&self) -> u64 {
        *self as u64
    }
}

impl ToSeed for u16 {
    fn to_seed(&self) -> u64 {
        *self as u64
    }
}

impl ToSeed for i16 {
    fn to_seed(&self) -> u64 {
        *self as u64
    }
}

impl ToSeed for u32 {
    fn to_seed(&self) -> u64 {
        *self as u64
    }
}

impl ToSeed for i32 {
    fn to_seed(&self) -> u64 {
        *self as u64
    }
}

impl ToSeed for u64 {
    fn to_seed(&self) -> u64 {
        *self
    }
}

impl ToSeed for i64 {
    fn to_seed(&self) -> u64 {
        *self as u64
    }
}

impl ToSeed for str {
    fn to_seed(&self) -> u64 {
        self.as_bytes().iter().enumerate().fold(0, |acc, (idx, b)| {
            acc ^ b
                .to_seed()
                .wrapping_mul((idx as u64).wrapping_mul(2967931333).wrapping_add(1))
        })
    }
}

impl<T: ToSeed> ToSeed for [T] {
    fn to_seed(&self) -> u64 {
        self.iter().enumerate().fold(0, |acc, (idx, b)| {
            acc ^ b
                .to_seed()
                .wrapping_mul((idx as u64).wrapping_mul(2967931333).wrapping_add(1))
        })
    }
}

impl<T: ToSeed> ToSeed for Vec<T> {
    fn to_seed(&self) -> u64 {
        self.iter().enumerate().fold(0, |acc, (idx, b)| {
            acc ^ b
                .to_seed()
                .wrapping_mul((idx as u64).wrapping_mul(2967931333).wrapping_add(1))
        })
    }
}

impl<T: ToSeed + Copy + Zero + One, const N: usize> ToSeed for Vector<T, N> {
    fn to_seed(&self) -> u64 {
        self.iter().enumerate().fold(0, |acc, (idx, b)| {
            acc ^ b
                .to_seed()
                .wrapping_mul((idx as u64).wrapping_mul(2967931333).wrapping_add(1))
        })
    }
}

impl<T0: ToSeed, T1: ToSeed> ToSeed for (T0, T1) {
    fn to_seed(&self) -> u64 {
        self.0.to_seed() ^ self.1.to_seed()
    }
}

impl<T0: ToSeed, T1: ToSeed, T2: ToSeed> ToSeed for (T0, T1, T2) {
    fn to_seed(&self) -> u64 {
        self.0.to_seed()
            ^ self.1.to_seed().wrapping_mul(2967931333)
            ^ self.2.to_seed().wrapping_mul(2967931333 * 2)
    }
}

impl<T0: ToSeed, T1: ToSeed, T2: ToSeed, T3: ToSeed> ToSeed for (T0, T1, T2, T3) {
    fn to_seed(&self) -> u64 {
        self.0.to_seed()
            ^ self.1.to_seed().wrapping_mul(2967931333)
            ^ self.2.to_seed().wrapping_mul(2967931333 * 2)
            ^ self.3.to_seed().wrapping_mul(2967931333 * 3)
    }
}

impl<T0: ToSeed, T1: ToSeed, T2: ToSeed, T3: ToSeed, T4: ToSeed> ToSeed for (T0, T1, T2, T3, T4) {
    fn to_seed(&self) -> u64 {
        self.0.to_seed()
            ^ self.1.to_seed().wrapping_mul(2967931333)
            ^ self.2.to_seed().wrapping_mul(2967931333 * 2)
            ^ self.3.to_seed().wrapping_mul(2967931333 * 3)
            ^ self.4.to_seed().wrapping_mul(2967931333 * 4)
    }
}

impl<T0: ToSeed, T1: ToSeed, T2: ToSeed, T3: ToSeed, T4: ToSeed, T5: ToSeed> ToSeed
    for (T0, T1, T2, T3, T4, T5)
{
    fn to_seed(&self) -> u64 {
        self.0.to_seed()
            ^ self.1.to_seed().wrapping_mul(2967931333)
            ^ self.2.to_seed().wrapping_mul(2967931333 * 2)
            ^ self.3.to_seed().wrapping_mul(2967931333 * 3)
            ^ self.4.to_seed().wrapping_mul(2967931333 * 4)
            ^ self.5.to_seed().wrapping_mul(2967931333 * 5)
    }
}

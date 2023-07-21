use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Rem, Sub},
};

use num::{integer::gcd, traits::AsPrimitive};

/// A fraction with a numerator and denominator.
#[derive(Clone, Copy, Hash)]
pub enum Fraction {
    Undefined,
    Normal { numerator: i64, denominator: i64 },
}

impl Fraction {
    /// Creates a new fraction with the given numerator and denominator.
    /// If the denominator is 0, the fraction is undefined.
    pub fn new(numerator: i64, denominator: i64) -> Self {
        Self::Normal {
            numerator,
            denominator,
        }
        .simplify()
    }

    /// Returns the numerator of the fraction.
    pub fn numerator(self) -> Option<i64> {
        match self {
            Self::Undefined => None,
            Self::Normal { numerator, .. } => Some(numerator),
        }
    }

    /// Returns the denominator of the fraction.
    pub fn denominator(self) -> i64 {
        match self {
            Self::Undefined => 0,
            Self::Normal { denominator, .. } => denominator,
        }
    }

    /// Simplifies the fraction to its simplest form.
    fn simplify(mut self) -> Self {
        // Normalize first to ensure the denominator is positive.
        self = self.normalized();
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            // Simplify the fraction to its simplest form.
            let gcd = gcd(numerator, denominator);
            Self::Normal {
                numerator: numerator / gcd,
                denominator: denominator / gcd,
            }
        } else {
            self
        }
    }

    /// Normalizes the fraction to a positive denominator.
    /// If the denominator is negative, the numerator is negated.
    fn normalized(self) -> Self {
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            // If the denominator is 0, the fraction is undefined.
            if denominator == 0 {
                Self::Undefined
            }
            // If the numerator is 0, the fraction is 0.
            else if denominator == 1 {
                Self::Normal {
                    numerator: 0,
                    denominator: 1,
                }
            }
            // If the denominator is negative, negate the numerator.
            else if denominator < 0 {
                Self::Normal {
                    numerator: -numerator,
                    denominator: -denominator,
                }
            }
            // Otherwise, the fraction is already normalized.
            else {
                self
            }
        } else {
            self
        }
    }

    /// Returns the approximate value of the fraction as the given type.
    /// Returns `None` if the fraction is undefined.
    pub fn approximate<T: Div<Output = T> + Copy + 'static>(self) -> Option<T>
    where
        i64: AsPrimitive<T>,
    {
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            Some(numerator.as_() / denominator.as_())
        } else {
            None
        }
    }

    /// Returns whether the fraction is undefined.
    pub fn is_undefined(self) -> bool {
        matches!(self, Self::Undefined)
    }

    /// Returns whether the fraction is negative.
    /// Always returns `false` if the fraction is undefined.
    pub fn is_negative(self) -> bool {
        if let Self::Normal { numerator, .. } = self {
            numerator < 0
        } else {
            false
        }
    }
}

impl<T> Add<T> for Fraction
where
    Self: From<T>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = Self::from(rhs);
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            if let Self::Normal {
                numerator: rhs_numerator,
                denominator: rhs_denominator,
            } = rhs
            {
                Self::new(
                    numerator * rhs_denominator + rhs_numerator * denominator,
                    denominator * rhs_denominator,
                )
            } else {
                Self::Undefined
            }
        } else {
            Self::Undefined
        }
    }
}

impl<T> Sub<T> for Fraction
where
    Self: From<T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs = Self::from(rhs);
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            if let Self::Normal {
                numerator: rhs_numerator,
                denominator: rhs_denominator,
            } = rhs
            {
                Self::new(
                    numerator * rhs_denominator - rhs_numerator * denominator,
                    denominator * rhs_denominator,
                )
            } else {
                Self::Undefined
            }
        } else {
            Self::Undefined
        }
    }
}

impl<T> Mul<T> for Fraction
where
    Self: From<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = Self::from(rhs);
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            if let Self::Normal {
                numerator: rhs_numerator,
                denominator: rhs_denominator,
            } = rhs
            {
                Self::new(numerator * rhs_numerator, denominator * rhs_denominator)
            } else {
                Self::Undefined
            }
        } else {
            Self::Undefined
        }
    }
}

impl<T> Div<T> for Fraction
where
    Self: From<T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let rhs = Self::from(rhs);
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            if let Self::Normal {
                numerator: rhs_numerator,
                denominator: rhs_denominator,
            } = rhs
            {
                Self::new(numerator * rhs_denominator, denominator * rhs_numerator)
            } else {
                Self::Undefined
            }
        } else {
            Self::Undefined
        }
    }
}

impl<T> Rem<T> for Fraction
where
    Self: From<T>,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        let rhs = Self::from(rhs);
        if let Self::Normal {
            numerator,
            denominator,
        } = self
        {
            if let Self::Normal {
                numerator: rhs_numerator,
                denominator: rhs_denominator,
            } = rhs
            {
                Self::new(
                    numerator * rhs_denominator % rhs_numerator * denominator,
                    denominator * rhs_denominator,
                )
            } else {
                Self::Undefined
            }
        } else {
            Self::Undefined
        }
    }
}

impl From<i64> for Fraction {
    fn from(n: i64) -> Self {
        Self::new(n, 1)
    }
}

impl From<(i64, i64)> for Fraction {
    fn from((numerator, denominator): (i64, i64)) -> Self {
        Self::new(numerator, denominator)
    }
}

impl PartialEq for Fraction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Normal {
                    numerator,
                    denominator,
                },
                Self::Normal {
                    numerator: rhs_numerator,
                    denominator: rhs_denominator,
                },
            ) => numerator == rhs_numerator && denominator == rhs_denominator,
            _ => false,
        }
    }
}

impl PartialOrd for Fraction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.approximate::<f64>()?
            .partial_cmp(&other.approximate::<f64>()?)
    }
}

impl Debug for Fraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undefined => write!(f, "Undefined"),
            Self::Normal {
                numerator,
                denominator,
            } => write!(f, "{}/{}", numerator, denominator),
        }
    }
}

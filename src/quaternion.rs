use std::{
    iter::Sum,
    ops::{Mul, MulAssign},
};

use num::traits::Float;

use crate::{
    float_ext::FloatExt,
    prelude::Matrix3x3,
    vector,
    vector_alias::{Vector, Vector3, Vector4},
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
/// A quaternion
pub struct Quaternion<T: Float> {
    components: Vector4<T>,
}

impl<T: Float> Quaternion<T> {
    /// Create a new quaternion.
    pub const fn new(components: [T; 4]) -> Self {
        Self {
            components: Vector::from_components(components),
        }
    }

    /// Create an identity quaternion.
    pub fn identity() -> Self {
        let zero = T::zero();
        let one = T::one();
        Self::new([zero, zero, zero, one])
    }

    /// Create a quaternion representing a rotation around the X axis.
    pub fn from_rotation_x(radians: T) -> Self {
        let zero = T::zero();
        let half_radians = radians * T::half();
        let sin = half_radians.sin();
        let cos = half_radians.cos();
        vector!(sin, zero, zero, cos).into()
    }

    /// Create a quaternion representing a rotation around the Y axis.
    pub fn from_rotation_y(radians: T) -> Self {
        let zero = T::zero();
        let half_radians = radians * T::half();
        let sin = half_radians.sin();
        let cos = half_radians.cos();
        vector!(zero, sin, zero, cos).into()
    }

    /// Create a quaternion representing a rotation around the Z axis.
    pub fn from_rotation_z(radians: T) -> Self {
        let zero = T::zero();
        let half_radians = radians * T::half();
        let sin = half_radians.sin();
        let cos = half_radians.cos();
        vector!(zero, zero, sin, cos).into()
    }

    /// Create a quaternion from euler angles.
    /// Assumes Y is up and Z is forward.
    /// This is equivalent to a `roll` rotation around the Z axis,
    /// followed by a `pitch` rotation around the X axis,
    /// followed by a `yaw` rotation around the Y axis.
    pub fn from_euler_yup(roll: T, pitch: T, yaw: T) -> Self
    where
        T: Sum,
    {
        Quaternion::from_rotation_z(roll)
            .and_then(&Quaternion::from_rotation_x(pitch))
            .and_then(&Quaternion::from_rotation_y(yaw))
    }

    /// Create a quaternion representing a rotation around an axis.
    pub fn from_axis_angle(axis: &Vector3<T>, radians: T) -> Self {
        let half_radians = radians * T::half();
        let sin = half_radians.sin();
        let cos = half_radians.cos();
        vector!(axis.x() * sin, axis.y() * sin, axis.z() * sin, cos).into()
    }

    /// Break the quaternion into a rotation around an axis (in radians).
    pub fn axis_angle(&self) -> (Vector3<T>, T)
    where
        T: Sum + 'static,
    {
        let norm = if self.w() > T::one() {
            self.normalized()
        } else {
            *self
        };
        let w = T::two() * norm.w().acos();
        let den = (T::one() - norm.w() * norm.w()).sqrt();
        if den >= T::epsilon() {
            (norm.xyz() / den, w)
        } else {
            (Vector3::unit_x(), T::zero())
        }
    }
    
    /// Calculate the length of the quaternion.
    pub fn length(&self) -> T
    where
        T: Sum,
    {
        self.components.length()
    }

    /// Calculate the squared length of the quaternion.
    pub fn length_squared(&self) -> T
    where
        T: Sum,
    {
        self.components.length_squared()
    }

    /// Creates a normalized copy of the quaternion.
    pub fn normalized(&self) -> Self
    where
        T: Sum + 'static,
    {
        Self {
            components: self.components.normalized(),
        }
    }

    /// Calculate the inverse of the quaternion.
    /// May panic if the quaternion has a length of zero.
    pub fn inverted(&self) -> Self
    where
        T: Sum,
    {
        let len_sq = self.length_squared();
        if len_sq.abs() < T::epsilon() {
            panic!("Cannot invert a zero-length quaternion");
        }
        let inv_norm = T::one() / len_sq;
        (self.components * vector!(-inv_norm, -inv_norm, -inv_norm, inv_norm)).into()
    }

    /// Interpolate between this quaternion and another using spherical linear interpolation
    pub fn slerp(&self, target: &Self, n: T) -> Self
    where
        T: Sum,
    {
        let cos_omega = self.components.dot(&target.components);
        let (cos_omega, flip) = if cos_omega < T::zero() {
            (-cos_omega, true)
        } else {
            (cos_omega, false)
        };

        let (s1, s2) = if cos_omega > T::one() - T::epsilon() {
            (T::one() - n, if flip { -n } else { n })
        } else {
            let omega = cos_omega.acos();
            let inv_sin_omega = T::one() / omega.sin();
            (
                ((T::one() - n) * omega).sin() * inv_sin_omega,
                if flip {
                    -(n * omega).sin() * inv_sin_omega
                } else {
                    (n * omega).sin() * inv_sin_omega
                },
            )
        };

        self.components
            .zip(&target.components, |a, b| s1 * *a + s2 * *b)
            .into()
    }

    /// Get the X component of the quaternion
    pub const fn x(&self) -> T {
        self.components.x()
    }

    /// Get the Y component of the quaternion
    pub const fn y(&self) -> T {
        self.components.y()
    }

    /// Get the Z component of the quaternion
    pub const fn z(&self) -> T {
        self.components.z()
    }

    /// Get the W component of the quaternion
    pub const fn w(&self) -> T {
        self.components.w()
    }

    /// Get the X, Y, and Z components of the quaternion
    pub const fn xyz(&self) -> Vector3<T> {
        self.components.xyz()
    }

    /// Concatenate this quaternion rotation and another to form a new combined quaternion rotation
    pub fn and_then(&self, next: &Self) -> Self
    where
        T: Sum,
    {
        let a = self;
        let b = next;
        let cross: Vector3<T> = a.components.xyz().cross(&b.components.xyz());
        let dot = a.components.xyz().dot(&b.components.xyz());
        vector!(
            a.x() * b.w() + b.x() * a.w() + cross.x(),
            a.y() * b.w() + b.y() * a.w() + cross.y(),
            a.z() * b.w() + b.z() * a.w() + cross.z(),
            a.w() * b.w() - dot,
        )
        .into()
    }

    /// Convert the rotation quaternion to a rotation matrix
    pub fn to_matrix(self) -> Matrix3x3<T>
    where
        T: Sum + 'static,
    {
        Matrix3x3::from(self)
    }
}

impl<T: Float> From<[T; 4]> for Quaternion<T> {
    fn from(v: [T; 4]) -> Self {
        Self {
            components: v.into(),
        }
    }
}

impl<T: Float> Into<[T; 4]> for Quaternion<T> {
    fn into(self) -> [T; 4] {
        self.components.into()
    }
}

impl<T: Float> From<Vector4<T>> for Quaternion<T> {
    fn from(v: Vector4<T>) -> Self {
        Self { components: v }
    }
}

impl<T: Float> Into<Vector4<T>> for Quaternion<T> {
    fn into(self) -> Vector4<T> {
        self.components
    }
}

impl<T: Float> Mul for Quaternion<T>
where
    T: Sum,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        rhs.and_then(&self)
    }
}

impl<T: Float> Mul<Quaternion<T>> for &Quaternion<T>
where
    T: Sum,
{
    type Output = Quaternion<T>;

    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        rhs.and_then(&self)
    }
}

impl<T: Float> Mul<&Self> for Quaternion<T>
where
    T: Sum,
{
    type Output = Self;

    fn mul(self, rhs: &Quaternion<T>) -> Self::Output {
        rhs.and_then(&self)
    }
}

impl<T: Float> Mul for &Quaternion<T>
where
    T: Sum,
{
    type Output = Quaternion<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        rhs.and_then(&self)
    }
}

impl<T: Float> MulAssign for Quaternion<T>
where
    T: Sum,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Float> MulAssign<&Quaternion<T>> for Quaternion<T>
where
    T: Sum,
{
    fn mul_assign(&mut self, rhs: &Quaternion<T>) {
        *self = *self * rhs;
    }
}
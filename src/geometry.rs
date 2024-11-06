use std::{iter::Sum, mem::swap, ops::Sub};

use num::traits::*;

use crate::prelude::*;

/// An n-dimensional box.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NBox<T: Copy + Zero + One, const DIMENSIONS: usize> {
    min: Vector<T, DIMENSIONS>,
    max: Vector<T, DIMENSIONS>,
}

impl<T: Copy + Zero + One, const DIMENSIONS: usize> NBox<T, DIMENSIONS> {
    pub fn from_corners(mut min: Vector<T, DIMENSIONS>, mut max: Vector<T, DIMENSIONS>) -> Self
    where
        T: PartialOrd,
    {
        // Ensure that min is less than max
        for idx in 0..DIMENSIONS {
            if min.component(idx).unwrap() > max.component(idx).unwrap() {
                swap(
                    min.component_mut(idx).unwrap(),
                    max.component_mut(idx).unwrap(),
                );
            }
        }
        Self { min, max }
    }

    /// Returns the minimum corner of the box.
    pub fn min(&self) -> &Vector<T, DIMENSIONS> {
        &self.min
    }

    /// Returns the maximum corner of the box.
    pub fn max(&self) -> &Vector<T, DIMENSIONS> {
        &self.max
    }

    /// Returns the center of the box.
    pub fn center(&self) -> Vector<T, DIMENSIONS>
    where
        T: Float + 'static,
    {
        (self.min + self.max) / T::two()
    }

    /// Returns the center of the box. Works for integer types.
    /// The result is rounded towards zero.
    pub fn center_int(&self) -> Vector<T, DIMENSIONS>
    where
        T: PrimInt + 'static,
    {
        (self.min + self.max) / (T::one() + T::one())
    }

    /// Returns the size of the box.
    pub fn size(&self) -> Vector<T, DIMENSIONS>
    where
        T: Sub<Output = T> + 'static,
    {
        self.max - self.min
    }

    /// Ensures that the minimum corner is less than the maximum corner.
    fn normalize(&mut self)
    where
        T: PartialOrd,
    {
        for idx in 0..DIMENSIONS {
            if self.min.component(idx).unwrap() > self.max.component(idx).unwrap() {
                swap(
                    self.min.component_mut(idx).unwrap(),
                    self.max.component_mut(idx).unwrap(),
                );
            }
        }
    }

    /// Sets the minimum corner of the box.
    /// Will swap the corners if the new minimum is greater than the maximum.
    pub fn set_min(&mut self, min: Vector<T, DIMENSIONS>)
    where
        T: PartialOrd,
    {
        self.min = min;
        self.normalize();
    }

    /// Sets the maximum corner of the box.
    /// Will swap the corners if the new maximum is less than the minimum.
    pub fn set_max(&mut self, max: Vector<T, DIMENSIONS>)
    where
        T: PartialOrd,
    {
        self.max = max;
        self.normalize();
    }

    /// Sets the center of the box.
    /// Will not change the size of the box.
    pub fn set_center(&mut self, center: Vector<T, DIMENSIONS>)
    where
        T: Float + 'static,
    {
        let half_size = self.size() / T::two();
        self.min = center - half_size;
        self.max = center + half_size;
    }

    /// Sets the center of the box; doesn't require the box to be constructed from floating points.
    /// Will not change the size of the box.
    pub fn set_center_int(&mut self, center: Vector<T, DIMENSIONS>)
    where
        T: PrimInt + 'static,
    {
        let old_center = self.center_int();
        let offset = center - old_center;
        self.min += offset;
        self.max += offset;
    }

    /// Sets the size of the box.
    /// Will not change the center of the box.
    /// If the size is negative, it will be treated as positive.
    pub fn set_size(&mut self, size: Vector<T, DIMENSIONS>)
    where
        T: Float + 'static,
    {
        let size = size.map(|x| x.abs());
        let center = self.center();
        let half_size = size / T::two();
        self.min = center - half_size;
        self.max = center + half_size;
        self.normalize();
    }

    /// Sets the size of the box; doesn't require the box to be constructed from floating points.
    /// Will not change the center of the box.
    /// If the size is negative, it will be treated as positive.
    /// If the size is even, the minimum corner will be the one that is closer to the center.
    /// The result is rounded towards zero.
    pub fn set_size_int(&mut self, size: Vector<T, DIMENSIONS>)
    where
        T: PrimInt + Sub<Output = T> + Signed + 'static,
    {
        let size = size.map(|x| x.abs());
        let center = self.center_int();
        let half_size = size / (T::one() + T::one());
        self.min = center - half_size;
        self.max = self.min + size;
    }

    /// Returns true if the box contains the given point.
    /// The box is considered to contain the point if the point is on the edge of the box.
    pub fn contains(&self, point: Vector<T, DIMENSIONS>) -> bool
    where
        T: PartialOrd,
    {
        for idx in 0..DIMENSIONS {
            if point.component(idx).unwrap() < self.min.component(idx).unwrap()
                || point.component(idx).unwrap() > self.max.component(idx).unwrap()
            {
                return false;
            }
        }
        true
    }
}

/// An n-dimensional line segment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NLineSegment<T: Copy + Zero + One, const DIMENSIONS: usize> {
    start: Vector<T, DIMENSIONS>,
    end: Vector<T, DIMENSIONS>,
}

impl<T: Copy + Zero + One, const DIMENSIONS: usize> NLineSegment<T, DIMENSIONS> {
    /// Creates a new line segment from the given start and end points.
    pub fn from_points(start: Vector<T, DIMENSIONS>, end: Vector<T, DIMENSIONS>) -> Self {
        Self { start, end }
    }

    /// Returns the start point of the line segment.
    pub fn start(&self) -> &Vector<T, DIMENSIONS> {
        &self.start
    }

    /// Returns the end point of the line segment.
    pub fn end(&self) -> &Vector<T, DIMENSIONS> {
        &self.end
    }

    /// Returns the length of the line segment.
    pub fn length(&self) -> T
    where
        T: Float + Sum + 'static,
    {
        (self.end - self.start).length()
    }

    /// Returns the squared length of the line segment.
    pub fn length_squared(&self) -> T
    where
        T: PrimInt + Sum + 'static,
    {
        (self.end - self.start).length_squared()
    }

    /// Returns the center of the line segment.
    pub fn center(&self) -> Vector<T, DIMENSIONS>
    where
        T: Float + 'static,
    {
        (self.start + self.end) / T::two()
    }

    /// Returns the center of the line segment; doesn't require the line segment to be constructed from floating points.
    /// The result is rounded towards zero.
    pub fn center_int(&self) -> Vector<T, DIMENSIONS>
    where
        T: PrimInt + 'static,
    {
        (self.start + self.end) / (T::one() + T::one())
    }

    /// Returns the direction of the line segment.
    /// The direction is a normalized vector pointing from the start to the end of the line segment.
    /// Returns None if the line segment has zero length.
    pub fn direction(&self) -> Option<Vector<T, DIMENSIONS>>
    where
        T: Float + Sum + 'static,
    {
        let direction = self.end - self.start;
        if direction.length_squared().is_zero() {
            None
        } else {
            Some(direction.normalized())
        }
    }

    /// Returns the line segment with the start and end points swapped.
    pub fn reversed(&self) -> Self {
        Self {
            start: self.end,
            end: self.start,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CollisionShape<T: Float + Sum, const DIMENSIONS: usize> {
    Box {
        min: Vector<T, DIMENSIONS>,
        max: Vector<T, DIMENSIONS>,
    },
    Sphere {
        center: Vector<T, DIMENSIONS>,
        radius: T,
    },
    Point {
        position: Vector<T, DIMENSIONS>,
    },
    LineSegment {
        start: Vector<T, DIMENSIONS>,
        end: Vector<T, DIMENSIONS>,
    },
    Plane {
        center: Vector<T, DIMENSIONS>,
        normal: Vector<T, DIMENSIONS>,
    },
}

pub trait Collide<T: Float + Sum + 'static, const DIMENSIONS: usize> {
    fn collision_shape(&self) -> CollisionShape<T, DIMENSIONS>;

    fn colliding<C: Collide<T, DIMENSIONS>>(&self, other: &C) -> bool {
        match self.collision_shape() {
            CollisionShape::Box {
                min: self_min,
                max: self_max,
            } => match other.collision_shape() {
                CollisionShape::Box {
                    min: other_min,
                    max: other_max,
                } => colliding_box_box(self_min, self_max, other_min, other_max),
                CollisionShape::Sphere {
                    center: other_center,
                    radius: other_radius,
                } => colliding_box_sphere(self_min, self_max, other_center, other_radius),
                CollisionShape::Point {
                    position: other_position,
                } => colliding_box_point(self_min, self_max, other_position),
                CollisionShape::LineSegment { start: _, end: _ } => {
                    unimplemented!("Box-LineSegment collision not implemented")
                }
                CollisionShape::Plane {
                    center: _,
                    normal: _,
                } => unimplemented!("Box-Plane collision not implemented"),
            },
            CollisionShape::Sphere {
                center: self_center,
                radius: self_radius,
            } => match other.collision_shape() {
                CollisionShape::Box {
                    min: other_min,
                    max: other_max,
                } => colliding_box_sphere(other_min, other_max, self_center, self_radius),
                CollisionShape::Sphere {
                    center: other_center,
                    radius: other_radius,
                } => colliding_sphere_sphere(self_center, self_radius, other_center, other_radius),
                CollisionShape::Point {
                    position: other_position,
                } => colliding_sphere_point(self_center, self_radius, other_position),
                CollisionShape::LineSegment { start: _, end: _ } => {
                    unimplemented!("Sphere-LineSegment collision not implemented")
                }
                CollisionShape::Plane {
                    center: _,
                    normal: _,
                } => unimplemented!("Sphere-Plane collision not implemented"),
            },
            CollisionShape::Point {
                position: self_position,
            } => match other.collision_shape() {
                CollisionShape::Box {
                    min: other_min,
                    max: other_max,
                } => colliding_box_point(other_min, other_max, self_position),
                CollisionShape::Sphere {
                    center: other_center,
                    radius: other_radius,
                } => colliding_sphere_point(other_center, other_radius, self_position),
                CollisionShape::Point {
                    position: other_position,
                } => self_position == other_position,
                CollisionShape::LineSegment { start: _, end: _ } => {
                    unimplemented!("Point-LineSegment collision not implemented")
                }
                CollisionShape::Plane {
                    center: _,
                    normal: _,
                } => unimplemented!("Point-Plane collision not implemented"),
            },
            CollisionShape::LineSegment { start: _, end: _ } => match other.collision_shape() {
                CollisionShape::Box { min: _, max: _ } => {
                    unimplemented!("LineSegment-Box collision not implemented")
                }
                CollisionShape::Sphere {
                    center: _,
                    radius: _,
                } => unimplemented!("LineSegment-Sphere collision not implemented"),
                CollisionShape::Point { position: _ } => {
                    unimplemented!("LineSegment-Point collision not implemented")
                }
                CollisionShape::LineSegment { start: _, end: _ } => {
                    unimplemented!("LineSegment-LineSegment collision not implemented")
                }
                CollisionShape::Plane {
                    center: _,
                    normal: _,
                } => unimplemented!("LineSegment-Plane collision not implemented"),
            },
            CollisionShape::Plane {
                center: _,
                normal: _,
            } => match other.collision_shape() {
                CollisionShape::Box { min: _, max: _ } => {
                    unimplemented!("Plane-Box collision not implemented")
                }
                CollisionShape::Sphere {
                    center: _,
                    radius: _,
                } => unimplemented!("Plane-Sphere collision not implemented"),
                CollisionShape::Point { position: _ } => {
                    unimplemented!("Plane-Point collision not implemented")
                }
                CollisionShape::LineSegment { start: _, end: _ } => {
                    unimplemented!("Plane-LineSegment collision not implemented")
                }
                CollisionShape::Plane {
                    center: _,
                    normal: _,
                } => unimplemented!("Plane-Plane collision not implemented"),
            },
        }
    }
}

/// Check if two boxes are colliding
fn colliding_box_box<T: Copy + Zero + One + PartialOrd, const DIMENSIONS: usize>(
    mut self_min: Vector<T, DIMENSIONS>,
    mut self_max: Vector<T, DIMENSIONS>,
    mut other_min: Vector<T, DIMENSIONS>,
    mut other_max: Vector<T, DIMENSIONS>,
) -> bool {
    // Fix the corners if they are incorrect
    for idx in 0..DIMENSIONS {
        if self_min.component(idx).unwrap() > self_max.component(idx).unwrap() {
            swap(
                self_min.component_mut(idx).unwrap(),
                self_max.component_mut(idx).unwrap(),
            );
        }
    }
    for idx in 0..DIMENSIONS {
        if other_min.component(idx).unwrap() > other_max.component(idx).unwrap() {
            swap(
                &mut other_min.component_mut(idx).unwrap(),
                &mut other_max.component_mut(idx).unwrap(),
            );
        }
    }
    // Check if the boxes are colliding
    for i in 0..DIMENSIONS {
        if self_min.as_slice()[i] > other_max.as_slice()[i]
            || self_max.as_slice()[i] < other_min.as_slice()[i]
        {
            return false;
        }
    }
    true
}

/// Check if a box and a sphere are colliding
fn colliding_box_sphere<T: Float + Sum + 'static, const DIMENSIONS: usize>(
    mut self_min: Vector<T, DIMENSIONS>,
    mut self_max: Vector<T, DIMENSIONS>,
    other_center: Vector<T, DIMENSIONS>,
    other_radius: T,
) -> bool {
    // Fix the corners if they are incorrect
    for idx in 0..DIMENSIONS {
        if self_min.component(idx).unwrap() > self_max.component(idx).unwrap() {
            swap(
                self_min.component_mut(idx).unwrap(),
                self_max.component_mut(idx).unwrap(),
            );
        }
    }
    // Find the closest point to the sphere on the box
    let mut closest_point = Vector::zero();
    for i in 0..DIMENSIONS {
        if other_center.as_slice()[i] < self_min.as_slice()[i] {
            closest_point.as_slice_mut()[i] = self_min.as_slice()[i];
        } else if other_center.as_slice()[i] > self_max.as_slice()[i] {
            closest_point.as_slice_mut()[i] = self_max.as_slice()[i];
        } else {
            closest_point.as_slice_mut()[i] = other_center.as_slice()[i];
        }
    }
    // Check if the closest point is inside the sphere
    (other_center - closest_point).length_squared() < other_radius * other_radius
}

// Check if a box and a point are colliding
fn colliding_box_point<T: Float + Sum + 'static, const DIMENSIONS: usize>(
    mut self_min: Vector<T, DIMENSIONS>,
    mut self_max: Vector<T, DIMENSIONS>,
    other_position: Vector<T, DIMENSIONS>,
) -> bool {
    // Fix the corners if they are incorrect
    for idx in 0..DIMENSIONS {
        if self_min.component(idx).unwrap() > self_max.component(idx).unwrap() {
            swap(
                self_min.component_mut(idx).unwrap(),
                self_max.component_mut(idx).unwrap(),
            );
        }
    }
    // Check if the point is inside the box
    for i in 0..DIMENSIONS {
        if other_position.as_slice()[i] < self_min.as_slice()[i]
            || other_position.as_slice()[i] > self_max.as_slice()[i]
        {
            return false;
        }
    }
    true
}

/// Check if two spheres are colliding
fn colliding_sphere_sphere<T: Float + Sum + 'static, const DIMENSIONS: usize>(
    self_center: Vector<T, DIMENSIONS>,
    self_radius: T,
    other_center: Vector<T, DIMENSIONS>,
    other_radius: T,
) -> bool {
    (self_center - other_center).length_squared() < (self_radius + other_radius).powi(2)
}

/// Check if a sphere and a point are colliding
fn colliding_sphere_point<T: Float + Sum + 'static, const DIMENSIONS: usize>(
    self_center: Vector<T, DIMENSIONS>,
    self_radius: T,
    other_position: Vector<T, DIMENSIONS>,
) -> bool {
    (self_center - other_position).length_squared() < self_radius.powi(2)
}

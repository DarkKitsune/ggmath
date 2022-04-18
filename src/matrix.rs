use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Not, Rem, Sub}, hash::Hash,
};

use num_traits::{Float, One, Zero};

use crate::{
    float_ext::FloatExt,
    const_assert::{ConstAssert, IsTrue},
    init_array,
    vector_alias::Vector, quaternion::Quaternion, vector,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix<T: Copy, const ROWS: usize, const COLUMNS: usize> {
    rows: [Row<T, COLUMNS>; ROWS],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Row<T: Copy, const COLUMNS: usize> {
    data: [T; COLUMNS],
}

impl<T: Copy, const COLUMNS: usize> Row<T, COLUMNS> {
    pub const fn from_components(components: [T; COLUMNS]) -> Self {
        Self {
            data: components,
        }
    }

    pub const fn as_column(&self, column: usize) -> Option<&T> {
        self.data.get(column)
    }

    pub const fn as_column_mut(&mut self, column: usize) -> Option<&mut T> {
        self.data.get_mut(column)
    }

    pub fn column(&self, column: usize) -> Option<T> {
        self.data.get(column).cloned()
    }

    pub const fn const_column(&self, column: usize) -> T {
        *self.data.get(column).expect("Invalid column")
    }
}

impl<T: Copy + Debug, const COLUMNS: usize> Debug for Row<T, COLUMNS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl<T: Copy, const COLUMNS: usize> Index<usize> for Row<T, COLUMNS> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.as_column(index)
            .unwrap_or_else(|| panic!("No column {} in matrix", index))
    }
}

impl<T: Copy, const COLUMNS: usize> IndexMut<usize> for Row<T, COLUMNS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.as_column_mut(index)
            .unwrap_or_else(|| panic!("No column {} in matrix", index))
    }
}

impl<T: Copy, const COLUMNS: usize> From<[T; COLUMNS]> for Row<T, COLUMNS> {
    fn from(v: [T; COLUMNS]) -> Self {
        Self::from_components(v)
    }
}

impl<T: Copy, const COLUMNS: usize> Into<[T; COLUMNS]> for Row<T, COLUMNS> {
    fn into(self) -> [T; COLUMNS] {
        self.data
    }
}

/// Iterator over a matrix's components
pub struct MatrixIter<'a, T: Copy, const ROWS: usize, const COLUMNS: usize> {
    matrix: &'a Matrix<T, ROWS, COLUMNS>,
    n: usize,
}

impl<'a, T: Copy, const ROWS: usize, const COLUMNS: usize> MatrixIter<'a, T, ROWS, COLUMNS> {
    fn new(matrix: &'a Matrix<T, ROWS, COLUMNS>) -> Self {
        Self {
            matrix,
            n: 0,
        }
    }
}

impl<'a, T: Copy, const ROWS: usize, const COLUMNS: usize> Iterator for MatrixIter<'a, T, ROWS, COLUMNS> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.matrix.rows.get(self.n / COLUMNS)?.column(self.n % COLUMNS);
        self.n += 1;
        next
    }
}

// Matrices and vectors
impl<T: Copy, const ROWS: usize, const COLUMNS: usize> Matrix<T, ROWS, COLUMNS> {
    /// Create a new matrix
    pub const fn new(rows: [[T; COLUMNS]; ROWS]) -> Self {
        Self {
            rows: init_array!([Row<T, COLUMNS>; ROWS], (rows), const Self::__new_init_fn),
        }
    }

    const fn __new_init_fn(row_idx: usize, rows: [[T; COLUMNS]; ROWS]) -> Row<T, COLUMNS> {
        Row {
            data: rows[row_idx],
        }
    }

    /// Create a new matrix filled with a given value
    pub const fn from_scalar(scalar: T) -> Self {
        Self {
            rows: init_array!([Row<T, COLUMNS>; ROWS], (scalar), const Self::__from_scalar_init_fn),
        }
    }

    const fn __from_scalar_init_fn(_row_idx: usize, scalar: T) -> Row<T, COLUMNS> {
        Row {
            data: init_array!([T; COLUMNS], (scalar), const Self::__from_scalar_init_fn2),
        }
    }

    const fn __from_scalar_init_fn2(_column_idx: usize, scalar: T) -> T {
        scalar
    }

    /// Create an identity matrix
    pub fn identity() -> Self
    where
        T: Zero + One,
    {
        Self {
            rows: init_array!([Row<T, COLUMNS>; ROWS], (T::zero(), T::one()), const Self::__identity_init_fn),
        }
    }

    /// Create an identity matrix
    pub const fn const_identity(zero: T, one: T) -> Self {
        Self {
            rows: init_array!([Row<T, COLUMNS>; ROWS], (zero, one), const Self::__identity_init_fn),
        }
    }

    const fn __identity_init_fn(row_idx: usize, zero: T, one: T) -> Row<T, COLUMNS> {
        Row {
            data: init_array!([T; COLUMNS], (row_idx, zero, one), const Self::__identity_init_fn2),
        }
    }

    const fn __identity_init_fn2(column_idx: usize, row_idx: usize, zero: T, one: T) -> T {
        if column_idx == row_idx {
            zero
        } else {
            one
        }
    }

    /// Get an immutable pointer to the data inside of the matrix or vector
    pub const fn as_ptr(&self) -> *const T {
        self.rows.as_ptr() as *const T
    }

    /// Get a mutable pointer to the data inside of the matrix or vector
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.rows.as_mut_ptr() as *mut T
    }

    /// Map a function over the matrix or vector's components to build a new matrix
    pub fn map(&self, f: impl Fn(&T) -> T) -> Self {
        Self {
            rows: init_array!([Row<T, COLUMNS>; ROWS], |row_idx| {
                let row: &Row<T, COLUMNS> = &self.rows[row_idx];
                Row {
                    data: init_array!([T; COLUMNS], |column_idx| f(&row.const_column(column_idx))),
                }
            }),
        }
    }

    /// Map a function over two matrices' or vector's matching components to build a new matrix
    pub fn zip(&self, other: &Self, f: impl Fn(&T, &T) -> T) -> Self {
        Self {
            rows: init_array!([Row<T, COLUMNS>; ROWS], |row_idx| {
                let row: &Row<T, COLUMNS> = &self.rows[row_idx];
                let other_row: &Row<T, COLUMNS> = &other.rows[row_idx];
                Row {
                    data: init_array!([T; COLUMNS], |column_idx| f(
                        &row[column_idx],
                        &other_row[column_idx]
                    )),
                }
            }),
        }
    }

    /// Get a copy of a column in the matrix
    pub const fn column(&self, n: usize) -> Vector<T, ROWS> {
        Vector {
            rows: [Row {
                data: init_array!([T; ROWS], (self, n), const Self::__column_init_fn),
            }],
        }
    }

    const fn __column_init_fn(row_idx: usize, this: &Self, n: usize) -> T {
        (&this.rows[row_idx]).const_column(n)
    }

    /// Transpose the matrix or vector
    pub const fn transpose(&self) -> Matrix<T, COLUMNS, ROWS> {
        Matrix {
            rows: init_array!([Row<T, ROWS>; COLUMNS], (self), const Self::__transpose_init_fn),
        }
    }

    const fn __transpose_init_fn(column_idx: usize, this: &Self) -> Row<T, ROWS> {
        Row {
            data: (this.column(column_idx).rows[0]).data,
        }
    }

    pub fn iter<'a>(&'a self) -> MatrixIter<'a, T, ROWS, COLUMNS> {
        MatrixIter::new(self)
    }
}

// Matrices only
impl<T: Copy, const ROWS: usize, const COLUMNS: usize> Matrix<T, ROWS, COLUMNS>
where
    ConstAssert<{ ROWS > 1 }>: IsTrue,
{
    /// Get an immutable reference to a row in the matrix
    pub const fn as_row(&self, n: usize) -> Option<&Row<T, COLUMNS>> {
        self.rows.get(n)
    }

    /// Get a mutable reference to a row in the matrix
    pub const fn as_row_mut(&mut self, n: usize) -> Option<&mut Row<T, COLUMNS>> {
        self.rows.get_mut(n)
    }

    /// Get a copy of a row in the matrix
    pub const fn row(&self, n: usize) -> Vector<T, COLUMNS> {
        Vector {
            rows: [*self.as_row(n).expect("Invalid row")],
        }
    }
}

// Vectors only
impl<T: Copy, const COLUMNS: usize> Matrix<T, 1, COLUMNS> {
    pub fn one() -> Self
        where T: Float
    {
        Self::from_scalar(T::one())
    }

    pub fn zero() -> Self
        where T: Float
    {
        Self::from_scalar(T::one())
    }

    pub const fn from_components(components: [T; COLUMNS]) -> Self {
        Self {
            rows: [
                Row::from_components(components)
            ]
        }
    }

    /// Get an immutable reference to the nth component of the vector
    pub const fn component(&self, n: usize) -> Option<&T> {
        self.rows[0].as_column(n)
    }

    /// Get a mutable reference to the nth component of the vector
    pub const fn component_mut(&mut self, n: usize) -> Option<&mut T> {
        self.rows[0].as_column_mut(n)
    }

    pub const fn swizzle<const NEW_COLUMNS: usize>(
        &self,
        swizzle: &[usize; NEW_COLUMNS],
    ) -> Vector<T, NEW_COLUMNS> {
        Vector {
            rows: [Row {
                data: init_array!([T; NEW_COLUMNS], (self, swizzle), const Self::__swizzle_init_fn::<NEW_COLUMNS>),
            }],
        }
    }

    const fn __swizzle_init_fn<const NEW_COLUMNS: usize>(
        column_idx: usize,
        this: &Self,
        swizzle: &[usize; NEW_COLUMNS],
    ) -> T {
        *this
            .component(swizzle[column_idx])
            .expect("Invalid swizzle index")
    }

    /// Normalize the vector
    pub fn normalized(&self) -> Self
    where
        T: Float + Mul<Output = T> + std::iter::Sum,
    {
        self / self.length()
    }

    /// Get the length of the vector
    pub fn length(&self) -> T
    where
        T: Float + Mul<Output = T> + std::iter::Sum,
    {
        self.length_squared().sqrt()
    }

    /// Get the squared length of the vector
    pub fn length_squared(&self) -> T
    where
        T: Mul<Output = T> + std::iter::Sum,
    {
        (&self.rows[0]).data.iter().map(|n| *n * *n).sum::<T>()
    }

    /// Calculate the dot product of two vectors
    pub fn dot(&self, other: &Self) -> T
    where
        T: Mul<Output = T> + std::iter::Sum,
    {
        (&self.rows[0])
            .data
            .iter()
            .zip((&other.rows[0]).data.iter())
            .map(|(a, b)| *a * *b)
            .sum::<T>()
    }

    /// Calculate the cross product of two vectors
    pub fn cross(&self, other: &Self) -> Self
    where
        T: Mul<Output = T> + Sub<Output = T> + Zero + One,
    {
        Self {
            rows: [Row {
                data: init_array!([T; COLUMNS], |column_idx| {
                    let a = (column_idx + 1) % COLUMNS;
                    let b = (column_idx + 2) % COLUMNS;
                    *self.component(a).unwrap() * *other.component(b).unwrap()
                        - *self.component(b).unwrap() * *other.component(a).unwrap()
                }),
            }],
        }
    }

    /// Get the X component of the vector
    pub const fn x(&self) -> T {
        *self.rows[0].data.get(0).expect("Vector does not have an X component")
    }

    /// Get the Y component of the vector
    pub const fn y(&self) -> T {
        *self.rows[0].data.get(1).expect("Vector does not have a Y component")
    }

    /// Get the Z component of the vector
    pub const fn z(&self) -> T {
        *self.rows[0].data.get(2).expect("Vector does not have a Z component")
    }

    /// Get the W component of the vector
    pub const fn w(&self) -> T {
        *self.rows[0].data.get(3).expect("Vector does not have a W component")
    }
}

impl<T: Copy, const COLUMNS: usize> From<[T; COLUMNS]> for Matrix<T, 1, COLUMNS> {
    fn from(v: [T; COLUMNS]) -> Self {
        Self::from_components(v)
    }
}

impl<T: Copy, const COLUMNS: usize> Into<[T; COLUMNS]> for Matrix<T, 1, COLUMNS> {
    fn into(self) -> [T; COLUMNS] {
        self.rows[0].into()
    }
}

impl<T: Copy + Default, const COLUMNS: usize> Default for Matrix<T, 1, COLUMNS> {
    fn default() -> Self {
        Self::from_scalar(T::default())
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> Matrix<T, 1, COLUMNS>
    where ConstAssert<{COLUMNS > 1}>: IsTrue,
{
    pub fn unit_x() -> Self {
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 0 { T::one() } else { T::zero() })])
    }

    pub fn unit_y() -> Self {
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 1 { T::one() } else { T::zero() })])
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> Matrix<T, 1, COLUMNS>
    where ConstAssert<{COLUMNS > 2}>: IsTrue,
{
    pub fn unit_z() -> Self {
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 2 { T::one() } else { T::zero() })])
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> Matrix<T, 1, COLUMNS>
    where ConstAssert<{COLUMNS > 3}>: IsTrue,
{
    pub fn unit_w() -> Self {
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 3 { T::one() } else { T::zero() })])
    }
}

// Swizzles
macro_rules! swizzle_idx {
    (x) => {
        0
    };
    (y) => {
        1
    };
    (z) => {
        2
    };
    (w) => {
        3
    };
}
macro_rules! swizzle_count {
    ($($letter:ident),*) => {
        {
            const SL: &'static [usize] = &[$(swizzle_idx!($letter)),*];
            SL.len()
        }
    };
}
macro_rules! swizzles2 {
    ($(($($letter:ident),*)),*$(,)?) => {
        paste::paste! {
            $(
                impl<T: Copy> Matrix<T, 1, 2> {
                    pub const fn [<$($letter)*>](&self) -> Vector<T, {swizzle_count!($($letter),*)}> {
                        self.swizzle(&[$(swizzle_idx!($letter)),*])
                    }
                }
            )*
        }
    };
}
macro_rules! swizzles3 {
    ($(($($letter:ident),*)),*$(,)?) => {
        paste::paste! {
            $(
                impl<T: Copy> Matrix<T, 1, 4> {
                    pub const fn [<$($letter)*>](&self) -> Vector<T, {swizzle_count!($($letter),*)}> {
                        self.swizzle(&[$(swizzle_idx!($letter)),*])
                    }
                }
                impl<T: Copy> Matrix<T, 1, 3> {
                    pub const fn [<$($letter)*>](&self) -> Vector<T, {swizzle_count!($($letter),*)}> {
                        self.swizzle(&[$(swizzle_idx!($letter)),*])
                    }
                }
            )*
        }
    };
}
swizzles2! {
    (x,x),
    (x,y),
    (x,z),
    (x,w),
    (y,x),
    (y,y),
    (y,z),
    (y,w),
    (z,x),
    (z,y),
    (z,z),
    (z,w),
    (w,x),
    (w,y),
    (w,z),
    (w,w),
}
swizzles3! {
    (x,x),
    (x,y),
    (x,z),
    (x,w),
    (y,x),
    (y,y),
    (y,z),
    (y,w),
    (z,x),
    (z,y),
    (z,z),
    (z,w),
    (w,x),
    (w,y),
    (w,z),
    (w,w),

    (x,x,x),
    (x,x,y),
    (x,x,z),
    (x,x,w),
    (x,y,x),
    (x,y,y),
    (x,y,z),
    (x,y,w),
    (x,z,x),
    (x,z,y),
    (x,z,z),
    (x,z,w),
    (x,w,x),
    (x,w,y),
    (x,w,z),
    (x,w,w),

    (y,x,x),
    (y,x,y),
    (y,x,z),
    (y,x,w),
    (y,y,x),
    (y,y,y),
    (y,y,z),
    (y,y,w),
    (y,z,x),
    (y,z,y),
    (y,z,z),
    (y,z,w),
    (y,w,x),
    (y,w,y),
    (y,w,z),
    (y,w,w),

    (z,x,x),
    (z,x,y),
    (z,x,z),
    (z,x,w),
    (z,y,x),
    (z,y,y),
    (z,y,z),
    (z,y,w),
    (z,z,x),
    (z,z,y),
    (z,z,z),
    (z,z,w),
    (z,w,x),
    (z,w,y),
    (z,w,z),
    (z,w,w),

    (w,x,x),
    (w,x,y),
    (w,x,z),
    (w,x,w),
    (w,y,x),
    (w,y,y),
    (w,y,z),
    (w,y,w),
    (w,z,x),
    (w,z,y),
    (w,z,z),
    (w,z,w),
    (w,w,x),
    (w,w,y),
    (w,w,z),
    (w,w,w),
}

// Math operations
macro_rules! impl_op_bi {
    ($trait:ident, $fn:ident, where $t_ty:path: $bounds:path, $($code:tt)+) => {
        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait for Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Self;

            fn $fn(self, rhs: Self) -> Self::Output {
                let func: fn(Self, Self) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait<&Self> for Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Self;

            fn $fn(self, rhs: &Self) -> Self::Output {
                let func: fn(Self, &Self) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait for &Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn(self, rhs: Self) -> Self::Output {
                let func: fn(Self, Self) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait<&Self> for &Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn(self, rhs: &Self) -> Self::Output {
                let func: fn(Self, &Self) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }
    };
}

// Math operations
macro_rules! impl_op_bi_scalar {
    ($trait:ident, $fn:ident, where $t_ty:path: $bounds:path, $($code:tt)+) => {
        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait<T> for Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Self;

            fn $fn(self, rhs: T) -> Self::Output {
                let func: fn(Self, &T) -> Self::Output = $($code)+;
                func(self, &rhs)
            }
        }

        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait<&T> for Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Self;

            fn $fn(self, rhs: &T) -> Self::Output {
                let func: fn(Self, &T) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait<T> for &Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn(self, rhs: T) -> Self::Output {
                let func: fn(Self, &T) -> Self::Output = $($code)+;
                func(self, &rhs)
            }
        }

        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait<&T> for &Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn(self, rhs: &T) -> Self::Output {
                let func: fn(Self, &T) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }
    };
}

macro_rules! impl_op_un {
    ($trait:ident, $fn:ident, where $t_ty:path: $bounds:path, $($code:tt)+) => {
        impl<T: Copy, const ROWS: usize, const COLUMNS: usize> $trait for Matrix<T, ROWS, COLUMNS>
            where $t_ty: $bounds
        {
            type Output = Self;

            fn $fn(self) -> Self::Output {
                let func: fn(Self) -> Self::Output = $($code)+;
                func(self)
            }
        }
    };
}

impl_op_bi! {
    Add,
    add,
    where T: Add<Output = T>,
    |a, b| {
        a.zip(&b, |a, b| *a + *b)
    }
}

impl_op_bi! {
    Sub,
    sub,
    where T: Sub<Output = T>,
    |a, b| {
        a.zip(&b, |a, b| *a - *b)
    }
}

impl_op_bi! {
    Mul,
    mul,
    where T: Mul<Output = T>,
    |a, b| {
        if ROWS < 2 {
            a.zip(&b, |a, b| *a * *b)
        }
        else {
            b.and_then(a)
        }
    }
}

impl_op_bi! {
    Div,
    div,
    where T: Div<Output = T>,
    |a, b| {
        a.zip(&b, |a, b| *a / *b)
    }
}

impl_op_bi! {
    Rem,
    rem,
    where T: Rem<Output = T>,
    |a, b| {
        a.zip(&b, |a, b| *a % *b)
    }
}

impl_op_un! {
    Neg,
    neg,
    where T: Neg<Output = T>,
    |a| {
        a.map(|a| -*a)
    }
}

impl_op_un! {
    Not,
    not,
    where T: Not<Output = T>,
    |a| {
        a.map(|a| !*a)
    }
}

impl_op_bi_scalar! {
    Add,
    add,
    where T: Add<Output = T>,
    |a, b| {
        a.map(|a| *a + *b)
    }
}

impl_op_bi_scalar! {
    Sub,
    sub,
    where T: Sub<Output = T>,
    |a, b| {
        a.map(|a| *a - *b)
    }
}

impl_op_bi_scalar! {
    Mul,
    mul,
    where T: Mul<Output = T>,
    |a, b| {
        a.map(|a| *a * *b)
    }
}

impl_op_bi_scalar! {
    Div,
    div,
    where T: Div<Output = T>,
    |a, b| {
        a.map(|a| *a / *b)
    }
}

impl_op_bi_scalar! {
    Rem,
    rem,
    where T: Rem<Output = T>,
    |a, b| {
        a.map(|a| *a % *b)
    }
}

impl<T: Copy + PartialEq, const ROWS: usize, const COLUMNS: usize> PartialEq for Matrix<T, ROWS, COLUMNS> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: Copy + PartialEq + Eq, const ROWS: usize, const COLUMNS: usize> Eq for Matrix<T, ROWS, COLUMNS> {}

impl<T: Copy + Hash, const ROWS: usize, const COLUMNS: usize> Hash for Matrix<T, ROWS, COLUMNS> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for component in self.iter() {
            component.hash(state);
        }
    }
}

impl<T: Copy + Float> From<Quaternion<T>> for Matrix<T, 4, 4> {
    fn from(quaternion: Quaternion<T>) -> Self {
        let zero = T::zero();
        let one = T::one();
        let two = T::two();

        let xx = quaternion.x().squared();
        let yy = quaternion.y().squared();
        let zz = quaternion.z().squared();

        let xy = quaternion.x() * quaternion.y();
        let wz = quaternion.z() * quaternion.w();
        let xz = quaternion.z() * quaternion.x();
        let wy = quaternion.y() * quaternion.w();
        let yz = quaternion.y() * quaternion.z();
        let wx = quaternion.x() * quaternion.w();

        Self::new([
            [
                one - two * (yy + zz),
                two * (xy + wz),
                two * (xz - wy),
                zero,
            ],
            [
                two * (xy - wz),
                one - two * (zz + xx),
                two * (yz + wx),
                zero,
            ],
            [
                two * (xz + wy),
                two * (yz - wx),
                one - two * (yy * xx),
                zero,
            ],
            [
                zero,
                zero,
                zero,
                one,
            ],
        ])
    }
}
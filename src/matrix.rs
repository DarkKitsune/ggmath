use std::{
    convert::identity,
    fmt::Debug,
    hash::Hash,
    iter::Sum,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Not, Rem, Sub, AddAssign},
};

use num_traits::{Float, One, Zero};

use crate::{
    float_ext::FloatExt,
    init_array,
    quaternion::Quaternion,
    vector,
    vector_alias::Vector,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// A matrix
pub struct Matrix<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> {
    rows: [Row<T, COLUMNS>; ROWS],
}

#[repr(C)]
#[derive(Clone, Copy)]
/// A row of a matrix
pub struct Row<T: Copy + Zero + One, const COLUMNS: usize> {
    data: [T; COLUMNS],
}

impl<T: Copy + Zero + One, const COLUMNS: usize> Row<T, COLUMNS> {
    /// Creates a new row using the given values.
    pub const fn from_components(components: [T; COLUMNS]) -> Self {
        Self { data: components }
    }

    /// Returns a reference to the `column`th column of this row.
    pub const fn as_column(&self, column: usize) -> Option<&T> {
        self.data.get(column)
    }

    /// Returns a mutable reference to the `column`th column of this row.
    pub const fn as_column_mut(&mut self, column: usize) -> Option<&mut T> {
        self.data.get_mut(column)
    }

    /// Returns a copy of the `column`th column of this row.
    pub fn column(&self, column: usize) -> Option<T> {
        self.data.get(column).cloned()
    }

    /// Returns a copy of the `column`th column of this row.
    pub const fn const_column(&self, column: usize) -> T {
        *self.data.get(column).expect("Invalid column")
    }
}

impl<T: Copy + Zero + One + Debug, const COLUMNS: usize> Debug for Row<T, COLUMNS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> Index<usize> for Row<T, COLUMNS> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.as_column(index)
            .unwrap_or_else(|| panic!("No column {} in matrix", index))
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> IndexMut<usize> for Row<T, COLUMNS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.as_column_mut(index)
            .unwrap_or_else(|| panic!("No column {} in matrix", index))
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> From<[T; COLUMNS]> for Row<T, COLUMNS> {
    fn from(v: [T; COLUMNS]) -> Self {
        Self::from_components(v)
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> Into<[T; COLUMNS]> for Row<T, COLUMNS> {
    fn into(self) -> [T; COLUMNS] {
        self.data
    }
}

/// Iterator over a matrix's components
pub struct MatrixIter<'a, T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> {
    matrix: &'a Matrix<T, ROWS, COLUMNS>,
    n: usize,
}

impl<'a, T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize>
    MatrixIter<'a, T, ROWS, COLUMNS>
{
    /// Creates a new iterator over the given matrix.
    fn new(matrix: &'a Matrix<T, ROWS, COLUMNS>) -> Self {
        Self { matrix, n: 0 }
    }
}

impl<'a, T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> Iterator
    for MatrixIter<'a, T, ROWS, COLUMNS>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self
            .matrix
            .rows
            .get(self.n / COLUMNS)?
            .column(self.n % COLUMNS);
        self.n += 1;
        next
    }
}

// Matrices and vectors
impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> Matrix<T, ROWS, COLUMNS> {
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
    pub fn identity() -> Self {
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
            one
        } else {
            zero
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

    /// Concatenate one transformation matrix to another.
    /// The left hand matrix must have equal or greater columns to the right hand matrix's rows
    pub fn and_then<const OTHER_ROWS: usize, const OTHER_COLUMNS: usize>(
        &self,
        other: &Matrix<T, OTHER_ROWS, OTHER_COLUMNS>,
    ) -> Self
    where
        T: Sum + Mul<Output = T>,
    {
        if COLUMNS < OTHER_ROWS {
            panic!("Matrix::and_then: left hand matrix must have equal or greater columns to the right hand matrix's rows");
        } else {
            Self {
                rows: init_array!([Row<T, COLUMNS>; ROWS], |row_idx| {
                    Row {
                        data: init_array!([T; COLUMNS], |column_idx| {
                            (0..OTHER_ROWS)
                                .map(|i| {
                                    identity::<&Row<T, COLUMNS>>(&self.rows[row_idx])
                                        .const_column(i)
                                        * identity::<&Row<T, OTHER_COLUMNS>>(&other.rows[i])
                                            .const_column(column_idx)
                                })
                                .sum()
                        }),
                    }
                }),
            }
        }
    }

    pub const fn as_size<const NEW_ROWS: usize, const NEW_COLUMNS: usize>(
        &self,
        filler: T,
    ) -> Matrix<T, NEW_ROWS, NEW_COLUMNS> {
        Matrix {
            rows: init_array!([Row<T, NEW_COLUMNS>; NEW_ROWS], (self, filler), const Self::__as_size_init_fn::<NEW_COLUMNS>),
        }
    }

    pub const fn __as_size_init_fn<const NEW_COLUMNS: usize>(
        row_idx: usize,
        this: &Self,
        filler: T,
    ) -> Row<T, NEW_COLUMNS> {
        Row {
            data: init_array!([T; NEW_COLUMNS], (row_idx, this, filler), const Self::__as_size_init_fn2),
        }
    }

    pub const fn __as_size_init_fn2(
        column_idx: usize,
        row_idx: usize,
        this: &Self,
        filler: T,
    ) -> T {
        if row_idx < ROWS && column_idx < COLUMNS {
            this.rows[row_idx].const_column(column_idx)
        } else {
            filler
        }
    }
}

// All matrices & vectors
impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> Matrix<T, ROWS, COLUMNS> {
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

    /// Create a rotation matrix with the Z axis facing in the 'forward' direction and the Y axis facing in the 'up' direction
    /// If the 'forward' and 'up' vectors are not orthogonal, the matrix will be orthogonalized along the 'up' direction
    pub fn from_forward_up(forward: &Vector<T, 3>, up: &Vector<T, 3>) -> Self
    where
        T: Mul<Output = T> + Sub<Output = T> + Neg<Output = T>,
    {
        #[cfg(debug_assertions)]
        {
            if ROWS < 3 {
                panic!("Matrix must have at least 3 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        let mut matrix = Self::identity();
        let right = Vector::derive_right(forward, up);
        let forward = Vector::derive_forward(&right, up);
        // Set X row
        let row = matrix.as_row_mut(0).unwrap();
        row.data[0] = right.x();
        row.data[1] = right.y();
        row.data[2] = right.z();
        // Set Y row
        let row = matrix.as_row_mut(1).unwrap();
        row.data[0] = up.x();
        row.data[1] = up.y();
        row.data[2] = up.z();
        // Set Z row
        let row = matrix.as_row_mut(2).unwrap();
        row.data[0] = -forward.x();
        row.data[1] = -forward.y();
        row.data[2] = -forward.z();
        // Return the matrix
        matrix
    }

    /// Set the scale component of a transformation matrix
    pub fn set_scale(&mut self, scale: &Vector<T, 3>) {
        #[cfg(debug_assertions)]
        {
            if ROWS < 3 {
                panic!("Matrix must have at least 3 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        self.rows[0].data[0] = scale.x();
        self.rows[1].data[1] = scale.y();
        self.rows[2].data[2] = scale.z();
    }

    /// Returns the component part of a transformation matrix
    pub const fn scale(&self) -> Vector<T, 3> {
        #[cfg(debug_assertions)]
        {
            if ROWS < 3 {
                panic!("Matrix must have at least 3 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        Vector::from_components([
            self.rows[0].const_column(0),
            self.rows[1].const_column(1),
            self.rows[2].const_column(2),
        ])
    }

    pub fn new_scale(scale: &Vector<T, 3>) -> Self {
        #[cfg(debug_assertions)]
        {
            if ROWS < 3 {
                panic!("Matrix must have at least 3 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        let mut matrix = Self::identity();
        matrix.set_scale(scale);
        matrix
    }

    /// Get the X axis component of a transformation matrix
    pub fn x_axis(&self) -> Vector<T, 3> {
        #[cfg(debug_assertions)]
        {
            if ROWS < 1 {
                panic!("Matrix must have at least 1 row");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        Vector::from_components(self.row(0).as_slice()[0..3].try_into().unwrap())
    }

    /// Set the X axis component of a transformation matrix
    /// Note: This will not be orthogonalized
    pub fn set_x_axis(&mut self, x_axis: &Vector<T, 3>) {
        #[cfg(debug_assertions)]
        {
            if ROWS < 1 {
                panic!("Matrix must have at least 1 row");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        self.rows[0].data[0] = x_axis.x();
        self.rows[0].data[1] = x_axis.y();
        self.rows[0].data[2] = x_axis.z();
    }

    /// Get the Y axis component of a transformation matrix
    pub fn y_axis(&self) -> Vector<T, 3> {
        #[cfg(debug_assertions)]
        {
            if ROWS < 2 {
                panic!("Matrix must have at least 2 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        Vector::from_components(self.row(1).as_slice()[0..3].try_into().unwrap())
    }

    /// Set the Y axis component of a transformation matrix
    /// Note: This will not be orthogonalized
    pub fn set_y_axis(&mut self, y_axis: &Vector<T, 3>) {
        #[cfg(debug_assertions)]
        {
            if ROWS < 2 {
                panic!("Matrix must have at least 2 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        self.rows[1].data[0] = y_axis.x();
        self.rows[1].data[1] = y_axis.y();
        self.rows[1].data[2] = y_axis.z();
    }

    /// Get the Z axis component of a transformation matrix
    pub fn z_axis(&self) -> Vector<T, 3> {
        #[cfg(debug_assertions)]
        {
            if ROWS < 3 {
                panic!("Matrix must have at least 3 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        Vector::from_components(self.row(2).as_slice()[0..3].try_into().unwrap())
    }

    /// Set the Z axis component of a transformation matrix
    /// Note: This will not be orthogonalized
    pub fn set_z_axis(&mut self, z_axis: &Vector<T, 3>) {
        #[cfg(debug_assertions)]
        {
            if ROWS < 3 {
                panic!("Matrix must have at least 3 rows");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        self.rows[2].data[0] = z_axis.x();
        self.rows[2].data[1] = z_axis.y();
        self.rows[2].data[2] = z_axis.z();
    }

    /// Sets the translation component of a transformation matrix
    pub fn set_translation(&mut self, translation: &Vector<T, 3>) {
        #[cfg(debug_assertions)]
        {
            if ROWS < 1 {
                panic!("Matrix must have at least 1 row");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        self.rows[ROWS - 1].data[0] = translation.x();
        self.rows[ROWS - 1].data[1] = translation.y();
        self.rows[ROWS - 1].data[2] = translation.z();
    }

    /// Returns the translation component of a transformation matrix
    pub fn translation(self) -> Vector<T, 3> {
        #[cfg(debug_assertions)]
        {
            if ROWS < 1 {
                panic!("Matrix must have at least 1 row");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        Vector::from_components(self.row(ROWS - 1).as_slice()[0..2].try_into().unwrap())
    }

    /// Creates a transformation matrix that applies the given translation
    pub fn new_translation(translation: &Vector<T, 3>) -> Self {
        #[cfg(debug_assertions)]
        {
            if ROWS < 1 {
                panic!("Matrix must have at least 1 row");
            }
            if COLUMNS < 3 {
                panic!("Matrix must have at least 3 columns");
            }
        }
        let mut matrix = Self::identity();
        let translation_row = matrix.as_row_mut(ROWS - 1).unwrap();
        translation_row.data[0] = translation.x();
        translation_row.data[1] = translation.y();
        translation_row.data[2] = translation.z();
        matrix
    }
}

// 4x4 matrices only
impl<T: Copy + Float> Matrix<T, 4, 4> {
    /// Creates a perspective projection matrix
    pub fn new_projection_perspective(fov: T, aspect_ratio: T, near: T, far: T) -> Self {
        #[cfg(debug_assertions)]
        {
            if fov <= T::zero() || fov >= T::from(std::f64::consts::PI).unwrap() {
                panic!("Invalid field of view");
            }
            if aspect_ratio <= T::zero() {
                panic!("Invalid aspect ratio");
            }
            if near <= T::zero() || near >= far {
                panic!("Invalid near/far planes");
            }
        }
        let y_scale = T::one() / (fov * T::half()).tan();
        let x_scale = y_scale / aspect_ratio;
        let z_scale = far / (near - far);
        let mut matrix = Self::new_scale(&vector!(x_scale, y_scale, z_scale));
        matrix.as_row_mut(2).unwrap().data[3] = -T::one();
        matrix.as_row_mut(3).unwrap().data[2] = near * z_scale;
        matrix
    }

    /// Creates an orthographic projection matrix
    pub fn new_projection_orthographic(size: Vector<T, 2>, near: T, far: T) -> Self {
        #[cfg(debug_assertions)]
        {
            if size.x() <= T::zero() || size.y() <= T::zero() {
                panic!("Invalid size");
            }
            if near <= T::zero() || near >= far {
                panic!("Invalid near/far planes");
            }
        }
        let x_scale = T::two() / size.x();
        let y_scale = T::two() / size.y();
        let z_range = near - far;
        let z_scale = T::one() / z_range;
        let mut matrix = Self::new_scale(&vector!(x_scale, y_scale, z_scale));
        matrix.as_row_mut(3).unwrap().data[2] = near / z_range;
        matrix
    }

    /// Creates a view matrix
    pub fn new_view(position: &Vector<T, 3>, target: &Vector<T, 3>, up: &Vector<T, 3>) -> Self
    where
        T: Sum + 'static,
    {
        #[cfg(debug_assertions)]
        {
            if (target - position).length() < T::epsilon() {
                panic!("Target and position are equal or too close");
            }
            if up.length() < T::epsilon() {
                panic!("Up vector is too close to zero");
            }
        }
        let mut matrix = Self::identity();
        // Calculate axes
        let z = (position - target).normalized();
        let x = up.cross(&z).normalized();
        let y = z.cross(&x);
        // Set X row
        let row = matrix.as_row_mut(0).unwrap();
        row.data[0] = x.x();
        row.data[1] = y.x();
        row.data[2] = z.x();
        // Set Y row
        let row = matrix.as_row_mut(1).unwrap();
        row.data[0] = x.y();
        row.data[1] = y.y();
        row.data[2] = z.y();
        // Set Z row
        let row = matrix.as_row_mut(2).unwrap();
        row.data[0] = x.z();
        row.data[1] = y.z();
        row.data[2] = z.z();
        // Set translation row
        let row = matrix.as_row_mut(3).unwrap();
        row.data[0] = -position.dot(&x);
        row.data[1] = -position.dot(&y);
        row.data[2] = -position.dot(&z);
        // Return the matrix
        matrix
    }
}

// Vectors only
impl<T: Copy + Zero + One, const COLUMNS: usize> Matrix<T, 1, COLUMNS> {
    /// Create a vector whose components are T::one()
    pub fn one() -> Self
    where
        T: Float,
    {
        Self::from_scalar(T::one())
    }

    /// Create a vector whose components are T::zero()
    pub fn zero() -> Self
    where
        T: Float,
    {
        Self::from_scalar(T::zero())
    }

    /// Create a vector from the given values
    pub const fn from_components(components: [T; COLUMNS]) -> Self {
        Self {
            rows: [Row::from_components(components)],
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

    /// Create a new vector from this vector's components
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
        T: Float + Mul<Output = T> + std::iter::Sum + 'static,
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
        #[cfg(debug_assertions)]
        {
            if COLUMNS < 1 {
                panic!("Vector must have at least 1 component");
            }
        }
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
        T: Mul<Output = T> + Sub<Output = T>,
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
        *self.rows[0]
            .data
            .get(0)
            .expect("Vector does not have an X component")
    }

    /// Get the Y component of the vector
    pub const fn y(&self) -> T {
        *self.rows[0]
            .data
            .get(1)
            .expect("Vector does not have a Y component")
    }

    /// Get the Z component of the vector
    pub const fn z(&self) -> T {
        *self.rows[0]
            .data
            .get(2)
            .expect("Vector does not have a Z component")
    }

    /// Get the W component of the vector
    pub const fn w(&self) -> T {
        *self.rows[0]
            .data
            .get(3)
            .expect("Vector does not have a W component")
    }

    /// Concatenate two vectors
    pub const fn concat<const EXTRA_COMPONENTS: usize>(
        &self,
        other: &Vector<T, EXTRA_COMPONENTS>,
    ) -> Vector<T, { COLUMNS + EXTRA_COMPONENTS }> {
        Vector::from_components(
            init_array!([T; COLUMNS + EXTRA_COMPONENTS], (self, other), const Self::__concat_init_fn),
        )
    }

    const fn __concat_init_fn<const EXTRA_COMPONENTS: usize>(
        column_idx: usize,
        this: &Self,
        other: &Vector<T, EXTRA_COMPONENTS>,
    ) -> T {
        if column_idx < EXTRA_COMPONENTS {
            *this.component(column_idx).unwrap()
        } else {
            *other.component(column_idx - EXTRA_COMPONENTS).unwrap()
        }
    }

    /// Get the vector's components as a slice
    pub const fn as_slice(&self) -> &[T] {
        &self.rows[0].data
    }

    /// Get the vector's components as a reference to an array
    pub const fn as_array(&self) -> &[T; COLUMNS] {
        &self.rows[0].data
    }

    /// Use a 'right' and 'up' vector to calculate the 'forward' vector
    pub fn derive_forward(right: &Self, up: &Self) -> Self
    where
        T: Mul<Output = T> + Sub<Output = T>,
    {
        up.cross(right)
    }

    /// Use a 'forward' and 'up' vector to calculate the 'right' vector
    pub fn derive_right(forward: &Self, up: &Self) -> Self
    where
        T: Mul<Output = T> + Sub<Output = T>,
    {
        forward.cross(up)
    }

    /// Use a 'forward' and 'right' vector to calculate the 'up' vector
    pub fn derive_up(forward: &Self, right: &Self) -> Self
    where
        T: Mul<Output = T> + Sub<Output = T>,
    {
        right.cross(forward)
    }

    /// Create a vector with its X component set to 1 and all other components set to 0
    pub fn unit_x() -> Self {
        if COLUMNS == 0 {
            panic!("Vector cannot have an X component");
        }
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 0 {
            T::one()
        } else {
            T::zero()
        })])
    }

    /// Create a vector with its Y component set to 1 and all other components set to 0
    pub fn unit_y() -> Self {
        if COLUMNS < 2 {
            panic!("Vector cannot have a Y component");
        }
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 1 {
            T::one()
        } else {
            T::zero()
        })])
    }

    /// Create a vector with its Z component set to 1 and all other components set to 0
    pub fn unit_z() -> Self {
        if COLUMNS < 3 {
            panic!("Vector cannot have a Z component");
        }
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 1 {
            T::one()
        } else {
            T::zero()
        })])
    }

    /// Create a vector with its W component set to 1 and all other components set to 0
    pub fn unit_w() -> Self {
        if COLUMNS < 4 {
            panic!("Vector cannot have a Z component");
        }
        Self::new([init_array!([T; COLUMNS], |column_idx| if column_idx == 1 {
            T::one()
        } else {
            T::zero()
        })])
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> From<[T; COLUMNS]> for Matrix<T, 1, COLUMNS> {
    fn from(v: [T; COLUMNS]) -> Self {
        Self::from_components(v)
    }
}

impl<T: Copy + Zero + One, const COLUMNS: usize> Into<[T; COLUMNS]> for Matrix<T, 1, COLUMNS> {
    fn into(self) -> [T; COLUMNS] {
        self.rows[0].into()
    }
}

impl<T: Copy + Zero + One + Default, const COLUMNS: usize> Default for Matrix<T, 1, COLUMNS> {
    fn default() -> Self {
        Self::from_scalar(T::default())
    }
}

// =============================
// Swizzle Definitions
// =============================

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
                impl<T: Copy + Zero + One> Matrix<T, 1, 2> {
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
                impl<T: Copy + Zero + One> Matrix<T, 1, 4> {
                    pub const fn [<$($letter)*>](&self) -> Vector<T, {swizzle_count!($($letter),*)}> {
                        self.swizzle(&[$(swizzle_idx!($letter)),*])
                    }
                }
                impl<T: Copy + Zero + One> Matrix<T, 1, 3> {
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

// =============================
// Operator Definitions
// I'm sorry.
// =============================

// Math operations
macro_rules! impl_op_bi {
    ($trait:ident, $fn:ident, $((where $t_ty:path: $($bounds:path)+))+, $($code:tt)+) => {
        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait for Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
        {
            type Output = Self;

            fn $fn(self, rhs: Self) -> Self::Output {
                let func: fn(Self, Self) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait<&Self> for Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
        {
            type Output = Self;

            fn $fn(self, rhs: &Self) -> Self::Output {
                let func: fn(Self, &Self) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait for &Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn(self, rhs: Self) -> Self::Output {
                let func: fn(Self, Self) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait<&Self> for &Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
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
    ($trait:ident, $fn:ident, $((where $t_ty:path: $($bounds:path)+))+, $($code:tt)+) => {
        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait<T> for Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
        {
            type Output = Self;

            fn $fn(self, rhs: T) -> Self::Output {
                let func: fn(Self, &T) -> Self::Output = $($code)+;
                func(self, &rhs)
            }
        }

        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait<&T> for Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
        {
            type Output = Self;

            fn $fn(self, rhs: &T) -> Self::Output {
                let func: fn(Self, &T) -> Self::Output = $($code)+;
                func(self, rhs)
            }
        }

        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait<T> for &Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
        {
            type Output = Matrix<T, ROWS, COLUMNS>;

            fn $fn(self, rhs: T) -> Self::Output {
                let func: fn(Self, &T) -> Self::Output = $($code)+;
                func(self, &rhs)
            }
        }

        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait<&T> for &Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
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
    ($trait:ident, $fn:ident, $((where $t_ty:path: $($bounds:path)+))+, $($code:tt)+) => {
        impl<T: Copy + Zero + One, const ROWS: usize, const COLUMNS: usize> $trait for Matrix<T, ROWS, COLUMNS>
            where$($t_ty: std::any::Any$(+$bounds)+),+
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
    (where T: Add<Output = T>),
    |a, b| {
        a.zip(&b, |a, b| *a + *b)
    }
}

impl_op_bi! {
    Sub,
    sub,
    (where T: Sub<Output = T>),
    |a, b| {
        a.zip(&b, |a, b| *a - *b)
    }
}

impl_op_bi! {
    Div,
    div,
    (where T: Div<Output = T>),
    |a, b| {
        a.zip(&b, |a, b| *a / *b)
    }
}

impl_op_bi! {
    Rem,
    rem,
    (where T: Rem<Output = T>),
    |a, b| {
        a.zip(&b, |a, b| *a % *b)
    }
}

impl_op_un! {
    Neg,
    neg,
    (where T: Neg<Output = T>),
    |a| {
        a.map(|a| -*a)
    }
}

impl_op_un! {
    Not,
    not,
    (where T: Not<Output = T>),
    |a| {
        a.map(|a| !*a)
    }
}

impl_op_bi_scalar! {
    Add,
    add,
    (where T: Add<Output = T>),
    |a, b| {
        a.map(|a| *a + *b)
    }
}

impl_op_bi_scalar! {
    Sub,
    sub,
    (where T: Sub<Output = T>),
    |a, b| {
        a.map(|a| *a - *b)
    }
}

impl_op_bi_scalar! {
    Mul,
    mul,
    (where T: Mul<Output = T>),
    |a, b| {
        a.map(|a| *a * *b)
    }
}

impl_op_bi_scalar! {
    Div,
    div,
    (where T: Div<Output = T>),
    |a, b| {
        a.map(|a| *a / *b)
    }
}

impl_op_bi_scalar! {
    Rem,
    rem,
    (where T: Rem<Output = T>),
    |a, b| {
        a.map(|a| *a % *b)
    }
}

impl<T: Copy + Zero + One + PartialEq, const ROWS: usize, const COLUMNS: usize> PartialEq
    for Matrix<T, ROWS, COLUMNS>
{
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: Copy + Zero + One + PartialEq + Eq, const ROWS: usize, const COLUMNS: usize> Eq
    for Matrix<T, ROWS, COLUMNS>
{
}

impl<T: Copy + Zero + One + PartialEq + Eq + 'static, const ROWS: usize, const COLUMNS: usize> AddAssign
    for Matrix<T, ROWS, COLUMNS>
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// Hashing implentation for Matrix
impl<T: Copy + Zero + One + Hash, const ROWS: usize, const COLUMNS: usize> Hash
    for Matrix<T, ROWS, COLUMNS>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for component in self.iter() {
            component.hash(state);
        }
    }
}

// Matrix * Matrix
impl<
        T: Copy + Zero + One,
        const ROWS: usize,
        const COLUMNS: usize,
        const OTHER_ROWS: usize,
        const OTHER_COLUMNS: usize,
    > Mul<Matrix<T, OTHER_ROWS, OTHER_COLUMNS>> for Matrix<T, ROWS, COLUMNS>
where
    T: Sum + Mul<Output = T>,
{
    type Output = Matrix<T, OTHER_ROWS, OTHER_COLUMNS>;
    fn mul(self, rhs: Matrix<T, OTHER_ROWS, OTHER_COLUMNS>) -> Self::Output {
        if ROWS == 1 && OTHER_ROWS == 1 {
            self.as_size(T::one()).zip(&rhs, |a, b| *a * *b)
        } else if OTHER_COLUMNS < ROWS {
            panic!("Matrix multiplication is not supported for matrices with dimensions {}x{} and {}x{}", ROWS, COLUMNS, OTHER_ROWS, OTHER_COLUMNS);
        } else {
            rhs.and_then(&self)
        }
    }
}

impl<
        T: Copy + Zero + One,
        const ROWS: usize,
        const COLUMNS: usize,
        const OTHER_ROWS: usize,
        const OTHER_COLUMNS: usize,
    > Mul<Matrix<T, OTHER_ROWS, OTHER_COLUMNS>> for &Matrix<T, ROWS, COLUMNS>
where
    T: Sum + Mul<Output = T>,
{
    type Output = Matrix<T, OTHER_ROWS, OTHER_COLUMNS>;
    fn mul(self, rhs: Matrix<T, OTHER_ROWS, OTHER_COLUMNS>) -> Self::Output {
        if ROWS == 1 && OTHER_ROWS == 1 {
            self.as_size(T::one()).zip(&rhs, |a, b| *a * *b)
        } else if OTHER_COLUMNS < ROWS {
            panic!("Matrix multiplication is not supported for matrices with dimensions {}x{} and {}x{}", ROWS, COLUMNS, OTHER_ROWS, OTHER_COLUMNS);
        } else {
            rhs.and_then(&self)
        }
    }
}

impl<
        T: Copy + Zero + One,
        const ROWS: usize,
        const COLUMNS: usize,
        const OTHER_ROWS: usize,
        const OTHER_COLUMNS: usize,
    > Mul<&Matrix<T, OTHER_ROWS, OTHER_COLUMNS>> for Matrix<T, ROWS, COLUMNS>
where
    T: Sum + Mul<Output = T>,
{
    type Output = Matrix<T, OTHER_ROWS, OTHER_COLUMNS>;
    fn mul(self, rhs: &Matrix<T, OTHER_ROWS, OTHER_COLUMNS>) -> Self::Output {
        if ROWS == 1 && OTHER_ROWS == 1 {
            self.as_size(T::one()).zip(&rhs, |a, b| *a * *b)
        } else if OTHER_COLUMNS < ROWS {
            panic!("Matrix multiplication is not supported for matrices with dimensions {}x{} and {}x{}", ROWS, COLUMNS, OTHER_ROWS, OTHER_COLUMNS);
        } else {
            rhs.and_then(&self)
        }
    }
}

impl<
        T: Copy + Zero + One,
        const ROWS: usize,
        const COLUMNS: usize,
        const OTHER_ROWS: usize,
        const OTHER_COLUMNS: usize,
    > Mul<&Matrix<T, OTHER_ROWS, OTHER_COLUMNS>> for &Matrix<T, ROWS, COLUMNS>
where
    T: Sum + Mul<Output = T>,
{
    type Output = Matrix<T, OTHER_ROWS, OTHER_COLUMNS>;
    fn mul(self, rhs: &Matrix<T, OTHER_ROWS, OTHER_COLUMNS>) -> Self::Output {
        if ROWS == 1 && OTHER_ROWS == 1 {
            self.as_size(T::one()).zip(&rhs, |a, b| *a * *b)
        } else if OTHER_COLUMNS < ROWS {
            panic!("Matrix multiplication is not supported for matrices with dimensions {}x{} and {}x{}", ROWS, COLUMNS, OTHER_ROWS, OTHER_COLUMNS);
        } else {
            rhs.and_then(&self)
        }
    }
}

// Converting a quaternion to a matrix with 3 rows and 3 columns; creates a rotation matrix
impl<T: Copy + Float> From<Quaternion<T>> for Matrix<T, 3, 3> {
    fn from(quaternion: Quaternion<T>) -> Self {
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
            ],
            [
                two * (xy - wz),
                one - two * (zz + xx),
                two * (yz + wx),
            ],
            [
                two * (xz + wy),
                two * (yz - wx),
                one - two * (yy * xx),
            ],
        ])
    }
}

// Converting a quaternion to a 4x4 matrix; creates a rotation matrix
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
            [zero, zero, zero, one],
        ])
    }
}

/// A matrix with 4 rows and 4 columns.
pub type Matrix4x4<T> = Matrix<T, 4, 4>;
/// A matrix with 3 rows and 3 columns.
pub type Matrix3x3<T> = Matrix<T, 3, 3>;
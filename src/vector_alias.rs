use crate::matrix::Matrix;

/// A vector
pub type Vector<T, const COMPONENTS: usize> = Matrix<T, 1, COMPONENTS>;

/// A vector with 2 components
pub type Vector2<T> = Vector<T, 2>;

/// A vector with 3 components
pub type Vector3<T> = Vector<T, 3>;

/// A vector with 4 components
pub type Vector4<T> = Vector<T, 4>;

#[macro_export]
macro_rules! vector {
    ($($component:expr),*$(,)?) => {
        $crate::vector_alias::Vector::new([[$($component),*]])
    };
}

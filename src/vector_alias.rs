use crate::matrix::Matrix;

pub type Vector<T, const COMPONENTS: usize> = Matrix<T, 1, COMPONENTS>;
pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector4<T> = Vector<T, 4>;

#[macro_export]
macro_rules! vector {
    ($($component:expr),*$(,)?) => {
        $crate::vector_alias::Vector::new([[$($component),*]])
    };
}

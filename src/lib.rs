#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(const_for)]
#![feature(const_ptr_write)]
#![feature(const_mut_refs)]
#![feature(const_maybe_uninit_as_mut_ptr)]
#![feature(inline_const)]
#![feature(const_slice_index)]
#![feature(const_option)]
#![feature(const_trait_impl)]
#![feature(generic_arg_infer)]

pub mod float_ext;
pub mod const_assert;
pub mod init_array;
pub mod matrix;
pub mod vector_alias;
pub mod quaternion;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    mod vector_tests {
        use crate::{vector, vector_alias::Vector};

        #[test]
        fn math() {
            assert_eq!(vector!(1, 5, -2) + vector!(-99, 0, -2), vector!(-98, 5, -4));
            assert_eq!(vector!(1.0, 5.0, -2.0) - vector!(-99.0, 0.0, -2.5), vector!(100.0, 5.0, 0.5));
            assert_eq!(vector!(40, 5, -2) / vector!(2, 2, -2), vector!(20, 2, 1));
            assert_eq!(vector!(1, 2, 3) * vector!(2, 6, -2), vector!(2, 12, -6));
            assert_eq!(vector!(40, 5, -7) % vector!(5, 2, 9), vector!(0, 1, -7));
            
            assert_eq!(vector!(1, 5, -2) + 2, vector!(3, 7, 0));
            assert_eq!(vector!(1, 5, -2) - 2, vector!(-1, 3, -4));
            assert_eq!(vector!(40, 5, -2) / 2, vector!(20, 2, -1));
            assert_eq!(vector!(1, 2, 3) * 2, vector!(2, 4, 6));
            assert_eq!(vector!(40, 5, -7) % 2, vector!(0, 1, -1));
            
            assert_eq!(-vector!(1, 2, 3), vector!(-1, -2, -3));
        }

        #[test]
        fn normalize() {
            let a: Vector<f32, _> = vector!(7.0, 22.6410, 2.406).normalized();
            assert!((a.length() - 1.0).abs() < 0.00000001);
        }

        #[test]
        fn dot_product() {
            let a: Vector<f32, _> = vector!(7.0, 22.6410, 2.406).normalized();
            assert!((a.dot(&a) - 1.0).abs() < 0.00000001);
            assert!((a.dot(&-a) + 1.0).abs() < 0.00000001);
        }

        #[test]
        fn cross_product() {
            let a: Vector<i32, _> = vector!(1, 0, 0);
            let b: Vector<i32, _> = vector!(0, 1, 0);
            let cross = a.cross(&b);
            assert_eq!(cross, vector!(0, 0, 1));
        }
    }

    mod matrix_tests {
        use crate::{quaternion::Quaternion, vector, vector_alias::{Vector, Vector3}};

        #[test]
        fn rotation() {
            let unit_x: Vector3<f64> = Vector3::unit_x();
            let rotation = Quaternion::from_axis_angle(Vector3::unit_y(), std::f64::consts::TAU * 0.25);

        }
    }
}

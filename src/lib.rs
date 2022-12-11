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
#![feature(const_convert)]
#![feature(const_result)]

pub mod float_ext;
pub mod geometry;
pub(crate) mod init_array;
pub mod matrix;
pub mod noise;
pub mod prelude;
pub mod quaternion;
pub mod random;
pub mod vector_alias;
pub mod fraction;

#[cfg(test)]
mod tests {
    mod vector_tests {
        use crate::prelude::*;

        #[test]
        fn math() {
            assert_eq!(vector!(1, 5, -2) + vector!(-99, 0, -2), vector!(-98, 5, -4));
            assert_eq!(
                vector!(1.0, 5.0, -2.0) - vector!(-99.0, 0.0, -2.5),
                vector!(100.0, 5.0, 0.5)
            );
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
        use crate::prelude::*;

        #[test]
        fn identity() {
            let ident = Matrix4x4::identity();
            assert_eq!(
                ident,
                Matrix4x4::new([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],])
            );
        }

        #[test]
        fn rotation() {
            let unit_x: Vector3<f64> = Vector3::unit_x();
            let rotation = Quaternion::from_rotation_y(std::f64::consts::PI * 0.5);
            let rotated = unit_x.rotated_by(&rotation);
            assert!((rotated - Vector3::unit_z()).length() < 0.000001);
        }
    }

    mod random_tests {
        use crate::prelude::*;

        #[test]
        fn deviation() {
            // Test regular LCG's deviation.
            let mut lcg = Lcg::new(71923);
            let mut accumulator = 0.0;
            for _ in 0..2000 {
                accumulator += lcg.next::<f32>();
            }
            let mean = accumulator / 2000.0;
            assert!((mean - 0.5).abs() < 0.01);

            // Test normally distributed LCG's deviation.
            let mut lcg = NormalLcg::<3>::new(9471);
            let mut accumulator = 0.0;
            for _ in 0..2000 {
                accumulator += lcg.next_f32();
            }
            let mean = accumulator / 2000.0;
            assert!((mean - 0.5).abs() < 0.01);
        }
    }

    mod noise_tests {
        use image::RgbImage;

        use crate::prelude::*;

        #[test]
        fn noise() {
            const SIZE: u32 = 512;

            // Create noise generator
            let noise = Noise::<2>::new(
                0xDEADBEEFu32, // Seed
                7,             // Number of levels
                2.0,           // Scale of level 0 (most detailed level)
                1.5,           // Smoothness value, lower value results in rougher noise
                0.2, // Detail strength, higher value results in lower (higher detail) levels contributing more to the end result
            );

            // Create image
            let mut image = RgbImage::new(SIZE, SIZE);

            // Sample noise for each pixel
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                // Sample value at pixel position
                let sample = noise.sample_f64(vector!(x as f64, y as f64));

                // Pixel brightness = sample value
                let byte = (sample * 255.0) as u8;
                *pixel = image::Rgb([byte, byte, byte]);
            }

            // Save image
            image.save("noise.png").unwrap();
        }
    }

    mod fraction_tests {
        use crate::prelude::*;

        #[test]
        fn fraction() {
            let a = Fraction::new(1, 2);
            let b = Fraction::new(1, 3);
            assert_eq!(a + b, Fraction::new(5, 6));
            assert_eq!(a - b, Fraction::new(1, 6));
            assert_eq!(a * b, Fraction::new(1, 6));
            assert_eq!(a / b, Fraction::new(3, 2));
            assert_eq!(Fraction::new(1, 2) - Fraction::new(2, 3), Fraction::new(-1, 6));
            assert!((Fraction::new(1, 2) - Fraction::new(2, 3)).is_negative());
            assert!((Fraction::new(1, 2) + Fraction::new(2, 3)).approximate::<f64>().unwrap() - 1.16666666667 < 0.00000000001);
            assert!(Fraction::new(1, 2) < Fraction::new(2, 3));
        }

        #[test]
        fn undefined() {
            let a = Fraction::new(1, 0);
            assert!(a.is_undefined());
            assert!((a + 1).is_undefined());
            assert!((a - Fraction::new(1, 2)).is_undefined());
        }
    }
}

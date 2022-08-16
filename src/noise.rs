use crate::{init_array, prelude::*};

pub struct Noise<const DIMENSIONS: usize> {
    seed: u64,
    levels: usize,
    scale: f64,
    smoothness: f64,
    detail_strength: f64,
}

impl<const DIMENSIONS: usize> Noise<DIMENSIONS> {
    pub fn new(seed: impl ToSeed, levels: usize, scale: f64, smoothness: f64, detail_strength: f64) -> Self {
        if levels == 0 {
            panic!("Noise::new(): levels must be greater than 0");
        }
        Self {
            seed: seed.to_seed(),
            levels,
            scale,
            smoothness,
            detail_strength,
        }
    }

    fn _raw_noise(&self, position: Vector<i64, DIMENSIONS>) -> f64 {
        (self.seed, position).into_random::<f64>()
    }

    fn _raw_noise_smooth(&self, position: Vector<f64, DIMENSIONS>) -> f64 {
        let min_corner = Vector::new([init_array!([i64; DIMENSIONS], |idx| position
            .component(idx)
            .unwrap()
            .floor()
            as i64)]);
        match DIMENSIONS {
            // 1 dimensional noise
            1 => self._raw_noise(min_corner).lerp(
                self._raw_noise(min_corner + 1),
                position.x() - min_corner.x() as f64,
            ),
            // 2 dimensional noise
            2 => (self._raw_noise(min_corner).lerp(
                self._raw_noise(min_corner + Vector::unit_x()),
                position.x() - min_corner.x() as f64,
            ))
            .lerp(
                self._raw_noise(min_corner + Vector::unit_y()).lerp(
                    self._raw_noise(min_corner + 1),
                    position.x() - min_corner.x() as f64,
                ),
                position.y() - min_corner.y() as f64,
            ),
            // 3 dimensional noise
            3 => ((self._raw_noise(min_corner).lerp(
                self._raw_noise(min_corner + Vector::unit_x()),
                position.x() - min_corner.x() as f64,
            ))
            .lerp(
                self._raw_noise(min_corner + Vector::unit_y()).lerp(
                    self._raw_noise(min_corner + 1),
                    position.x() - min_corner.x() as f64,
                ),
                position.y() - min_corner.y() as f64,
            ))
            .lerp(
                (self._raw_noise(min_corner + Vector::unit_z()).lerp(
                    self._raw_noise(min_corner + Vector::unit_x() + Vector::unit_z()),
                    position.x() - min_corner.x() as f64,
                ))
                    .lerp(
                        self
                            ._raw_noise(min_corner + Vector::unit_y() + Vector::unit_z())
                            .lerp(
                                self._raw_noise(min_corner + 1),
                                position.x() - min_corner.x() as f64,
                            ),
                        position.y() - min_corner.y() as f64,
                    ),
                position.z() - min_corner.z() as f64,
            ),
            _ => panic!(
                "Noise::sample() is not implemented for {} dimensions",
                DIMENSIONS
            ),
        }
    }

    pub fn sample_f64(&self, mut position: Vector<f64, DIMENSIONS>) -> f64 {
        position = position / self.scale + self.seed.into_random::<f64>() * 111098765.4321;
        let mut sum = self._raw_noise_smooth(position) * self.detail_strength;
        let mut total_strength = self.detail_strength;
        for idx in 1..self.levels {
            position = position / (1.0 + self.smoothness) + (idx as u64, self.seed).into_random::<f64>() * 111098765.4321;
            let this_strength = self.detail_strength.lerp(1.0, idx as f64 / self.levels as f64);
            sum += self._raw_noise_smooth(position) * this_strength;
            total_strength += this_strength;
        }
        sum / total_strength
    }
}

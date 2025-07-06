// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use num_traits::Float;
use std::iter::Sum;

/// Error returned when two points have a different dimension.
#[derive(Debug, PartialEq, Eq)]
pub struct IncompatiblePointsError;

/// Computes the L^`p` distance between `x` and `y`.
///
/// Works for any `p`>0. An [`IncompatiblePointsError`] is returned when `x` and `y` have different
/// lengths.
pub fn lp_distance<T>(x: &[T], y: &[T], p: i32) -> Result<T, IncompatiblePointsError>
where
    T: Float + Sum + From<i32>,
{
    if x.len() != y.len() {
        Err(IncompatiblePointsError {})
    } else {
        Ok(x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| T::powi(T::abs(b - a), p))
            .sum::<T>()
            .powf(T::one() / p.into()))
    }
}

/// Computes the Euclidean distance between `x` and `y`.
///
/// An [`IncompatiblePointsError`] is returned when `x` and `y` have different lengths.
pub fn euclidean_distance<T>(x: &[T], y: &[T]) -> Result<T, IncompatiblePointsError>
where
    T: Float + Sum + From<i32>,
{
    lp_distance(x, y, 2)
}

/// Computes the maximum distance (Chebyshev distance or L^infinity distance) between `x` and `y`.
///
/// An [`IncompatiblePointsError`] is returned when `x` and `y` have different lengths.
pub fn maximum_distance<T: Float>(x: &[T], y: &[T]) -> Result<T, IncompatiblePointsError> {
    if x.len() != y.len() {
        Err(IncompatiblePointsError {})
    } else {
        Ok(x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| T::abs(b - a))
            .reduce(Float::max)
            .unwrap_or(T::zero()))
    }
}

/// Computes the Euclidean dot product between points the unit n-sphere, where n is the length of
/// `angles1` and `angles2`.
///
/// No check is done on the lengths of `angles1` and `angles2`. The Euclidean dot product is also
/// the cosine of the angular distance between `angles1` and `angles2`.
fn euclidean_dot_product<T: Float>(angles1: &[T], angles2: &[T]) -> T {
    let mut total = T::zero();
    let mut sin_prod = T::one();
    let d = angles1.len();
    for (i, (t1, t2)) in angles1.iter().zip(angles2.iter()).enumerate() {
        total = T::mul_add(sin_prod, t1.cos() * t2.cos(), total);
        sin_prod = sin_prod * t1.sin() * t2.sin();
        if i == d - 1 {
            total = total + sin_prod;
        }
    }
    total
}

/// Computes the distance between the points `angles1` and `angles2` on the unit n-sphere, where n
/// is the length of `angles1` and `angles2`.
///
/// The last element of `angles1` and `angles2` is assumed to be in [0, 2pi] or [-pi, pi] (and the
/// other elements are in [0, pi]). An [`IncompatiblePointsError`] is returned when `angles1` and
/// `angles2` have different lengths.
pub fn angular_distance<T: Float>(
    angles1: &[T],
    angles2: &[T],
) -> Result<T, IncompatiblePointsError> {
    if angles1.len() != angles2.len() {
        Err(IncompatiblePointsError {})
    } else {
        Ok(euclidean_dot_product(angles1, angles2).acos().abs())
    }
}

/// Computes the hyperbolic distance between two points in polar coordinates.
///
/// `r1` and `r2` are the distances to the origin and the last element of `angles1` and `angles2`
/// is assumed to be in [0, 2pi] or [-pi, pi] (and the other elements are in [0, pi]). An
/// [`IncompatiblePointsError`] is returned when `angles1` and `angles2` have different lengths.
pub fn polar_hyperbolic_distance<T: Float>(
    r1: T,
    angles1: &[T],
    r2: T,
    angles2: &[T],
) -> Result<T, IncompatiblePointsError> {
    if angles1.len() != angles2.len() {
        Err(IncompatiblePointsError {})
    } else {
        let arg = (r1 - r2).cosh()
            + (T::one() - euclidean_dot_product(angles1, angles2)) * r1.sinh() * r2.sinh();
        Ok(if arg < T::one() {
            T::zero()
        } else {
            arg.acosh()
        })
    }
}

/// Computes the hyperbolic distance between the points `x1` and `x2` in the hyperboloid model.
///
/// The "time" coordinate (opposite sign in the metric) is inferred from the others and should not
/// be included in `x1` and `x2`. An [`IncompatiblePointsError`] is returned when `x` and `y` have
/// different lengths.
pub fn hyperboloid_hyperbolic_distance<T: Float>(
    x: &[T],
    y: &[T],
) -> Result<T, IncompatiblePointsError> {
    if x.len() != y.len() {
        Err(IncompatiblePointsError {})
    } else {
        let mut sum_x_squared = T::zero();
        let mut sum_y_squared = T::zero();
        let mut sum_xy = T::zero();
        for (x_i, y_i) in x.iter().zip(y.iter()) {
            sum_x_squared = T::mul_add(*x_i, *x_i, sum_x_squared);
            sum_y_squared = T::mul_add(*y_i, *y_i, sum_y_squared);
            sum_xy = T::mul_add(*x_i, *y_i, sum_xy);
        }
        let arg = (T::one() + sum_x_squared).sqrt() * (T::one() + sum_y_squared).sqrt() - sum_xy;
        Ok(if arg < T::one() {
            T::zero()
        } else {
            arg.acosh()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::{
        angular_distance, euclidean_distance, hyperboloid_hyperbolic_distance, lp_distance,
        maximum_distance, polar_hyperbolic_distance, IncompatiblePointsError,
    };

    #[test]
    fn test_l4_dist() {
        assert!(
            (lp_distance(&[1., 2., 3.], &[5., 3., 1.], 4).unwrap()
                - (256_f64 + 1. + 16.).sqrt().sqrt())
            .abs()
                < 1e-15
        );
    }

    #[test]
    fn test_l1_dist() {
        assert!(
            (lp_distance(&[1., 2., 3.], &[5., 3., 1.], 1).unwrap() - (4. + 1. + 2_f64)).abs()
                < 1e-15
        );
    }

    #[test]
    fn test_lp_dist_incompatible_error() {
        assert_eq!(
            lp_distance(&[0., 0.], &[0., 0., 0.], 1),
            Err(IncompatiblePointsError {})
        );
        assert_eq!(
            lp_distance(&[0., 0.], &[0., 0., 0.], 4),
            Err(IncompatiblePointsError {})
        );
    }

    #[test]
    fn test_euclidean_dist() {
        assert!(
            (euclidean_distance(&[1., 2., 3.], &[5., 3., 1.]).unwrap() - (16_f64 + 1. + 4.).sqrt())
                .abs()
                < 1e-15
        );
    }

    #[test]
    fn test_euclidean_dist_incompatible_error() {
        assert_eq!(
            euclidean_distance(&[0., 0.], &[0., 0., 0.]),
            Err(IncompatiblePointsError {})
        );
    }

    #[test]
    fn test_maximum_dist() {
        assert!((maximum_distance(&[1., 2., 3.], &[5., 3., 1.]).unwrap() - 4_f64).abs() < 1e-15);
    }

    #[test]
    fn test_maximum_dist_incompatible_error() {
        assert_eq!(
            maximum_distance(&[0., 0.], &[0., 0., 0.]),
            Err(IncompatiblePointsError {})
        );
    }

    #[test]
    fn test_angular_dist() {
        assert!((angular_distance(&[0.3], &[0.5]).unwrap() - 0.2_f64).abs() < 1e-15);
        assert!((angular_distance(&[0.5 * PI, PI], &[0.5 * PI, 0.]).unwrap() - PI).abs() < 1e-15);
        assert!((angular_distance(&[0.2, 0., 1.], &[0., 0., 1.]).unwrap() - 0.2_f64).abs() < 1e-15);
        assert!(
            (angular_distance(&[2. * PI, 0., 1.], &[0., 0., 1.]).unwrap() - 0_f64).abs() < 1e-15
        );
    }

    #[test]
    fn test_angular_dist_incompatible_error() {
        assert_eq!(
            angular_distance(&[0.], &[0., 0.]),
            Err(IncompatiblePointsError {})
        );
    }

    #[test]
    fn test_polar_hyperbolic_dist() {
        assert_eq!(
            polar_hyperbolic_distance(3., &[0.], 0.5, &[PI]).unwrap(),
            3.5
        );
    }

    #[test]
    fn test_polar_hyperbolic_dist_incompatible_error() {
        assert_eq!(
            polar_hyperbolic_distance(1., &[0.], 1., &[0., 0.]),
            Err(IncompatiblePointsError {})
        );
    }

    #[test]
    fn test_hyperboloid_dist() {
        assert_eq!(
            hyperboloid_hyperbolic_distance(&[3_f64.sinh(), 0.], &[-0.5_f64.sinh(), 0.]).unwrap(),
            3.5
        );
    }

    #[test]
    fn test_hyperboloid_dist_inf() {
        assert!(
            hyperboloid_hyperbolic_distance(&[f64::INFINITY, 0.], &[0., 0.])
                .unwrap()
                .is_nan()
        );
    }
    #[test]
    fn test_hyperboloid_dist_length_error() {
        assert_eq!(
            euclidean_distance(&[0., 0.], &[0., 0., 0.]),
            Err(IncompatiblePointsError {})
        );
    }
}

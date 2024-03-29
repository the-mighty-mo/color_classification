//! This module provides common linear
//! algebra interfaces, such as points.
//!
//! Author: Benjamin Hall

use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use crate::Complex;

/// Stores a point vector.
#[derive(Clone, Default, PartialEq)]
pub struct Point(pub Vec<Complex>);

impl Debug for Point {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl Point {
    /// Returns the magnitude of the point vector.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::Point;
    /// let point = Point(vec![3.0.into(), 4.0.into()]);
    /// assert!((point.magnitude() - 5.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        f64::sqrt(
            self.0
                .iter()
                .map(|&x| x * x.conjugate())
                .sum::<Complex>()
                .magnitude(),
        )
    }

    /// Scales the point by the given scalar.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::Point;
    /// let point = Point(vec![1.0.into(), 2.0.into()]);
    /// assert_eq!(point.scale(2.0), Point(vec![2.0.into(), 4.0.into()]));
    /// ```
    #[inline]
    #[must_use]
    pub fn scale(mut self, scalar: f64) -> Self {
        for x in &mut self.0 {
            *x = x.scale(scalar);
        }
        self
    }

    /// Computes the dot product of two points.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::{Complex, Point};
    /// let a = Point(vec![1.0.into(), 2.0.into()]);
    /// let b = Point(vec![2.0.into(), 3.0.into()]);
    /// assert_eq!(a.dot(&b), 8.0.into());
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> Complex {
        // dot product -- multiply each non-zero term and sum
        self.0.iter().zip(&other.0).map(|(&a, &b)| a * b).sum()
    }
}

impl Neg for Point {
    type Output = Self;

    #[inline]
    #[must_use]
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

impl Add for Point {
    type Output = Self;

    #[must_use]
    fn add(self, rhs: Self) -> Self::Output {
        // use whichever point has more dimensions
        if self.0.len() >= rhs.0.len() {
            // self + rhs, where dimensions not present in rhs are 0
            let mut point = self;
            for (x1, x2) in point.0.iter_mut().zip(rhs.0) {
                *x1 += x2;
            }
            point
        } else {
            // rhs + self, where dimensions not present in self are 0
            let mut point = rhs;
            for (x1, x2) in point.0.iter_mut().zip(self.0) {
                *x1 += x2;
            }
            point
        }
    }
}

impl Add for &Point {
    type Output = Point;

    #[must_use]
    fn add(self, rhs: Self) -> Self::Output {
        // use whichever point has more dimensions
        if self.0.len() >= rhs.0.len() {
            // self + rhs, where dimensions not present in rhs are 0
            let mut point = self.clone();
            for (x1, &x2) in point.0.iter_mut().zip(&rhs.0) {
                *x1 += x2;
            }
            point
        } else {
            // rhs + self, where dimensions not present in self are 0
            let mut point = rhs.clone();
            for (x1, &x2) in point.0.iter_mut().zip(&self.0) {
                *x1 += x2;
            }
            point
        }
    }
}

impl Add<&Self> for Point {
    type Output = Self;

    #[must_use]
    fn add(self, rhs: &Self) -> Self::Output {
        // self + rhs, where dimensions not present in rhs are 0
        let mut point = self;
        for (x1, &x2) in point.0.iter_mut().zip(&rhs.0) {
            *x1 += x2;
        }
        // add any additional dimensions present in rhs
        point.0.extend(rhs.0.iter().skip(point.0.len()));
        point
    }
}

impl Add<Point> for &Point {
    type Output = Point;

    #[must_use]
    fn add(self, rhs: Point) -> Self::Output {
        // self + rhs, where dimensions not present in rhs are 0
        let mut point = rhs;
        for (x1, &x2) in point.0.iter_mut().zip(&self.0) {
            *x1 += x2;
        }
        // add any additional dimensions present in self
        point.0.extend(self.0.iter().skip(point.0.len()));
        point
    }
}

impl AddAssign<&Self> for Point {
    fn add_assign(&mut self, rhs: &Self) {
        // self + rhs, where dimensions not present in rhs are 0
        for (x1, &x2) in self.0.iter_mut().zip(&rhs.0) {
            *x1 += x2;
        }
        // add any additional dimensions present in rhs
        self.0.extend(rhs.0.iter().skip(self.0.len()));
    }
}

impl AddAssign for Point {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl Sub for Point {
    type Output = Self;

    #[must_use]
    fn sub(self, rhs: Self) -> Self::Output {
        // use whichever point has more dimensions
        if self.0.len() >= rhs.0.len() {
            // self - rhs, where dimensions not present in rhs are 0
            let mut point = self;
            for (x1, x2) in point.0.iter_mut().zip(rhs.0) {
                *x1 -= x2;
            }
            point
        } else {
            // -(rhs - self), where dimensions not present in self are 0
            let mut point = rhs;
            for (x1, x2) in point.0.iter_mut().zip(self.0) {
                *x1 -= x2;
            }
            -point
        }
    }
}

impl Sub for &Point {
    type Output = Point;

    #[must_use]
    fn sub(self, rhs: Self) -> Self::Output {
        // use whichever point has more dimensions
        if self.0.len() >= rhs.0.len() {
            // self - rhs, where dimensions not present in rhs are 0
            let mut point = self.clone();
            for (x1, &x2) in point.0.iter_mut().zip(&rhs.0) {
                *x1 -= x2;
            }
            point
        } else {
            // -(rhs - self), where dimensions not present in self are 0
            let mut point = rhs.clone();
            for (x1, &x2) in point.0.iter_mut().zip(&self.0) {
                *x1 -= x2;
            }
            -point
        }
    }
}

impl Sub<&Self> for Point {
    type Output = Self;

    #[must_use]
    fn sub(self, rhs: &Self) -> Self::Output {
        // self - rhs, where dimensions not present in rhs are 0
        let mut point = self;
        for (x1, &x2) in point.0.iter_mut().zip(&rhs.0) {
            *x1 -= x2;
        }
        // subtract any additional dimensions present in rhs
        point
            .0
            .extend(rhs.0.iter().skip(point.0.len()).map(|&x| -x));
        point
    }
}

impl Sub<Point> for &Point {
    type Output = Point;

    #[must_use]
    fn sub(self, rhs: Point) -> Self::Output {
        // -(rhs - self), where dimensions not present in self are 0
        let mut point = rhs;
        for (x1, &x2) in point.0.iter_mut().zip(&self.0) {
            *x1 -= x2;
        }
        // subtract any additional dimensions present in self
        point
            .0
            .extend(self.0.iter().skip(point.0.len()).map(|&x| -x));
        -point
    }
}

impl SubAssign<&Self> for Point {
    fn sub_assign(&mut self, rhs: &Self) {
        // self - rhs, where dimensions not present in rhs are 0
        for (x1, &x2) in self.0.iter_mut().zip(&rhs.0) {
            *x1 -= x2;
        }
        // subtract any additional dimensions present in rhs
        self.0.extend(rhs.0.iter().skip(self.0.len()).map(|&x| -x));
    }
}

impl SubAssign for Point {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl Sum for Point {
    #[inline]
    #[must_use]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for Point {
    #[inline]
    #[must_use]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn point_ops() {
        let a = Point(vec![1.0.into(), 2.0.into()]);
        let b = Point(vec![3.0.into(), 4.0.into(), 5.0.into()]);

        let sum = &a + &b;
        assert_eq!(sum, Point(vec![4.0.into(), 6.0.into(), 5.0.into()]));

        let diff = &b - &a;
        assert_eq!(diff, Point(vec![2.0.into(), 2.0.into(), 5.0.into()]));

        let sum_iter: Point = [a, b].iter().sum();
        assert_eq!(sum_iter, sum);
    }
}

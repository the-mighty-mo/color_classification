//! This module provides interfaces for
//! working with complex numbers.
//!
//! Author: Benjamin Hall

use std::{
    fmt::{Debug, Display},
    iter::Sum,
    num::ParseFloatError,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    /// Creates a complex number from polar coordinates.
    /// The angle `theta` should be given in radians.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::Complex;
    /// let polar = Complex::from_polar(1.0, 0.0);
    /// let rect = Complex { re: 1.0, im: 0.0 };
    /// assert!((polar.re - rect.re).abs() < f64::EPSILON);
    /// assert!((polar.im - rect.im).abs() < f64::EPSILON);
    ///
    /// let polar = Complex::from_polar(2.0, std::f64::consts::FRAC_PI_2);
    /// let rect = Complex { re: 0.0, im: 2.0 };
    /// assert!((polar.re - rect.re).abs() < f64::EPSILON);
    /// assert!((polar.im - rect.im).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let re = r * theta.cos();
        let im = r * theta.sin();
        Self { re, im }
    }

    /// Calculates the magnitude of the complex number.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::Complex;
    /// let complex = Complex { re: 4.0, im: 3.0 };
    /// assert!((complex.magnitude() - 5.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        if self.im == 0.0 {
            self.re
        } else {
            self.re.hypot(self.im)
        }
    }

    /// Calculates the angle of the complex number in radians.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::Complex;
    /// let complex = Complex { re: 1.0, im: 1.0 };
    /// assert!((complex.angle() - std::f64::consts::FRAC_PI_4).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn angle(&self) -> f64 {
        f64::atan2(self.im, self.re)
    }

    /// Scales the complex number.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::Complex;
    /// let complex = Complex { re: 1.0, im: 2.0 };
    /// assert_eq!(complex.scale(2.0), Complex { re: 2.0, im: 4.0 });
    /// ```
    #[inline]
    #[must_use]
    pub fn scale(mut self, scalar: f64) -> Self {
        self.re *= scalar;
        self.im *= scalar;
        self
    }

    /// Calculates the conjugate of the complex number.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// # use data_utils::Complex;
    /// let complex = Complex { re: 1.0, im: 2.0 };
    /// assert_eq!(complex.conjugate(), Complex { re: 1.0, im: -2.0 });
    /// ```
    #[inline]
    #[must_use]
    pub fn conjugate(mut self) -> Self {
        self.im *= -1.0;
        self
    }
}

impl Neg for Complex {
    type Output = Self;

    #[inline]
    #[must_use]
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Add for Complex {
    type Output = Self;

    #[inline]
    #[must_use]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;

    #[inline]
    #[must_use]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        *self = Self { re, im };
    }
}

impl Mul for Complex {
    type Output = Self;

    #[inline]
    #[must_use]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl DivAssign for Complex {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        let rhs_conj = rhs.conjugate();
        let num = *self * rhs_conj;
        // complex num * its conjugate = real number
        let denom = (rhs * rhs_conj).re;
        *self = num.scale(denom.recip());
    }
}

impl Div for Complex {
    type Output = Self;

    #[inline]
    #[must_use]
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl Sum for Complex {
    #[inline]
    #[must_use]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

impl Debug for Complex {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im == 0.0 {
            write!(f, "{}", self.re)
        } else if self.im > 0.0 {
            write!(f, "{} + {}i", self.re, self.im)
        } else {
            write!(f, "{} - {}i", self.re, self.im.abs())
        }
    }
}

impl Display for Complex {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im == 0.0 {
            write!(f, "{}", self.re)
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

impl From<f64> for Complex {
    #[inline]
    fn from(value: f64) -> Self {
        Self { re: value, im: 0.0 }
    }
}

impl FromStr for Complex {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (re, im) = s.split_once('+').unwrap_or((s, "0.0i"));
        let im = im.split('i').next().unwrap_or("0.0");
        let (re, im) = (re.parse()?, im.parse()?);
        Ok(Self { re, im })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn complex_from_str() {
        let complex = Complex::from_str("1.0").unwrap();
        assert_eq!(complex, Complex::from(1.0));

        let complex = Complex::from_str("1.0+2.0i").unwrap();
        assert_eq!(complex, Complex { re: 1.0, im: 2.0 });
    }

    #[test]
    fn complex_ops() {
        let a = Complex { re: 1.0, im: 2.0 };
        let b = Complex { re: 2.0, im: 3.0 };

        let sum = a + b;
        assert_eq!(sum, Complex { re: 3.0, im: 5.0 });

        let diff = b - a;
        assert_eq!(diff, Complex { re: 1.0, im: 1.0 });

        let mult = a * b;
        assert_eq!(mult, Complex { re: -4.0, im: 7.0 });

        let div = b / a;
        assert!((div.re - 1.6).abs() < f64::EPSILON);
        assert!((div.im + 0.2).abs() < f64::EPSILON);

        let sum_iter: Complex = [a, b].into_iter().sum();
        assert_eq!(sum_iter, sum);
    }
}

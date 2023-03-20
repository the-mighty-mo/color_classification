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
    #[inline]
    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let re = r * theta.cos();
        let im = r * theta.sin();
        Self { re, im }
    }

    /// Calculates the magnitude of the complex number.
    #[inline]
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        if self.im == 0.0 {
            self.re
        } else {
            (self.re.powi(2) + self.im.powi(2)).sqrt()
        }
    }

    /// Calculates the angle of the complex number in radians.
    #[inline]
    #[must_use]
    pub fn angle(&self) -> f64 {
        f64::atan2(self.im, self.re)
    }

    /// Scales the complex number.
    #[inline]
    #[must_use]
    pub fn scale(mut self, scalar: f64) -> Self {
        self.re *= scalar;
        self.im *= scalar;
        self
    }

    /// Calculates the conjugate of the complex number.
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    fn div_assign(&mut self, rhs: Self) {
        let rhs_conj = rhs.conjugate();
        let num = *self * rhs_conj;
        // complex num * its conjugate = real number
        let denom = (rhs * rhs_conj).re;
        *self = num.scale(1.0 / denom);
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
        iter.fold(Complex::default(), |a, b| a + b)
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
        Complex { re: value, im: 0.0 }
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

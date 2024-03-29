//! This crate provides utilities to work with
//! and classify data points.
//!
//! Author: Benjamin Hall

pub mod classify;
pub mod color;
pub mod complex;
pub mod io;
pub mod lin_alg;
pub mod sort;

pub use complex::Complex;
pub use lin_alg::Point;
use std::{
    error::Error,
    fmt::{Debug, Display},
    str::FromStr,
};

/// Stores a point of data.
#[derive(Clone, Debug, PartialEq)]
pub struct DataPoint<T> {
    // The data point, represented as a point vector
    pub point: Point,
    // The data's classification
    pub class: T,
}

impl<T> TryFrom<&str> for DataPoint<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::error::Error + 'static,
{
    type Error = Box<dyn Error>;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        // split on whitespace
        let data: Vec<_> = value.split_whitespace().collect();
        // last element is classification
        let Some((&last, rest)) = data.split_last() else {
            return Err("Cannot parse empty line of data".into());
        };

        // parse classification
        let class = last.parse::<T>()?;
        // map all other elements to components of a point vector
        let point: Vec<_> = rest
            .iter()
            .map(|&p| p.parse::<Complex>())
            .collect::<Result<Vec<_>, _>>()?;
        // wrap displacement vector in a Point
        let point = Point(point);
        // return the DataPoint
        Ok(Self { point, class })
    }
}

impl<T> Display for DataPoint<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for v in &self.point.0 {
            write!(f, "{}   ", *v)?;
        }
        write!(f, "{}", self.class)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn data_point_from_str() {
        let data = DataPoint::<String>::try_from("0.954+0.3i   1.0   0.7   red").unwrap();
        let expected = DataPoint::<String> {
            point: Point(vec![
                Complex { re: 0.954, im: 0.3 },
                Complex::from_polar(1.0, 0.0),
                Complex::from_polar(0.7, 0.0),
            ]),
            class: String::from("red"),
        };
        assert_eq!(data, expected);
    }
}

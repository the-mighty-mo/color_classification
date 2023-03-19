//! This module provides interfaces to run
//! classification algorithms on data.
//!
//! Author: Benjamin Hall

pub mod bayes;
pub mod knn;
pub mod perceptron;

use crate::{DataPoint, Debug};
pub use {bayes::*, knn::*, perceptron::*};

/// Stores the result of a classification algorithm.
#[derive(Copy, Clone, Debug)]
pub struct Classification<'a, T> {
    /// The data point that has been classified
    pub data: &'a DataPoint<T>,
    /// The classification algorithm's guess for the data element's classification
    pub class_guess: T,
}

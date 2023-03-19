//! This module provides interfaces to run the
//! Single-Layer Perceptron algorithm on data.
//!
//! Author: Benjamin Hall

use std::{collections::HashSet, hash::Hash};

use super::Classification;
use crate::{DataPoint, Point};

/// Generates a Point of random weights in the range -1.0..1.0.
#[inline]
#[must_use]
fn generate_random_weights(size: usize) -> Point {
    Point(
        (0..size)
            .map(|_| fastrand::f64().mul_add(2.0, -1.0))
            .collect(),
    )
}

/// Runs the Single-Layer Perceptron algorithm with the given training
/// data on the given test data with the given learning rate.
///
/// Note that this algorithm requires that the data can only be
/// split into two classifications.
#[must_use]
pub fn single_layer_perceptron<'a, T>(
    train_data: &[DataPoint<T>],
    test_data: &'a [DataPoint<T>],
    learning_rate: f64,
    threshold: f64,
) -> Vec<Classification<'a, T>>
where
    T: Clone + Eq,
{
    // initialize random weights, [w_i0, w_i] = 1 + dimension of training data points
    let mut weights = generate_random_weights(train_data[0].point.0.len() + 1);

    // let the first training data point be class 1 (g(x) > 0)
    let pos_class = &train_data[0].class;
    // any other classifications will be treated as class 2 (g(x) < 0)
    let neg_class = train_data
        .iter()
        .map(|d| &d.class)
        .find(|&c| c != pos_class)
        .expect("The data provided to the SLP algorithm does not have two classifications");

    // calculate the mean of the training data so we can offset data points
    let train_mean = train_data
        .iter()
        .map(|d| &d.point)
        .sum::<Point>()
        .scale(1.0 / train_data.len() as f64);

    // map all the training data to [1, x]
    let mut y: Vec<_> = train_data
        .iter()
        .map(|d| {
            // offset training point by the mean
            let mut point = &d.point - &train_mean;
            // y = [1, x]
            point.0.insert(0, 1.0);
            DataPoint {
                point,
                class: &d.class,
            }
        })
        .collect();

    // loop at most 10,000 times, otherwise we may overtrain
    // or enter an infinite loop if the weights cannot converge
    for _ in 0..10_000 {
        // shuffle data set -- this prevents oscillations and overtraining
        fastrand::shuffle(&mut y);

        let mut misclassified = 0;

        // perceptron iterative algorithm
        for d in &y {
            // get linear classifier value using the dot product of the data point and the weights
            let lin_class_value = weights.dot(&d.point);
            // convert linear classifier value to binary classification: 1.0 or -1.0
            let class_guess_value = lin_class_value.signum();
            // convert actual class to binary classification: 1.0 or -1.0
            let class_value = if d.class == pos_class { 1.0 } else { -1.0 };
            // calculate error in classification
            let error = class_value - class_guess_value;
            // if error is non-zero, adjust weights and add to misclassified count
            if error != 0.0 {
                misclassified += 1;

                // scale point by the error
                let weight_error = d.point.clone().scale(error);
                // scale by learning rate
                let weight_adjustment = weight_error.scale(learning_rate);
                // update weights
                weights += weight_adjustment;
            }
        }

        // continue until no misclassifications
        if (misclassified as f64) < (threshold * train_data.len() as f64) {
            break;
        }
    }

    // run SLP on all test data and collect the results
    test_data
        .iter()
        .map(|data| {
            // run the SLP on this data
            let mut point = &data.point - &train_mean;
            point.0.insert(0, 1.0);
            let class_result = weights.dot(&point);

            // determine if positive or negative class
            let class_guess = if class_result > 0.0 {
                pos_class.clone()
            } else {
                neg_class.clone()
            };
            // wrap in a Classification
            Classification { data, class_guess }
        })
        .collect()
}

/// Runs the Multiclass Single-Layer Perceptron algorithm with the given
/// training data on the given test data with the given learning rate.
///
/// This algorithm uses the one-vs-rest method to transform the multiclass
/// problem to multiple binary classifications.
#[must_use]
pub fn multiclass_single_layer_perceptron<'a, T>(
    train_data: &[DataPoint<T>],
    test_data: &'a [DataPoint<T>],
    learning_rate: f64,
    threshold: f64,
) -> Vec<Classification<'a, T>>
where
    T: Clone + Default + Eq + Hash,
{
    /// Stores weights for a classification.
    struct Weights<'a, T> {
        class: &'a T,
        /// w = 2µ
        w: Point,
    }

    let mut weights_vec: Vec<_> = {
        // create set of classifications
        let mut classes = HashSet::new();
        for d in train_data {
            classes.insert(&d.class);
        }

        // initialize random weights for each classification, -1.0..1.0
        classes
            .into_iter()
            .map(|class| Weights {
                class,
                w: generate_random_weights(test_data[0].point.0.len() + 1),
            })
            .collect()
    };

    // calculate the mean of the training data so we can offset data points
    let train_mean = train_data
        .iter()
        .map(|d| &d.point)
        .sum::<Point>()
        .scale(1.0 / train_data.len() as f64);

    // map all the training data to [1, x]
    let mut y: Vec<_> = train_data
        .iter()
        .map(|d| {
            // offset training point by the mean
            let mut point = &d.point - &train_mean;
            // y = [1, x]
            point.0.insert(0, 1.0);
            DataPoint {
                point,
                class: &d.class,
            }
        })
        .collect();

    // loop at most 10,000 times, otherwise we may overtrain
    // or enter an infinite loop if the weights cannot converge
    for _ in 0..10_000 {
        // shuffle data set -- this prevents oscillations and overtraining
        fastrand::shuffle(&mut y);

        let mut misclassified = 0;

        // perceptron iterative algorithm
        for d in &y {
            // get linear classifier values using the dot product of the data point and the weights
            let class_results = weights_vec
                .iter()
                .map(|weights| (weights.class, weights.w.dot(&d.point)));
            // find the maximum classification value, pull out class
            let class_guess = class_results
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(class, _)| class)
                .unwrap();
            // if classification is wrong, adjust weights
            if class_guess != d.class {
                misclassified += 1;

                // scale point by learning rate
                let weight_adjustment = d.point.clone().scale(learning_rate);
                // update weights
                for weights in weights_vec.iter_mut() {
                    if weights.class == d.class {
                        // increase the weights of the correct class
                        weights.w += &weight_adjustment;
                    } else {
                        // decrease the weights of the incorrect classes
                        weights.w -= &weight_adjustment;
                    }
                }
            }
        }

        // continue until no misclassifications
        if (misclassified as f64) < (threshold * train_data.len() as f64) {
            break;
        }
    }

    // run SLP on all test data and collect the results
    test_data
        .iter()
        .map(|data| {
            // run the SLP on this data
            let mut point = &data.point - &train_mean;
            point.0.insert(0, 1.0);

            // find the maximum classification value, pull out class
            let class_guess = weights_vec
                .iter()
                .map(|weights| (weights.class, weights.w.dot(&point)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or_else(T::default, |(class, _)| class.clone());

            // wrap in a Classification
            Classification { data, class_guess }
        })
        .collect()
}

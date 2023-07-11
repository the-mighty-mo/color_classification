//! This module provides interfaces to run the
//! Bayesian plug-in rule algorithm on data.
//!
//! Author: Benjamin Hall

use super::Classification;
use crate::{Complex, DataPoint, Point};
use std::collections::BTreeMap;

/// Runs the Bayesian plug-in rule with the given training data
/// on the given test data.
#[must_use]
pub fn bayes_plug_in<'a, T>(
    train_data: &[DataPoint<T>],
    test_data: &'a [DataPoint<T>],
) -> Vec<Classification<'a, T>>
where
    T: Clone + Default + Ord,
{
    /// Stores plug-in weights for a classification.
    struct Weights {
        /// w = 2µ
        w: Point,
        /// w_0 = µ⋅µ
        w_0: Complex,
    }

    // group training data by classification
    let mut train_data_grp: BTreeMap<&T, Vec<&Point>> = BTreeMap::new();
    for d in train_data {
        if let Some(g) = train_data_grp.get_mut(&d.class) {
            g.push(&d.point);
        } else {
            train_data_grp.insert(&d.class, vec![&d.point]);
        }
    }

    // calculate means for each classification
    let train_data_means = train_data_grp.into_iter().map(|(class, points)| {
        let cnt = points.len() as f64;
        // sum together points and scale by 1/cnt
        let mean = points.into_iter().sum::<Point>().scale(cnt.recip());
        (class, mean)
    });

    // calculate the two weight components for each classification
    let train_data_weights: Vec<_> = train_data_means
        .map(|(class, mean)| {
            // create a point with the conjugates of all complex numbers
            let mut mean_conj = mean.clone();
            for m in &mut mean_conj.0 {
                *m = m.conjugate();
            }

            // dot the mean with its conjugate
            let w_0 = mean.dot(&mean_conj);
            let weights = Weights {
                // conjugate mean for weight offset
                w: mean_conj.scale(2.0),
                w_0,
            };
            (class, weights)
        })
        .collect();

    // run Bayesian plug-in rule on all test data and collect the results
    test_data
        .iter()
        .map(|data| {
            // run the plug-in rule on this data for all classifications
            let class_results = train_data_weights
                .iter()
                .map(|(class, weights)| (*class, weights.w.dot(&data.point) - weights.w_0));
            // find the maximum classification value, pull out class
            let class_guess = class_results
                .max_by(|a, b| a.1.re.total_cmp(&b.1.re))
                .map_or_else(T::default, |(class, _)| class.clone());
            // wrap in a Classification
            Classification { data, class_guess }
        })
        .collect()
}

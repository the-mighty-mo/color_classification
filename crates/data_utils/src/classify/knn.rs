//! This module provides interfaces to run the
//! k-nearest neighbor algorithm on data.
//!
//! Author: Benjamin Hall

use super::Classification;
use crate::{sort::PartialSort, DataPoint};
use std::{collections::HashMap, hash::Hash};

/// Runs the k-nearest neighbor algorithm with the given training data
/// on the given test data for the specified number of neighbors.
#[must_use]
pub fn k_nearest_neighbor<'a, T>(
    train_data: &[DataPoint<T>],
    test_data: &'a [DataPoint<T>],
    num_neighbors: usize,
) -> Vec<Classification<'a, T>>
where
    T: Clone + Default + Eq + Hash,
{
    assert!(
        train_data.len() >= num_neighbors,
        "Not enough training data for {num_neighbors} neighbors"
    );

    // run k-nearest neighbor on all test data and collect the results
    test_data
        .iter()
        .map(|data| {
            /// Stores the distance between a training data point and the current test data point.
            struct Dist<'a, T> {
                /// The training data point
                data: &'a DataPoint<T>,
                /// The distance from the test data point
                dist: f64,
            }

            // calculate distance between training data points and the test data point
            let mut distances: Vec<_> = train_data
                .iter()
                .map(|d| Dist {
                    data: d,
                    dist: (&d.point - &data.point).magnitude(),
                })
                .collect();
            // perform a partial sort of the training data distances, up to num_neighbors
            distances.partial_sort_by(num_neighbors, |d1, d2| d1.dist.total_cmp(&d2.dist));

            // pull out the nearest neighbors
            let nearest = &distances[0..num_neighbors];
            // count how many votes are present for each classification
            let mut votes: HashMap<&T, usize> = HashMap::with_capacity(num_neighbors);
            for d in nearest {
                if let Some(v) = votes.get_mut(&d.data.class) {
                    *v += 1;
                } else {
                    votes.insert(&d.data.class, 1);
                }
            }

            // majority vote: max by value, pull out classification
            let class_guess = votes
                .iter()
                .max_by(|a, b| a.1.cmp(b.1))
                .map_or_else(T::default, |(&class, _)| class.clone());
            // wrap in a Classification
            Classification { data, class_guess }
        })
        .collect()
}

//! This module provides functions to perform
//! various sorting algorithms on slices of data.
//!
//! Author: Benjamin Hall

use std::cmp::Ordering;

/// Adds functionality for partial sorting.
pub trait PartialSort<T> {
    /// Partially sorts the first `num_sorted` elements.
    #[inline]
    fn partial_sort(&mut self, num_sorted: usize)
    where
        T: Ord,
    {
        self.partial_sort_by(num_sorted, T::cmp);
    }

    /// Partially sorts the first `num_sorted` elements
    /// with the given comparison function.
    fn partial_sort_by<F>(&mut self, num_sorted: usize, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering;
}

impl<T> PartialSort<T> for [T] {
    /// Partially sorts the first `num_sorted` elements of a slice
    /// with the given comparison function.
    ///
    /// This implementation uses a reverse bubble sort, i.e. starting
    /// at the end and swapping the smallest element to the start.
    #[inline]
    fn partial_sort_by<F>(&mut self, num_sorted: usize, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        for i in 0..num_sorted {
            for j in (i..(self.len() - 1)).rev() {
                if compare(&self[j], &self[j + 1]).is_gt() {
                    self.swap(j, j + 1);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn partial_sort_vec() {
        let mut vec = vec![1.0, 5.0, 4.0, 7.0, 3.0];
        vec.partial_sort_by(3, f64::total_cmp);
        assert_eq!(vec[..3], [1.0, 3.0, 4.0]);
    }
}

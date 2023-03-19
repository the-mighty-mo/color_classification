//! This module provides helper functions
//! for file IO.
//!
//! Author: Benjamin Hall

use std::{
    fs::File,
    io::{self, BufReader, Read},
};

/// Loads data from a file into a String.
///
/// Any errors, such as the file not existing or not having
/// read access, will be propagated up to the caller.
#[inline]
pub fn read_file(file: io::Result<File>) -> io::Result<String> {
    let file = file?;

    let mut buffer = String::new();
    let mut reader = BufReader::new(file);
    reader.read_to_string(&mut buffer).map(|_| buffer)
}

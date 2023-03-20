//! This program runs the Single-Layer Perceptron algorithm
//! on a set of data using specified training data, classifying
//! the data into two groups. The resulting classifications are
//! printed to the terminal.
//!
//! Author: Benjamin Hall

use std::{
    env,
    fs::File,
    io::{self, Write},
};

/// Runs the Single-Layer Perceptron algorithm and outputs the results.
/// Data is only classified into two groups.
///
/// Program input is the filename of the training data, the filename
/// of the test data, and the number of neighbors used in the algorithm.
fn main() {
    // get program arguments
    let args: Vec<_> = env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        /* invalid number of arguments, print a help message */
        let mut lock = io::stdout().lock();
        writeln!(lock, "Binary Single-Layer Perceptron").unwrap();
        writeln!(lock, "Author: Benjamin Hall").unwrap();
        writeln!(
            lock,
            "Usage: ./color_class_binary_slp [train data filename] [test data filename] [learning rate = 1.0] [threshold = 0.0]"
        )
        .unwrap();
        writeln!(lock).unwrap();

        writeln!(
            lock,
            "Runs the binary single-layer perceptron algorithm on a set of data using the given training data and learning rate."
        )
        .unwrap();
        writeln!(
            lock,
            "The training threshold specifies the percentage of the training data that can be misclassified before stopping training."
        ).unwrap();
        writeln!(
            lock,
            "The data can be n-dimensional, but the dimensions of the training data and of the test data should match."
        )
        .unwrap();

        return;
    }

    // pull out training data file name
    let train_data_file_name = args[1].as_str();
    if train_data_file_name.is_empty() {
        println!("Error: no training data specified");
        return;
    }

    // pull out test data file name
    let test_data_file_name = args[2].as_str();
    if test_data_file_name.is_empty() {
        println!("Error: no test data specified");
        return;
    }

    // pull out learning rate
    let learning_rate = if let Some(arg) = args.get(3) {
        let Ok(learning_rate) = str::parse::<f64>(arg) else {
            println!("Error: invalid learning rate");
            return;
        };
        learning_rate
    } else {
        1.0
    };

    // pull out training threshold
    let threshold = if let Some(arg) = args.get(4) {
        let Ok(threshold) = str::parse::<f64>(arg) else {
            println!("Error: invalid training threshold");
            return;
        };
        threshold
    } else {
        0.0
    };

    // parse training data
    let train_data = {
        // open and read file
        let train_data_file = File::open(train_data_file_name);
        let Ok(train_data_file_contents) = data_utils::io::read_file(train_data_file) else {
            println!("Error: could not read training data");
            return;
        };

        // map lines to DataPoints
        let train_data: Result<Vec<_>, _> = train_data_file_contents
            .lines()
            .map(data_utils::DataPoint::<String>::try_from)
            .collect();
        let Ok(train_data) = train_data else {
            println!("Error: could not parse training data");
            return;
        };
        train_data
    };

    // parse test data
    let test_data = {
        // open and read file
        let test_data_file = File::open(test_data_file_name);
        let Ok(test_data_file_contents) = data_utils::io::read_file(test_data_file) else {
            println!("Error: could not read test data");
            return;
        };

        // map lines to DataPoints
        let test_data: Result<Vec<_>, _> = test_data_file_contents
            .lines()
            .map(data_utils::DataPoint::try_from)
            .collect();
        let Ok(test_data) = test_data else {
            println!("Error: could not parse test data");
            return;
        };
        test_data
    };

    // run single-layer perceptron algorithm
    let test_res = data_utils::classify::single_layer_perceptron(
        &train_data,
        &test_data,
        learning_rate,
        threshold,
    );
    // print the results (formatted)
    println!("{test_res:#?}");
}

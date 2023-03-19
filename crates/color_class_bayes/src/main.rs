//! This program runs a Bayesian plug-in rule algorithm
//! on a set of data using specified training data. The
//! resulting classifications are printed to the terminal.
//!
//! Author: Benjamin Hall

use std::{
    env,
    fs::File,
    io::{self, Write},
};

/// Runs the Bayesian plug-in rule and outputs the results.
///
/// Program input is the filename of the training data, and the
/// filename of the test data.
fn main() {
    // get program arguments
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        /* invalid number of arguments, print a help message */
        let mut lock = io::stdout().lock();
        writeln!(lock, "Bayesian Plug-In Rule").unwrap();
        writeln!(lock, "Author: Benjamin Hall").unwrap();
        writeln!(
            lock,
            "Usage: ./color_class_bayes [train data file name] [test data file name]"
        )
        .unwrap();
        writeln!(lock).unwrap();

        writeln!(
            lock,
            "Runs a Bayesian plug-in rule on a set of data using the given training data."
        )
        .unwrap();
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
            .map(data_utils::DataPoint::try_from)
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
            .map(data_utils::DataPoint::<String>::try_from)
            .collect();
        let Ok(test_data) = test_data else {
            println!("Error: could not parse test data");
            return;
        };
        test_data
    };

    // run Bayesian plug-in rule
    let test_res = data_utils::classify::bayes_plug_in(&train_data, &test_data);
    // print the results (formatted)
    println!("{test_res:#?}");
}

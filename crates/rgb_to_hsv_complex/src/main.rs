//! This program converts a file of RGB colors
//! to a file of HSV colors, using a complex
//! number on the unit circle to represent the hue.
//!
//! Author: Benjamin Hall

use std::{
    env,
    fs::File,
    io::{self, BufWriter, Write},
};

use data_utils::{
    color::{Hsv, Rgb},
    Complex,
};

/// Runs the RGB to HSV (Complex) conversion on an input file,
/// writing the results to an output file.
///
/// Program input is the input filename and the output filename.
fn main() {
    // get program arguments
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        /* invalid number of arguments, print a help message */
        let mut lock = io::stdout().lock();
        writeln!(lock, "RGB to HSV (Complex) Converter").unwrap();
        writeln!(lock, "Author: Benjamin Hall").unwrap();
        writeln!(
            lock,
            "Usage: ./rgb_to_hsv [input filename] [output filename]"
        )
        .unwrap();
        writeln!(lock).unwrap();

        writeln!(
            lock,
            "Converts the data in the input file from RGB to HSV, writing the results to the output file."
        )
        .unwrap();
        writeln!(
            lock,
            "The hue is represented as a complex number using Euler's formula: e^ix = cos(x) + i*sin(x)."
        )
        .unwrap();
        writeln!(
            lock,
            "This can be useful in applications that need to handle wrapping near the 0/360 boundary."
        )
        .unwrap();

        return;
    }

    // pull out input data file name
    let input_data_file_name = args[1].as_str();
    if input_data_file_name.is_empty() {
        println!("Error: no input data specified");
        return;
    }

    // pull out output data file name
    let output_data_file_name = args[2].as_str();
    if output_data_file_name.is_empty() {
        println!("Error: no output file specified");
        return;
    }

    // parse input data
    let input_data = {
        // open and read file
        let input_data_file = File::open(input_data_file_name);
        let Ok(input_data_file_contents) = data_utils::io::read_file(input_data_file) else {
            println!("Error: could not read input data");
            return;
        };

        // map lines to DataPoints
        let input_data: Result<Vec<_>, _> = input_data_file_contents
            .lines()
            .map(data_utils::DataPoint::<String>::try_from)
            .collect();
        let Ok(input_data) = input_data else {
            println!("Error: could not parse input data");
            return;
        };
        input_data
    };

    // convert to HSV output data
    let output_data = input_data.into_iter().map(|mut d| {
        let [r, g, b] = d.point.0[0..3] else {
            unreachable!()
        };
        let (r, g, b) = (r.re as u8, g.re as u8, b.re as u8);
        let Hsv { h, s, v } = Hsv::from(Rgb { r, g, b });
        d.point.0[0..3].copy_from_slice(&[
            // complex value on the unit circle
            Complex::from_polar(1.0, h.to_radians()),
            s.into(),
            v.into(),
        ]);
        d.to_string()
    });

    {
        let Ok(output_file) = File::create(output_data_file_name) else {
            println!("Error: could not open output file");
            return;
        };
        let mut writer = BufWriter::new(output_file);
        for line in output_data {
            writeln!(writer, "{line}").unwrap();
        }
        writer.flush().unwrap();
    }
}

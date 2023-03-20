//! This module provides interfaces to store
//! and convert between colors.
//!
//! Author: Benjamin Hall

/// Stores an RGB color.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Stores an HSV color.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hsv {
    pub h: f64,
    pub s: f64,
    pub v: f64,
}

/// An enum of possible colors.
pub enum Color {
    Rgb(Rgb),
    Hsv(Hsv),
}

impl From<Rgb> for Hsv {
    fn from(value: Rgb) -> Self {
        let r = (value.r as f64) / 255.0;
        let g = (value.g as f64) / 255.0;
        let b = (value.b as f64) / 255.0;

        let c_max = r.max(g).max(b);
        let c_min = r.min(g).min(b);
        let range = c_max - c_min;

        let h = if range == 0.0 {
            0.0
        } else if c_max == r {
            60.0 * ((g - b) / range).rem_euclid(6.0)
        } else if c_max == g {
            60.0 * ((b - r) / range + 2.0)
        } else {
            60.0 * ((r - g) / range + 4.0)
        };
        let s = if c_max == 0.0 { 0.0 } else { range / c_max };
        let v = c_max;

        Self { h, s, v }
    }
}

impl From<Hsv> for Rgb {
    fn from(value: Hsv) -> Self {
        let c = value.v * value.s;
        let x = c * (1.0 - ((value.h / 60.0) % 2.0 - 1.0).abs());
        let m = value.v - c;

        let (r, g, b) = if value.h >= 0.0 && value.h < 60.0 {
            (c, x, 0.0)
        } else if value.h >= 60.0 && value.h < 120.0 {
            (x, c, 0.0)
        } else if value.h >= 120.0 && value.h < 180.0 {
            (0.0, c, x)
        } else if value.h >= 180.0 && value.h < 240.0 {
            (0.0, x, c)
        } else if value.h >= 240.0 && value.h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        let (r, g, b) = ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0);
        let (r, g, b) = (r as u8, g as u8, b as u8);
        Self { r, g, b }
    }
}

impl From<Color> for Hsv {
    fn from(value: Color) -> Self {
        match value {
            Color::Rgb(rgb) => Self::from(rgb),
            Color::Hsv(hsv) => hsv,
        }
    }
}

impl From<Color> for Rgb {
    fn from(value: Color) -> Self {
        match value {
            Color::Rgb(rgb) => rgb,
            Color::Hsv(hsv) => Self::from(hsv),
        }
    }
}

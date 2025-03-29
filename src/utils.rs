use std::{fmt::Display, fs, path::PathBuf};

use log::*;
use serde::{Deserialize, Serialize};

/// returns true if the cell on LEFT exists and is available (free, true), otherwise returns false
pub fn test_left(cells: &[bool], pos: usize, size: usize) -> bool {
    return pos % size != 0 && cells[pos - 1];
}

/// returns true if the cell on RIGHT exists and is available (free, true), otherwise returns false
pub fn test_right(cells: &[bool], pos: usize, size: usize) -> bool {
    return ((pos + 1) % size != 0) && cells[pos + 1];
}

/// returns true if the cell DOWN exists and is available (free, true), otherwise returns false
pub fn test_down(cells: &[bool], pos: usize, size: usize) -> bool {
    return pos < size * size - size && cells[pos + size];
}

/// returns true if the cell UP exists and is available (free, true), otherwise returns false
pub fn test_up(cells: &[bool], pos: usize, size: usize) -> bool {
    return pos >= size && cells[pos - size];
}

/// Loads a Field from given file
pub fn parse_field(path: &PathBuf) -> Field {
    let Ok(contents) = fs::read_to_string(path) else {
        error!("Failed to read file {}", path.display());
        std::process::exit(exitcode::IOERR);
    };
    let size = contents.lines().count();
    let mut mushrooms: Vec<Coord> = vec![];
    let mut players: Vec<Coord> = vec![];
    let mut cells: Vec<bool> = vec![];
    contents.lines().enumerate().for_each(|(y, line)| {
        let line = line.replace(" ", "");
        let line_len = line.chars().count();
        if line_len != size {
            error!(
                "inconsistent size (map has {size} lines, but line {y} has {line_len} characters, expected {size})"
            );
            std::process::exit(exitcode::DATAERR);
        };
        line.chars().enumerate().for_each(|(x, c)| {
            match c {
                'M' => {
                    mushrooms.push(Coord { x, y });
                    cells.push(true) // mushroom is not obstacle
                }
                'P' => {
                    players.push(Coord { x, y });
                    cells.push(true) // player is not obstacle
                }
                'X' => cells.push(false),
                '-' => cells.push(true),
                _ => {
                    error!("encountered unrecognized character: '{c}'");
                    std::process::exit(exitcode::DATAERR);
                }
            }
        });
    });
    return Field {
        size,
        mushrooms,
        players,
        cells,
    };
}

/// Playing field
#[derive(Clone, Debug)]
pub struct Field {
    /// size of the playing field, assuming it is a square, so value of N means field N-by-N
    pub size: usize,
    /// list of coordinates where mushrooms are located
    pub mushrooms: Vec<Coord>,
    /// list of coordinates where players are located
    pub players: Vec<Coord>,
    /// booleans representing the cells, where true means the player can step on that cell
    /// the shape is 2D matrix stored in a single vec, row-after-row
    pub cells: Vec<bool>,
}

/// General structure that represents named [X,Y] value
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Coord {
    /// X-coordinate
    pub x: usize,
    /// Y-coordinate
    pub y: usize,
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "[{}, {}]", self.x, self.y);
    }
}

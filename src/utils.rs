use std::{fmt::Display, fs, path::PathBuf};

use log::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
#[allow(clippy::missing_docs_in_private_items)]
/// One of the 4 cardinal directions
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

// transpose NxM matrix into MxN matrix
pub fn transpose<T: Clone>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut transposed = vec![vec![matrix[0][0].clone(); rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j].clone();
        }
    }

    return transposed;
}

impl Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Up => return write!(f, "Up"),
            Direction::Down => return write!(f, "Down"),
            Direction::Left => return write!(f, "Left"),
            Direction::Right => return write!(f, "Right"),
        }
    }
}

impl Direction {
    /// Get a list of all directions
    pub fn get_all_dirs() -> Vec<Direction> {
        return vec![
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];
    }

    /// Transform direction into coordinate-difference
    pub fn as_diff(&self) -> (i32, i32) {
        return match self {
            Direction::Up => (0, -1), // Y is counted downwards
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        };
    }

    /// Transform direction into character code
    pub fn as_char(&self) -> char {
        return match self {
            Direction::Up => 'u',
            Direction::Down => 'd',
            Direction::Left => 'l',
            Direction::Right => 'r',
        };
    }

    /// Transform coordinate difference into a Direction
    pub fn from_diff(diff: (i32, i32)) -> Option<Direction> {
        return match diff {
            (0, -1) => Some(Direction::Up), // Y is counted downwards
            (0, 1) => Some(Direction::Down),
            (-1, 0) => Some(Direction::Left),
            (1, 0) => Some(Direction::Right),
            _ => None,
        };
    }

    #[allow(dead_code)]
    /// Apply step to a coordinate to get a new position
    pub fn apply_step(&self, coord: &Coord) -> Coord {
        let diff = self.as_diff();
        return Coord {
            x: coord.x + diff.0,
            y: coord.y + diff.1,
        };
    }

    /// Transform char code into a Direction
    pub fn from_char(c: char) -> Result<Direction, String> {
        return match c {
            'u' => Ok(Direction::Up),
            'd' => Ok(Direction::Down),
            'l' => Ok(Direction::Left),
            'r' => Ok(Direction::Right),
            c => Err(format!("no Direction has code '{c}'")),
        };
    }
}

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
        let y = y as i32;
        let line = line.replace(" ", "");
        let line_len = line.chars().count();
        if line_len != size {
            error!(
                "inconsistent size (map has {size} lines, but line {y} has {line_len} characters, expected {size})"
            );
            std::process::exit(exitcode::DATAERR);
        };
        line.chars().enumerate().for_each(|(x, c)| {
            let x = x as i32;
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
    pub x: i32,
    /// Y-coordinate
    pub y: i32,
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "[{}, {}]", self.x, self.y);
    }
}

use std::{fmt::Display, fs, path::PathBuf};

use log::*;

/// Helper function that will print out formatted matrix
pub fn print_matrix(mat: &Vec<Vec<usize>>) {
    let max_item = mat
        .iter()
        .map(|row: &Vec<usize>| return row.iter().max().unwrap_or(&0))
        .max()
        .unwrap_or(&0);
    let col_width = f32::log10(*max_item as f32).ceil() as usize + 2;
    for row in mat {
        for cell in row {
            print!("{cell:col_width$}");
        }
        println!()
    }
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

impl Field {
    /// render field into parsable ascii art
    pub fn render(&self) -> String {
        let mut buf = String::new();
        let player_flat_idxs: Vec<usize> = self
            .players
            .iter()
            .map(|c| return c.x + c.y * self.size)
            .collect();
        let mush_flat_idxs: Vec<usize> = self
            .mushrooms
            .iter()
            .map(|c| return c.x + c.y * self.size)
            .collect();
        for (idx, free) in self.cells.iter().enumerate() {
            if idx > 0 && idx % self.size == 0 {
                buf += "\n"
            }
            if !free {
                buf += "X ";
            } else if player_flat_idxs.contains(&idx) {
                buf += "P ";
            } else if mush_flat_idxs.contains(&idx) {
                buf += "M ";
            } else {
                buf += "- ";
            }
        }
        return buf;
    }
}

/// General structure that represents named [X,Y] value
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
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

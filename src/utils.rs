use std::{fmt::Display, fs, path::PathBuf};

use itertools::Itertools;
use log::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::missing_docs_in_private_items)]
/// One of the 4 cardinal directions
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
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
    #[allow(dead_code)]
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

/// Loads a Field from given file
pub fn parse_field(path: &PathBuf) -> Field {
    let Ok(contents) = fs::read_to_string(path) else {
        error!("Failed to read file {}", path.display());
        std::process::exit(exitcode::IOERR);
    };
    return parse_field_str(&contents);
}

/// Parses a Field from the textual map representation
pub fn parse_field_str(contents: &str) -> Field {
    let size = contents.lines().count();
    let mut mushrooms: Vec<Coord> = vec![];
    let mut players: Vec<Coord> = vec![];
    let mut cells: Vec<bool> = Vec::with_capacity(size * size);
    for (y, line) in contents.lines().enumerate() {
        let row_start = cells.len();
        for c in line.chars().filter(|&c| return c != ' ') {
            let pos = Coord {
                x: (cells.len() - row_start) as i32,
                y: y as i32,
            };
            match c {
                'M' => mushrooms.push(pos),
                'P' => players.push(pos),
                'X' | '-' => {}
                _ => {
                    error!("encountered unrecognized character: '{c}'");
                    std::process::exit(exitcode::DATAERR);
                }
            }
            // mushrooms and players are not obstacles
            cells.push(c != 'X');
        }
        let line_len = cells.len() - row_start;
        if line_len != size {
            error!(
                "inconsistent size (map has {size} lines, but line {y} has {line_len} characters, expected {size})"
            );
            std::process::exit(exitcode::DATAERR);
        }
    }
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
    /// render field into pretty ascii art
    pub fn render_pretty(&self) -> String {
        let mut buf: Vec<Vec<char>> = Vec::with_capacity(self.size);
        let wall = '█';
        let path = '░';
        let player = 'P';
        let mush = '*';
        let player_pos_flattened = self
            .players
            .iter()
            .map(|p| return p.x + p.y * self.size as i32)
            .collect_vec();
        let mushrooms_pos_flattened = self
            .mushrooms
            .iter()
            .map(|p| return p.x + p.y * self.size as i32)
            .collect_vec();
        for r in 0..self.size {
            let mut row: Vec<char> = Vec::with_capacity(self.size);
            for c in 0..self.size {
                let pos = r * self.size + c;
                let cell = self.cells[pos];
                if cell {
                    if player_pos_flattened.contains(&(pos as i32)) {
                        row.push(player);
                    } else if mushrooms_pos_flattened.contains(&(pos as i32)) {
                        row.push(mush);
                    } else {
                        row.push(path);
                    }
                } else {
                    row.push(wall);
                }
            }
            buf.push(row);
        }
        return buf
            .into_iter()
            .map(|row| return row.into_iter().collect::<String>())
            .collect::<Vec<String>>()
            .join("\n");
    }

    /// render field into parsable ascii art
    pub fn render_parsable(&self) -> String {
        let mut buf = String::new();
        let player_flat_idxs = self
            .players
            .iter()
            .map(|c| return c.x + c.y * self.size as i32)
            .collect_vec();
        let mush_flat_idxs = self
            .mushrooms
            .iter()
            .map(|c| return c.x + c.y * self.size as i32)
            .collect_vec();
        for (idx, free) in self.cells.iter().enumerate() {
            if idx > 0 && idx % self.size == 0 {
                buf += "\n"
            }
            let idx = idx as i32;
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
    pub x: i32,
    /// Y-coordinate
    pub y: i32,
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "[{}, {}]", self.x, self.y);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a Field from an inline map string
    pub(crate) fn field(s: &str) -> Field {
        return parse_field_str(s);
    }

    #[test]
    fn direction_char_round_trip() {
        for d in Direction::get_all_dirs() {
            let back = Direction::from_char(d.as_char()).unwrap();
            assert_eq!(back.as_char(), d.as_char());
        }
        assert!(Direction::from_char('x').is_err());
    }

    #[test]
    fn direction_diff_round_trip() {
        for d in Direction::get_all_dirs() {
            let back = Direction::from_diff(d.as_diff()).unwrap();
            assert_eq!(back.as_char(), d.as_char());
        }
        assert!(Direction::from_diff((1, 1)).is_none());
        assert!(Direction::from_diff((0, 0)).is_none());
    }

    #[test]
    fn apply_step_y_grows_downwards() {
        let c = Coord { x: 2, y: 2 };
        assert_eq!(Direction::Up.apply_step(&c), Coord { x: 2, y: 1 });
        assert_eq!(Direction::Down.apply_step(&c), Coord { x: 2, y: 3 });
        assert_eq!(Direction::Left.apply_step(&c), Coord { x: 1, y: 2 });
        assert_eq!(Direction::Right.apply_step(&c), Coord { x: 3, y: 2 });
    }

    #[test]
    fn parse_with_and_without_spaces() {
        let a = field("P - M\nX - -\nM - P\n");
        let b = field("P-M\nX--\nM-P\n");
        assert_eq!(a.size, 3);
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.players, b.players);
        assert_eq!(a.mushrooms, b.mushrooms);
    }

    #[test]
    fn parse_collects_entities_in_reading_order() {
        let f = field("P - M\nX - -\nM - P\n");
        assert_eq!(f.players, vec![Coord { x: 0, y: 0 }, Coord { x: 2, y: 2 }]);
        assert_eq!(
            f.mushrooms,
            vec![Coord { x: 2, y: 0 }, Coord { x: 0, y: 2 }]
        );
        assert_eq!(
            f.cells,
            vec![true, true, true, false, true, true, true, true, true]
        );
    }
}

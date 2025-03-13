use itertools::Itertools;

use crate::{Coord, Field};

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

impl Field {
    /// render field into pretty ascii art
    pub fn render_pretty(&self, paths: Option<&Vec<Vec<Coord>>>) -> String {
        let mut buf: Vec<Vec<char>> = Vec::with_capacity(self.size);
        let wall = '█';
        //let path = '░';
        let path = ' ';
        let player = 'P';
        let mush = '*';
        let picked = 'o';
        let stepped = '·';
        let end = 'X';
        let player_pos_flattened: Vec<usize> = self
            .players
            .iter()
            .map(|p| return p.x + p.y * self.size)
            .collect();
        let mushrooms_pos_flattened: Vec<usize> = self
            .mushrooms
            .iter()
            .map(|p| return p.x + p.y * self.size)
            .collect();
        for r in 0..self.size {
            let mut row: Vec<char> = Vec::with_capacity(self.size);
            for c in 0..self.size {
                let pos = r * self.size + c;
                let cell = self.cells[pos];
                if cell {
                    if player_pos_flattened.contains(&pos) {
                        row.push(player);
                    } else if mushrooms_pos_flattened.contains(&pos) {
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
        if let Some(paths) = paths {
            for p in paths {
                for (i, step) in p.iter().enumerate() {
                    if i == 0 {
                        // skip the first coord, since it is a player
                        continue;
                    }
                    let cell = buf.get_mut(step.y).unwrap().get_mut(step.x).unwrap();
                    if cell == &mush {
                        *cell = picked;
                    } else if cell == &path {
                        *cell = stepped;
                    };
                }
            }
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

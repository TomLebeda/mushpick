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
    pub fn render_pretty(&self) -> String {
        let mut buf = String::new();
        let outer_wall = String::from("█");
        let wall = String::from("▒");
        let path = String::from(" ");
        let player = String::from("P");
        let mush = String::from("*");
        buf += &outer_wall.repeat(self.size + 2);
        buf += "\n";
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
            buf += &outer_wall;
            for c in 0..self.size {
                let pos = r * self.size + c;
                let cell = self.cells[pos];
                if cell {
                    if player_pos_flattened.contains(&pos) {
                        buf += &player;
                    } else if mushrooms_pos_flattened.contains(&pos) {
                        buf += &mush;
                    } else {
                        buf += &path;
                    }
                } else {
                    buf += &wall;
                }
            }
            buf += &outer_wall;
            buf += "\n";
        }
        buf += &outer_wall.repeat(self.size + 2);
        return buf;
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

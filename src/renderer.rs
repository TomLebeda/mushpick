use std::{collections::HashMap, path::PathBuf};

use itertools::Itertools;
use log::*;

use crate::{Coord, Field, Result, parse_field, utils::Direction};

/// Render the inputs into a tikz code and save it into provided file
pub fn render_tikz(map_file: PathBuf, output_file: PathBuf, solution_file: Option<PathBuf>) {
    let field = parse_field(&map_file);
    let mut buf = String::new();

    let wall_color = "gray";
    let numbers = true;
    let scale = 1.0;
    let mush_scale = 1.2;
    let mush_leg_width = 0.07 * mush_scale;
    let mush_leg_height = 0.3 * mush_scale;
    let mush_head_width = 0.2 * mush_scale;
    let mush_head_height = 0.2 * mush_scale;
    let mush_leg_color = "yellow!20!brown!20!white";
    let mush_head_color = "brown";
    let mush_offset = (
        0.5,
        -(1.0 - (mush_head_width + mush_head_height)) / 2.0 + 0.05,
    );

    let mushpic = format!(
        r"\tikzset{{
        mush/.pic = {{
            \draw[rounded corners={mush_leg_width}cm,fill={mush_leg_color}] (-{mush_leg_width}, 0) rectangle ({mush_leg_width}, {mush_leg_height});
            \draw[fill={mush_head_color}] (-{mush_head_width}, {mush_head_height}) -- ({mush_head_width}, {mush_head_height})
                arc(0:180:{mush_head_width}) --cycle;
        }}
    }}"
    );

    let person_head_radius = 0.17;
    let person_head_height = 0.5;
    let person_torso_width = 0.3;
    let person_torso_height = 0.4;
    let person_torso_corner_radius = 0.1;
    let person_head_color = "yellow";
    let person_torso_color = "blue!30";
    let person_offset = (0.5, -0.2);
    let label_y = person_torso_height / 2.0 - 0.05;
    let personpic = format!(
        r"\tikzset{{
        person/.pic = {{
            \draw[rounded corners={person_torso_corner_radius}cm,fill={person_torso_color}]
                (0,{person_torso_height}) to[out=-10,in=100]
                ({person_torso_width},0) to[bend left=15]
                (-{person_torso_width},0) to[out=80,in=190]
                (0,{person_torso_height});
            \draw[fill={person_head_color}] (0, {person_head_height}) circle [radius={person_head_radius}];
            \node[anchor=center] at (0, {label_y}) {{\scriptsize P\tikzpictext}};
        }}
    }}"
    );

    buf += &format!(r"\begin{{tikzpicture}}[scale={scale},yscale=-1.0]");
    buf += "\n";
    buf += &mushpic;
    buf += "\n";
    buf += &personpic;
    buf += "\n";

    // draw the walls
    for (cell_idx, cell) in field.cells.iter().enumerate() {
        if !*cell {
            let x = (cell_idx % field.size) as f32;
            let y = ((cell_idx - (cell_idx % field.size)) / field.size) as f32;
            buf += &format!(
                r"\draw[draw=none,fill={}] ({},{}) rectangle ({},{});",
                wall_color,
                x,
                y,
                x + 1.0,
                y + 1.0,
            );
            buf += "\n";
        }
    }

    // draw the grid
    buf += &format!(r"\draw (0,0) grid[step=1] ({},{});", field.size, field.size);
    buf += "\n";

    // draw the numbers
    if numbers {
        for i in 0..field.size {
            let x = i as f32 + 0.5;
            buf += &format!(r"\node[anchor=south] at ({x},0) {{{i}}};");
            buf += "\n";
            let y = i as f32 + 0.5;
            buf += &format!(r"\node[anchor=east] at (0,{y}) {{{i}}};");
            buf += "\n";
        }
    }

    // draw the solutions
    if let Some(solution_file) = solution_file {
        let Ok(contents) = std::fs::read_to_string(&solution_file) else {
            error!("Failed to read file {}", &solution_file.display());
            std::process::exit(exitcode::IOERR);
        };

        let solution = contents
            .lines()
            .map(|line| return line.chars())
            .map(|chars| {
                return chars
                    .map(|c| match Direction::from_char(c) {
                        Ok(d) => return d,
                        Err(e) => {
                            error!("can't parse solution: {e}");
                            println!("solution parsable: false");
                            std::process::exit(exitcode::DATAERR);
                        }
                    })
                    .collect_vec();
            })
            .collect_vec();

        for (player_idx, steps) in solution.iter().enumerate() {
            let first_coord = field.players[player_idx];
            let coords = steps.iter().fold(vec![first_coord], |mut acc, dir| {
                let new = dir.apply_step(acc.last().unwrap());
                acc.push(new);
                return acc;
            });
            buf += &format!(
                r"\draw[very thick, red, rounded corners=0.2cm] {} ;",
                coords
                    .iter()
                    .map(|s| return format!("({},{})", s.x as f32 + 0.5, s.y as f32 + 0.5))
                    .join(" -- ")
            );

            buf += "\n";
        }
    }

    // draw the mushrooms
    for m in field.mushrooms {
        let x = m.x as f32 + mush_offset.0;
        let y = (m.y + 1) as f32 + mush_offset.1;
        buf += &format!(r"\pic at ({x},{y}) {{mush}};");
        buf += "\n";
    }

    // draw the players
    for (i, p) in field.players.iter().enumerate() {
        let x = p.x as f32 + person_offset.0;
        let y = (p.y + 1) as f32 + person_offset.1;
        buf += &format!(r"\pic[pic text={{{i}}}] at ({x},{y}) {{person}};");
        buf += "\n";
    }

    buf += r"\end{tikzpicture}";

    if std::fs::write(&output_file, buf).is_err() {
        error!(
            "Failed to write output tikz code into file {}. (invalid path?)",
            output_file.display()
        );
    } else {
        info!(
            "Tikz code succesfully written into {}.",
            output_file.display()
        );
    }
}

/// Helper function that will print out formatted matrix
#[allow(dead_code)]
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
        let path = '░';
        // let path = ' ';
        let player = 'P';
        let mush = '*';
        let picked = 'o';
        let stepped = '·';
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
        if let Some(paths) = paths {
            for p in paths {
                for (i, step) in p.iter().enumerate() {
                    if i == 0 {
                        // skip the first coord, since it is a player
                        continue;
                    }
                    let cell = buf
                        .get_mut(step.y as usize)
                        .unwrap()
                        .get_mut(step.x as usize)
                        .unwrap();
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

//! CLI tool to solve mushroom-picker game using BFS algorithm

use clap::Parser;
use cli::Cli;
use itertools::Itertools;
use log::*;
use pathfinding::is_field_accessible;
use rand::seq::SliceRandom;
use solver::solve;
use utils::*;

/// cli types and definitions
mod cli;
/// path-finding related functions
mod pathfinding;
/// functions related to the solver itself
mod solver;
/// various utility functions and types
mod utils;

fn main() {
    let cli = Cli::parse();
    env_logger::Builder::new().init();

    match cli.command {
        cli::Commands::Check { map } => {
            let field = parse_field(&map);
            info!("map parsed successfully");
            match is_field_accessible(&field.cells, field.size) {
                true => println!("OK: all cells are accessible"),
                false => println!("ERR: flood-fill couldn't reach all cells"),
            }
        }
        cli::Commands::Solve { map } => {
            solve(map);
        }
        cli::Commands::Generate {
            size,
            players,
            mushrooms,
            walls,
        } => {
            if walls > size * size {
                error!("more walls ({walls}) than cells ({})", size * size);
                return;
            }
            if players + mushrooms > size * size - walls {
                error!("not enough empty cells for players and mushrooms");
                return;
            }

            let mut rng = rand::rng();
            let mut map: Vec<bool> = vec![true; size * size - walls]
                .into_iter()
                .chain(vec![false; walls])
                .collect();

            map.shuffle(&mut rng); // the first shuffle is important!
            while !is_field_accessible(&map, size) {
                map.shuffle(&mut rng);
            }

            let mut available_indices: Vec<usize> = map
                .iter()
                .enumerate()
                .filter_map(|(i, b)| match b {
                    true => return Some(i),
                    false => return None,
                })
                .collect();

            available_indices.shuffle(&mut rng);
            let player_coords = available_indices
                .get(0..players)
                .unwrap()
                .iter()
                .map(|i| {
                    return Coord {
                        x: i % size,
                        y: (i - (i % size)) / size,
                    };
                })
                .collect_vec();
            let mush_coords = available_indices
                .get(players..players + mushrooms)
                .unwrap()
                .iter()
                .map(|i| {
                    return Coord {
                        x: i % size,
                        y: (i - (i % size)) / size,
                    };
                })
                .collect_vec();

            let field = Field {
                size,
                mushrooms: mush_coords,
                players: player_coords,
                cells: map,
            };
            println!("{}", field.render());
        }
    };
}

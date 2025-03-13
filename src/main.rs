//! CLI tool to solve mushroom-picker game using BFS algorithm

use clap::Parser;
use cli::Cli;
use generator::generate_field;
use log::*;
use pathfinding::is_field_accessible;
use solver::solve;
use utils::*;

/// cli types and definitions
mod cli;
/// functions related to generating stuff
mod generator;
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
        } => match generate_field(size, walls, players, mushrooms) {
            Ok(field) => println!("{}", field.render()),
            Err(e) => error!("can not generate maze: {e}"),
        },
    };
}

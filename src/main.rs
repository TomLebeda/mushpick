//! CLI tool to solve mushroom-picker game using BFS algorithm

use std::path::PathBuf;

use clap::Parser;
use cli::Cli;
use generator::*;
use itertools::Itertools;
use log::*;
use pathfinding::*;
use renderer::*;
use solver::*;
use utils::*;

/// cli types and definitions
mod cli;
/// functions related to generating stuff
mod generator;
/// path-finding related functions
mod pathfinding;
/// functions related to rendering
mod renderer;
/// functions related to the solver itself
mod solver;
/// various utility functions and types
mod utils;

fn main() {
    let cli = Cli::parse();
    env_logger::Builder::new().init();

    match cli.command {
        cli::Commands::Check {
            map_file,
            solution_file,
        } => run_check(map_file, solution_file),
        cli::Commands::Solve { map_file } => solve(map_file),
        cli::Commands::Generate {
            size,
            players,
            mushrooms,
            walls,
            pretty,
            save,
        } => run_generate(size, players, mushrooms, walls, pretty, save),
        cli::Commands::Render {
            map,
            paths,
            out_file,
        } => render_tikz(map, paths, out_file),
    };
}

/// Run the command `check`
fn run_check(map_file: PathBuf, solution_file: Option<PathBuf>) {
    let field = parse_field(&map_file);
    trace!("map parsed successfully");
    match is_field_accessible(&field.cells, field.size) {
        true => println!("map OK: all cells are accessible"),
        false => println!("map invalid: flood-fill couldn't reach all cells"),
    }

    if let Some(solution_file) = solution_file {
        let Ok(contents) = std::fs::read_to_string(&solution_file) else {
            error!("Failed to read file {}", &solution_file.display());
            std::process::exit(exitcode::IOERR);
        };
        let lines = contents.lines();
        let solution_map: HashMap<i32, Vec<char>>
    };
}

/// Run the `generate` command
#[allow(clippy::too_many_arguments)]
fn run_generate(
    size: usize,
    players: usize,
    mushrooms: usize,
    walls: usize,
    pretty: bool,
    save: bool,
) {
    match generate_field(size, walls, players, mushrooms) {
        Ok(field) => {
            let parsable = field.render_parsable();
            if save {
                let timestamp = chrono::Local::now().format("%Y-%m-%d-%H-%M-%S-%f");
                let path = format!("map_n{size}w{walls}p{players}m{mushrooms}_{timestamp}.txt");
                match std::fs::write(&path, &parsable) {
                    Err(e) => {
                        error!("can't write the output file {}, err: {e}", path);
                        info!("here is the content, so you don't loose it:\n{}", &parsable);
                    }
                    Ok(_) => info!("output written into {}", path),
                }
            }
            if pretty {
                println!("{}", field.render_pretty(None));
            } else {
                println!("{}", parsable);
            }
        }
        Err(e) => error!("can not generate maze: {e}"),
    }
}

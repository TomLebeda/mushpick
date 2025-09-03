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
    env_logger::Builder::new()
        .filter_module("mushpick", cli.log_level.to_log_filter())
        .format_timestamp_micros()
        .init();

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
            out_file,
            solution,
        } => render_tikz(map, out_file, solution),
    };
}

/// Run the command `check`
fn run_check(map_file: PathBuf, solution_file: Option<PathBuf>) {
    trace!("checking map");
    let field = parse_field(&map_file);
    println!("map parsable: true");
    println!(
        "map accessible: {}",
        is_field_accessible(&field.cells, field.size)
    );

    if let Some(solution_file) = solution_file {
        trace!("checking solution");
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
        println!("solution parsable: true");
        let mut free_mushrooms = field.mushrooms.clone();
        let mut picked_mushrooms: Vec<Vec<Coord>> = vec![vec![]; solution.len()];

        for (player_idx, steps) in solution.iter().enumerate() {
            trace!("checking player {player_idx}");
            let Some(mut pos) = field.players.get(player_idx).cloned() else {
                error!("map doesn't have player {player_idx}");
                println!("solution passed: false");
                std::process::exit(exitcode::DATAERR);
            };
            for step in steps {
                let new_pos = step.apply_step(&pos);
                // check if the new position is in-bounds:
                if new_pos.x < 0
                    || new_pos.y < 0
                    || new_pos.x >= field.size as i32
                    || new_pos.y >= field.size as i32
                {
                    error!("player {player_idx} run out of map");
                    println!("solution passed: false");
                    std::process::exit(exitcode::DATAERR);
                }
                // check if the new position is free space:
                let new_pos_flat = new_pos.x + new_pos.y * field.size as i32;
                if !field.cells[new_pos_flat as usize] {
                    error!("player {player_idx} run into a wall");
                    println!("solution passed: false");
                    std::process::exit(exitcode::DATAERR);
                }
                // update the position
                pos = new_pos;
                // check if the new position has a mushroom:
                if let Some(mush_idx) = free_mushrooms
                    .iter()
                    .position(|p| return p.x == pos.x && p.y == pos.y)
                {
                    trace!("player {player_idx} picked a mushroom at {pos}");
                    free_mushrooms.swap_remove(mush_idx); // remove the picked mushroom
                    picked_mushrooms[player_idx].push(pos); // add the picked mushroom to the player
                }
            }
        }
        trace!("left mushroom: {}", free_mushrooms.len());
        println!("solution passed: {}", free_mushrooms.is_empty());
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
                let ts = chrono::Local::now().format("%Y-%m-%d-%H-%M-%S-%f");
                let path = format!("maps/map_n{size}w{walls}p{players}m{mushrooms}_{ts}.txt");
                match std::fs::write(&path, &parsable) {
                    Err(e) => {
                        error!("can't write the output file {path}, err: {e}");
                        info!("here is the content, so you don't loose it:\n{}", &parsable);
                    }
                    Ok(_) => info!("output written into {path}"),
                }
            }
            if pretty {
                println!("{}", field.render_pretty(None));
            } else {
                println!("{parsable}");
            }
        }
        Err(e) => error!("can not generate maze: {e}"),
    }
}

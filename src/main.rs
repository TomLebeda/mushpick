//! CLI tool to solve mushroom-picker game using BFS algorithm

use clap::Parser;
use cli::Cli;
use generator::*;
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
        cli::Commands::Check { map } => {
            let field = parse_field(&map);
            info!("map parsed successfully");
            match is_field_accessible(&field.cells, field.size) {
                true => println!("OK: all cells are accessible"),
                false => println!("ERR: flood-fill couldn't reach all cells"),
            }
        }
        cli::Commands::Solve { map, out, pretty } => {
            solve(map, out, pretty);
        }
        cli::Commands::Generate {
            size,
            players,
            mushrooms,
            walls,
            pretty,
            save,
        } => match generate_field(size, walls, players, mushrooms) {
            Ok(field) => {
                let parsable = field.render_parsable();
                if let Some(path) = save {
                    if std::fs::write(&path, &parsable).is_err() {
                        error!("can't write the output file {}", path.display());
                        info!("here is the content, so you don't loose it:\n{}", &parsable);
                    } else {
                        info!("output written into {}", path.display());
                    }
                }
                if pretty {
                    println!("{}", field.render_pretty(None));
                } else {
                    println!("{}", parsable);
                }
            }
            Err(e) => error!("can not generate maze: {e}"),
        },
        cli::Commands::Render {
            map,
            paths,
            out_file,
        } => render_tikz(map, paths, out_file),
    };
}

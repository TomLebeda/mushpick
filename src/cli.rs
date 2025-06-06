use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author = "Tomáš Lebeda <tom.lebeda@gmail.com>")]
#[command(about = "Helper software for mushroom-picking game.")]
/// Basic command-line-interface structure
pub struct Cli {
    /// Command to execute
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
/// Basic commands available via the CLI
pub enum Commands {
    /// Check if the map is ok
    ///
    /// This command will attempt to parse the file and then use flood-fill algorithm to check if all
    /// non-obstacle cells are accessible.
    Check {
        /// Path to the map file
        map: PathBuf,
    },
    /// Solve the provided map
    ///
    /// This command will attempt to parse the file and find the best solution.
    Solve {
        /// Path to the map file
        map: PathBuf,
        /// Path to the output file where the results will be stored
        out: PathBuf,
        /// if set, print out the result as an ascii-art
        #[arg(long, short = 'P')]
        pretty: bool,
    },
    /// Generate a new map that is guaranteed to be solvable
    #[clap(alias = "gen")]
    Generate {
        /// size of the map, value of N will produce N-by-N map
        #[arg(long, short = 'n')]
        size: usize,

        /// number of walls placed in the map
        #[arg(long, short)]
        walls: usize,

        /// number of players placed randomly in the map
        #[arg(long, short)]
        players: usize,

        /// number of mushrooms placed randomly in the map
        #[arg(long, short)]
        mushrooms: usize,

        /// if enabled, print pretty (non-parsable!) version into stdout
        #[arg(long, short = 'P')]
        pretty: bool,

        /// repeat the random generation multiple times and save/show all the results at once
        #[arg(long, short, default_value_t = 1)]
        batch: u32,

        /// string used to separate fields when using batch generation
        #[arg(long, alias = "sep", short = 's', default_value_t = String::from("====="))]
        batch_separator: String,

        /// save the maze (as parsable text) into given file instead of printing out, useful with --pretty
        #[arg(long, short)]
        save: Option<PathBuf>,
    },
    /// Render input map and optionally paths as tikz code for pretty pictures
    Render {
        /// file where the map is stored
        map: PathBuf,
        /// file where the output tikz code will be saved
        out_file: PathBuf,
        /// file where the paths are stored
        paths: Option<PathBuf>,
    },
}

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
    },
    /// Generate a new map that is guaranteed to be solvable
    #[clap(alias = "gen")]
    Generate {
        /// size of the map, value of N will produce N-by-N map
        #[arg(long, short, alias = "n")]
        size: usize,

        /// number of walls placed randomly in the map
        ///
        /// If the walls happen to be placed in such a way that some areas are not accessible, the
        /// attempt will be scrapped and new one will be generated.
        #[arg(long, short, alias = "w")]
        walls: usize,

        /// number of players placed randomly in the map
        #[arg(long, short)]
        players: usize,

        /// number of mushrooms placed randomly in the map
        #[arg(long, short)]
        mushrooms: usize,
    },
}

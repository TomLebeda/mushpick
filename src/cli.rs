use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(author = "Tomáš Lebeda <tom.lebeda@gmail.com>")]
#[command(about = "Helper software for mushroom-picking game.")]
/// Basic command-line-interface structure
pub struct Cli {
    /// Command to execute
    #[command(subcommand)]
    pub command: Commands,

    /// level of logging details (into stderr)
    #[arg(short, long, value_enum, default_value_t = LogLevel::Trace)]
    pub log_level: LogLevel,

    /// print the logging information into stdout instead of stderr
    #[arg(long = "logstdout", default_value_t = false)]
    pub log_to_stdout: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
#[allow(clippy::missing_docs_in_private_items)]
pub enum LogLevel {
    Warn,
    Info,
    Debug,
    Trace,
    Error,
    Off,
}

impl LogLevel {
    /// converts LogLevel into log::LevelFilter for env_logger initialization
    pub fn to_log_filter(self) -> log::LevelFilter {
        match self {
            LogLevel::Trace => return log::LevelFilter::Trace,
            LogLevel::Warn => return log::LevelFilter::Warn,
            LogLevel::Info => return log::LevelFilter::Info,
            LogLevel::Debug => return log::LevelFilter::Debug,
            LogLevel::Error => return log::LevelFilter::Error,
            LogLevel::Off => return log::LevelFilter::Off,
        }
    }
}

#[derive(Subcommand, Debug)]
/// Basic commands available via the CLI
pub enum Commands {
    /// Check if the map is ok and optionally if the provided solution solves the provided map.
    ///
    /// This command will attempt to parse the file and then use flood-fill algorithm to check if all
    /// non-obstacle cells are accessible.
    Check {
        /// Path to the map file
        map_file: PathBuf,
        /// Path to the solution file
        solution_file: Option<PathBuf>,
    },

    /// Solve the provided map
    ///
    /// This command will attempt to parse the file and find the best solution.
    /// The solution will be printed into stdout in the format `P<player-index>:<steps>`
    /// where steps is string made out of "l" "r" "u" "d" for 4-direction steps.
    /// One player-steps per output line
    Solve {
        /// Path to the map file
        map_file: PathBuf,
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

        /// save the maze (as parsable text) into a text file instead of printing out, useful with --pretty
        #[arg(long, short)]
        save: bool,
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

# mushpick

CLI helper for a multi-player mushroom-picking grid game. It can generate solvable maps,
solve them and verify solutions.

## The game

The map is an N×N grid with walls, players and mushrooms. Players move in 4 directions
(`l`, `r`, `u`, `d`). A mushroom is picked when any player steps on it. The goal is to pick
all mushrooms with the minimal total number of steps across all players.

## Build

```sh
cargo build --release
```

The binary is `target/release/mushpick`.

## Usage

```sh
mushpick [-l <level>] <command>
```

`-l, --log-level`: `trace` (default), `debug`, `info`, `warn`, `error`, `off`. Logs go to stderr.

### generate (alias `gen`)

Generate a random map that is guaranteed to be solvable.

```sh
mushpick gen -s 100 -w 3000 -p 3 -m 10
```

| Option | Meaning |
| --- | --- |
| `-s, --size N` | map is N×N |
| `-w, --walls N` | number of walls |
| `-p, --players N` | number of players (default 1) |
| `-m, --mushrooms N` | number of mushrooms (default 1) |
| `-P, --pretty` | print a human-readable version to stdout |
| `--save [PATH]` | also save the parsable map; without PATH saves to `maps/map_n<size>w<walls>p<players>m<mushrooms>_<timestamp>.txt` |

### solve

```sh
mushpick solve maps/map_s50_p3_m15.txt > solution.txt
```

Prints one line per player in the format `P<index>:<steps>`. `-f, --fast` skips the
accessibility check and the exhaustive search, returning the best heuristic solution
(not guaranteed optimal).

### check

```sh
mushpick check <map> [solution]
```

Checks that the map parses and that all free cells are reachable (flood fill).
With a solution file, it also simulates the moves and reports whether all mushrooms
were picked without leaving the map or hitting a wall.

## Map format

One text line per row, cells optionally separated by spaces:

| Char | Meaning |
| --- | --- |
| `-` | free cell |
| `X` | wall |
| `P` | player start (indexed in reading order, top-left first) |
| `M` | mushroom |

```
- X - - P
- X M X -
- - - X -
M X - - -
- - - X M
```

Example maps are in `maps/` (`map_s<size>_p<players>_m<mushrooms>.txt`).

## How the solver works

1. BFS computes the shortest paths and distances player-to-mushroom and mushroom-to-mushroom.
2. Two heuristics (greedy and lookahead) produce initial assignments of mushrooms to players.
3. Each player's visiting order is optimized with a DFS branch-and-bound using MST lower bounds.
4. Unless `--fast` is used, all assignments are then searched exhaustively, pruned by
   MST-based lower bounds against the best cost found so far.

## Tests

```sh
cargo test --no-fail-fast
```

- Unit tests live in `#[cfg(test)]` modules inside each `src/*.rs` file.
- `tests/cli.rs` drives the built binary end to end: generate, solve and check.
  It compares `solve` results against a brute-force optimum on small random maps.

## Project layout

| Path | Content |
| --- | --- |
| `src/main.rs` | entry point, `check` and `generate` commands |
| `src/cli.rs` | clap CLI definitions |
| `src/solver.rs` | splitting and ordering search |
| `src/pathfinding.rs` | BFS distance/path computation, flood fill |
| `src/generator.rs` | random map generation |
| `src/utils.rs` | map parsing and printing, shared types |

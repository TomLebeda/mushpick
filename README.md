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

Prints one line per player in the format `P<index>:<steps>`. The solution is always optimal.
Maps with up to about 100 mushrooms usually take milliseconds to seconds; the time grows
exponentially in the worst case, progress is logged at the `info` level.
`-f, --fast` uses only the heuristics (milliseconds even for 1000 mushrooms, not guaranteed optimal). Fails if some mushroom
can't be reached by any player; mushrooms reachable only by some players are fine.

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

1. One BFS per mushroom gives the mushroom-to-mushroom and player-to-mushroom distances
   (unreachable pairs are infinite) and a BFS tree used to print the paths.
2. Heuristics build an initial solution: greedy, lookahead and cheapest insertion, each
   followed by local search (2-opt, Or-opt, moving and swapping mushrooms between players,
   exchanging route ends). Routes of up to 10 mushrooms are ordered optimally.
3. Unless `--fast` is used, the exact solver proves the optimum:
   - `src/solver/search.rs`, any size: depth-first branch and bound that builds the routes
     player by player. The lower bound is the cheapest spanning forest of the remaining
     mushrooms rooted at the current route end and the later players, tightened by
     Lagrangian degree penalties (Held-Karp bound). States reached again at no lower cost
     are pruned.
   - `src/solver/exact.rs`, up to 20 mushrooms: if the search exceeds a node budget, tables
     over all mushroom subsets give the exact route cost of every player for every subset,
     and branch and bound (or dynamic programming) over these tables splits the mushrooms.
     This bounds the worst case for small maps.

## Tests

```sh
cargo test --no-fail-fast
```

- Unit tests live in `#[cfg(test)]` modules inside each `src/*.rs` file.
- `tests/cli.rs` drives the built binary end to end: generate, solve and check.
  It compares `solve` results against a brute-force optimum on small random maps and
  checks the known optimal costs of the maps in `maps/`.

## Project layout

| Path | Content |
| --- | --- |
| `src/main.rs` | entry point, `check` and `generate` commands |
| `src/cli.rs` | clap CLI definitions |
| `src/solver/mod.rs` | `solve` command, distance matrices (`Costs`) |
| `src/solver/exact.rs` | exact solver entry, subset tables for up to 20 mushrooms |
| `src/solver/search.rs` | exact branch and bound for any size (Held-Karp bound) |
| `src/solver/heuristic.rs` | construction heuristics and local search |
| `src/pathfinding.rs` | BFS distances and paths, flood fill |
| `src/generator.rs` | random map generation |
| `src/utils.rs` | map parsing and printing, shared types |

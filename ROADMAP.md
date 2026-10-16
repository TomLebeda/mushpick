# Roadmap: faster solver (exact + scalable fast mode)

Status: planned, not started. Pick up from "Order of work" below.

## Goal
Competition score = runtime + total steps of all agents (sum over agents).
- `solve` (exact) must always return a provably optimal total, as fast as possible.
- `solve --fast` must scale to large inputs (many agents, hundreds+ of mushrooms) with good quality.

## Current state (measured 2026-10-16, release build, 16 cores)
- Small maps take ~12.5 ms, of which ~9 ms is the whole-map accessibility check
  (`is_field_accessible` runs a rayon `par_iter` per BFS layer plus a hashing `unique()`).
  `solve --fast` (skips the check) takes 3.6 ms.
- Exact search ranges from 12 ms to 3.5 s (`map_s100_p3_m17`). Runtime depends mostly on how
  good the initial heuristic is (that map: fast 1530 vs optimum 1412), not on map size.
- `perf` profile of `map_s100_p3_m17`:

  | Share | Where |
  | --- | --- |
  | 54% | `optimize_split::dfs` (per-player order) |
  | ~19% | SipHash `HashMap<u64,u32>` MST cache + rehash |
  | 10% | `get_split_cost_lower_bound_mst` (allocates `Vec<Vec>` per call) |
  | ~5% | `sort_unstable_by_key` of candidates per DFS node |
  | ~5% | malloc/free (e.g. `current_split.clone()` per search node) |

- Fast mode does not scale: it still runs an exact per-player TSP, uses `u64` masks (breaks at
  64 mushrooms per player), a fixed `MAX_PLAYERS = 128` array, and BFS that is
  O(layers x mushrooms) per source and stores all n^2 paths.
- Unreachable mushrooms silently get distance 0 in fast mode (BFS leaves an empty path).

### Baseline costs (exact must match these after the rewrite)

| Map | Exact | Fast |
| --- | --- | --- |
| map_s100_p3_m5 | 691 | 691 |
| map_s100_p3_m6 | 616 | 616 |
| map_s100_p3_m7 | 414 | 414 |
| map_s100_p3_m8 | 599 | 613 |
| map_s100_p3_m9 | 802 | 878 |
| map_s100_p3_m10 | 923 | 942 |
| map_s100_p3_m11 | 617 | 726 |
| map_s100_p3_m12 | 1064 | 1126 |
| map_s100_p3_m13 | 1198 | 1203 |
| map_s100_p3_m14 | 1205 | 1278 |
| map_s100_p3_m15 | 1094 | 1094 |
| map_s100_p3_m16 | 1288 | 1321 |
| map_s100_p3_m17 | 1412 | 1530 |
| map_s100_p3_m18 | 1123 | 1145 |
| map_s100_p3_m19 | 706 | 737 |
| map_s100_p3_m20 | 1269 | 1269 |
| map_s50_p3_m15 | 597 | 597 |

Exact runtimes before: m5 12 ms, m8 14 ms, m10 15 ms, m12 26 ms, m14 131 ms, m15 71 ms,
m16 162 ms, m17 3486 ms, m18 272 ms, m19 32 ms, m20 304 ms.

## Design

### 1. Distances and paths (`src/pathfinding.rs`)
- Delete the ~270 lines of commented-out code.
- `bfs_tree(field, start) -> BfsTree { dist: Vec<u32>, dir: Vec<u8> }`: plain sequential
  queue BFS, reusing the existing direction encoding (1..5) for backtracking.
- `Distances::build(field)`: one BFS per player and per mushroom (rayon over sources),
  extract flat `p2m: Vec<u32>` (P x n) and `m2m: Vec<u32>` (n x n), unreachable = `INF`.
  Keep `dir` arrays only to rebuild the paths of legs used in the answer.
- `path(tree, target) -> Vec<Direction>` (from the existing `get_path` logic).
- `is_field_accessible`: sequential flood fill with a visited array (no rayon, no `unique()`).
- `solve` replaces the whole-map accessibility check with "every mushroom is reachable by at
  least one player" (free from the BFS, done in both modes). Error otherwise.

### 2. Solver core (split `src/solver.rs` into `src/solver/`)
- `instance.rs`: `Instance { n, p, m2m, p2m }` with `d(i,j)`, `s(p,i)`, route cost,
  per-mushroom neighbor lists sorted by distance (used by both modes).
- `route.rs` (single player, fixed start, open path):
  - exact B&B over a `u64` mask (adapted from current `optimize_split::dfs`): global mushroom
    indices, bound `cost + min edge from last + MST(remaining)`, candidates from presorted
    neighbor lists instead of sorting per node, an upper limit for early exit (returns the
    exact cost or "at least limit");
  - heuristic: nearest neighbor + 2-opt + Or-opt (segments 1..3) for open paths, used when
    the route is too long for exact.
- `mst.rs`: MST cost of a mushroom set. One cache shared by all players and splits (the MST of
  a set does not depend on the player): dense `Vec<AtomicU32>` indexed by mask when n <= 22
  (16 MB), otherwise `rustc_hash::FxHashMap` per thread. Prim with reused buffers.
- `fast.rs` (fast mode, any size):
  1. initial splits: existing greedy and lookahead (lookahead duplicate bug already fixed);
  2. intra-route: exact for routes <= 12 mushrooms, else 2-opt + Or-opt;
  3. inter-route local search until no improvement: relocate (move a mushroom to its cheapest
     insertion in another route) and swap (exchange two mushrooms), restricted to k nearest
     neighbors; re-optimize changed routes;
  4. return the best. No masks, no player limit, deterministic.
- `exact.rs` (exact mode, n <= 64, otherwise a clear error suggesting `--fast`):
  - upper bound = fast-mode result (tighter start => far more pruning);
  - DFS assigning mushrooms to players with per-player `u64` masks (no `Vec` clones);
    mushroom order: farthest from all players first; player order: cheapest insertion first;
  - per-player lower bound `min_{i in S} s(p,i) + MST(S)`: at least as tight as the current
    `MST(S + player)` and valid because skipping mushrooms never makes a route longer
    (shortest-path distances satisfy the triangle inequality); maintained incrementally;
  - leaf: sum of exact route costs, memoized per thread by `(player, mask)` (exact value or a
    proven lower bound), early exit once the running sum reaches the best;
  - parallel: enumerate assignment prefixes until there are >= 8 x threads tasks, rayon
    `par_iter`, shared best in `AtomicU32`, best split in a `Mutex` with a deterministic
    tie-break (lexicographically smallest masks) so output is reproducible.
- `mod.rs`: `solve(map, fast)`: build distances, check reachability, run fast or exact, print
  `P<i>:<steps>` from BFS trees. Remove `struct Result`, `MAX_PLAYERS`, old split code.

### 3. Build
- `Cargo.toml`: `[profile.release] lto = "fat", codegen-units = 1, panic = "abort"`.
- `cargo add rustc-hash` (only new dependency); `cargo remove regex serde_json` (unused).
- README: mention optional `RUSTFLAGS="-C target-cpu=native"` (only if the competition machine
  matches the build machine).

## Tests
- Rewrite solver unit tests for the new API:
  - exact route == brute-force permutations; heuristic route >= exact and is a permutation;
  - lower bound <= exact route cost; MST on a known graph;
  - exact solver == brute-force optimum on many random small instances (1-5 players, up to
    8 mushrooms), including mushrooms unreachable for some players;
  - fast result covers every mushroom exactly once and is >= exact;
  - large instance (e.g. 60x60, 10 players, 300 mushrooms) in fast mode finishes and is valid.
- Pathfinding tests adapted to `Distances` (compare with the reference BFS, walk rebuilt paths).
- Integration (`tests/cli.rs`): keep existing; add more players to the optimality test; add
  `repo_maps_exact_costs` asserting the baseline table (fast maps always, the full set
  `#[ignore]`, run with `cargo test --release -- --ignored`); unreachable mushroom => error.

## Order of work (run tests + benchmark after each step)
1. Release profile, sequential flood fill, new BFS/`Distances`, reachability check.
2. Instance, MST cache, route B&B, single-threaded exact search; costs must equal the baseline.
3. Parallel exact search.
4. Fast mode heuristics + large-instance tests.
5. Dead code/deps cleanup, README (solver section, limits, benchmark command).

## Verification
- `cargo test --release --no-fail-fast` and `cargo test --release -- --ignored`.
- `cargo clippy --all-targets`: no new warnings.
- Benchmark before/after for every map, both modes:
  `hyperfine -N -w 3 './target/release/mushpick -l off solve maps/<map>'`;
  report a table (time and cost) old vs new.
- Large fast-mode run on generated maps (e.g. `gen -s 200 -w 12000 -p 20 -m 500`), checked
  with `mushpick check`.

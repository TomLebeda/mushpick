# Roadmap: solver performance

Status: the solver rewrite is done (2026-10-17). Open ideas are at the end.

## Goal
Competition score = runtime + total steps of all agents (sum over agents).
- `solve` always returns a provably optimal total, for any map size (it may take long).
- `solve --fast` uses heuristics that scale to many players and mushrooms.

## What changed
- Pathfinding: one sequential, branchless queue BFS per mushroom on a grid with a wall border
  (rayon over sources). Players get their distances from the mushroom trees, so there is no BFS
  per player. Unreachable pairs are `INF`. The whole-map accessibility check is gone from
  `solve`: it now only fails when a mushroom can't be reached by any player.
- Exact search (`src/solver/search.rs`), any size: branch and bound building the routes player
  by player, Held-Karp style bound (spanning forest rooted at the route end and the later
  players, Lagrangian degree penalties, warm started from the parent), forests that are paths
  become new solutions, a memo of (player, route end, picked set) prunes repeated states
  (capped at 512 MiB).
- Subset tables (`src/solver/exact.rs`), up to 20 mushrooms, used when the search exceeds its
  node budget: subset table of walks (player independent), exact route cost of each player for
  each subset, then branch and bound over the split, or subset DP (3^n per player).
- Heuristics (`src/solver/heuristic.rs`): greedy, lookahead, cheapest insertion, then local
  search (2-opt, Or-opt, relocate, swap, tail exchange); routes of up to 10 mushrooms are
  ordered by DP. No mask or player limits.
- Removed: MST bounds, `MAX_PLAYERS`, ~270 lines of commented-out code, unused `regex`,
  `serde`, `serde_json`. A release profile with LTO was measured and gave no gain.

## Results (release build, Ryzen 7 7700X, 16 threads, hyperfine mean)

| Map | Steps | Before | After |
| --- | --- | --- | --- |
| map_s100_p3_m14 | 1205 | 104 ms | 1.7 ms |
| map_s100_p3_m16 | 1288 | 128 ms | 1.1 ms |
| map_s100_p3_m17 | 1412 | 2802 ms | 2.6 ms |
| map_s100_p3_m18 | 1123 | 208 ms | 1.2 ms |
| map_s100_p3_m19 | 706 | 16 ms | 1.2 ms |
| map_s100_p3_m20 | 1269 | 210 ms | 1.4 ms |
| map_s50_p3_m15 | 597 | 7.2 ms | 1.2 ms |

All repo maps keep their optimal costs (asserted in `tests/cli.rs`); the exact search and the
subset tables agree on all repo maps and 30 random maps with 12 to 20 mushrooms.

Exact mode on generated maps (size, walls, players, mushrooms):

| Map | Heuristic | Optimum | Time |
| --- | --- | --- | --- |
| 100, 4000, 3, 30 | 1309 | 1305 | 4 ms |
| 100, 4000, 5, 40 | 1991 | 1931 | 62 ms |
| 100, 3000, 5, 50 | 749 | 738 | 75 ms |
| 200, 15000, 5, 75 | 2776 | 2724 | 3.1 s |
| 200, 15000, 8, 100 | 3548 | 3409 | 12.7 s |
| 500, 70000, 10, 200 | 7239 | not proven | > 5 min (best 7210, bound 7027) |
| 200, 12000, 20, 500 | 5326 | not proven | > 5 min (bound 5151) |

Fast mode on large maps (before: old `--fast`):

| Map | Before | After |
| --- | --- | --- |
| 200, 12000, 20, 500 | 4.6 s, 5527 steps | 51 ms, 5326 steps |
| 500, 70000, 10, 200 | 84 s, 7550 steps | 30 ms, 7239 steps |
| 300, 30000, 50, 1000 | > 120 s | 0.29 s, 11968 steps |

Heuristic vs optimum: optimal on 20 of 30 random maps with 12 to 20 mushrooms (at most 3.9 %
worse), 2 to 5 % worse at 40 to 100 mushrooms.

## Open ideas
- Parallel exact search (subtrees on rayon threads, shared best cost): up to ~8x on long runs.
- Stronger heuristic (iterated local search with a fixed seed): better `--fast` results and a
  tighter starting bound, which shrinks the exact search.
- Print the best solution found so far when an exact run is interrupted (Ctrl-C).
- BFS dominates on large maps with few mushrooms (about 4 ms for 15 mushrooms on 500x500).

## Verification
- `cargo test --release --no-fail-fast`, `cargo clippy --all-targets`.
- Benchmark: `hyperfine -N -w 3 './target/release/mushpick -l off solve maps/<map>'`.
- Large maps: `mushpick gen -s 200 -w 12000 -p 20 -m 500 > big.txt`, solve, then
  `mushpick check big.txt solution.txt`.

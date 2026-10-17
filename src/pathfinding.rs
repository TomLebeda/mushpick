use rayon::prelude::*;

use crate::{Coord, Field, solver::Costs, utils::Direction};

/// Distance of an unreachable pair
pub const INF: u32 = u32::MAX;

// Cell codes of a BFS tree. A visited cell stores the step that leads one cell closer to the
// BFS source, so walking the codes from any visited cell gives a shortest path to the source.
/// free cell not visited yet
const UNSEEN: u8 = 0;
/// step right to get closer to the source
const STEP_RIGHT: u8 = 1;
/// step up to get closer to the source
const STEP_UP: u8 = 2;
/// step left to get closer to the source
const STEP_LEFT: u8 = 3;
/// step down to get closer to the source
const STEP_DOWN: u8 = 4;
/// the BFS source itself
const SOURCE: u8 = 5;
/// wall or the border around the map
const WALL: u8 = 6;
/// free cell holding a BFS target (a mushroom or a player), not visited yet
const TARGET: u8 = 8;

/// Marks a cell that is not a BFS target
const NO_TARGET: u32 = u32::MAX;

/// Keep the BFS trees for path reconstruction only while they fit into this many bytes,
/// otherwise rebuild them when printing the solution
const TREE_MEMORY_BUDGET: usize = 512 << 20;

/// Map surrounded by a one-cell wall border, so neighbors never need bounds checks
struct Grid {
    /// width (and height) including the border
    width: usize,
    /// row-major cells: `UNSEEN` or `TARGET` for free cells, `WALL` otherwise
    cells: Vec<u8>,
}

impl Grid {
    fn new(cells: &[bool], size: usize) -> Grid {
        let width = size + 2;
        let mut padded = vec![WALL; width * width];
        for y in 0..size {
            for x in 0..size {
                if cells[y * size + x] {
                    padded[(y + 1) * width + x + 1] = UNSEEN;
                }
            }
        }
        return Grid {
            width,
            cells: padded,
        };
    }

    /// flat index of a map coordinate
    fn index(&self, c: &Coord) -> usize {
        return (c.y as usize + 1) * self.width + c.x as usize + 1;
    }

    /// Run BFS from `source`, returning the BFS tree and the number of visited cells.
    ///
    /// `target_slot` maps `TARGET` cells to indices into `dist`, which receives the distances of
    /// the reached targets. The search stops early once all `dist.len()` targets are reached
    /// (never when there are no targets, so it can be used as a flood fill).
    fn bfs(
        &self,
        source: usize,
        target_slot: &[u32],
        dist: &mut [u32],
        queue: &mut Vec<u32>,
    ) -> (Vec<u8>, usize) {
        let w = self.width;
        let mut tree = self.cells.clone();
        tree[source] = SOURCE;
        // every cell enters the queue at most once
        if queue.len() < tree.len() {
            queue.resize(tree.len(), 0);
        }
        queue[0] = source as u32;

        let n_targets = dist.len();
        let mut found = 0;
        let mut reach = |cell: usize, d: u32, found: &mut usize| {
            let slot = target_slot[cell];
            if slot != NO_TARGET {
                dist[slot as usize] = d;
                *found += 1;
            }
        };
        if self.cells[source] == TARGET {
            reach(source, 0, &mut found);
        }

        let (mut head, mut tail, mut level_end, mut d) = (0, 1, 1, 0);
        while head < tail && (found < n_targets || n_targets == 0) {
            if head == level_end {
                d += 1;
                level_end = tail;
            }
            let cell = queue[head] as usize;
            head += 1;
            // neighbor and the step from the neighbor back to `cell`
            for (next, step) in [
                (cell + 1, STEP_LEFT),
                (cell - 1, STEP_RIGHT),
                (cell + w, STEP_UP),
                (cell - w, STEP_DOWN),
            ] {
                let code = tree[next];
                // branchless on purpose: walls are random, so a branch here mispredicts a lot
                let open = code & !TARGET == 0; // UNSEEN or TARGET
                tree[next] = if open { step } else { code };
                queue[tail] = next as u32;
                tail += open as usize;
                if code == TARGET {
                    reach(next, d + 1, &mut found);
                }
            }
        }
        return (tree, tail);
    }

    /// Steps from `from` to the source of `tree`, along the tree
    fn path_to_source(&self, tree: &[u8], from: usize) -> Vec<Direction> {
        let mut path = vec![];
        let mut cell = from;
        loop {
            let step = match tree[cell] {
                STEP_RIGHT => Direction::Right,
                STEP_UP => Direction::Up,
                STEP_LEFT => Direction::Left,
                STEP_DOWN => Direction::Down,
                SOURCE => return path,
                _ => unreachable!("cell {cell} is not in the BFS tree"),
            };
            cell = match step {
                Direction::Right => cell + 1,
                Direction::Up => cell - self.width,
                Direction::Left => cell - 1,
                Direction::Down => cell + self.width,
            };
            path.push(step);
        }
    }
}

/// Shortest distances between players and mushrooms, with the paths behind them
pub struct Distances {
    /// distance matrices used by the solver
    pub costs: Costs,
    /// padded map
    grid: Grid,
    /// grid index of each player
    player_cells: Vec<usize>,
    /// grid index of each mushroom
    mushroom_cells: Vec<usize>,
    /// BFS tree of each mushroom, `None` when they did not fit into memory
    trees: Option<Vec<Vec<u8>>>,
}

impl Distances {
    /// Run one BFS from every mushroom, reaching all mushrooms and players
    pub fn build(field: &Field) -> Distances {
        let grid = Grid::new(&field.cells, field.size);
        let player_cells: Vec<usize> = field.players.iter().map(|c| return grid.index(c)).collect();
        let mushroom_cells: Vec<usize> =
            field.mushrooms.iter().map(|c| return grid.index(c)).collect();
        let (n, p) = (mushroom_cells.len(), player_cells.len());

        // targets: mushrooms first, then players
        let mut grid = grid;
        let mut target_slot = vec![NO_TARGET; grid.cells.len()];
        for (slot, &cell) in mushroom_cells.iter().chain(&player_cells).enumerate() {
            target_slot[cell] = slot as u32;
            grid.cells[cell] = TARGET;
        }

        let keep_trees = n.saturating_mul(grid.cells.len()) <= TREE_MEMORY_BUDGET;
        let results: Vec<(Vec<u32>, Option<Vec<u8>>)> = mushroom_cells
            .par_iter()
            .map_init(Vec::new, |queue, &cell| {
                let mut dist = vec![INF; n + p];
                let (tree, _) = grid.bfs(cell, &target_slot, &mut dist, queue);
                return (dist, keep_trees.then_some(tree));
            })
            .collect();

        let mut m2m = vec![INF; n * n];
        let mut p2m = vec![INF; p * n];
        let mut trees = keep_trees.then(Vec::new);
        for (i, (dist, tree)) in results.into_iter().enumerate() {
            m2m[i * n..(i + 1) * n].copy_from_slice(&dist[..n]);
            for pi in 0..p {
                p2m[pi * n + i] = dist[n + pi];
            }
            if let (Some(trees), Some(tree)) = (&mut trees, tree) {
                trees.push(tree);
            }
        }

        return Distances {
            costs: Costs { n, p, m2m, p2m },
            grid,
            player_cells,
            mushroom_cells,
            trees,
        };
    }

    /// Steps of each player walking its route (mushroom indices in visiting order)
    pub fn paths(&self, routes: &[Vec<usize>]) -> Vec<Vec<Direction>> {
        return routes
            .par_iter()
            .enumerate()
            .map_init(Vec::new, |queue, (pi, route)| {
                let mut path = vec![];
                let mut from = self.player_cells[pi];
                for &m in route {
                    // the tree of the destination mushroom leads from any cell to it
                    let tree = match &self.trees {
                        Some(trees) => std::borrow::Cow::Borrowed(&trees[m]),
                        None => {
                            let slot = vec![NO_TARGET; self.grid.cells.len()];
                            let (tree, _) = self.grid.bfs(self.mushroom_cells[m], &slot, &mut [], queue);
                            std::borrow::Cow::Owned(tree)
                        }
                    };
                    path.extend(self.grid.path_to_source(&tree, from));
                    from = self.mushroom_cells[m];
                }
                return path;
            })
            .collect();
    }
}

/// Run flood fill to check if all free cells are accessible
pub fn is_field_accessible(cells: &[bool], size: usize) -> bool {
    let Some(start) = cells.iter().position(|free| return *free) else {
        log::warn!("the whole field is made out of obstacles");
        return false;
    };
    let grid = Grid::new(cells, size);
    let start = grid.index(&Coord {
        x: (start % size) as i32,
        y: (start / size) as i32,
    });
    let slot = vec![NO_TARGET; grid.cells.len()];
    let (_, visited) = grid.bfs(start, &slot, &mut [], &mut Vec::new());
    return visited == cells.iter().filter(|free| return **free).count();
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::generator::generate_field;
    use crate::utils::tests::field;

    /// Reference BFS distance between two coordinates, None if unreachable
    fn ref_dist(f: &Field, from: Coord, to: Coord) -> Option<usize> {
        let n = f.size as i32;
        let idx = |c: Coord| return (c.y * n + c.x) as usize;
        let mut dist = vec![usize::MAX; f.cells.len()];
        let mut q = VecDeque::from([from]);
        dist[idx(from)] = 0;
        while let Some(c) = q.pop_front() {
            if c == to {
                return Some(dist[idx(c)]);
            }
            for d in Direction::get_all_dirs() {
                let nc = d.apply_step(&c);
                if nc.x < 0 || nc.y < 0 || nc.x >= n || nc.y >= n {
                    continue;
                }
                if f.cells[idx(nc)] && dist[idx(nc)] == usize::MAX {
                    dist[idx(nc)] = dist[idx(c)] + 1;
                    q.push_back(nc);
                }
            }
        }
        return None;
    }

    /// Walk the path from `from` and assert it stays on free cells and ends at `to`
    fn assert_valid_path(f: &Field, from: Coord, to: Coord, path: &[Direction]) {
        let n = f.size as i32;
        let mut pos = from;
        for d in path {
            pos = d.apply_step(&pos);
            assert!(
                pos.x >= 0 && pos.y >= 0 && pos.x < n && pos.y < n,
                "left the map"
            );
            assert!(
                f.cells[(pos.y * n + pos.x) as usize],
                "walked into a wall at {pos}"
            );
        }
        assert_eq!(pos, to, "path does not end at the target");
    }

    /// Distance as stored in the matrices, None if unreachable
    fn stored(d: u32) -> Option<usize> {
        return (d != INF).then_some(d as usize);
    }

    /// Check the distance matrices and the rebuilt paths against the reference BFS
    fn check_distances(f: &Field) {
        let dist = Distances::build(f);
        let c = &dist.costs;
        assert_eq!((c.n, c.p), (f.mushrooms.len(), f.players.len()));
        for (i, &mi) in f.mushrooms.iter().enumerate() {
            for (j, &mj) in f.mushrooms.iter().enumerate() {
                assert_eq!(stored(c.d(i, j)), ref_dist(f, mi, mj), "m2m {i}->{j}");
            }
            for (pi, &p) in f.players.iter().enumerate() {
                assert_eq!(stored(c.s(pi, i)), ref_dist(f, p, mi), "p2m {pi}->{i}");
            }
        }

        // every player walks to every reachable mushroom, then on to the next one
        for pi in 0..c.p {
            let reachable: Vec<usize> = (0..c.n).filter(|&i| return c.s(pi, i) != INF).collect();
            let mut routes = vec![vec![]; c.p];
            routes[pi] = reachable.clone();
            let path = &dist.paths(&routes)[pi];
            assert_eq!(path.len() as u32, c.route_cost(pi, &reachable));
            let end = reachable.last().map_or(f.players[pi], |&m| return f.mushrooms[m]);
            assert_valid_path(f, f.players[pi], end, path);
        }
    }

    #[test]
    fn accessible_open_map() {
        let f = field("- - -\n- - -\n- - -\n");
        assert!(is_field_accessible(&f.cells, f.size));
    }

    #[test]
    fn accessible_maze_with_walls() {
        let f = field("- X -\n- X -\n- - -\n");
        assert!(is_field_accessible(&f.cells, f.size));
    }

    #[test]
    fn not_accessible_when_split_by_wall() {
        let f = field("- X -\n- X -\n- X -\n");
        assert!(!is_field_accessible(&f.cells, f.size));
    }

    #[test]
    fn not_accessible_when_cell_enclosed() {
        let f = field("- - - -\n- - X -\n- X - X\n- - X -\n");
        assert!(!is_field_accessible(&f.cells, f.size));
    }

    #[test]
    fn not_accessible_when_all_walls() {
        let f = field("X X\nX X\n");
        assert!(!is_field_accessible(&f.cells, f.size));
    }

    #[test]
    fn distances_with_detour() {
        // the player has to go around the wall to reach the mushrooms
        let f = field("P X M -\n- X X -\n- - - -\nM X - -\n");
        let c = Distances::build(&f).costs;
        assert_eq!(c.s(0, 0), 8);
        assert_eq!(c.s(0, 1), 3);
        check_distances(&f);
    }

    #[test]
    fn distances_on_hand_made_maps() {
        check_distances(&field("P M\n- -\n"));
        check_distances(&field("P - - M\nX X - X\nM - - -\nP X X M\n"));
        check_distances(&field(
            "P - X - M\n- X - - X\n- - - X -\nX X - - -\nM - - X P\n",
        ));
    }

    #[test]
    fn unreachable_pairs_are_inf() {
        // the right column is cut off by walls
        let f = field("P X M\nM X -\n- X P\n");
        let c = Distances::build(&f).costs;
        assert_eq!(c.s(0, 1), 1);
        assert_eq!(c.s(0, 0), INF);
        assert_eq!(c.d(0, 1), INF);
        assert_eq!(c.s(1, 0), 2);
        check_distances(&f);
    }

    #[test]
    fn distances_on_generated_maps() {
        for size in [5, 8, 13] {
            for _ in 0..10 {
                let f = generate_field(size, size * size / 3, 3, 6).unwrap();
                check_distances(&f);
            }
        }
    }
}

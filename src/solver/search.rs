//! Exact branch and bound for maps of any size.
//!
//! Routes are built in player order: a node either extends the route of the current player by
//! one mushroom or closes it and moves on to the next player.
//!
//! Lower bound: in any completion, every remaining mushroom has one predecessor (another
//! mushroom, the end of the current route or the start of a later player), so the predecessor
//! edges form a spanning forest with one root per tree. The cheapest such forest is a lower
//! bound. Lagrangian penalties on the degrees (at most 2 for a mushroom, 1 for a root) make it
//! much tighter (Held-Karp bound). When the forest happens to be a set of paths it is a complete
//! solution, which may improve the best one.
//!
//! A state reached again (same player, same route end, same picked mushrooms) at a cost that is
//! not lower than before is pruned.

use log::*;
use rustc_hash::FxHashMap;

use super::Costs;
use crate::pathfinding::INF;

/// Subgradient iterations at the root
const ROOT_ITERATIONS: usize = 300;
/// Subgradient iterations at the other nodes
const NODE_ITERATIONS: usize = 10;
/// Memory for remembered states
const SEEN_BYTES: usize = 512 << 20;
/// Log progress every this many nodes
const PROGRESS_NODES: u64 = 1 << 18;

/// Start of a tree in the bound forest
#[derive(Clone, Copy)]
enum Root {
    /// end of the current route
    Mushroom(usize),
    /// start of a later player
    Player(usize),
}

/// Depth-first search state
struct Search<'a> {
    /// distances
    c: &'a Costs,
    /// cost of the best solution found
    best: u32,
    /// routes of the best solution found
    best_routes: Vec<Vec<usize>>,
    /// routes of the current node
    routes: Vec<Vec<usize>>,
    /// mushrooms not picked yet, ascending
    left: Vec<usize>,
    /// picked mushrooms as a bitset
    picked: Vec<u64>,
    /// cheapest cost each state was reached at
    seen: FxHashMap<Box<[u64]>, u32>,
    /// most states to remember
    max_seen: usize,
    /// nodes visited
    nodes: u64,
    /// lower bound at the root
    root_bound: u32,
    /// stop after this many nodes
    max_nodes: u64,
}

/// Forest computed by the bound, indices are positions in `Search::left`
struct Forest {
    /// cost with penalties, minus the penalty constants (the Lagrangian bound)
    bound: f64,
    /// parent of each mushroom: a root index, or `roots.len() + position` of a mushroom
    parent: Vec<usize>,
    /// degree of each root
    root_deg: Vec<i32>,
    /// degree of each mushroom
    deg: Vec<i32>,
}

impl Search<'_> {
    /// cost of the leg from a root to mushroom `m`
    #[inline(always)]
    fn root_leg(&self, root: Root, m: usize) -> u32 {
        return match root {
            Root::Mushroom(e) => self.c.d(e, m),
            Root::Player(p) => self.c.s(p, m),
        };
    }

    /// index of a root into the penalty vector
    #[inline(always)]
    fn root_pi(&self, root: Root) -> usize {
        return match root {
            Root::Mushroom(e) => e,
            Root::Player(p) => self.c.n + p,
        };
    }

    /// Cheapest forest under the penalties `pi`, `None` when some mushroom can't be connected
    fn forest(&self, roots: &[Root], pi: &[f64]) -> Option<Forest> {
        let r = self.left.len();
        let mut key = vec![f64::INFINITY; r];
        let mut parent = vec![usize::MAX; r];
        for (ri, &root) in roots.iter().enumerate() {
            let pr = pi[self.root_pi(root)];
            for (v, &m) in self.left.iter().enumerate() {
                let w = self.root_leg(root, m);
                if w != INF {
                    let val = w as f64 + pr + pi[m];
                    if val < key[v] {
                        key[v] = val;
                        parent[v] = ri;
                    }
                }
            }
        }

        // Prim from all roots at once
        let mut in_tree = vec![false; r];
        let mut total = 0.0;
        let mut root_deg = vec![0; roots.len()];
        let mut deg = vec![0; r];
        for _ in 0..r {
            let mut u = usize::MAX;
            let mut best = f64::INFINITY;
            for v in 0..r {
                if !in_tree[v] && key[v] < best {
                    best = key[v];
                    u = v;
                }
            }
            if u == usize::MAX {
                return None;
            }
            in_tree[u] = true;
            total += best;
            deg[u] += 1;
            if parent[u] < roots.len() {
                root_deg[parent[u]] += 1;
            } else {
                deg[parent[u] - roots.len()] += 1;
            }
            let mu = self.left[u];
            let pu = pi[mu];
            for v in 0..r {
                if !in_tree[v] {
                    let w = self.c.d(mu, self.left[v]);
                    if w != INF {
                        let val = w as f64 + pu + pi[self.left[v]];
                        if val < key[v] {
                            key[v] = val;
                            parent[v] = roots.len() + u;
                        }
                    }
                }
            }
        }

        let penalties: f64 = self.left.iter().map(|&m| return 2.0 * pi[m]).sum::<f64>()
            + roots.iter().map(|&root| return pi[self.root_pi(root)]).sum::<f64>();
        return Some(Forest {
            bound: total - penalties,
            parent,
            root_deg,
            deg,
        });
    }

    /// Record the forest as a solution if it is a set of paths that beats the best one
    fn try_forest_solution(&mut self, k: usize, roots: &[Root], forest: &Forest, partial: u32) {
        if forest.deg.iter().any(|&d| return d > 2) || forest.root_deg.iter().any(|&d| return d > 1)
        {
            return;
        }
        // every node has at most one child, follow the chains from the roots
        let mut child = vec![usize::MAX; roots.len() + self.left.len()];
        let mut cost = partial;
        for (v, &par) in forest.parent.iter().enumerate() {
            child[par] = v;
            let leg = match par.checked_sub(roots.len()) {
                Some(u) => self.c.d(self.left[u], self.left[v]),
                None => self.root_leg(roots[par], self.left[v]),
            };
            cost = cost.saturating_add(leg);
        }
        if cost >= self.best {
            return;
        }
        let mut routes = self.routes.clone();
        for (ri, &root) in roots.iter().enumerate() {
            let p = match root {
                Root::Mushroom(_) => k,
                Root::Player(p) => p,
            };
            let mut v = child[ri];
            while v != usize::MAX {
                routes[p].push(self.left[v]);
                v = child[roots.len() + v];
            }
        }
        debug_assert_eq!(self.c.total_cost(&routes), cost);
        self.best = cost;
        self.best_routes = routes;
    }

    /// Improve the penalties `pi` by subgradient steps. Returns the lower bound on the cost of
    /// the best completion, or `None` when no completion beats the best solution.
    fn bound(
        &mut self,
        k: usize,
        end: Option<usize>,
        partial: u32,
        pi: &mut [f64],
        iterations: usize,
    ) -> Option<u32> {
        let mut roots = vec![match end {
            Some(m) => Root::Mushroom(m),
            None => Root::Player(k),
        }];
        roots.extend((k + 1..self.c.p).map(Root::Player));

        let mut best_bound = 0;
        let mut step = 2.0;
        let mut stale = 0;
        for _ in 0..iterations {
            let forest = self.forest(&roots, pi)?;
            let bound = (forest.bound - 1e-6).ceil().max(0.0) as u32;
            if bound > best_bound {
                best_bound = bound;
                stale = 0;
            } else {
                stale += 1;
                if stale >= 3 {
                    step /= 2.0;
                    stale = 0;
                }
            }
            self.try_forest_solution(k, &roots, &forest, partial);
            if partial.saturating_add(best_bound) >= self.best {
                return None;
            }

            // subgradient: degree above the limit raises the penalty, below it lowers it
            let mut norm = 0.0;
            for &d in forest.deg.iter() {
                norm += ((d - 2) * (d - 2)) as f64;
            }
            for &d in forest.root_deg.iter() {
                norm += ((d - 1) * (d - 1)) as f64;
            }
            if norm == 0.0 {
                break;
            }
            let gap = (self.best - partial) as f64 - forest.bound;
            let t = step * gap.max(1.0) / norm;
            for (v, &m) in self.left.iter().enumerate() {
                pi[m] = (pi[m] + t * (forest.deg[v] - 2) as f64).max(0.0);
            }
            for (ri, &root) in roots.iter().enumerate() {
                let i = self.root_pi(root);
                pi[i] = (pi[i] + t * (forest.root_deg[ri] - 1) as f64).max(0.0);
            }
        }
        return Some(best_bound);
    }

    /// True when the state was already reached at most as cheaply, remembers it otherwise
    fn dominated(&mut self, k: usize, end: Option<usize>, partial: u32) -> bool {
        let end_id = end.unwrap_or(self.c.n) as u64;
        let mut key = Vec::with_capacity(self.picked.len() + 1);
        key.push((k as u64) << 32 | end_id);
        key.extend_from_slice(&self.picked);
        let key = key.into_boxed_slice();
        if let Some(cost) = self.seen.get_mut(&key) {
            if *cost <= partial {
                return true;
            }
            *cost = partial;
        } else if self.seen.len() < self.max_seen {
            self.seen.insert(key, partial);
        }
        return false;
    }

    /// Explore the node: player `k` is at `end` (its start when `None`), picking cost `partial`
    fn dfs(&mut self, k: usize, end: Option<usize>, partial: u32, mut pi: Vec<f64>, iterations: usize) {
        if self.nodes >= self.max_nodes {
            return;
        }
        self.nodes += 1;
        if self.nodes.is_multiple_of(PROGRESS_NODES) {
            info!(
                "exact search: {} nodes, best {}, lower bound {}, {} states remembered",
                self.nodes,
                self.best,
                self.root_bound,
                self.seen.len()
            );
        }
        if self.left.is_empty() {
            if partial < self.best {
                self.best = partial;
                self.best_routes = self.routes.clone();
            }
            return;
        }
        if self.dominated(k, end, partial) {
            return;
        }
        let Some(bound) = self.bound(k, end, partial, &mut pi, iterations) else {
            return;
        };
        if self.nodes == 1 {
            self.root_bound = bound;
            info!("exact search: lower bound {bound}, heuristic {}", self.best);
        }

        // children: (sort key, mushroom to pick next, or None to close the route)
        let mut kids: Vec<(u32, Option<usize>)> = self
            .left
            .iter()
            .map(|&m| {
                return (
                    match end {
                        Some(e) => self.c.d(e, m),
                        None => self.c.s(k, m),
                    },
                    Some(m),
                );
            })
            .filter(|&(w, _)| return w != INF)
            .collect();
        if k + 1 < self.c.p {
            // closing is ranked by the cheapest first leg of a later player
            let next = (k + 1..self.c.p)
                .flat_map(|p| return self.left.iter().map(move |&m| return (p, m)))
                .map(|(p, m)| return self.c.s(p, m))
                .min()
                .unwrap();
            if next != INF {
                kids.push((next, None));
            }
        }
        kids.sort_by_key(|&(w, _)| return w);

        for (w, kid) in kids {
            match kid {
                Some(m) => {
                    if partial + w >= self.best {
                        continue;
                    }
                    let pos = self.left.binary_search(&m).unwrap();
                    self.left.remove(pos);
                    self.picked[m / 64] |= 1 << (m % 64);
                    self.routes[k].push(m);
                    self.dfs(k, Some(m), partial + w, pi.clone(), NODE_ITERATIONS);
                    self.routes[k].pop();
                    self.picked[m / 64] ^= 1 << (m % 64);
                    self.left.insert(pos, m);
                }
                None => self.dfs(k + 1, None, partial, pi.clone(), NODE_ITERATIONS),
            }
        }
    }
}

/// Optimal routes for all players (`c.n >= 1`, `c.p >= 1`), given a feasible solution `upper`.
/// With `max_nodes` the search may stop early, then the result is only the best routes found
/// and the second value is false.
pub fn solve(c: &Costs, upper: &[Vec<usize>], max_nodes: Option<u64>) -> (Vec<Vec<usize>>, bool) {
    let mut search = Search {
        c,
        best: c.total_cost(upper),
        best_routes: upper.to_vec(),
        routes: vec![vec![]; c.p],
        left: (0..c.n).collect(),
        picked: vec![0; c.n.div_ceil(64)],
        seen: FxHashMap::default(),
        // key, its allocation and the map entry
        max_seen: SEEN_BYTES / (8 * (c.n.div_ceil(64) + 1) + 48),
        nodes: 0,
        root_bound: 0,
        max_nodes: max_nodes.unwrap_or(u64::MAX),
    };
    search.dfs(0, None, 0, vec![0.0; c.n + c.p], ROOT_ITERATIONS);
    let complete = search.nodes < search.max_nodes;
    if complete {
        info!("exact search: optimum {} after {} nodes", search.best, search.nodes);
    }
    return (search.best_routes, complete);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{exact, heuristic, tests::*};

    #[test]
    fn matches_brute_force_on_random_maps() {
        for players in 1..=4 {
            for mushrooms in 1..=6 {
                for _ in 0..5 {
                    let c = random_costs(7, 15, players, mushrooms);
                    // start from a poor solution so the search has to do the work
                    let mut upper = vec![vec![]; c.p];
                    for m in 0..c.n {
                        let p = (0..c.p).find(|&p| return c.s(p, m) != INF).unwrap();
                        upper[p].push(m);
                    }
                    let routes = solve(&c, &upper, None).0;
                    assert_complete(&c, &routes);
                    assert_eq!(c.total_cost(&routes), brute_force_optimum(&c));
                }
            }
        }
    }

    #[test]
    fn matches_subset_solver_on_larger_maps() {
        for (size, walls, players, mushrooms) in [(20, 120, 3, 12), (30, 300, 4, 14), (25, 200, 2, 15)]
        {
            for _ in 0..3 {
                let c = random_costs(size, walls, players, mushrooms);
                let upper = heuristic::solve(&c);
                let routes = solve(&c, &upper, None).0;
                assert_complete(&c, &routes);
                let optimal = exact::with_tables(&c, &upper);
                assert_eq!(c.total_cost(&routes), c.total_cost(&optimal));
            }
        }
    }

    #[test]
    fn mushrooms_unreachable_for_some_players() {
        let c = costs(
            "P - X - M -\nM - X - - -\n- M X M - P\n- - X - - -\nM - X - M -\n- - X - - M\n",
        );
        let routes = solve(&c, &heuristic::solve(&c), None).0;
        assert_complete(&c, &routes);
        assert_eq!(c.total_cost(&routes), brute_force_optimum(&c));
    }

    #[test]
    fn stops_after_max_nodes() {
        let c = random_costs(30, 300, 4, 14);
        let upper = heuristic::solve(&c);
        let (routes, complete) = solve(&c, &upper, Some(0));
        assert!(!complete);
        assert_eq!(routes, upper);
    }
}

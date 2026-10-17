//! Exact solver.
//!
//! The branch and bound in `search` runs first, it handles any number of mushrooms and is usually
//! fast. Its worst case is exponential, so for up to 20 mushrooms it gets a node budget, and when
//! that runs out, the solver over subsets of mushrooms below takes over (bounded time and memory):
//!
//! 1. `walk[S][i]`: cheapest walk that starts at mushroom `i` and visits every mushroom of `S`.
//!    It does not depend on the player, so it is computed once.
//! 2. `alone[p][S] = min_i s(p, i) + walk[S][i]`: cheapest route of player `p` picking exactly `S`.
//! 3. The mushrooms are split between the players by branch and bound, which starts from the
//!    heuristic solution and usually needs only a few thousand nodes. If it runs out of its
//!    node budget, dynamic programming finishes the job:
//!    `team[k][S] = min_{T ⊆ S} team[k-1][S \ T] + alone[k][T]` (3^n steps per player).
//!
//! Walking through a mushroom picks it on the way, but since distances are shortest paths,
//! visiting it explicitly at that moment costs the same, so the optimum is not affected.

use rayon::prelude::*;

use super::{Costs, search};
use crate::pathfinding::INF;

/// Largest mushroom count for the exact solver (the walk table then takes 80 MiB)
const MAX_MUSHROOMS: usize = 20;
/// Largest memory spent on the walk and route tables
const MAX_TABLE_BYTES: usize = 2 << 30;
/// Largest number of submask steps spent combining the players by dynamic programming
const MAX_COMBINE_STEPS: f64 = 2e10;
/// Smallest node budget of the branch and bound
const MIN_NODES: f64 = 1e5;
/// Smallest node budget of `search` before switching to the subset tables
const MIN_SEARCH_NODES: u64 = 20_000;

/// Iterate over the indices of the set bits
#[inline(always)]
fn bits(mut mask: usize) -> impl Iterator<Item = usize> {
    return std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let i = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        return Some(i);
    });
}

/// `walk[S * n + i]`: cheapest walk starting at mushroom `i` and visiting all of `S` (`i` in `S`)
fn walks(c: &Costs) -> Vec<u32> {
    let n = c.n;
    let full = 1usize << n;
    let mut walk = vec![INF; full * n];
    for i in 0..n {
        walk[(1 << i) * n + i] = 0;
    }

    // a set only depends on sets with one mushroom less, so each size is computed in parallel
    let mut by_size: Vec<Vec<usize>> = vec![vec![]; n + 1];
    for s in 1..full {
        by_size[s.count_ones() as usize].push(s);
    }
    for sets in &by_size[2..] {
        let rows: Vec<[u32; MAX_MUSHROOMS]> = sets
            .par_iter()
            .map(|&s| {
                let mut row = [INF; MAX_MUSHROOMS];
                for i in bits(s) {
                    let rest = s ^ (1 << i);
                    let rest_walk = &walk[rest * n..(rest + 1) * n];
                    row[i] = bits(rest)
                        .map(|j| return c.d(i, j).saturating_add(rest_walk[j]))
                        .min()
                        .unwrap();
                }
                return row;
            })
            .collect();
        for (&s, row) in sets.iter().zip(&rows) {
            walk[s * n..(s + 1) * n].copy_from_slice(&row[..n]);
        }
    }
    return walk;
}

/// Start mushroom of player `p` picking exactly `s`, with the cost of the route
#[inline(always)]
fn best_start(c: &Costs, walk: &[u32], p: usize, s: usize) -> (u32, usize) {
    return bits(s)
        .map(|i| return (c.s(p, i).saturating_add(walk[s * c.n + i]), i))
        .min()
        .unwrap();
}

/// Optimal visiting order of player `p` picking exactly `s`
fn route(c: &Costs, walk: &[u32], p: usize, s: usize) -> Vec<usize> {
    if s == 0 {
        return vec![];
    }
    let (_, mut current) = best_start(c, walk, p, s);
    let mut route = vec![current];
    let mut rest = s ^ (1 << current);
    while rest != 0 {
        let target = walk[(rest | (1 << current)) * c.n + current];
        current = bits(rest)
            .find(|&j| return c.d(current, j).saturating_add(walk[rest * c.n + j]) == target)
            .unwrap();
        route.push(current);
        rest ^= 1 << current;
    }
    return route;
}

/// Minimum of `team[S \ T] + alone[T]` over the subsets `T` of `S`, with the best `T`
#[inline(always)]
fn best_split(team: &[u32], alone: &[u32], s: usize) -> (u32, usize) {
    let mut best = (INF, 0);
    let mut t = s;
    loop {
        let cost = team[s ^ t].saturating_add(alone[t]);
        if cost < best.0 {
            best = (cost, t);
        }
        if t == 0 {
            return best;
        }
        t = (t - 1) & s;
    }
}


/// Players' mushroom sets of the cheapest split
fn combine(c: &Costs, alone: &[Vec<u32>]) -> Vec<usize> {
    let full = 1usize << c.n;
    // team[k][S]: cheapest way for players 0..=k to pick exactly S, the last player only needs S = all
    let mut team: Vec<Vec<u32>> = vec![alone[0].clone()];
    for k in 1..c.p.saturating_sub(1) {
        let next = (0..full)
            .into_par_iter()
            .map(|s| return best_split(&team[k - 1], &alone[k], s).0)
            .collect();
        team.push(next);
    }

    // walk back from the last player, splitting off the set of each player
    let mut sets = vec![0; c.p];
    let mut rest = full - 1;
    for k in (1..c.p).rev() {
        let (_, t) = best_split(&team[k - 1], &alone[k], rest);
        sets[k] = t;
        rest ^= t;
    }
    sets[0] = rest;
    return sets;
}

/// Depth-first branch and bound over the assignment of mushrooms to players
struct Search<'a> {
    /// distances
    c: &'a Costs,
    /// exact route cost of each player and set
    alone: &'a [Vec<u32>],
    /// mushrooms in the order they are assigned
    order: Vec<usize>,
    /// current set of each player
    sets: Vec<usize>,
    /// cost of the best split found, or the cost to beat
    best: u32,
    /// sets of the best split found
    best_sets: Option<Vec<usize>>,
    /// children buffer of each depth: (added cost, player)
    kids: Vec<Vec<(u32, usize)>>,
    /// nodes left before giving up
    nodes_left: u64,
}

impl Search<'_> {
    /// Assign `order[depth..]`, the sets cost `partial` so far. Returns false when out of nodes.
    fn dfs(&mut self, depth: usize, partial: u32) -> bool {
        if self.nodes_left == 0 {
            return false;
        }
        self.nodes_left -= 1;
        if depth == self.c.n {
            // bounds below guarantee partial < best
            self.best = partial;
            self.best_sets = Some(self.sets.clone());
            return true;
        }
        // route costs only grow with the set, so each mushroom still adds at least its
        // cheapest increment, and the largest of those is a lower bound for the rest
        let mut extra = 0;
        for &m in &self.order[depth..] {
            let inc = (0..self.c.p)
                .map(|p| {
                    let s = self.sets[p];
                    return self.alone[p][s | 1 << m].saturating_sub(self.alone[p][s]);
                })
                .min()
                .unwrap();
            extra = extra.max(inc);
        }
        if partial.saturating_add(extra) >= self.best {
            return true;
        }

        let m = self.order[depth];
        let mut kids = std::mem::take(&mut self.kids[depth]);
        kids.clear();
        for p in 0..self.c.p {
            let s = self.sets[p];
            let next = self.alone[p][s | 1 << m];
            if next != INF {
                kids.push((next - self.alone[p][s], p));
            }
        }
        // cheapest first finds good splits early
        kids.sort_unstable();
        let mut complete = true;
        for &(inc, p) in &kids {
            if partial + inc >= self.best {
                break;
            }
            self.sets[p] |= 1 << m;
            complete = self.dfs(depth + 1, partial + inc);
            self.sets[p] ^= 1 << m;
            if !complete {
                break;
            }
        }
        self.kids[depth] = kids;
        return complete;
    }
}

/// Players' mushroom sets of the cheapest split that costs at most `limit`,
/// `None` when there is none or when the search used up `nodes`
fn branch_and_bound(c: &Costs, alone: &[Vec<u32>], limit: u32, nodes: u64) -> Option<Vec<usize>> {
    // mushrooms far from every player first, their placement matters the most
    let mut order: Vec<usize> = (0..c.n).collect();
    order.sort_by_key(|&m| return std::cmp::Reverse((0..c.p).map(|p| return c.s(p, m)).min()));
    let mut search = Search {
        c,
        alone,
        order,
        sets: vec![0; c.p],
        best: limit.saturating_add(1),
        best_sets: None,
        kids: vec![vec![]; c.n],
        nodes_left: nodes,
    };
    let complete = search.dfs(0, 0);
    log::debug!("branch and bound used {} nodes", nodes - search.nodes_left);
    return if complete { search.best_sets } else { None };
}

/// True when the subset tables fit into memory
fn tables_fit(c: &Costs) -> bool {
    return c.n <= MAX_MUSHROOMS && (c.n + c.p).saturating_mul(4 << c.n) <= MAX_TABLE_BYTES;
}

/// Optimal routes for all players (`c.n >= 1`, `c.p >= 1`), given a feasible solution `upper`
pub fn solve(c: &Costs, upper: &[Vec<usize>]) -> Vec<Vec<usize>> {
    if !tables_fit(c) {
        log::info!("{} mushrooms, the exact search may take long", c.n);
        return search::solve(c, upper, None).0;
    }
    let max_nodes = MIN_SEARCH_NODES.max((1u64 << c.n) / 20);
    let (routes, complete) = search::solve(c, upper, Some(max_nodes));
    if complete {
        return routes;
    }
    log::info!("exact search used {max_nodes} nodes, switching to the subset tables");
    return with_tables(c, &routes);
}

/// Optimal routes by the subset tables (`tables_fit(c)` must hold)
pub fn with_tables(c: &Costs, upper: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let full = 1usize << c.n;
    let walk = walks(c);
    let alone: Vec<Vec<u32>> = (0..c.p)
        .into_par_iter()
        .map(|p| {
            return (0..full)
                .map(|s| return if s == 0 { 0 } else { best_start(c, &walk, p, s).0 })
                .collect();
        })
        .collect();

    // spend at most about as much on the branch and bound as the dynamic programming would take
    let combine_steps = match c.p {
        1 | 2 => full as f64,
        p => 3f64.powi(c.n as i32) * (p - 2) as f64,
    };
    let dp_feasible = combine_steps <= MAX_COMBINE_STEPS;
    let node_cost = (c.n * c.p) as f64;
    let nodes = (combine_steps.min(MAX_COMBINE_STEPS) / node_cost).max(MIN_NODES) as u64;

    let sets = match branch_and_bound(c, &alone, c.total_cost(upper), nodes) {
        Some(sets) => sets,
        None if dp_feasible => {
            log::debug!("branch and bound ran out of nodes, combining by dynamic programming");
            combine(c, &alone)
        }
        None => {
            log::info!("branch and bound ran out of nodes, using the exact search");
            return search::solve(c, upper, None).0;
        }
    };
    let routes: Vec<Vec<usize>> = sets
        .iter()
        .enumerate()
        .map(|(p, &s)| return route(c, &walk, p, s))
        .collect();
    return routes;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{heuristic, tests::*};

    /// Exact routes by the subset tables, starting from the heuristic
    fn exact(c: &Costs) -> Vec<Vec<usize>> {
        return with_tables(c, &heuristic::solve(c));
    }

    /// Route cost tables as `solve` builds them
    fn alone(c: &Costs) -> Vec<Vec<u32>> {
        let walk = walks(c);
        return (0..c.p)
            .map(|p| {
                return (0..1usize << c.n)
                    .map(|s| return if s == 0 { 0 } else { best_start(c, &walk, p, s).0 })
                    .collect();
            })
            .collect();
    }

    /// Total cost of a split given as sets
    fn sets_cost(alone: &[Vec<u32>], sets: &[usize]) -> u32 {
        return sets.iter().enumerate().map(|(p, &s)| return alone[p][s]).sum();
    }

    #[test]
    fn single_player_order_is_optimal() {
        for _ in 0..20 {
            let c = random_costs(8, 20, 1, 7);
            let routes = exact(&c);
            assert_complete(&c, &routes);
            let all: Vec<usize> = (0..c.n).collect();
            assert_eq!(c.total_cost(&routes), brute_force_route(&c, 0, &all));
        }
    }

    #[test]
    fn matches_brute_force_on_random_maps() {
        for players in 1..=4 {
            for mushrooms in 1..=6 {
                for _ in 0..5 {
                    let c = random_costs(7, 15, players, mushrooms);
                    let best = brute_force_optimum(&c);
                    let routes = exact(&c);
                    assert_complete(&c, &routes);
                    assert_eq!(c.total_cost(&routes), best);

                    // both ways of splitting the mushrooms are optimal on their own
                    let alone = alone(&c);
                    assert_eq!(sets_cost(&alone, &combine(&c, &alone)), best);
                    let sets = branch_and_bound(&c, &alone, INF - 1, u64::MAX).unwrap();
                    assert_eq!(sets_cost(&alone, &sets), best);
                }
            }
        }
    }

    #[test]
    fn branch_and_bound_limits() {
        let c = random_costs(10, 30, 3, 8);
        let alone = alone(&c);
        let best = sets_cost(&alone, &combine(&c, &alone));
        // nothing cheaper than the optimum exists
        assert!(branch_and_bound(&c, &alone, best - 1, u64::MAX).is_none());
        assert!(branch_and_bound(&c, &alone, best, u64::MAX).is_some());
        // out of nodes
        assert!(branch_and_bound(&c, &alone, INF - 1, 3).is_none());
    }

    #[test]
    fn mushrooms_unreachable_for_some_players() {
        // walls split the map into a left and a right part, each with one player
        let c = costs(
            "P - X - M -\nM - X - - -\n- M X M - P\n- - X - - -\nM - X - M -\n- - X - - M\n",
        );
        let routes = exact(&c);
        assert_complete(&c, &routes);
        assert_eq!(c.total_cost(&routes), brute_force_optimum(&c));
    }

    #[test]
    fn more_mushrooms_than_the_tables_allow() {
        let c = random_costs(12, 30, 2, 22);
        let upper = heuristic::solve(&c);
        let routes = solve(&c, &upper);
        assert_complete(&c, &routes);
        assert!(c.total_cost(&routes) <= c.total_cost(&upper));
    }
}

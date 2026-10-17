//! Heuristic solver for maps of any size: construction heuristics followed by local search.
//!
//! Routes are open paths that start at the player. All moves keep the solution feasible
//! (no route uses an unreachable leg) and only apply strict improvements, so the search ends.

use rayon::prelude::*;

use super::Costs;
use crate::pathfinding::INF;

/// Routes up to this length are ordered optimally by dynamic programming
const EXACT_ROUTE_LEN: usize = 10;
/// Longest segment moved by Or-opt
const OR_OPT_LEN: usize = 3;

/// Cost of an unreachable leg in the signed move arithmetic, larger than any real gain
const BIG: i64 = 1 << 50;

/// Heuristic routes for all players
pub fn solve(c: &Costs) -> Vec<Vec<usize>> {
    let starts = [greedy(c), lookahead(c), cheapest_insertion(c)];
    let improved: Vec<Vec<Vec<usize>>> = starts
        .into_par_iter()
        .map(|routes| return improve(c, routes))
        .collect();
    // first of the cheapest, so the result does not depend on thread timing
    return improved
        .into_iter()
        .min_by_key(|routes| return c.total_cost(routes))
        .unwrap();
}

/// signed cost of a leg
#[inline(always)]
fn leg(cost: u32) -> i64 {
    return if cost == INF { BIG } else { cost as i64 };
}

/// leg from `prev` (or the player when `None`) to mushroom `m`
#[inline(always)]
fn edge(c: &Costs, p: usize, prev: Option<usize>, m: usize) -> i64 {
    return leg(match prev {
        Some(q) => c.d(q, m),
        None => c.s(p, m),
    });
}

/// mushroom before position `i` of the route, `None` means the player
#[inline(always)]
fn before(route: &[usize], i: usize) -> Option<usize> {
    return i.checked_sub(1).map(|j| return route[j]);
}

/// Repeatedly give the globally closest free mushroom to the player that is closest to it
fn greedy(c: &Costs) -> Vec<Vec<usize>> {
    let mut routes: Vec<Vec<usize>> = vec![vec![]; c.p];
    let mut free = vec![true; c.n];
    loop {
        let mut best: Option<(u32, usize, usize)> = None;
        for (p, route) in routes.iter().enumerate() {
            for m in (0..c.n).filter(|&m| return free[m]) {
                let dist = match route.last() {
                    Some(&last) => c.d(last, m),
                    None => c.s(p, m),
                };
                if dist != INF && best.is_none_or(|b| return dist < b.0) {
                    best = Some((dist, p, m));
                }
            }
        }
        let Some((_, p, m)) = best else {
            return routes;
        };
        routes[p].push(m);
        free[m] = false;
    }
}

/// Simulate all players walking at the same speed, each towards its nearest free mushroom
fn lookahead(c: &Costs) -> Vec<Vec<usize>> {
    let mut routes: Vec<Vec<usize>> = vec![vec![]; c.p];
    let mut free = vec![true; c.n];
    // steps walked by each player towards its next mushroom
    let mut walked = vec![0u32; c.p];
    let mut left = c.n;
    while left > 0 {
        // nearest free mushroom of each player and the distance to it
        let nearest: Vec<Option<(u32, usize)>> = (0..c.p)
            .map(|p| {
                return (0..c.n)
                    .filter(|&m| return free[m])
                    .map(|m| match routes[p].last() {
                        Some(&last) => return (c.d(last, m), m),
                        None => return (c.s(p, m), m),
                    })
                    .filter(|&(d, _)| return d != INF)
                    .min();
            })
            .collect();
        // time until the first player reaches its mushroom
        let Some(tick) = (0..c.p)
            .filter_map(|p| return nearest[p].map(|(d, _)| return d.saturating_sub(walked[p])))
            .min()
        else {
            break;
        };
        for p in 0..c.p {
            let Some((d, m)) = nearest[p] else {
                continue;
            };
            walked[p] += tick;
            // another player may have taken the same mushroom in this tick
            if walked[p] >= d && free[m] {
                walked[p] -= d;
                routes[p].push(m);
                free[m] = false;
                left -= 1;
            }
        }
    }
    return routes;
}

/// cheapest position to insert mushroom `m` into the route of player `p`: (added cost, position)
fn best_insertion(c: &Costs, p: usize, route: &[usize], m: usize) -> (i64, usize) {
    return (0..=route.len())
        .map(|k| {
            let prev = before(route, k);
            let added = match route.get(k) {
                Some(&next) => {
                    edge(c, p, prev, m) + leg(c.d(m, next)) - edge(c, p, prev, next)
                }
                None => edge(c, p, prev, m),
            };
            return (added, k);
        })
        .min()
        .unwrap();
}

/// Repeatedly insert the mushroom whose cheapest insertion into any route adds the least
fn cheapest_insertion(c: &Costs) -> Vec<Vec<usize>> {
    let mut routes: Vec<Vec<usize>> = vec![vec![]; c.p];
    // insertion[m * p + r]: cheapest insertion of free mushroom m into route r
    let mut insertion: Vec<(i64, usize)> = (0..c.n)
        .flat_map(|m| return (0..c.p).map(move |p| return (edge(c, p, None, m), 0)))
        .collect();
    let mut free: Vec<usize> = (0..c.n).collect();
    while !free.is_empty() {
        let ins = &insertion;
        let (_, fi, r) = free
            .iter()
            .enumerate()
            .flat_map(|(fi, &m)| {
                return (0..c.p).map(move |r| return (ins[m * c.p + r].0, fi, r));
            })
            .min()
            .unwrap();
        let m = free.swap_remove(fi);
        routes[r].insert(insertion[m * c.p + r].1, m);
        for &f in &free {
            insertion[f * c.p + r] = best_insertion(c, r, &routes[r], f);
        }
    }
    return routes;
}

/// Local search until no move improves the solution
fn improve(c: &Costs, mut routes: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for (p, route) in routes.iter_mut().enumerate() {
        optimize_route(c, p, route);
    }
    loop {
        let mut improved = false;
        improved |= relocate(c, &mut routes);
        improved |= swap(c, &mut routes);
        improved |= exchange_tails(c, &mut routes);
        if !improved {
            return routes;
        }
    }
}

/// Improve the order of one route, returns true if it changed
fn optimize_route(c: &Costs, p: usize, route: &mut Vec<usize>) -> bool {
    if route.len() <= EXACT_ROUTE_LEN {
        let before = c.route_cost(p, route);
        let best = exact_route(c, p, route);
        if c.route_cost(p, &best) < before {
            *route = best;
            return true;
        }
        return false;
    }
    let mut changed = false;
    while two_opt(c, p, route) | or_opt(c, p, route) {
        changed = true;
    }
    return changed;
}

/// Optimal order of a short route, by dynamic programming over its subsets
fn exact_route(c: &Costs, p: usize, route: &[usize]) -> Vec<usize> {
    let k = route.len();
    if k < 2 {
        return route.to_vec();
    }
    let full = 1usize << k;
    // cost[S * k + j]: cheapest path from the player over S ending at route[j]
    let mut cost = vec![INF; full * k];
    for j in 0..k {
        cost[(1 << j) * k + j] = c.s(p, route[j]);
    }
    for s in 1..full {
        for j in (0..k).filter(|&j| return s >> j & 1 == 1) {
            let here = cost[s * k + j];
            if here == INF {
                continue;
            }
            for next in (0..k).filter(|&next| return s >> next & 1 == 0) {
                let t = s | 1 << next;
                let new = here.saturating_add(c.d(route[j], route[next]));
                if new < cost[t * k + next] {
                    cost[t * k + next] = new;
                }
            }
        }
    }
    // walk back from the cheapest end
    let mut s = full - 1;
    let mut j = (0..k).min_by_key(|&j| return cost[s * k + j]).unwrap();
    let mut order = vec![route[j]];
    while s != 1 << j {
        let prev_s = s ^ (1 << j);
        let target = cost[s * k + j];
        j = (0..k)
            .filter(|&i| return prev_s >> i & 1 == 1)
            .find(|&i| return cost[prev_s * k + i].saturating_add(c.d(route[i], route[j])) == target)
            .unwrap();
        s = prev_s;
        order.push(route[j]);
    }
    order.reverse();
    return order;
}

/// Reverse a segment of the route when it shortens it, returns true if anything changed
fn two_opt(c: &Costs, p: usize, route: &mut [usize]) -> bool {
    let mut changed = false;
    for i in 0..route.len() {
        for j in i + 1..route.len() {
            let prev = before(route, i);
            let mut delta = edge(c, p, prev, route[j]) - edge(c, p, prev, route[i]);
            if let Some(&next) = route.get(j + 1) {
                delta += leg(c.d(route[i], next)) - leg(c.d(route[j], next));
            }
            if delta < 0 {
                route[i..=j].reverse();
                changed = true;
            }
        }
    }
    return changed;
}

/// Move a short segment (optionally reversed) elsewhere in the route, returns true if anything changed
fn or_opt(c: &Costs, p: usize, route: &mut Vec<usize>) -> bool {
    let mut changed = false;
    for len in 1..=OR_OPT_LEN.min(route.len().saturating_sub(1)) {
        let mut i = 0;
        while i + len <= route.len() {
            let (first, last) = (route[i], route[i + len - 1]);
            let prev = before(route, i);
            let removed = match route.get(i + len) {
                Some(&next) => {
                    edge(c, p, prev, first) + leg(c.d(last, next)) - edge(c, p, prev, next)
                }
                None => edge(c, p, prev, first),
            };
            let rest: Vec<usize> = route[..i].iter().chain(&route[i + len..]).copied().collect();
            // best insertion position in the rest and orientation of the segment
            let mut best = (0i64, 0usize, false);
            for k in 0..=rest.len() {
                let u = before(&rest, k);
                for reversed in [false, true] {
                    let (a, b) = if reversed { (last, first) } else { (first, last) };
                    let added = match rest.get(k) {
                        Some(&w) => edge(c, p, u, a) + leg(c.d(b, w)) - edge(c, p, u, w),
                        None => edge(c, p, u, a),
                    };
                    if added - removed < best.0 {
                        best = (added - removed, k, reversed);
                    }
                }
            }
            if best.0 < 0 {
                let mut segment = route[i..i + len].to_vec();
                if best.2 {
                    segment.reverse();
                }
                let mut new_route = rest;
                new_route.splice(best.1..best.1, segment);
                *route = new_route;
                changed = true;
            }
            i += 1;
        }
    }
    return changed;
}

/// cost saved by removing position `i` from the route of player `p`
fn removal_gain(c: &Costs, p: usize, route: &[usize], i: usize) -> i64 {
    let prev = before(route, i);
    let m = route[i];
    return match route.get(i + 1) {
        Some(&next) => edge(c, p, prev, m) + leg(c.d(m, next)) - edge(c, p, prev, next),
        None => edge(c, p, prev, m),
    };
}

/// Move single mushrooms to the cheapest position in another route
fn relocate(c: &Costs, routes: &mut [Vec<usize>]) -> bool {
    let mut changed = false;
    for a in 0..c.p {
        let mut i = 0;
        while i < routes[a].len() {
            let m = routes[a][i];
            let gain = removal_gain(c, a, &routes[a], i);
            let best = (0..c.p)
                .filter(|&b| return b != a)
                .map(|b| {
                    let (added, k) = best_insertion(c, b, &routes[b], m);
                    return (added, b, k);
                })
                .min();
            match best {
                Some((added, b, k)) if added < gain => {
                    routes[a].remove(i);
                    routes[b].insert(k, m);
                    optimize_route(c, a, &mut routes[a]);
                    optimize_route(c, b, &mut routes[b]);
                    changed = true;
                    i = 0;
                }
                _ => i += 1,
            }
        }
    }
    return changed;
}

/// cost change of putting mushroom `m` at position `i` of the route instead of the current one
fn replace_delta(c: &Costs, p: usize, route: &[usize], i: usize, m: usize) -> i64 {
    let prev = before(route, i);
    let old = route[i];
    let mut delta = edge(c, p, prev, m) - edge(c, p, prev, old);
    if let Some(&next) = route.get(i + 1) {
        delta += leg(c.d(m, next)) - leg(c.d(old, next));
    }
    return delta;
}

/// Exchange two mushrooms between different routes
fn swap(c: &Costs, routes: &mut [Vec<usize>]) -> bool {
    let mut changed = false;
    for a in 0..c.p {
        for b in a + 1..c.p {
            let mut pair_changed = false;
            for i in 0..routes[a].len() {
                for j in 0..routes[b].len() {
                    let (x, y) = (routes[a][i], routes[b][j]);
                    let delta = replace_delta(c, a, &routes[a], i, y)
                        + replace_delta(c, b, &routes[b], j, x);
                    if delta < 0 {
                        routes[a][i] = y;
                        routes[b][j] = x;
                        pair_changed = true;
                    }
                }
            }
            if pair_changed {
                changed = true;
                optimize_route(c, a, &mut routes[a]);
                optimize_route(c, b, &mut routes[b]);
            }
        }
    }
    return changed;
}

/// Exchange the ends of two routes
fn exchange_tails(c: &Costs, routes: &mut [Vec<usize>]) -> bool {
    let mut changed = false;
    for a in 0..c.p {
        for b in a + 1..c.p {
            let mut best = (0i64, 0, 0);
            for i in 0..=routes[a].len() {
                for j in 0..=routes[b].len() {
                    let (ra, rb) = (&routes[a], &routes[b]);
                    let (pa, pb) = (before(ra, i), before(rb, j));
                    let mut delta = 0;
                    if let Some(&x) = ra.get(i) {
                        delta += edge(c, b, pb, x) - edge(c, a, pa, x);
                    }
                    if let Some(&y) = rb.get(j) {
                        delta += edge(c, a, pa, y) - edge(c, b, pb, y);
                    }
                    if delta < best.0 {
                        best = (delta, i, j);
                    }
                }
            }
            if best.0 < 0 {
                let (_, i, j) = best;
                let tail_a = routes[a].split_off(i);
                let tail_b = routes[b].split_off(j);
                routes[a].extend(tail_b);
                routes[b].extend(tail_a);
                optimize_route(c, a, &mut routes[a]);
                optimize_route(c, b, &mut routes[b]);
                changed = true;
            }
        }
    }
    return changed;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{exact, tests::*};

    #[test]
    fn constructions_assign_every_mushroom_once() {
        for _ in 0..20 {
            let c = random_costs(10, 30, 3, 8);
            assert_complete(&c, &greedy(&c));
            assert_complete(&c, &lookahead(&c));
            assert_complete(&c, &cheapest_insertion(&c));
        }
    }

    #[test]
    fn exact_route_is_optimal() {
        for _ in 0..20 {
            let c = random_costs(8, 20, 1, 7);
            let route: Vec<usize> = (0..c.n).collect();
            let best = exact_route(&c, 0, &route);
            assert_eq!(c.route_cost(0, &best), brute_force_route(&c, 0, &route));
        }
    }

    #[test]
    fn long_route_moves_never_make_it_worse() {
        for _ in 0..10 {
            let c = random_costs(20, 100, 1, 30);
            let mut route = greedy(&c).concat();
            let start = c.route_cost(0, &route);
            optimize_route(&c, 0, &mut route);
            assert!(c.route_cost(0, &route) <= start);
            assert_complete(&c, &[route]);
        }
    }

    #[test]
    fn never_better_than_exact() {
        for _ in 0..20 {
            let c = random_costs(12, 40, 3, 10);
            let routes = solve(&c);
            assert_complete(&c, &routes);
            assert!(c.total_cost(&routes) >= c.total_cost(&exact::solve(&c, &routes)));
        }
    }

    #[test]
    fn unreachable_mushrooms_stay_with_players_that_reach_them() {
        let c = costs(
            "P - X - M -\nM - X - - -\n- M X M - P\n- - X - - -\nM - X - M -\n- - X - - M\n",
        );
        let routes = solve(&c);
        assert_complete(&c, &routes);
        assert!(c.total_cost(&routes) < INF);
    }

    #[test]
    fn large_instance() {
        let c = random_costs(60, 1000, 10, 300);
        let routes = solve(&c);
        assert_complete(&c, &routes);
        assert!(c.total_cost(&routes) < INF);
    }
}

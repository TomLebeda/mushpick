use std::fmt::Write;
use std::path::PathBuf;

use log::*;

use crate::{
    parse_field,
    pathfinding::{Distances, INF},
};

/// exact dynamic programming over mushroom subsets
mod exact;
/// construction heuristics and local search
mod heuristic;
/// exact branch and bound for any number of mushrooms
mod search;

/// Distances between players and mushrooms, `INF` when unreachable
#[derive(Clone, Debug)]
pub struct Costs {
    /// number of mushrooms
    pub n: usize,
    /// number of players
    pub p: usize,
    /// mushroom-to-mushroom distances, `m2m[i * n + j]`
    pub m2m: Vec<u32>,
    /// player-to-mushroom distances, `p2m[p * n + i]`
    pub p2m: Vec<u32>,
}

impl Costs {
    /// distance between mushrooms `i` and `j`
    #[inline(always)]
    pub fn d(&self, i: usize, j: usize) -> u32 {
        return self.m2m[i * self.n + j];
    }

    /// distance from player `p` to mushroom `i`
    #[inline(always)]
    pub fn s(&self, p: usize, i: usize) -> u32 {
        return self.p2m[p * self.n + i];
    }

    /// steps of player `p` visiting `route` in order
    pub fn route_cost(&self, p: usize, route: &[usize]) -> u32 {
        let Some(&first) = route.first() else {
            return 0;
        };
        return route
            .windows(2)
            .fold(self.s(p, first), |acc, w| return acc.saturating_add(self.d(w[0], w[1])));
    }

    /// total steps of all players
    pub fn total_cost(&self, routes: &[Vec<usize>]) -> u32 {
        return routes
            .iter()
            .enumerate()
            .fold(0u32, |acc, (p, r)| return acc.saturating_add(self.route_cost(p, r)));
    }
}

/// Find the routes for every player: optimal unless `fast` is set, heuristic otherwise
pub fn find_routes(costs: &Costs, fast: bool) -> Vec<Vec<usize>> {
    if costs.n == 0 {
        return vec![vec![]; costs.p];
    }
    let routes = heuristic::solve(costs);
    trace!("heuristic solution: {} steps", costs.total_cost(&routes));
    if fast {
        return routes;
    }
    return exact::solve(costs, &routes);
}

/// Solve the map on provided file path
pub fn solve(map_file: PathBuf, fast: bool) {
    let field = parse_field(&map_file);
    trace!("field parsed");
    let dist = Distances::build(&field);
    let costs = &dist.costs;
    trace!("distances computed");

    if let Some(m) = (0..costs.n).find(|&m| return (0..costs.p).all(|p| return costs.s(p, m) == INF))
    {
        error!(
            "mushroom {m} at {} can't be reached by any player",
            field.mushrooms[m]
        );
        std::process::exit(exitcode::DATAERR);
    }

    let routes = find_routes(costs, fast);
    info!("total steps: {}", costs.total_cost(&routes));

    let mut out = String::new();
    for (pi, path) in dist.paths(&routes).iter().enumerate() {
        write!(out, "P{pi}:").unwrap();
        out.extend(path.iter().map(|d| return d.as_char()));
        out.push('\n');
    }
    print!("{out}");
}

#[cfg(test)]
pub(crate) mod tests {
    use itertools::Itertools;

    use super::*;
    use crate::generator::generate_field;
    use crate::utils::tests::field;

    /// Costs of a map given as text
    pub(crate) fn costs(map: &str) -> Costs {
        return Distances::build(&field(map)).costs;
    }

    /// Costs of a random map
    pub(crate) fn random_costs(size: usize, walls: usize, players: usize, mushrooms: usize) -> Costs {
        return Distances::build(&generate_field(size, walls, players, mushrooms).unwrap()).costs;
    }

    /// Cost of one player's best order, by trying all permutations
    pub(crate) fn brute_force_route(c: &Costs, p: usize, mushrooms: &[usize]) -> u32 {
        return mushrooms
            .iter()
            .copied()
            .permutations(mushrooms.len())
            .map(|r| return c.route_cost(p, &r))
            .min()
            .unwrap();
    }

    /// Optimal total cost over all assignments and orders
    pub(crate) fn brute_force_optimum(c: &Costs) -> u32 {
        let mut best = INF;
        for assignment in (0..c.n).map(|_| return 0..c.p).multi_cartesian_product() {
            let mut split = vec![vec![]; c.p];
            for (m, &p) in assignment.iter().enumerate() {
                split[p].push(m);
            }
            let cost = (0..c.p).fold(0u32, |acc, p| {
                return acc.saturating_add(brute_force_route(c, p, &split[p]));
            });
            best = best.min(cost);
        }
        return best;
    }

    /// Every mushroom appears in exactly one route
    #[track_caller]
    pub(crate) fn assert_complete(c: &Costs, routes: &[Vec<usize>]) {
        assert_eq!(routes.len(), c.p);
        let all = routes.concat().into_iter().sorted().collect_vec();
        assert_eq!(all, (0..c.n).collect_vec());
    }

    #[test]
    fn route_and_total_cost_hand_computed() {
        let c = costs("P M - M\n- - - -\n- - - -\nM - - P\n");
        // P0 -> M0 (1) -> M1 (2), P1 -> M2 (3)
        assert_eq!(c.total_cost(&[vec![0, 1], vec![2]]), 1 + 2 + 3);
        // empty players cost nothing
        assert_eq!(c.total_cost(&[vec![], vec![2]]), 3);
        assert_eq!(c.total_cost(&[vec![], vec![]]), 0);
    }

    #[test]
    fn unreachable_route_costs_inf() {
        let c = costs("P X M\nM X -\n- X P\n");
        assert_eq!(c.route_cost(0, &[0]), INF);
        assert_eq!(c.route_cost(0, &[1, 0]), INF);
        assert_eq!(c.route_cost(1, &[0]), 2);
    }

    #[test]
    fn no_mushrooms_gives_empty_routes() {
        let c = costs("P - -\n- - -\n- - P\n");
        assert_eq!(find_routes(&c, false), vec![Vec::<usize>::new(); 2]);
        assert_eq!(find_routes(&c, true), vec![Vec::<usize>::new(); 2]);
    }
}

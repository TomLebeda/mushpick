use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use itertools::Itertools;
use log::*;

use crate::{
    Coord, parse_field,
    pathfinding::{bfs_m2m, bfs_p2m, is_field_accessible},
};

/// global static bound for number of players (so I can use arrays instead of vectors for speed)
static MAX_PLAYERS: usize = 128;

#[derive(Serialize, Deserialize)]
/// Solution for single player
pub struct Result {
    /// indices of mushrooms picked by given player
    pub mushrooms: Vec<usize>,
    /// list of coordinates where the player moved
    pub steps: Vec<Coord>,
    /// total cost of the path
    pub cost: usize,
}

/// Solve the map on provided file path
pub fn solve(map_file: PathBuf, fast: bool) {
    let field = parse_field(&map_file);
    trace!("field parsed");
    if !fast {
        match is_field_accessible(&field.cells, field.size) {
            true => info!("all cells are accessible"),
            false => {
                error!("flood-fill couldn't reach all cells");
                std::process::exit(exitcode::DATAERR);
            }
        }
    }
    let n_mush = field.mushrooms.len();
    let n_plr = field.players.len();

    trace!("starting search");
    let m2m_paths = bfs_m2m(&field);
    let m2m_dist = m2m_paths
        .iter()
        .enumerate()
        .map(|(i, row)| {
            return row
                .iter()
                .enumerate()
                .map(|(j, path)| {
                    if i == j {
                        return u32::MAX;
                    } else {
                        return path.len() as u32;
                    }
                })
                .collect_vec();
        })
        .collect_vec();
    trace!(" - found m2m dist");

    let p2m_paths = bfs_p2m(&field);
    let p2m_dist = p2m_paths
        .iter()
        .map(|row| {
            return row
                .iter()
                .map(|path| return path.len() as u32)
                .collect_vec();
        })
        .collect_vec();
    trace!(" - found p2m dist");

    // how the players split the mushrooms
    let mut greedy_split: Vec<Vec<usize>> = vec![Vec::with_capacity(n_mush); n_plr];
    // movement-potential for each player
    let mut free_steps: Vec<u32> = vec![0; n_plr];
    // indicator if mushroom at index is assigned
    let mut mushrooms_free = vec![true; n_mush];
    // vector of (player_index, nearest_mush_idx, nearest_mush_dist, ticks_required)
    let mut nearest: [(_, _, _, _); MAX_PLAYERS] = [(0, 0, u32::MAX, 0); MAX_PLAYERS];

    // while there are still some mushrooms left to be assigned...
    while mushrooms_free.iter().any(|&free| return free) {
        // how many ticks are required to pick up next mushroom (by anyone)
        let mut min_ticks = u32::MAX;
        // find how many ticks are required for any player to pick any (nearest) mushroom
        for pi in 0..n_plr {
            let dist_row = match greedy_split[pi].last() {
                Some(pos) => &m2m_dist[*pos], // player is already standing on a mushroom
                None => &p2m_dist[pi],        // player hasn't moved yet
            };
            let mut nearest_mush_idx = 0;
            let mut nearest_mush_dist = u32::MAX;
            // find nearest free mushroom for current player
            for mi in 0..mushrooms_free.len() {
                // branch-less checking if mushroom is free: mask = 1 if free, 0 if not
                let mask = mushrooms_free[mi] as u32;
                let dist = dist_row[mi];
                let use_dist = dist * mask + (1 - mask) * u32::MAX;
                if use_dist < nearest_mush_dist {
                    nearest_mush_idx = mi;
                    nearest_mush_dist = use_dist;
                }
            }
            let ticks_required = nearest_mush_dist - free_steps[pi];
            nearest[pi] = (pi, nearest_mush_idx, nearest_mush_dist, ticks_required);
            min_ticks = min_ticks.min(ticks_required);
        }

        // increment free ticks for all players
        for val in &mut free_steps[0..n_plr] {
            *val += min_ticks;
        }

        // check all players if they have enough free ticks to pick up some mushrooms
        for &(pi, nearest_mush_idx, nearest_mush_dist, _ticks_required) in &nearest[0..n_plr] {
            if free_steps[pi] >= nearest_mush_dist {
                free_steps[pi] -= nearest_mush_dist; // subtract the used steps
                greedy_split[pi].push(nearest_mush_idx); // push the mushroom into the split
                mushrooms_free[nearest_mush_idx] = false; // mark the mushroom as assigned
            }
        }
    }

    let greedy_split_cost = get_split_cost(&greedy_split, &m2m_dist, &p2m_dist);

    trace!("GREEDY split:");
    greedy_split
        .iter()
        .enumerate()
        .for_each(|(p_idx, mush_seq)| {
            trace!("   - P{}: {:?}", p_idx, mush_seq);
        });
    trace!(" - cost: {greedy_split_cost}");

    let optimized_greedy_split =
        optimize_split(greedy_split_cost, &greedy_split, &m2m_dist, &p2m_dist);
    let optimized_greedy_split_cost = get_split_cost(&optimized_greedy_split, &m2m_dist, &p2m_dist);

    trace!("optimized GREEDY split:");
    optimized_greedy_split
        .iter()
        .enumerate()
        .for_each(|(p_idx, mush_seq)| {
            trace!("   - P{}: {:?}", p_idx, mush_seq);
        });
    trace!(" - cost: {optimized_greedy_split_cost}");

    let mut best_split_cost = optimized_greedy_split_cost;
    let mut best_split = optimized_greedy_split;
    trace!(" - best split: {best_split:?}");

    let mut t = 0; // how many full splits were explored
    let mut o = 0; // how many splits were optimized
    let mut l = 0; // how many lower boundaries were computed
    if !fast {
        trace!("searching for best split...");
        // keep the lower bound outside of loop and mutable so we can early-stop while still building it
        let mut stack = vec![(0, vec![Vec::new(); n_plr])];
        while let Some((mush_idx, mut current_split)) = stack.pop() {
            if mush_idx == n_mush {
                // we have a full split => check it
                t += 1;
                let lower_bound = get_split_cost_lower_bound_mst(
                    &current_split,
                    &m2m_dist,
                    &p2m_dist,
                    n_plr,
                    best_split_cost,
                );
                l += 1;
                // seems promising, we need to fully optimize
                if lower_bound >= best_split_cost {
                    continue;
                }
                o += 1;
                let optimized_split =
                    optimize_split(best_split_cost, &current_split, &m2m_dist, &p2m_dist);
                let optimized_split_cost = get_split_cost(&optimized_split, &m2m_dist, &p2m_dist);
                if optimized_split_cost < best_split_cost {
                    best_split_cost = optimized_split_cost;
                    best_split = optimized_split;
                }
            } else {
                for i in 0..n_plr {
                    current_split[i].push(mush_idx);
                    let lower_bound = get_split_cost_lower_bound_mst(
                        &current_split,
                        &m2m_dist,
                        &p2m_dist,
                        n_plr,
                        best_split_cost,
                    );
                    l += 1;
                    if lower_bound < best_split_cost {
                        stack.push((mush_idx + 1, current_split.clone()))
                    }
                    current_split[i].pop();
                }
            }
        }
    }

    let a = (n_plr as u128).pow(n_mush as u32);
    trace!(
        " === all possible {a}, full splits explored: {t}, optimized {o}, lower-bounds computed: {l}",
    );

    trace!("BEST split:");
    best_split.iter().enumerate().for_each(|(p_idx, mush_seq)| {
        trace!("   - P{}: {:?}", p_idx, mush_seq);
    });
    trace!(" - cost: {best_split_cost}");

    trace!("pathfinding the best split");
    best_split
        .iter()
        .enumerate()
        .for_each(|(player_idx, mush_seq)| {
            print!("P{player_idx}:");
            if mush_seq.is_empty() {
                println!();
            } else {
                // print out the path from player to first mushroom
                p2m_paths[player_idx][mush_seq[0]]
                    .iter()
                    .for_each(|d| print!("{}", d.as_char()));

                for (mush_idx_1, mush_idx_2) in mush_seq.iter().tuple_windows() {
                    // print the path from mush to mush
                    m2m_paths[*mush_idx_1][*mush_idx_2]
                        .iter()
                        .for_each(|d| print!("{}", d.as_char()));
                }
                println!() // separate lines for each player
            }
        });

    trace!("DONE");
}

/// Compute the lower bound cost for given split, using early-stopping MST.
///
/// If the cost becomes larger or equal to best_split_cost, the algorithm
/// will short-circuit and return the best_split_cost immediately.
#[inline]
fn get_split_cost_lower_bound_mst(
    split: &[Vec<usize>],
    m2m_dist: &[Vec<u32>],
    p2m_dist: &[Vec<u32>],
    n_plr: usize,
    best_split_cost: u32,
) -> u32 {
    let mut lower_bound = 0;

    // for each player, find the minimum spanning tree of their assigned mush sequence
    for pi in 0..n_plr {
        if lower_bound >= best_split_cost {
            return best_split_cost;
        }
        let mush_count = split[pi].len();
        if mush_count == 0 {
            continue;
        }
        let n = mush_count + 1; // size of the matrix
        // build a extended distance matrix including the players starting position and assigned mushrooms
        let mut dist_mat = vec![vec![u32::MAX; n]; n];
        // build the player-to-mush row and column
        for col in 0..mush_count {
            let mi = split[pi][col];
            dist_mat[0][col + 1] = p2m_dist[pi][mi];
            dist_mat[col + 1][0] = p2m_dist[pi][mi];
        }
        // build the mush-to-mush rows and columns
        for row in 1..n {
            for col in row..n {
                let from_mi = split[pi][row - 1];
                let to_mi = split[pi][col - 1];
                dist_mat[row][col] = m2m_dist[from_mi][to_mi];
                dist_mat[col][row] = m2m_dist[from_mi][to_mi];
            }
        }

        // how much we can increase the cost before it's > best
        let threshold = best_split_cost - lower_bound;
        let mst_cost = get_mst_cost(&dist_mat, threshold);

        lower_bound += mst_cost;
    }
    return lower_bound;
}

/// compute the cost of Minimum spanning tree using Prim's algorithm
///
/// If the cost becomes larger or equal to threshold, the algorithm
/// will short-circuit and return the threshold
#[inline]
fn get_mst_cost(dist_matrix: &[Vec<u32>], threshold: u32) -> u32 {
    let n = dist_matrix.len();

    let mut in_mst = vec![false; n];
    let mut min_edge = vec![u32::MAX; n];
    min_edge[0] = 0;

    let mut total_cost: u32 = 0;

    for _ in 0..n {
        // Find vertex with minimal edge
        let mut u = 0;
        let mut best = u32::MAX;

        // manual scan, avoid extra branching
        for v in 0..n {
            let val = min_edge[v];
            if !in_mst[v] && val < best {
                best = val;
                u = v;
            }
        }

        in_mst[u] = true;
        total_cost += best;
        if total_cost >= threshold {
            return threshold;
        }

        // Update distances to other vertices
        let row = &dist_matrix[u];
        for v in 0..n {
            let d = row[v];
            if !in_mst[v] && d < min_edge[v] {
                min_edge[v] = d;
            }
        }
    }

    return total_cost;
}

use std::collections::HashMap;

pub fn optimize_split(
    best_cost: u32,
    split: &[Vec<usize>],
    m2m_dist: &[Vec<u32>],
    p2m_dist: &[Vec<u32>],
) -> Vec<Vec<usize>> {
    let n_plr = p2m_dist.len();
    let mut optimized_split = split.to_vec();

    for pi in 0..n_plr {
        let mushrooms = &split[pi];
        let n_mush = mushrooms.len();
        if n_mush < 2 {
            continue;
        }

        let mut best_seq = mushrooms.clone();
        let mut best_seq_cost = best_cost;
        let mut path = vec![0usize; n_mush];

        // MST cache
        let mut mst_cache: HashMap<u64, u32> = HashMap::new();

        // preallocated MST scratch
        let mut nodes = vec![0usize; n_mush];
        let mut in_mst = vec![false; n_mush];
        let mut min_edge = vec![u32::MAX; n_mush];

        // preallocated candidates buffer
        let mut candidates = Vec::with_capacity(n_mush);

        #[inline(always)]
        fn compute_mst(
            mask: u64,
            n_mush: usize,
            mushrooms: &[usize],
            m2m_dist: &[Vec<u32>],
            cache: &mut HashMap<u64, u32>,
            nodes: &mut [usize],
            in_mst: &mut [bool],
            min_edge: &mut [u32],
        ) -> u32 {
            if mask == 0 {
                return 0;
            }
            if let Some(&v) = cache.get(&mask) {
                return v;
            }

            let mut k = 0;
            for i in 0..n_mush {
                if (mask >> i) & 1 != 0 {
                    nodes[k] = i;
                    k += 1;
                }
            }

            for i in 0..k {
                in_mst[i] = false;
                min_edge[i] = u32::MAX;
            }

            in_mst[0] = true;
            let a = mushrooms[nodes[0]];
            for j in 1..k {
                let b = mushrooms[nodes[j]];
                min_edge[j] = m2m_dist[a][b];
            }

            let mut total = 0u32;
            for _ in 1..k {
                let mut best_j = usize::MAX;
                let mut best_cost = u32::MAX;
                for j in 0..k {
                    if !in_mst[j] && min_edge[j] < best_cost {
                        best_cost = min_edge[j];
                        best_j = j;
                    }
                }
                if best_j == usize::MAX {
                    break;
                }
                total += best_cost;
                in_mst[best_j] = true;

                let u_global = mushrooms[nodes[best_j]];
                for v in 0..k {
                    if !in_mst[v] {
                        let v_global = mushrooms[nodes[v]];
                        let duv = m2m_dist[u_global][v_global];
                        if duv < min_edge[v] {
                            min_edge[v] = duv;
                        }
                    }
                }
            }

            cache.insert(mask, total);
            total
        }

        #[inline(always)]
        fn dfs(
            depth: usize,
            last_idx: usize,
            cost: u32,
            visited_mask: u64,
            n_mush: usize,
            mushrooms: &[usize],
            m2m_dist: &[Vec<u32>],
            path: &mut [usize],
            best_seq: &mut [usize],
            best_seq_cost: &mut u32,
            mst_cache: &mut HashMap<u64, u32>,
            nodes: &mut [usize],
            in_mst: &mut [bool],
            min_edge: &mut [u32],
            candidates: &mut Vec<(u32, usize)>,
        ) {
            let full_mask = if n_mush >= 64 {
                u64::MAX
            } else {
                (1u64 << n_mush) - 1
            };
            let rem_mask = full_mask & !visited_mask;

            if rem_mask == 0 {
                if cost < *best_seq_cost {
                    *best_seq_cost = cost;
                    best_seq.copy_from_slice(path);
                }
                return;
            }

            let mst_cost = compute_mst(
                rem_mask, n_mush, mushrooms, m2m_dist, mst_cache, nodes, in_mst, min_edge,
            );

            let last_mush = mushrooms[last_idx];
            let mut min_from_last = u32::MAX;

            candidates.clear();
            let mut bit = rem_mask;
            while bit != 0 {
                let tz = bit.trailing_zeros() as usize;
                let next_idx = tz;
                let next_mush = mushrooms[next_idx];
                let e = m2m_dist[last_mush][next_mush];
                candidates.push((e, next_idx));
                if e < min_from_last {
                    min_from_last = e;
                }
                bit &= bit - 1;
            }

            let lower_bound = cost.saturating_add(mst_cost).saturating_add(min_from_last);
            if lower_bound >= *best_seq_cost {
                return;
            }

            candidates.sort_unstable_by_key(|&(e, _)| e);

            // snapshot to avoid mutation conflicts
            let snapshot: Vec<(u32, usize)> = candidates.clone();

            for (edge_cost, next_idx) in snapshot {
                let new_cost = cost.saturating_add(edge_cost);
                if new_cost >= *best_seq_cost {
                    continue;
                }
                let next_mush = mushrooms[next_idx];
                path[depth] = next_mush;
                dfs(
                    depth + 1,
                    next_idx,
                    new_cost,
                    visited_mask | (1u64 << next_idx),
                    n_mush,
                    mushrooms,
                    m2m_dist,
                    path,
                    best_seq,
                    best_seq_cost,
                    mst_cache,
                    nodes,
                    in_mst,
                    min_edge,
                    candidates,
                );
            }
        }

        for (idx, &mush) in mushrooms.iter().enumerate() {
            path[0] = mush;
            let start_cost = p2m_dist[pi][mush];
            if start_cost >= best_seq_cost {
                continue;
            }
            dfs(
                1,
                idx,
                start_cost,
                1u64 << idx,
                n_mush,
                mushrooms,
                m2m_dist,
                &mut path,
                &mut best_seq,
                &mut best_seq_cost,
                &mut mst_cache,
                &mut nodes,
                &mut in_mst,
                &mut min_edge,
                &mut candidates,
            );
        }

        optimized_split[pi].copy_from_slice(&best_seq);
    }

    optimized_split
}
/// Compute a minimum costs and best sequences for given mushroom-split (for each player)
/// returns a vector of best_seq mush indexes for each player
///
/// If the cost reaches best_cost, the algorithm will short-circuit and returns empty vector
pub fn optimize_split_old2(
    best_cost: u32,
    split: &[Vec<usize>],
    m2m_dist: &[Vec<u32>],
    p2m_dist: &[Vec<u32>],
) -> Vec<Vec<usize>> {
    let n_plr = p2m_dist.len();
    let mut optimized_split = split.to_vec();

    for pi in 0..n_plr {
        let mushrooms = &split[pi];
        let n_mush = mushrooms.len();
        if n_mush < 2 {
            continue;
        }

        // best sequence (global mushroom indices) and its cost (initialized to cutoff)
        let mut best_seq = mushrooms.clone();
        let mut best_seq_cost = best_cost;

        // current path (global mushroom indices)
        let mut path = vec![0usize; n_mush];

        // cache for MST costs keyed by bitmask of remaining nodes
        let mut mst_cache: HashMap<u64, u32> = HashMap::new();

        #[inline(always)]
        fn compute_mst(
            mask: u64,
            n_mush: usize,
            mushrooms: &[usize],
            m2m_dist: &[Vec<u32>],
            cache: &mut HashMap<u64, u32>,
        ) -> u32 {
            if mask == 0 {
                return 0;
            }
            if let Some(&v) = cache.get(&mask) {
                return v;
            }

            // collect node indices (indices into `mushrooms` slice)
            let mut nodes = Vec::with_capacity(mask.count_ones() as usize);
            for i in 0..n_mush {
                if (mask >> i) & 1 != 0 {
                    nodes.push(i);
                }
            }

            let k = nodes.len();
            debug_assert!(k > 0);

            // Prim's algorithm on the induced subgraph of `nodes`
            let mut in_mst = vec![false; k];
            let mut min_edge = vec![u32::MAX; k];

            // pick first as starting point
            in_mst[0] = true;
            for j in 1..k {
                let a = mushrooms[nodes[0]];
                let b = mushrooms[nodes[j]];
                min_edge[j] = m2m_dist[a][b];
            }

            let mut total = 0u32;
            for _ in 1..k {
                // find cheapest edge to add
                let mut best_j = None;
                let mut best_cost = u32::MAX;
                for j in 0..k {
                    if !in_mst[j] && min_edge[j] < best_cost {
                        best_cost = min_edge[j];
                        best_j = Some(j);
                    }
                }
                let u = best_j.expect("there must be a next MST node");
                // add and update
                total = total.saturating_add(best_cost);
                in_mst[u] = true;

                let u_global = mushrooms[nodes[u]];
                for v in 0..k {
                    if !in_mst[v] {
                        let v_global = mushrooms[nodes[v]];
                        let duv = m2m_dist[u_global][v_global];
                        if duv < min_edge[v] {
                            min_edge[v] = duv;
                        }
                    }
                }
            }

            cache.insert(mask, total);
            total
        }

        #[inline(always)]
        fn dfs(
            depth: usize,
            last_idx: usize,
            cost: u32,
            visited_mask: u64,
            n_mush: usize,
            mushrooms: &[usize],
            m2m_dist: &[Vec<u32>],
            path: &mut [usize],
            best_seq: &mut [usize],
            best_seq_cost: &mut u32,
            mst_cache: &mut HashMap<u64, u32>,
        ) {
            // full_mask for n_mush bits
            let full_mask = if n_mush >= 64 {
                u64::MAX
            } else {
                (1u64 << n_mush) - 1
            };
            let rem_mask = full_mask & !visited_mask;

            // finished
            if rem_mask == 0 {
                if cost < *best_seq_cost {
                    *best_seq_cost = cost;
                    best_seq.copy_from_slice(path);
                }
                return;
            }

            // MST lower bound on remaining nodes
            let mst_cost = compute_mst(rem_mask, n_mush, mushrooms, m2m_dist, mst_cache);

            // min edge from last node into remaining nodes
            let last_mush = mushrooms[last_idx];
            let mut min_from_last = u32::MAX;

            // collect candidates (edge_cost, next_idx) to explore promising ones first
            let mut candidates: Vec<(u32, usize)> = Vec::new();
            let mut bit = rem_mask;
            while bit != 0 {
                let tz = bit.trailing_zeros() as usize;
                let next_idx = tz;
                let next_mush = mushrooms[next_idx];
                let e = m2m_dist[last_mush][next_mush];
                candidates.push((e, next_idx));
                if e < min_from_last {
                    min_from_last = e;
                }
                bit &= bit - 1;
            }

            // admissible LB: must at least add an edge from last to the remaining cluster + MST among them
            let lower_bound = cost.saturating_add(mst_cost).saturating_add(min_from_last);
            if lower_bound >= *best_seq_cost {
                return;
            }

            // explore cheaper edges first
            candidates.sort_unstable_by_key(|&(e, _)| e);

            for (edge_cost, next_idx) in candidates {
                let new_cost = cost.saturating_add(edge_cost);
                if new_cost >= *best_seq_cost {
                    continue;
                }
                let next_mush = mushrooms[next_idx];
                path[depth] = next_mush;
                dfs(
                    depth + 1,
                    next_idx,
                    new_cost,
                    visited_mask | (1u64 << next_idx),
                    n_mush,
                    mushrooms,
                    m2m_dist,
                    path,
                    best_seq,
                    best_seq_cost,
                    mst_cache,
                );
            }
        }

        // initialize dfs for each possible starting mushroom
        for (idx, &mush) in mushrooms.iter().enumerate() {
            path[0] = mush;
            let start_cost = p2m_dist[pi][mush];
            if start_cost >= best_seq_cost {
                // pruning: start already worse than best known
                continue;
            }
            dfs(
                1,
                idx,
                start_cost,
                1u64 << idx,
                n_mush,
                mushrooms,
                m2m_dist,
                &mut path,
                &mut best_seq,
                &mut best_seq_cost,
                &mut mst_cache,
            );
        }

        // write best sequence back (keeps original order if no better solution found)
        optimized_split[pi].copy_from_slice(&best_seq);
    }

    optimized_split
}

/// Compute a minimum costs and best sequences for given mushroom-split (for each player)
/// returns a vector of best_seq mush indexes for each player
///
/// If the cost reaches best_cost, the algorithm will short-circuit and returns empty vector
pub fn optimize_split_old(
    best_cost: u32,
    split: &[Vec<usize>],
    m2m_dist: &[Vec<u32>],
    p2m_dist: &[Vec<u32>],
) -> Vec<Vec<usize>> {
    let n_plr = p2m_dist.len();
    let mut optimized_split = split.to_vec();

    for pi in 0..n_plr {
        let mushrooms = &split[pi];
        let n_mush = mushrooms.len();
        if n_mush < 2 {
            continue;
        }

        let mut best_seq = mushrooms.clone();
        let mut best_seq_cost = best_cost;

        let mut path = vec![0usize; n_mush];

        #[allow(clippy::too_many_arguments)]
        fn dfs(
            depth: usize,
            last_idx: usize,
            cost: u32,
            visited_mask: u64,
            mushrooms: &[usize],
            m2m_dist: &[Vec<u32>],
            path: &mut [usize],
            best_seq: &mut [usize],
            best_seq_cost: &mut u32,
        ) {
            let n_mush = mushrooms.len();
            if depth == n_mush {
                if cost < *best_seq_cost {
                    *best_seq_cost = cost;
                    best_seq.copy_from_slice(path);
                }
                return;
            }

            for next_idx in 0..n_mush {
                if visited_mask & (1 << next_idx) != 0 {
                    continue;
                }

                let next_mush = mushrooms[next_idx];
                let last_mush = mushrooms[last_idx];
                let new_cost = cost + m2m_dist[last_mush][next_mush];

                if new_cost >= *best_seq_cost {
                    continue;
                }

                path[depth] = next_mush;
                dfs(
                    depth + 1,
                    next_idx,
                    new_cost,
                    visited_mask | (1 << next_idx),
                    mushrooms,
                    m2m_dist,
                    path,
                    best_seq,
                    best_seq_cost,
                );
            }
        }

        // initialize dfs for each starting mushroom
        for (idx, &mush) in mushrooms.iter().enumerate() {
            path[0] = mush;
            dfs(
                1,
                idx,
                p2m_dist[pi][mush],
                1 << idx,
                mushrooms,
                m2m_dist,
                &mut path,
                &mut best_seq,
                &mut best_seq_cost,
            );
        }

        optimized_split[pi].copy_from_slice(&best_seq);
    }

    return optimized_split;
}

/// returns the cost of given split
fn get_split_cost(split: &[Vec<usize>], m2m_dist: &[Vec<u32>], p2m_dist: &[Vec<u32>]) -> u32 {
    return split
        .iter()
        .enumerate()
        .map(|(player_idx, mush_seq)| -> u32 {
            if mush_seq.is_empty() {
                return 0;
            }
            return mush_seq
                .windows(2)
                .map(|w| {
                    return m2m_dist[w[0]][w[1]];
                })
                .sum::<u32>()
                + p2m_dist[player_idx][mush_seq[0]];
        })
        .sum();
}

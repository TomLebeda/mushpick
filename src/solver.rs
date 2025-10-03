use concorde_rs::{LowerDistanceMatrix, solver};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use itertools::Itertools;
use log::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    Coord, Field, parse_field,
    pathfinding::{bfs_m2m, bfs_p2m, is_field_accessible},
    renderer::print_matrix,
    utils::transpose,
};

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
    let min_mush_cost = get_mush_min_costs(&m2m_dist, &p2m_dist);

    // trace!("OLD GREEDY SPLIT:");
    // let old_greedy_split = get_greedy_split(&field, &p2m_dist);
    // old_greedy_split
    //     .iter()
    //     .enumerate()
    //     .for_each(|(p_idx, mush_seq)| {
    //         trace!("   - P{}: {:?}", p_idx, mush_seq);
    //     });
    // let old_greedy_split_cost: u32 = old_greedy_split
    //     .iter()
    //     .enumerate()
    //     .map(|(player_idx, mush_seq)| -> u32 {
    //         return (mush_seq
    //             .windows(2)
    //             .map(|w| {
    //                 return m2m_dist[w[0]][w[1]];
    //             })
    //             .sum::<u32>()
    //             + p2m_dist[player_idx][mush_seq[0]]) as u32;
    //     })
    //     .sum();
    // trace!(" - cost: {old_greedy_split_cost}");
    //
    // trace!("OPTIMIZED OLD GREEDY SPLIT:");
    // let optimized_old_greedy_split =
    //     optimize_split(u32::MAX, &old_greedy_split, &m2m_dist, &p2m_dist, fast);
    // optimized_old_greedy_split
    //     .iter()
    //     .enumerate()
    //     .for_each(|(p_idx, mush_seq)| {
    //         trace!("   - P{}: {:?}", p_idx, mush_seq.0);
    //     });
    // let optimized_old_greedy_split_cost: u32 = optimized_old_greedy_split
    //     .iter()
    //     .map(|x| return x.clone().0)
    //     .enumerate()
    //     .map(|(player_idx, mush_seq)| -> u32 {
    //         return (mush_seq
    //             .windows(2)
    //             .map(|w| {
    //                 return m2m_dist[w[0]][w[1]];
    //             })
    //             .sum::<u32>()
    //             + p2m_dist[player_idx][mush_seq[0]]) as u32;
    //     })
    //     .sum();
    // trace!(" - cost: {optimized_old_greedy_split_cost}");

    // how the players split the mushrooms
    let mut greedy_split: Vec<Vec<usize>> = vec![Vec::with_capacity(n_mush); n_plr];
    // movement-potential for each player
    let mut free_steps: Vec<u32> = vec![0; n_plr];
    // indicator if mushroom at index is assigned
    let mut mushrooms_free = vec![true; n_mush];

    // while there are still some mushrooms left to be assigned...
    while mushrooms_free.iter().any(|free| return *free) {
        // find how many ticks are required for any player to pick any (nearest) mushroom
        let nearest = (0..n_plr)
            // find nearest free mushrooms for all players
            .map(|pi| {
                if let Some(pos) = greedy_split[pi].last() {
                    // the player is already standing at some mushroom position => use m2m
                    let mut nearest_mush_idx = 0;
                    let mut nearest_mush_dist = u32::MAX;
                    for mi in 0..n_mush {
                        if mushrooms_free[mi] && m2m_dist[*pos][mi] < nearest_mush_dist {
                            nearest_mush_idx = mi;
                            nearest_mush_dist = m2m_dist[*pos][mi];
                        }
                    }
                    let ticks_required = nearest_mush_dist - free_steps[pi];
                    return (pi, nearest_mush_idx, nearest_mush_dist, ticks_required);
                } else {
                    // the player hasn't moved yet => use p2m
                    let mut nearest_mush_idx = 0;
                    let mut nearest_mush_dist = u32::MAX;
                    for mi in 0..n_mush {
                        if mushrooms_free[mi] && p2m_dist[pi][mi] < nearest_mush_dist {
                            nearest_mush_idx = mi;
                            nearest_mush_dist = p2m_dist[pi][mi];
                        }
                    }
                    let ticks_required = nearest_mush_dist - free_steps[pi];
                    return (pi, nearest_mush_idx, nearest_mush_dist, ticks_required);
                };
            })
            .collect_vec();

        let ticks_required = &nearest
            .iter()
            .min_by_key(
                |(_pi, _nearest_mush_idx, _nearest_mush_dist, ticks_required)| {
                    return *ticks_required;
                },
            )
            .unwrap()
            .3;

        // increment the ticks by the minimum amount
        for pi in 0..n_plr {
            free_steps[pi] += ticks_required;
        }
        // check for all players if they can pick some mushrooms
        nearest.iter().for_each(
            |(pi, nearest_mush_idx, nearest_mush_dist, ticks_required)| {
                if &free_steps[*pi] >= nearest_mush_dist {
                    free_steps[*pi] -= nearest_mush_dist; // remove the steps
                    greedy_split[*pi].push(*nearest_mush_idx); // assign the mushroom
                    mushrooms_free[*nearest_mush_idx] = false; // mark the mushroom as picked
                }
            },
        )
    }

    // trace!("GREEDY SPLIT:");
    // greedy_split
    //     .iter()
    //     .enumerate()
    //     .for_each(|(p_idx, mush_seq)| {
    //         trace!("   - P{}: {:?}", p_idx, mush_seq);
    //     });
    let greedy_split_cost: u32 = greedy_split
        .iter()
        .enumerate()
        .map(|(player_idx, mush_seq)| -> u32 {
            return (mush_seq
                .windows(2)
                .map(|w| {
                    return m2m_dist[w[0]][w[1]];
                })
                .sum::<u32>()
                + p2m_dist[player_idx][mush_seq[0]]) as u32;
        })
        .sum();
    // trace!(" - cost: {new_greedy_split_cost}");
    //
    //
    // trace!("OPTIMIZED NEW GREEDY SPLIT");
    // optimized_new_greedy_split
    //     .iter()
    //     .enumerate()
    //     .for_each(|(p_idx, (mush_seq, _split_cost))| {
    //         trace!("   - P{}: {:?}", p_idx, mush_seq);
    //     });
    // let optimized_new_greedy_split_cost: u32 = optimized_new_greedy_split
    //     .iter()
    //     .map(|x| return x.clone().0)
    //     .enumerate()
    //     .map(|(player_idx, mush_seq)| -> u32 {
    //         return (mush_seq
    //             .windows(2)
    //             .map(|w| {
    //                 return m2m_dist[w[0]][w[1]];
    //             })
    //             .sum::<u32>()
    //             + p2m_dist[player_idx][mush_seq[0]]) as u32;
    //     })
    //     .sum();
    // trace!(" - cost: {optimized_new_greedy_split_cost}");

    let optimized_greedy_split =
        optimize_split(greedy_split_cost, &greedy_split, &m2m_dist, &p2m_dist, fast);
    let mut best_split_cost = greedy_split_cost;
    let mut best_split = optimized_greedy_split;
    // let mut best_split_cost = optimized_old_greedy_split
    //     .iter()
    //     .map(|x| return x.1)
    //     .max()
    //     .unwrap_or(u32::MAX);
    // trace!(" - found cost of optimized greedy split: {best_split_cost}");

    if !fast {
        trace!("generating splits");
        let split_generator =
            generate_splits(field.mushrooms.len(), field.players.len()).enumerate();
        for (_split_idx, split) in split_generator {
            let lower_bound = get_split_cost_lower_bound(&split, &min_mush_cost);
            if lower_bound >= best_split_cost {
                continue;
            }
            let optimized_split =
                optimize_split(best_split_cost, &split, &m2m_dist, &p2m_dist, fast);
            let optimized_split_cost = optimized_split.iter().max().unwrap();
            if optimized_split_cost < best_split_cost {
                best_split_cost = optimized_split_cost;
                best_split = optimized_split;
            }
        }
    }
    trace!("BEST SPLIT");
    best_split
        .iter()
        .enumerate()
        .for_each(|(p_idx, (mush_seq, _split_cost))| {
            trace!("   - P{}: {:?}", p_idx, mush_seq);
        });
    trace!(" - cost: {best_split_cost}");

    trace!("pathfinding split");
    best_split
        .iter()
        .enumerate()
        .for_each(|(player_idx, (mush_seq, _split_cost))| {
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

/// Generate all ways to split N elements into M groups
pub fn generate_splits(
    set_size: usize,
    group_count: usize,
) -> impl Iterator<Item = Vec<Vec<usize>>> {
    return vec![0..group_count; set_size]
        .into_iter()
        .multi_cartesian_product()
        .map(move |assignment| {
            let mut groups = vec![Vec::new(); group_count];
            for (i, &group) in assignment.iter().enumerate() {
                groups[group].push(i);
            }
            return groups;
        });
}

/// Compute the best sequence of mushrooms for given player, checking all possible permutations
#[allow(dead_code)]
#[deprecated]
pub fn get_best_seq_all_perms(
    best_cost: usize,
    mush_idxs: &[usize],
    player_idx: usize,
    dist_mat: &[Vec<usize>],
    field: &Field,
) -> (Vec<usize>, usize) {
    let player_count = field.players.len();
    let mut best_perm = Vec::with_capacity(mush_idxs.len());
    let mut best_perm_cost = best_cost;
    for perm in mush_idxs.iter().copied().permutations(mush_idxs.len()) {
        let first_target = perm.first().copied().unwrap_or(player_idx);
        let mut perm_cost = dist_mat[player_idx][first_target + player_count];
        for (a, b) in perm.iter().tuple_windows() {
            perm_cost += dist_mat[*a + player_count][*b + player_count];
            if perm_cost >= best_perm_cost {
                break;
            }
        }
        if perm_cost < best_perm_cost {
            best_perm = perm;
            best_perm_cost = perm_cost;
        }
    }
    return (best_perm, best_perm_cost);
}

/// produce a greedy split, simply by assigning each mushroom to the nearest player
pub fn get_greedy_split(field: &Field, p2m_dist: &[Vec<u32>]) -> Vec<Vec<usize>> {
    let mut split: Vec<Vec<usize>> = vec![vec![]; field.players.len()];
    let p2m_dist = transpose(p2m_dist.to_vec());
    for (mi, _) in field.mushrooms.iter().enumerate() {
        let nearest_player_idx = p2m_dist[mi].iter().position_min().unwrap();
        split[nearest_player_idx].push(mi);
    }
    return split;
}

/// Compute a minimum costs and best sequences for given mushroom-split (for each player)
/// returns a vector of best_seq mush indexes for each player
pub fn optimize_split(
    best_cost: u32,
    split: &[Vec<usize>],
    m2m_dist: &[Vec<u32>],
    p2m_dist: &[Vec<u32>],
    fast: bool,
) -> Vec<Vec<usize>> {
    // for each player, find the best sequence of their mushrooms
    let optimized_sequences = split
        .iter()
        .enumerate()
        .map(|(player_idx, mushrooms)| -> Vec<usize> {
            // now find the actual best sequence with better threshold
            return get_best_permutation(
                best_cost, mushrooms, player_idx, m2m_dist, p2m_dist, fast,
            )
            .0;
        })
        .collect_vec();

    // total cost of this split is sum of the best costs of each player
    return optimized_sequences;
}

/// Compute the minimum cost of each mushroom
pub fn get_mush_min_costs(m2m_dist: &[Vec<u32>], p2m_dist: &[Vec<u32>]) -> Vec<u32> {
    let mush_count = m2m_dist.len();
    // create a local copy of the mush-to-mush distance matrix where the diagonal values will be
    // replaced by usize::MAX so that they are not picked as the minimum value
    // cloning here is fine, because this function should run only once per map
    let mut m2m_dist: Vec<Vec<u32>> = m2m_dist.into();
    (0..m2m_dist.len()).for_each(|i| m2m_dist[i][i] = u32::MAX);

    let p2m_dist = transpose(p2m_dist.to_vec());

    let mut min_costs: Vec<u32> = Vec::with_capacity(mush_count);

    for (idx, row) in m2m_dist.iter().enumerate() {
        let min_m2m = row.iter().min().unwrap();
        let min_p2m = p2m_dist[idx].iter().min().unwrap();
        min_costs.push(std::cmp::min(*min_m2m, *min_p2m));
    }

    return min_costs;
}

/// Compute the lower-bound cost of given split
///
/// The lower bound is computed as a sum of minimum costs all mushrooms in this split.
pub fn get_split_cost_lower_bound(split: &[Vec<usize>], min_mush_cost: &[u32]) -> u32 {
    return split
        .par_iter()
        .map(|mushrooms| -> u32 {
            return mushrooms
                .par_iter()
                .map(|m| return min_mush_cost[*m as usize])
                .sum();
        })
        .max() // total cost of the split is given by maximum cost across players
        .unwrap_or(0);
}

/// Find the greedy sequence of mushrooms from given player position
pub fn get_greedy_seq(
    mush_idxs: &[usize],
    player_idx: usize,
    m2m: &[Vec<u32>],
    p2m: &[Vec<u32>],
) -> (Vec<usize>, u32) {
    let mut seq: Vec<usize> = Vec::with_capacity(mush_idxs.len());
    let used: Vec<bool> = vec![false; mush_idxs.len()];
    let mut zipped = std::iter::zip(used, mush_idxs).collect_vec();
    let mut cost: u32 = 0;
    for _ in 0..mush_idxs.len() {
        let (pos_to_flip, nearest_mush_idx, nearest_mush_cost) = zipped
            .iter()
            .enumerate()
            .filter_map(|(pos, (used, mush_idx))| {
                if *used {
                    return None;
                }
                let dist = match seq.last() {
                    None => p2m[player_idx][**mush_idx],
                    Some(j) => m2m[*j][**mush_idx],
                };
                return Some((pos, mush_idx, dist));
            })
            .min_by_key(|x| return x.2)
            .unwrap();
        cost += nearest_mush_cost;
        seq.push(**nearest_mush_idx);
        zipped[pos_to_flip].0 = true;
    }
    return (seq, cost);
}

/// Compute the best sequence using backtracking
fn get_best_permutation(
    best_cost: u32,
    mush_idxs: &[usize],
    player_idx: usize,
    m2m: &[Vec<u32>],
    p2m: &[Vec<u32>],
    fast: bool,
) -> (Vec<usize>, u32) {
    if mush_idxs.is_empty() {
        // if there are no mushrooms to pick up, the cost is zero
        return (vec![], 0);
    }
    if mush_idxs.len() == 1 {
        // if there is only one mushroom to pick up, the cost is simply the distance from player
        return (vec![mush_idxs[0]], p2m[player_idx][mush_idxs[0]]);
    }

    // get the greedy permutation as a warm-up, to better leverage the pruning optimization
    let (greedy_perm, greedy_cost) = get_greedy_seq(mush_idxs, player_idx, m2m, p2m);

    let mut best_perm: Vec<usize> = greedy_perm;
    if !fast {
        // now run the backtracking to actually find the best permutation
        let mut best_cost = std::cmp::min(best_cost, greedy_cost);
        let mut mush_idxs: Vec<usize> = mush_idxs.into();
        backtrack(
            0,
            0,
            &mut mush_idxs,
            &mut best_perm,
            &mut best_cost,
            player_idx,
            m2m,
            p2m,
        );
    }
    return (best_perm, best_cost);
}

/// backtracking algorithm to find best permutation
#[allow(clippy::too_many_arguments)]
fn backtrack(
    index: usize,
    current_cost: u32,
    numbers: &mut Vec<usize>,
    best_perm: &mut Vec<usize>,
    best_cost: &mut u32,
    player_idx: usize,
    m2m: &[Vec<u32>],
    p2m: &[Vec<u32>],
) {
    if index == numbers.len() {
        let from = numbers[index - 1];
        let to = numbers[index - 2];
        let total_cost = current_cost + m2m[from][to];
        if total_cost < *best_cost {
            *best_cost = total_cost;
            *best_perm = numbers.clone();
        }
        return;
    }

    for i in index..numbers.len() {
        numbers.swap(index, i);
        let added_cost = match index {
            0 => 0, // no elements => no cost
            1 => {
                // one elements => distance from player to first mush
                p2m[player_idx][numbers[0]]
            }
            _ => {
                // multiple elements => cost is the last step mush-to-mush
                let last = numbers[index - 1];
                let second_last = numbers[index - 2];
                m2m[second_last][last]
            }
        };
        let new_cost = current_cost + added_cost;
        if new_cost < *best_cost {
            backtrack(
                index + 1,
                new_cost,
                numbers,
                best_perm,
                best_cost,
                player_idx,
                m2m,
                p2m,
            );
        }
        numbers.swap(index, i); // undo swap (backtracking)
    }
}

use std::path::PathBuf;

use itertools::Itertools;
use log::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    Field, parse_field,
    pathfinding::{get_m2m_dist_matrix, get_p2m_dist_matrix, is_field_accessible, pathfind_split},
};

/// Solve the map on provided file path
pub fn solve(map: PathBuf) {
    let field = parse_field(&map);
    match is_field_accessible(&field.cells, field.size) {
        true => info!("all cells are accessible"),
        false => {
            error!("flood-fill couldn't reach all cells");
            std::process::exit(exitcode::DATAERR);
        }
    }

    //let split_count = field.players.len().pow(field.mushrooms.len() as u32);
    //println!("- map size: {}", field.size);
    //println!("- player count: {}", field.players.len());
    //println!("- mushroom count: {}", field.mushrooms.len());
    //println!(
    //    "- # of possible ways to split mushrooms between players: {}^{} = {}",
    //    field.players.len(),
    //    field.mushrooms.len(),
    //    split_count
    //);
    //println!(
    //    "- # of possible ways to pick up mushrooms: {}^{} * {}!",
    //    field.players.len(),
    //    field.mushrooms.len(),
    //    field.mushrooms.len(),
    //);

    let m2m_dist = get_m2m_dist_matrix(&field);
    //println!("m2m_dist: ");
    //print_matrix(&m2m_dist);
    let p2m_dist = get_p2m_dist_matrix(&field);
    //println!("p2m_dist: ");
    //print_matrix(&p2m_dist);
    let min_mush_cost = get_mush_min_costs(&m2m_dist, &p2m_dist);
    //println!("min mush cost: {min_mush_cost:?}");

    let greedy_split = get_greedy_split(&field, &p2m_dist);
    //println!("\ngreedy split:");
    //for (pi, m) in greedy_split.iter().enumerate() {
    //    println!("- P{pi}: {m:?}");
    //}

    let optimized_greedy_split = optimize_split(usize::MAX, &greedy_split, &m2m_dist, &p2m_dist);
    //println!("\noptimized greedy split:");
    //for (player_idx, (mush_seq, cost)) in optimized_greedy_split.iter().enumerate() {
    //    println!("- P{player_idx}: {mush_seq:?} (cost {cost})");
    //}
    let mut best_split_cost = optimized_greedy_split
        .iter()
        .map(|x| return x.1)
        .max()
        .unwrap_or(usize::MAX);
    //println!("- total cost: {best_split_cost}");
    let mut best_split = optimized_greedy_split;

    let split_generator = generate_splits(field.mushrooms.len(), field.players.len()).enumerate();
    for (_split_idx, split) in split_generator {
        //println!("\nsplit {split_idx}: {split:?}");
        let lower_bound = get_split_cost_lower_bound(&split, &min_mush_cost);
        //println!("- lower bound cost: {lower_bound}");
        if lower_bound >= best_split_cost {
            //println!("- skipping: lower bound too expensive, best is {best_split_cost})");
            continue;
        }
        let optimized_split = optimize_split(best_split_cost, &split, &m2m_dist, &p2m_dist);
        let optimized_split_cost = optimized_split.iter().map(|x| return x.1).max().unwrap();
        //for (p, (mush, cost)) in optimized_split.iter().enumerate() {
        //    println!("- P{p}: {mush:?} (cost {cost})")
        //}
        //println!("- total cost: {optimized_split_cost}");
        if optimized_split_cost < best_split_cost {
            //println!("FOUND NEW BEST!");
            best_split_cost = optimized_split_cost;
            best_split = optimized_split;
        }
    }
    println!("FINAL BEST SPLIT (total cost {best_split_cost}): ");
    for (p, (mush, cost)) in best_split.iter().enumerate() {
        println!("- P{p}: {mush:?} (cost {cost})");
    }

    println!("\nFINAL SPLIT PATHS: ");
    let paths = pathfind_split(&best_split, &field);
    for (player, path) in paths.iter().enumerate() {
        println!(
            "- P{player}: {{{}}}",
            path.iter().map(|c| format!("{c}")).join(", ")
        );
    }
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
pub fn get_greedy_split(field: &Field, p2m_dist: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut split: Vec<Vec<usize>> = vec![vec![]; field.players.len()];
    for (mi, _) in field.mushrooms.iter().enumerate() {
        let nearest_player_idx = p2m_dist[mi].iter().position_min().unwrap();
        split[nearest_player_idx].push(mi);
    }
    return split;
}

/// Compute a minimum costs and best sequences for given mushroom-split (for each player)
/// returns a vector of tuples (best_seq, best_seq_cost) for each player
pub fn optimize_split(
    best_cost: usize,
    split: &[Vec<usize>],
    m2m_dist: &[Vec<usize>],
    p2m_dist: &[Vec<usize>],
) -> Vec<(Vec<usize>, usize)> {
    // for each player, find the best sequence of their mushrooms
    let optimized_sequences = split
        .iter()
        .enumerate()
        .map(|(player_idx, mushrooms)| -> (Vec<usize>, usize) {
            // now find the actual best sequence with better threshold
            let (best_seq, cost) =
                get_best_permutation(best_cost, mushrooms, player_idx, m2m_dist, p2m_dist);
            return (best_seq, cost);
        })
        .collect_vec();

    // total cost of this split is sum of the best costs of each player
    return optimized_sequences;
}

/// Compute the minimum cost of each mushroom
pub fn get_mush_min_costs(m2m_dist: &[Vec<usize>], p2m_dist: &[Vec<usize>]) -> Vec<usize> {
    // create a local copy of the mush-to-mush distance matrix where the diagonal zeros will be
    // replaced by usize::MAX so that they are not picked as the minimum value
    // cloning here is fine, because this function should run only once per map
    let mut m2m_dist: Vec<Vec<usize>> = m2m_dist.into();
    (0..m2m_dist.len()).for_each(|i| m2m_dist[i][i] = usize::MAX);

    let mut min_costs: Vec<usize> = Vec::with_capacity(m2m_dist.len());
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
pub fn get_split_cost_lower_bound(split: &[Vec<usize>], min_mush_cost: &[usize]) -> usize {
    return split
        .par_iter()
        .map(|mushrooms| -> usize {
            return mushrooms.par_iter().map(|m| return min_mush_cost[*m]).sum();
        })
        .max() // total cost of the split is given by maximum cost across players
        .unwrap_or(0);
}

/// Find the greedy sequence of mushrooms from given player position
pub fn get_greedy_seq(
    mush_idxs: &[usize],
    player_idx: usize,
    m2m: &[Vec<usize>],
    p2m: &[Vec<usize>],
) -> (Vec<usize>, usize) {
    let mut seq: Vec<usize> = Vec::with_capacity(mush_idxs.len());
    let used: Vec<bool> = vec![false; mush_idxs.len()];
    let mut zipped = std::iter::zip(used, mush_idxs).collect_vec();
    let mut cost: usize = 0;
    for _ in 0..mush_idxs.len() {
        let (pos_to_flip, nearest_mush_idx, nearest_mush_cost) = zipped
            .iter()
            .enumerate()
            .filter_map(|(pos, (used, mush_idx))| {
                if *used {
                    return None;
                }
                let dist = match seq.last() {
                    None => p2m[**mush_idx][player_idx],
                    Some(j) => m2m[*j][**mush_idx],
                };
                return Some((pos, mush_idx, dist));
                //let start = match seq.last() {
                //    None => player_idx,
                //    Some(j) => j + player_count,
                //};
                //let goal = **mush_idx + player_count;
                //return Some((pos, mush_idx, dist_mat[start][goal]));
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
    best_cost: usize,
    mush_idxs: &[usize],
    player_idx: usize,
    m2m: &[Vec<usize>],
    p2m: &[Vec<usize>],
) -> (Vec<usize>, usize) {
    if mush_idxs.is_empty() {
        // if there are no mushrooms to pick up, the cost is zero
        return (vec![], 0);
    }
    if mush_idxs.len() == 1 {
        // if there is only one mushroom to pick up, the cost is simply the distance from player
        return (vec![mush_idxs[0]], p2m[mush_idxs[0]][player_idx]);
    }

    // get the greedy permutation as a warm-up, to better leverage the pruning optimization
    let (greedy_perm, greedy_cost) = get_greedy_seq(mush_idxs, player_idx, m2m, p2m);

    // now run the backtracking to actually find the best permutation
    let mut best_perm: Vec<usize> = greedy_perm;
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
    return (best_perm, best_cost);
}

/// backtracking algorithm to find best permutation
#[allow(clippy::too_many_arguments)]
fn backtrack(
    index: usize,
    current_cost: usize,
    numbers: &mut Vec<usize>,
    best_perm: &mut Vec<usize>,
    best_cost: &mut usize,
    player_idx: usize,
    m2m: &[Vec<usize>],
    p2m: &[Vec<usize>],
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
                p2m[numbers[0]][player_idx]
                //let from = player_idx;
                //let to = numbers[0];
                //dist_mat[from][to]
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

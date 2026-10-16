use itertools::Itertools;
use log::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{Coord, Field, test_down, test_left, test_right, test_up, utils::Direction};

// /// Find a shortest path using parallelized BFS algorithm
// pub fn find_min_path(field_size: i32, cells: &[bool], start: &Coord, goal: &Coord) -> Vec<Coord> {
//     #[derive(Debug)]
//     struct Node {
//         /// flattened XY coordinate of the cell
//         coord: i32,
//         /// index of the parent node in the previous layer/step
//         prev_coord: Option<i32>,
//     }
//
//     let mut cells = cells.to_owned();
//     let mut layers: Vec<Vec<Node>> = vec![vec![Node {
//         coord: start.x + field_size * start.y,
//         prev_coord: None,
//     }]];
//
//     loop {
//         let last_layer: &Vec<Node> = layers.last().unwrap();
//         let new_layer: Vec<Node> = last_layer
//             .par_iter()
//             .flat_map(|n: &Node| -> Vec<Node> {
//                 let mut new_nodes = Vec::<Node>::with_capacity(4);
//                 // test move RIGHT
//                 if test_right(&cells, n.coord as usize, field_size as usize) {
//                     new_nodes.push(Node {
//                         coord: n.coord + 1,
//                         prev_coord: Some(n.coord),
//                     });
//                 }
//                 // test move DOWN
//                 if test_down(&cells, n.coord as usize, field_size as usize) {
//                     new_nodes.push(Node {
//                         coord: n.coord + field_size,
//                         prev_coord: Some(n.coord),
//                     })
//                 }
//                 // test move LEFT
//                 if test_left(&cells, n.coord as usize, field_size as usize) {
//                     new_nodes.push(Node {
//                         coord: n.coord - 1,
//                         prev_coord: Some(n.coord),
//                     })
//                 }
//                 // test move UP
//                 if test_up(&cells, n.coord as usize, field_size as usize) {
//                     new_nodes.push(Node {
//                         coord: n.coord - field_size,
//                         prev_coord: Some(n.coord),
//                     })
//                 }
//                 return new_nodes;
//             })
//             .collect();
//
//         if new_layer.is_empty() {
//             println!("can't find the mushroom");
//             return vec![];
//         }
//
//         // keep each cell only once
//         let new_layer = new_layer
//             .into_iter()
//             .unique_by(|n| return n.coord)
//             .collect_vec();
//
//         // check if we found the target
//         if let Some(found_goal) = new_layer
//             .par_iter()
//             .find_any(|n| return n.coord == goal.x + goal.y * field_size)
//         {
//             let mut path: Vec<Coord> = vec![Coord {
//                 x: found_goal.coord % field_size,
//                 y: (found_goal.coord - (found_goal.coord % field_size)) / field_size,
//             }];
//             let mut prev_coord = found_goal.prev_coord.unwrap();
//             for i in (1..layers.len()).rev() {
//                 if let Some(prev_node) = layers[i].iter().find(|n| return n.coord == prev_coord) {
//                     path.push(Coord {
//                         x: prev_node.coord % field_size,
//                         y: (prev_node.coord - (prev_node.coord % field_size)) / field_size,
//                     });
//                     prev_coord = prev_node.prev_coord.unwrap();
//                 }
//             }
//             path.push(*start);
//             return path.into_iter().rev().collect_vec();
//         }
//
//         // mark the new layer as visited
//         new_layer
//             .iter()
//             .for_each(|n| cells[n.coord as usize] = false);
//
//         // add the new layer onto the stack
//         layers.push(new_layer);
//     }
// }

// /// Find shortest path distance using parallelized BFS algorithm
// pub fn find_min_path_dist(
//     field_size: i32,
//     cells: &[bool],
//     start: &Coord,
//     goal: &Coord,
// ) -> Option<usize> {
//     let mut cells = cells.to_owned();
//     let mut step_count = 0;
//     let mut last_layer = vec![start.x + field_size * start.y];
//     loop {
//         step_count += 1;
//         let new_layer: Vec<i32> = last_layer
//             .par_iter()
//             .flat_map(|n| {
//                 let mut new_nodes = Vec::<i32>::with_capacity(4);
//                 // test move RIGHT
//                 if test_right(&cells, *n as usize, field_size as usize) {
//                     new_nodes.push(n + 1);
//                 }
//                 // test move DOWN
//                 if test_down(&cells, *n as usize, field_size as usize) {
//                     new_nodes.push(n + field_size)
//                 }
//                 // test move LEFT
//                 if test_left(&cells, *n as usize, field_size as usize) {
//                     new_nodes.push(n - 1)
//                 }
//                 // test move UP
//                 if test_up(&cells, *n as usize, field_size as usize) {
//                     new_nodes.push(n - field_size)
//                 }
//                 return new_nodes;
//             })
//             .collect();
//         if new_layer.is_empty() {
//             return None;
//         }
//         // keep each cell only once
//         let new_layer = new_layer.into_iter().unique().collect_vec();
//         // check if we found the target
//         if let Some(_found_goal) = new_layer
//             .par_iter()
//             .find_any(|n| return **n == goal.x + goal.y * field_size)
//         {
//             return Some(step_count);
//         }
//         // mark the new layer as visited
//         new_layer.iter().for_each(|n| cells[*n as usize] = false);
//         // add the new layer onto the stack
//         last_layer = new_layer;
//     }
// }

// /// Compute the distance between players (columns) and mushrooms (rows).
// ///
// /// Will panic if some point can't be reached (assumes all non-obstacle cells are accessible).
// pub fn get_p2m_dist_matrix(field: &Field) -> Vec<Vec<usize>> {
//     let player_count = field.players.len();
//     let mush_count = field.mushrooms.len();
//     let mut mat: Vec<Vec<usize>> = vec![vec![0; player_count]; mush_count];
//     for p in 0..player_count {
//         (0..mush_count).for_each(|m| {
//             let start = field.players[p];
//             let goal = field.mushrooms[m];
//             let dist = match find_min_path_dist(field.size as i32, &field.cells, &start, &goal) {
//                 Some(d) => d,
//                 None => {
//                     unreachable!("can't find path from {start:?} to {goal:?}");
//                 }
//             };
//             mat[m][p] = dist;
//         });
//     }
//     return mat;
// }

// /// Compute the distance between all mushrooms
// ///
// /// Will panic if some point can't be reached (assumes all non-obstacle cells are accessible).
// pub fn get_m2m_dist_matrix(field: &Field) -> Vec<Vec<usize>> {
//     let mush_count = field.mushrooms.len();
//     let mut mat: Vec<Vec<usize>> = vec![vec![0; mush_count]; mush_count];
//     for i in 0..mush_count {
//         for j in i + 1..mush_count {
//             let start = field.mushrooms[i];
//             let goal = field.mushrooms[j];
//             let dist = match find_min_path_dist(field.size as i32, &field.cells, &start, &goal) {
//                 Some(d) => d,
//                 None => unreachable!("can't find path from {start:?} to {goal:?}"),
//             };
//             mat[i][j] = dist;
//             mat[j][i] = dist;
//         }
//     }
//     return mat;
// }

/// Find player-to-mush shortest paths using BFS algorithm
pub fn bfs_p2m(field: &Field) -> Vec<Vec<Vec<Direction>>> {
    /// get the path from given direction matrix and ending point
    fn get_path(dirmat: &[u8], end: usize, field_size: usize) -> Vec<Direction> {
        let mut end2start: Vec<Direction> = vec![];
        let mut current = end;

        while dirmat[current] != 5 {
            match dirmat[current] {
                1 => {
                    // move right
                    current += 1;
                    end2start.push(Direction::Right);
                }
                2 => {
                    // move up
                    current -= field_size;
                    end2start.push(Direction::Up);
                }
                3 => {
                    // move left
                    current -= 1;
                    end2start.push(Direction::Left);
                }
                4 => {
                    // move down
                    current += field_size;
                    end2start.push(Direction::Down);
                }
                _ => unreachable!(),
            }
        }
        // now we have path end -> start, so reverse it as well
        let start2end = end2start
            .iter()
            .rev()
            .map(|d| match d {
                Direction::Up => return Direction::Down,
                Direction::Down => return Direction::Up,
                Direction::Left => return Direction::Right,
                Direction::Right => return Direction::Left,
            })
            .collect();
        return start2end;
    }

    let mush_count = field.mushrooms.len();

    // final matrix with paths from player to mushroom
    let path_matrix = field
        .players
        .par_iter()
        .map(|player| -> Vec<Vec<Direction>> {
            let origin_flattened = player.x + field.size as i32 * player.y; // flattened coordinate of the origin mushroom

            // direction matrix for BFS flood-fill with backtracking
            let mut mat: Vec<u8> = vec![0; field.size * field.size]; // flattened grid, row-major
            // INFO: directions:
            // 0 = not visited yet,
            // 1 = right,
            // 2 = up,
            // 3 = left,
            // 4 = down,
            // 5 = beginning point

            mat[origin_flattened as usize] = 5;
            let mut open: Vec<usize> = vec![origin_flattened as usize]; // list of coordinates that are waiting for expansion

            // prepare the paths-buffer for this player
            let mut paths: Vec<Vec<Direction>> = vec![Vec::new(); mush_count];

            while !open.is_empty() {
                let new_open: Vec<usize> = open
                    .iter()
                    .flat_map(|c| {
                        let mut inner_buf: Vec<usize> = Vec::with_capacity(4);
                        // check above
                        if *c >= field.size {
                            let above = c - field.size;
                            if mat[above] == 0 && field.cells[above] {
                                mat[above] = 4; // 4 means "move down" (for backtracking)
                                inner_buf.push(above);
                            }
                        }
                        // check below
                        if *c < field.size * field.size - field.size {
                            let below = c + field.size;
                            if mat[below] == 0 && field.cells[below] {
                                mat[below] = 2; // 2 mean "move up" (for backtracking)
                                inner_buf.push(below);
                            }
                        }
                        // check right
                        if c % field.size != field.size - 1 {
                            let right = c + 1;
                            if mat[right] == 0 && field.cells[right] {
                                mat[right] = 3; // 3 means "move left" (for backtracking)
                                inner_buf.push(right)
                            }
                        }
                        // check left
                        if c % field.size != 0 {
                            let left = c - 1;
                            if mat[left] == 0 && field.cells[left] {
                                mat[left] = 1; // 1 means "move right" (for backtracking)
                                inner_buf.push(left)
                            }
                        }
                        return inner_buf;
                    })
                    .sorted_unstable()
                    .dedup()
                    .collect();

                // check if there are mushrooms left to find
                for (mush_idx, mush_coord) in field.mushrooms.iter().enumerate() {
                    let mush_coord_flattened =
                        mush_coord.x as usize + mush_coord.y as usize * field.size;
                    if new_open.contains(&mush_coord_flattened) {
                        // we found a mushroom in the last iteration
                        let path = get_path(&mat, mush_coord_flattened, field.size);
                        paths[mush_idx] = path;
                        // path_matrix[origin_mush_idx][*mush_idx] = path;
                    }
                }

                // check for early stopping
                if paths.iter().any(|p| return p.is_empty()) {
                    open = new_open;
                } else {
                    // all mushrooms were found already => skip the rest
                    open = vec![];
                }
            }
            return paths;
        })
        .collect();
    return path_matrix;
}

/// Find mush-to-mush shortest paths using BFS algorithm
pub fn bfs_m2m(field: &Field) -> Vec<Vec<Vec<Direction>>> {
    let mush_count = field.mushrooms.len();

    fn get_paths(dirmat: &[u8], end: usize, field_size: usize) -> (Vec<Direction>, Vec<Direction>) {
        let mut end2start: Vec<Direction> = vec![];
        let mut current = end;

        while dirmat[current] != 5 {
            match dirmat[current] {
                1 => {
                    // move right
                    current += 1;
                    end2start.push(Direction::Right);
                }
                2 => {
                    // move up
                    current -= field_size;
                    end2start.push(Direction::Up);
                }
                3 => {
                    // move left
                    current -= 1;
                    end2start.push(Direction::Left);
                }
                4 => {
                    // move down
                    current += field_size;
                    end2start.push(Direction::Down);
                }
                _ => unreachable!(),
            }
        }
        // now we have path end -> start, so reverse it as well
        let start2end = end2start
            .iter()
            .rev()
            .map(|d| match d {
                Direction::Up => return Direction::Down,
                Direction::Down => return Direction::Up,
                Direction::Left => return Direction::Right,
                Direction::Right => return Direction::Left,
            })
            .collect();
        return (start2end, end2start);
    }

    // final matrix with paths from mushroom to mushroom
    let mut path_matrix: Vec<Vec<Vec<Direction>>> = vec![vec![Vec::new(); mush_count]; mush_count];

    #[allow(clippy::needless_range_loop)]
    for i in 0..mush_count {
        // on the diagonal, fill up the paths with some dummy paths to avoid confusing the fill-checks
        path_matrix[i][i] = vec![Direction::Right];
    }

    for target_row_idx in 0..mush_count {
        let origin_mush = field.mushrooms[target_row_idx]; // the starting mushroom
        let origin_mush_idx = target_row_idx; // the starting mushroom index
        let origin_flattened = origin_mush.x + field.size as i32 * origin_mush.y; // flattened coordinate of the origin mushroom

        let target_mushrooms: Vec<(usize, &Coord)> = field
            .mushrooms
            .iter()
            .enumerate()
            .filter(|(mush_idx, _c)| {
                // drop the origin mushroom
                return *mush_idx != target_row_idx;
            })
            .filter(|(mush_idx, _c)| {
                // drop mushrooms that are fully mapped
                let has_some_empty = path_matrix[*mush_idx].iter().any(|p| return p.is_empty());
                return has_some_empty;
            })
            .collect();
        if target_mushrooms.is_empty() {
            break;
        }

        let mut mat: Vec<u8> = vec![0; field.size * field.size]; // flattened grid, row-major
        // INFO: directions:
        // 0 = not visited yet,
        // 1 = right,
        // 2 = up,
        // 3 = left,
        // 4 = down,
        // 5 = beginning point

        mat[origin_flattened as usize] = 5;
        let mut open: Vec<usize> = vec![origin_flattened as usize]; // list of coordinates that are waiting for expansion

        while !open.is_empty() {
            let new_open: Vec<usize> = open
                .iter()
                .flat_map(|c| {
                    let mut inner_buf: Vec<usize> = Vec::with_capacity(4);
                    // check above
                    if *c >= field.size {
                        let above = c - field.size;
                        if mat[above] == 0 && field.cells[above] {
                            mat[above] = 4; // 4 means "move down" (for backtracking)
                            inner_buf.push(above);
                        }
                    }
                    // check below
                    if *c < field.size * field.size - field.size {
                        let below = c + field.size;
                        if mat[below] == 0 && field.cells[below] {
                            mat[below] = 2; // 2 mean "move up" (for backtracking)
                            inner_buf.push(below);
                        }
                    }
                    // check right
                    if c % field.size != field.size - 1 {
                        let right = c + 1;
                        if mat[right] == 0 && field.cells[right] {
                            mat[right] = 3; // 3 means "move left" (for backtracking)
                            inner_buf.push(right)
                        }
                    }
                    // check left
                    if c % field.size != 0 {
                        let left = c - 1;
                        if mat[left] == 0 && field.cells[left] {
                            mat[left] = 1; // 1 means "move right" (for backtracking)
                            inner_buf.push(left)
                        }
                    }

                    return inner_buf;
                })
                .sorted_unstable()
                .dedup()
                .collect();

            // check if there are mushrooms left to find
            for (idx, coord) in &target_mushrooms {
                let flattened = coord.x as usize + coord.y as usize * field.size;
                if new_open.contains(&flattened) {
                    // we found a target mushroom in the last iteration
                    let (path, reverse_path) = get_paths(&mat, flattened, field.size);
                    path_matrix[origin_mush_idx][*idx] = path;

                    // and fill the reverse path as well
                    path_matrix[*idx][origin_mush_idx] = reverse_path;
                }
            }

            // check for early stopping
            if path_matrix[target_row_idx]
                .iter()
                .any(|p| return p.is_empty())
            {
                open = new_open;
            } else {
                // all other mushrooms were found already => skip the rest
                open = vec![];
            }
        }
    }

    return path_matrix;
}

/// Run flood-fill algorithm to check if all cells are accessible
pub fn is_field_accessible(cells: &[bool], size: usize) -> bool {
    let mut cells: Vec<bool> = cells.into();
    let field_size: usize = size;
    let Some(start) = cells.iter().position(|x| return *x) else {
        warn!("the whole field is made out of obstacles");
        return false;
    };
    let mut last_layer: Vec<usize> = vec![start];
    loop {
        if last_layer.is_empty() {
            // if all cells are 'false', then all cells have been accessed (or obstacles)
            return cells.iter().all(|b| return !b);
        }
        let new_layer: Vec<usize> = last_layer
            .par_iter()
            .flat_map(|n: &usize| -> Vec<usize> {
                let mut new_nodes = Vec::<usize>::with_capacity(4);
                // test move RIGHT
                if test_right(&cells, *n, field_size) {
                    new_nodes.push(n + 1);
                }
                // test move DOWN
                if test_down(&cells, *n, field_size) {
                    new_nodes.push(n + field_size)
                }
                // test move LEFT
                if test_left(&cells, *n, field_size) {
                    new_nodes.push(n - 1)
                }
                // test move UP
                if test_up(&cells, *n, field_size) {
                    new_nodes.push(n - field_size)
                }
                return new_nodes;
            })
            .collect();
        // keep each cell only once
        let new_layer = new_layer.into_iter().unique().collect_vec();
        // mark the new layer as visited
        new_layer.iter().for_each(|n| cells[*n] = false);
        // add the new layer onto the stack
        last_layer = new_layer;
    }
}

// /// Returns a flat vector of distances (row-major order).
// /// -1 means unreachable.
// pub fn bfs_distances(size: usize, cells: &[bool], start: Coord) -> Vec<i32> {
//     let n = size * size;
//     let mut dist: Vec<i32> = vec![-1; n];
//
//     let start_idx = start.y as usize * size + start.x as usize;
//     dist[start_idx] = 0;
//
//     let mut q: VecDeque<usize> = VecDeque::with_capacity(n);
//
//     q.push_back(start_idx);
//
//     while let Some(idx) = q.pop_front() {
//         let d = dist[idx];
//         let x = idx % size;
//         let y = idx / size;
//
//         // 4-neighborhood
//         if x + 1 < size {
//             let n_idx = y * size + (x + 1);
//             if cells[n_idx] && dist[n_idx] == -1 {
//                 dist[n_idx] = d + 1;
//                 q.push_back(n_idx);
//             }
//         }
//         if x > 0 {
//             let n_idx = y * size + (x - 1);
//             if cells[n_idx] && dist[n_idx] == -1 {
//                 dist[n_idx] = d + 1;
//                 q.push_back(n_idx);
//             }
//         }
//         if y + 1 < size {
//             let n_idx = (y + 1) * size + x;
//             if cells[n_idx] && dist[n_idx] == -1 {
//                 dist[n_idx] = d + 1;
//                 q.push_back(n_idx);
//             }
//         }
//         if y > 0 {
//             let n_idx = (y - 1) * size + x;
//             if cells[n_idx] && dist[n_idx] == -1 {
//                 dist[n_idx] = d + 1;
//                 q.push_back(n_idx);
//             }
//         }
//     }
//
//     return dist;
// }

// /// Find the paths for given split for each player.
// /// Return a vector of coordinate sequences for each player
// pub fn pathfind_split(split: &[(Vec<usize>, usize)], field: &Field) -> Vec<Vec<Coord>> {
//     return split
//         .iter()
//         .enumerate()
//         .map(|(player_idx, (mushrooms, _cost))| -> Vec<Coord> {
//             let player_coord = &field.players[player_idx];
//             let checkpoints = std::iter::once(*player_coord).chain(
//                 mushrooms
//                     .iter()
//                     .map(|m| return &field.mushrooms[*m])
//                     .cloned(),
//             );
//             let path = checkpoints
//                 .tuple_windows()
//                 .flat_map(|(a, b)| return find_min_path(field.size as i32, &field.cells, &a, &b))
//                 .dedup() // remove consecutive same coordinates
//                 .collect_vec();
//             return path;
//         })
//         .collect_vec();
// }

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

    /// Check bfs_p2m and bfs_m2m against the reference BFS
    fn check_paths(f: &Field) {
        let p2m = bfs_p2m(f);
        assert_eq!(p2m.len(), f.players.len());
        for (pi, row) in p2m.iter().enumerate() {
            assert_eq!(row.len(), f.mushrooms.len());
            for (mi, path) in row.iter().enumerate() {
                let (from, to) = (f.players[pi], f.mushrooms[mi]);
                assert_eq!(Some(path.len()), ref_dist(f, from, to));
                assert_valid_path(f, from, to, path);
            }
        }

        let m2m = bfs_m2m(f);
        assert_eq!(m2m.len(), f.mushrooms.len());
        for (i, row) in m2m.iter().enumerate() {
            assert_eq!(row.len(), f.mushrooms.len());
            for (j, path) in row.iter().enumerate() {
                if i == j {
                    continue;
                }
                let (from, to) = (f.mushrooms[i], f.mushrooms[j]);
                assert_eq!(Some(path.len()), ref_dist(f, from, to));
                assert_eq!(path.len(), m2m[j][i].len(), "m2m not symmetric");
                assert_valid_path(f, from, to, path);
            }
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
    fn paths_with_detour() {
        // the player has to go around the wall to reach the mushrooms
        let f = field("P X M -\n- X X -\n- - - -\nM X - -\n");
        let p2m = bfs_p2m(&f);
        assert_eq!(p2m[0][0].len(), 8);
        assert_eq!(p2m[0][1].len(), 3);
        check_paths(&f);
    }

    #[test]
    fn paths_on_hand_made_maps() {
        check_paths(&field("P M\n- -\n"));
        check_paths(&field("P - - M\nX X - X\nM - - -\nP X X M\n"));
        check_paths(&field(
            "P - X - M\n- X - - X\n- - - X -\nX X - - -\nM - - X P\n",
        ));
    }

    #[test]
    fn paths_on_generated_maps() {
        for size in [5, 8, 13] {
            for _ in 0..10 {
                let f = generate_field(size, size * size / 3, 3, 6).unwrap();
                check_paths(&f);
            }
        }
    }
}

use itertools::Itertools;
use log::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{Coord, Field};

/// Find a shortest path using parallelized BFS algorithm
pub fn find_min_path(field_size: usize, cells: &[bool], start: &Coord, goal: &Coord) -> Vec<Coord> {
    #[derive(Debug)]
    struct Node {
        /// flattened XY coordinate of the cell
        coord: usize,
        /// index of the parent node in the previous layer/step
        prev_coord: Option<usize>,
    }

    let mut cells = cells.to_owned();
    let mut layers: Vec<Vec<Node>> = vec![vec![Node {
        //x: start.x,
        //y: start.y,
        coord: start.x + field_size * start.y,
        prev_coord: None,
    }]];

    loop {
        let last_layer: &Vec<Node> = layers.last().unwrap();
        let new_layer: Vec<Node> = last_layer
            .par_iter()
            .flat_map(|n: &Node| -> Vec<Node> {
                let mut new_nodes = Vec::<Node>::with_capacity(4);
                // test move RIGHT
                if ((n.coord + 1) % field_size != 0) && cells[n.coord + 1] {
                    new_nodes.push(Node {
                        coord: n.coord + 1,
                        prev_coord: Some(n.coord),
                    });
                }
                // test move DOWN
                if n.coord < field_size * field_size - field_size && cells[n.coord + field_size] {
                    new_nodes.push(Node {
                        coord: n.coord + field_size,
                        prev_coord: Some(n.coord),
                    })
                }
                // test move LEFT
                if n.coord % field_size != 0 && cells[n.coord - 1] {
                    new_nodes.push(Node {
                        coord: n.coord - 1,
                        prev_coord: Some(n.coord),
                    })
                }
                // test move UP
                if n.coord >= field_size && cells[n.coord - field_size] {
                    new_nodes.push(Node {
                        coord: n.coord - field_size,
                        prev_coord: Some(n.coord),
                    })
                }
                return new_nodes;
            })
            .collect();

        if new_layer.is_empty() {
            println!("can't find the mushroom");
            return vec![];
        }

        // keep each cell only once
        let new_layer = new_layer
            .into_iter()
            .unique_by(|n| return n.coord)
            .collect_vec();

        // check if we found the target
        if let Some(found_goal) = new_layer
            .par_iter()
            .find_any(|n| return n.coord == goal.x + goal.y * field_size)
        {
            let mut path: Vec<Coord> = vec![Coord {
                x: found_goal.coord % field_size,
                y: (found_goal.coord - (found_goal.coord % field_size)) / field_size,
            }];
            let mut prev_coord = found_goal.prev_coord.unwrap();
            for i in (1..layers.len()).rev() {
                if let Some(prev_node) = layers[i].iter().find(|n| return n.coord == prev_coord) {
                    path.push(Coord {
                        x: prev_node.coord % field_size,
                        y: (prev_node.coord - (prev_node.coord % field_size)) / field_size,
                    });
                    prev_coord = prev_node.prev_coord.unwrap();
                }
            }
            path.push(*start);
            return path.into_iter().rev().collect_vec();
        }

        // mark the new layer as visited
        new_layer.iter().for_each(|n| cells[n.coord] = false);

        // add the new layer onto the stack
        layers.push(new_layer);
    }
}

/// Find shortest path distance using parallelized BFS algorithm
pub fn find_min_path_dist(
    field_size: usize,
    cells: &[bool],
    start: &Coord,
    goal: &Coord,
) -> Option<usize> {
    let mut cells = cells.to_owned();
    let mut step_count = 0;
    let mut last_layer: Vec<usize> = vec![start.x + field_size * start.y];
    loop {
        step_count += 1;
        let new_layer: Vec<usize> = last_layer
            .par_iter()
            .flat_map(|n: &usize| -> Vec<usize> {
                let mut new_nodes = Vec::<usize>::with_capacity(4);
                // test move RIGHT
                if ((n + 1) % field_size != 0) && cells[n + 1] {
                    new_nodes.push(n + 1);
                }
                // test move DOWN
                if *n < field_size * field_size - field_size && cells[n + field_size] {
                    new_nodes.push(n + field_size)
                }
                // test move LEFT
                if n % field_size != 0 && cells[n - 1] {
                    new_nodes.push(n - 1)
                }
                // test move UP
                if *n >= field_size && cells[n - field_size] {
                    new_nodes.push(n - field_size)
                }
                return new_nodes;
            })
            .collect();
        if new_layer.is_empty() {
            return None;
        }
        // keep each cell only once
        let new_layer = new_layer.into_iter().unique().collect_vec();
        // check if we found the target
        if let Some(_found_goal) = new_layer
            .par_iter()
            .find_any(|n| return **n == goal.x + goal.y * field_size)
        {
            return Some(step_count);
        }
        // mark the new layer as visited
        new_layer.iter().for_each(|n| cells[*n] = false);
        // add the new layer onto the stack
        last_layer = new_layer;
    }
}

/// Compute the distance between players (columns) and mushrooms (rows).
///
/// Will panic if some point can't be reached (assumes all non-obstacle cells are accessible).
pub fn get_p2m_dist_matrix(field: &Field) -> Vec<Vec<usize>> {
    let player_count = field.players.len();
    let mush_count = field.mushrooms.len();
    let mut mat: Vec<Vec<usize>> = vec![vec![0; player_count]; mush_count];
    for p in 0..player_count {
        (0..mush_count).for_each(|m| {
            let start = field.players[p];
            let goal = field.mushrooms[m];
            let dist = match find_min_path_dist(field.size, &field.cells, &start, &goal) {
                Some(d) => d,
                None => {
                    unreachable!("can't find path from {start:?} to {goal:?}");
                }
            };
            mat[m][p] = dist;
        });
    }
    return mat;
}

/// Compute the distance between all mushrooms
///
/// Will panic if some point can't be reached (assumes all non-obstacle cells are accessible).
pub fn get_m2m_dist_matrix(field: &Field) -> Vec<Vec<usize>> {
    let mush_count = field.mushrooms.len();
    let mut mat: Vec<Vec<usize>> = vec![vec![0; mush_count]; mush_count];
    for i in 0..mush_count {
        for j in i + 1..mush_count {
            let start = field.mushrooms[i];
            let goal = field.mushrooms[j];
            let dist = match find_min_path_dist(field.size, &field.cells, &start, &goal) {
                Some(d) => d,
                None => unreachable!("can't find path from {start:?} to {goal:?}"),
            };
            mat[i][j] = dist;
            mat[j][i] = dist;
        }
    }
    return mat;
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
                if ((n + 1) % field_size != 0) && cells[n + 1] {
                    new_nodes.push(n + 1);
                }
                // test move DOWN
                if *n < field_size * field_size - field_size && cells[n + field_size] {
                    new_nodes.push(n + field_size)
                }
                // test move LEFT
                if n % field_size != 0 && cells[n - 1] {
                    new_nodes.push(n - 1)
                }
                // test move UP
                if *n >= field_size && cells[n - field_size] {
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

/// Find the paths for given split for each player.
/// Return a vector of coordinate sequences for each player
pub fn pathfind_split(split: &[(Vec<usize>, usize)], field: &Field) -> Vec<Vec<Coord>> {
    return split
        .iter()
        .enumerate()
        .map(|(player_idx, (mushrooms, _cost))| -> Vec<Coord> {
            let player_coord = &field.players[player_idx];
            let checkpoints = std::iter::once(*player_coord).chain(
                mushrooms
                    .iter()
                    .map(|m| return &field.mushrooms[*m])
                    .cloned(),
            );
            let path = checkpoints
                .tuple_windows()
                .flat_map(|(a, b)| return find_min_path(field.size, &field.cells, &a, &b))
                .dedup() // remove consecutive same coordinates
                .collect_vec();
            return path;
        })
        .collect_vec();
}

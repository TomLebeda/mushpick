use itertools::Itertools;
use log::*;
use rand::{Rng, rngs::ThreadRng, seq::SliceRandom};

use crate::{Coord, Field, utils::Direction};

/// generate a new random field using provided parameters
pub fn generate_field(
    size: usize,
    walls: usize,
    players: usize,
    mushrooms: usize,
) -> Result<Field, String> {
    trace!("checking configuration");
    if walls > size * size {
        return Err(format!("more walls ({walls}) than cells ({})", size * size));
    }
    if players + mushrooms > size * size - walls {
        return Err(String::from(
            "not enough empty cells for players and mushrooms",
        ));
    }

    trace!("generating maze");
    let mut map: Vec<bool> = generate_maze(size, walls as i32).concat();

    trace!("correcting wall density");
    let wall_count = map.iter().filter(|b| return !(**b)).count();
    let walls_to_change: i32 = wall_count as i32 - walls as i32;
    let mut rng = rand::rng();
    if walls_to_change > 0 {
        // we need to remove more walls
        let mut wall_indicies: Vec<usize> = map
            .iter()
            .enumerate()
            .filter(|(_i, b)| return !(**b))
            .map(|(i, _b)| return i)
            .collect();
        wall_indicies.shuffle(&mut rng);
        for i in 0..walls_to_change {
            map[wall_indicies[i as usize]] = true;
        }
    }

    trace!("locating free cells");
    let mut available_indices: Vec<usize> = map
        .iter()
        .enumerate()
        .filter_map(|(i, b)| match b {
            true => return Some(i),
            false => return None,
        })
        .collect();
    available_indices.shuffle(&mut rng);

    trace!("placing players");
    let player_coords = available_indices
        .get(0..players)
        .unwrap()
        .iter()
        .map(|i| {
            return Coord {
                x: (i % size) as i32,
                y: ((i - (i % size)) / size) as i32,
            };
        })
        .collect_vec();

    trace!("placing mushrooms");
    let mush_coords = available_indices
        .get(players..players + mushrooms)
        .unwrap()
        .iter()
        .map(|i| {
            return Coord {
                x: (i % size) as i32,
                y: ((i - (i % size)) / size) as i32,
            };
        })
        .collect_vec();

    let field = Field {
        size,
        mushrooms: mush_coords,
        players: player_coords,
        cells: map,
    };
    trace!("field constructed");
    return Ok(field);
}

/// generates a random perfect maze in a square grid of given size using recursive backtracking
fn generate_maze(size: usize, target_wall_count: i32) -> Vec<Vec<bool>> {
    let mut grid = vec![vec![false; size]; size];
    let mut rng = rand::rng();

    // Helper function to check if a cell is within bounds
    fn in_bounds(x: isize, y: isize, size: usize) -> bool {
        return x >= 0 && x < size as isize && y >= 0 && y < size as isize;
    }

    // Function to count neighboring cells that are paths (true)
    fn count_neighbors(grid: &[Vec<bool>], x: usize, y: usize) -> usize {
        let directions = Direction::get_all_dirs()
            .iter()
            .map(|d| return d.as_diff())
            .collect_vec();
        let mut count = 0;

        for (dx, dy) in directions.iter() {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if in_bounds(nx as isize, ny as isize, grid.len()) && grid[ny][nx] {
                count += 1;
            }
        }

        return count;
    }

    /// Non-recursive method for carving out path in the grid
    fn carve_path(
        target_wall_count: i32,
        grid: &mut [Vec<bool>],
        start_x: usize,
        start_y: usize,
        size: usize,
        rng: &mut ThreadRng,
    ) {
        let mut wall_count: i32 = grid.iter().flatten().count() as i32;
        let mut stack = Vec::new();
        stack.push((start_x, start_y));

        while let Some((x, y)) = stack.pop() {
            if wall_count == target_wall_count {
                // early stopping for dense mazes
                break;
            }
            if !in_bounds(x as isize, y as isize, size) || grid[y][x] {
                continue;
            }

            let neighbor_count = count_neighbors(grid, x, y);
            if neighbor_count > 1 {
                continue; // Only carve if it has exactly 1 neighbor
            }

            grid[y][x] = true;
            wall_count -= 1; // we carved out a single wall

            // Randomize directions
            let mut directions = Direction::get_all_dirs()
                .iter()
                .map(|d| return d.as_diff())
                .collect_vec();
            directions.shuffle(rng);

            for (dx, dy) in directions {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if in_bounds(nx as isize, ny as isize, size) {
                    let nx_u = nx as usize;
                    let ny_u = ny as usize;
                    if !grid[ny_u][nx_u] {
                        stack.push((nx_u, ny_u));
                    }
                }
            }
        }
    }

    // Recursive backtracking to carve paths
    #[allow(dead_code)]
    fn carve_path_recursive(
        grid: &mut Vec<Vec<bool>>,
        x: usize,
        y: usize,
        size: usize,
        rng: &mut ThreadRng,
    ) {
        grid[y][x] = true; // Mark the current cell as part of the path

        // Directions shuffled for randomness
        let mut directions = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        directions.shuffle(rng);

        // Try all directions
        for (dx, dy) in directions.iter() {
            let nx = (x as isize + dx) as usize;
            let ny = (y as isize + dy) as usize;

            // Check if the neighbor is within bounds and is a wall
            if in_bounds(nx as isize, ny as isize, size) && !grid[ny][nx] {
                let neighbor_count = count_neighbors(grid, nx, ny);

                // Only carve a path if the cell has exactly 1 neighboring path
                if neighbor_count == 1 {
                    grid[ny][nx] = true; // Mark the cell as a path
                    carve_path_recursive(grid, nx, ny, size, rng); // Recur to continue carving
                }
            }
        }
    }

    // Start from a random position (1,1) avoiding outer walls
    // carve_path_recursive(&mut grid, 1, 1, size, &mut rng);
    carve_path(target_wall_count, &mut grid, 1, 1, size, &mut rng);

    return grid;
}

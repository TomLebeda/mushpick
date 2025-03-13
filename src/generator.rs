use itertools::Itertools;
use rand::{
    rngs::ThreadRng,
    seq::{IndexedRandom, SliceRandom},
};

use crate::{
    Coord, Field, pathfinding::is_field_accessible, test_down, test_left, test_right, test_up,
};

/// generate a new random field using provided parameters
pub fn generate_field(
    size: usize,
    walls: usize,
    players: usize,
    mushrooms: usize,
) -> Result<Field, String> {
    if walls > size * size {
        return Err(format!("more walls ({walls}) than cells ({})", size * size));
    }
    if players + mushrooms > size * size - walls {
        return Err(String::from(
            "not enough empty cells for players and mushrooms",
        ));
    }
    let mut map: Vec<bool> = generate_maze(size).concat();
    let wall_count = map.iter().filter(|b| return !(**b)).count();
    let walls_to_change: i32 = wall_count as i32 - walls as i32;

    let mut rng = rand::rng();
    if walls_to_change > 0 {
        // we need to remove some walls
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
    if walls_to_change < 0 {
        let walls_to_fill = -walls_to_change;
        for _ in 0..walls_to_fill {
            let dead_end_indicies: Vec<usize> = map
                .iter()
                .enumerate()
                .filter(|(i, b)| -> bool {
                    // keep only the cells that are free and have exactly one free neighbor (= dead ends)
                    let free = **b;
                    if !free {
                        return false;
                    }
                    let directions = [test_left, test_right, test_up, test_down];
                    let free_neighbor_count = directions
                        .iter()
                        .filter(|&dir| return dir(&map, *i, size))
                        .count();
                    return free_neighbor_count == 1;
                })
                .map(|(i, _b)| return i)
                .collect();
            let dead_end_to_fill = dead_end_indicies.choose(&mut rng).unwrap();
            map[*dead_end_to_fill] = false;
        }
    }
    let mut available_indices: Vec<usize> = map
        .iter()
        .enumerate()
        .filter_map(|(i, b)| match b {
            true => return Some(i),
            false => return None,
        })
        .collect();

    available_indices.shuffle(&mut rng);
    let player_coords = available_indices
        .get(0..players)
        .unwrap()
        .iter()
        .map(|i| {
            return Coord {
                x: i % size,
                y: (i - (i % size)) / size,
            };
        })
        .collect_vec();
    let mush_coords = available_indices
        .get(players..players + mushrooms)
        .unwrap()
        .iter()
        .map(|i| {
            return Coord {
                x: i % size,
                y: (i - (i % size)) / size,
            };
        })
        .collect_vec();

    let field = Field {
        size,
        mushrooms: mush_coords,
        players: player_coords,
        cells: map,
    };
    return Ok(field);
}

/// generates a random perfect maze in a square grid of given size using recursive backtracking
fn generate_maze(size: usize) -> Vec<Vec<bool>> {
    let mut grid = vec![vec![false; size]; size];
    let mut rng = rand::rng();

    // Helper function to check if a cell is within bounds
    fn in_bounds(x: isize, y: isize, size: usize) -> bool {
        return x >= 0 && x < size as isize && y >= 0 && y < size as isize;
    }

    // Function to count neighboring cells that are paths (true)
    fn count_neighbors(grid: &[Vec<bool>], x: usize, y: usize) -> usize {
        let directions = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        let mut count = 0;

        for (dx, dy) in directions.iter() {
            let nx = (x as isize + dx) as usize;
            let ny = (y as isize + dy) as usize;

            if in_bounds(nx as isize, ny as isize, grid.len()) && grid[ny][nx] {
                count += 1;
            }
        }

        return count;
    }

    // Recursive backtracking to carve paths
    fn carve_path(grid: &mut Vec<Vec<bool>>, x: usize, y: usize, size: usize, rng: &mut ThreadRng) {
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
                    carve_path(grid, nx, ny, size, rng); // Recur to continue carving
                }
            }
        }
    }

    // Start from a random position (1,1) avoiding outer walls
    carve_path(&mut grid, 1, 1, size, &mut rng);

    return grid;
}

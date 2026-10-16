use itertools::Itertools;
use log::*;
use rand::{rngs::ThreadRng, seq::SliceRandom};

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
    let mut walls_to_remove = walls_to_change.max(0) as usize;
    // remove walls in rounds: a wall enclosed by other walls becomes removable
    // once one of its neighbors is removed in an earlier round
    while walls_to_remove > 0 {
        let mut wall_indicies: Vec<usize> = map
            .iter()
            .enumerate()
            .filter(|(_idx, free)| return !(**free)) // keep only those that are not free
            .filter(|(idx, _free)| {
                // keep only those that touch some other free cells, so the field stays connected
                let idx = *idx;
                let left_edge = idx % size == 0;
                let right_edge = idx % size == size - 1;
                let top_edge = idx < size;
                let bottom_edge = idx >= size * size - size;
                return (!right_edge && map[idx + 1])
                    || (!left_edge && map[idx - 1])
                    || (!bottom_edge && map[idx + size])
                    || (!top_edge && map[idx - size]);
            })
            .map(|(idx, _free)| return idx)
            .collect();
        if wall_indicies.is_empty() {
            return Err(String::from("maze has no free cells to grow from"));
        }
        wall_indicies.shuffle(&mut rng);
        wall_indicies.truncate(walls_to_remove);
        for &idx in &wall_indicies {
            map[idx] = true;
        }
        walls_to_remove -= wall_indicies.len();
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

    // Start from a random position (1,1) avoiding outer walls
    carve_path(target_wall_count, &mut grid, 1, 1, size, &mut rng);

    return grid;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathfinding::is_field_accessible;

    #[test]
    fn generated_field_matches_parameters() {
        for (size, walls, players, mushrooms) in [(5, 5, 1, 1), (10, 30, 3, 8), (20, 150, 4, 20)] {
            for _ in 0..10 {
                let f = generate_field(size, walls, players, mushrooms).unwrap();
                assert_eq!(f.size, size);
                assert_eq!(f.cells.len(), size * size);
                assert_eq!(f.cells.iter().filter(|c| return !**c).count(), walls);
                assert_eq!(f.players.len(), players);
                assert_eq!(f.mushrooms.len(), mushrooms);
            }
        }
    }

    #[test]
    fn generated_entities_are_on_distinct_free_cells() {
        for _ in 0..20 {
            let f = generate_field(10, 40, 3, 10).unwrap();
            let all = f.players.iter().chain(&f.mushrooms).collect_vec();
            for c in &all {
                assert!(c.x >= 0 && c.y >= 0 && c.x < 10 && c.y < 10);
                assert!(
                    f.cells[(c.y * 10 + c.x) as usize],
                    "entity on a wall at {c}"
                );
            }
            assert_eq!(
                all.iter().map(|c| return (c.x, c.y)).unique().count(),
                all.len()
            );
        }
    }

    #[test]
    fn generated_field_is_accessible() {
        for walls in [10, 30, 45, 60] {
            for _ in 0..10 {
                let f = generate_field(10, walls, 2, 3).unwrap();
                assert!(is_field_accessible(&f.cells, f.size), "walls={walls}");
            }
        }
    }

    #[test]
    fn generates_field_without_walls() {
        for size in [5, 10] {
            for _ in 0..20 {
                let f = generate_field(size, 0, 1, 1).unwrap();
                assert!(f.cells.iter().all(|c| return *c));
            }
        }
    }

    #[test]
    fn rejects_impossible_configurations() {
        assert!(generate_field(5, 26, 1, 1).is_err());
        assert!(generate_field(5, 20, 3, 3).is_err());
    }
}

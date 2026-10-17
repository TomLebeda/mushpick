//! End-to-end tests driving the `mushpick` binary

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Run the binary with logging disabled
fn run(args: &[&str]) -> Output {
    return Command::new(env!("CARGO_BIN_EXE_mushpick"))
        .args(["-l", "off"])
        .args(args)
        .output()
        .expect("failed to run mushpick");
}

fn stdout(out: &Output) -> String {
    return String::from_utf8(out.stdout.clone()).unwrap();
}

/// Write content into a unique temporary file
fn temp_file(content: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("mushpick-{}-{n}.txt", std::process::id()));
    std::fs::write(&path, content).unwrap();
    return path;
}

fn repo_map(name: &str) -> String {
    return format!("{}/maps/{name}", env!("CARGO_MANIFEST_DIR"));
}

/// Generate a map through the CLI and return its text
fn generate(size: usize, walls: usize, players: usize, mushrooms: usize) -> String {
    let (s, w, p, m) = (
        size.to_string(),
        walls.to_string(),
        players.to_string(),
        mushrooms.to_string(),
    );
    let out = run(&["gen", "-s", &s, "-w", &w, "-p", &p, "-m", &m]);
    assert!(
        out.status.success(),
        "gen failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    return stdout(&out);
}

/// Solve the map file and return the solution text
fn solve(map: &Path, fast: bool) -> String {
    let mut args = vec!["solve", map.to_str().unwrap()];
    if fast {
        args.push("--fast");
    }
    let out = run(&args);
    assert!(
        out.status.success(),
        "solve failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    return stdout(&out);
}

/// Run `check` with a solution and return its stdout
fn check(map: &Path, solution: &str) -> String {
    let sol = temp_file(solution);
    let out = run(&["check", map.to_str().unwrap(), sol.to_str().unwrap()]);
    return stdout(&out);
}

/// Total number of steps over all players in a solution
fn total_steps(solution: &str) -> usize {
    return solution
        .lines()
        .map(|l| return l.split(':').nth(1).unwrap().len())
        .sum();
}

/// Parsed map for the reference solver: (size, free cells, players, mushrooms)
type Map = (usize, Vec<bool>, Vec<usize>, Vec<usize>);

fn parse_map(text: &str) -> Map {
    let rows: Vec<String> = text.lines().map(|l| return l.replace(' ', "")).collect();
    let size = rows.len();
    let (mut cells, mut players, mut mushrooms) = (vec![], vec![], vec![]);
    for (i, c) in rows.concat().chars().enumerate() {
        cells.push(c != 'X');
        match c {
            'P' => players.push(i),
            'M' => mushrooms.push(i),
            _ => {}
        }
    }
    return (size, cells, players, mushrooms);
}

/// BFS distances from one cell to all cells
fn distances(size: usize, cells: &[bool], from: usize) -> Vec<usize> {
    let mut dist = vec![usize::MAX; cells.len()];
    dist[from] = 0;
    let mut q = VecDeque::from([from]);
    while let Some(i) = q.pop_front() {
        let (x, y) = (i % size, i / size);
        let mut next = vec![];
        if x > 0 {
            next.push(i - 1);
        }
        if x + 1 < size {
            next.push(i + 1);
        }
        if y > 0 {
            next.push(i - size);
        }
        if y + 1 < size {
            next.push(i + size);
        }
        for n in next {
            if cells[n] && dist[n] == usize::MAX {
                dist[n] = dist[i] + 1;
                q.push_back(n);
            }
        }
    }
    return dist;
}

/// Optimal total steps via DP over (player, set of picked mushrooms, last mushroom)
fn optimal_steps(text: &str) -> usize {
    let (size, cells, players, mushrooms) = parse_map(text);
    let n = mushrooms.len();
    let full = 1usize << n;
    let m_dist: Vec<Vec<usize>> = mushrooms
        .iter()
        .map(|&m| return distances(size, &cells, m))
        .collect();

    // best[mask] = cheapest cost for one player to pick exactly the mushrooms in mask
    let per_player: Vec<Vec<usize>> = players
        .iter()
        .map(|&p| {
            let p_dist = distances(size, &cells, p);
            // tsp[mask][last]
            let mut tsp = vec![vec![usize::MAX; n]; full];
            for (j, &m) in mushrooms.iter().enumerate() {
                tsp[1 << j][j] = p_dist[m];
            }
            for mask in 1..full {
                for last in 0..n {
                    let cost = tsp[mask][last];
                    if cost == usize::MAX {
                        continue;
                    }
                    for next in 0..n {
                        if mask & (1 << next) == 0 {
                            let c = cost + m_dist[last][mushrooms[next]];
                            let nm = mask | (1 << next);
                            tsp[nm][next] = tsp[nm][next].min(c);
                        }
                    }
                }
            }
            let mut best: Vec<usize> = tsp
                .iter()
                .map(|row| return *row.iter().min().unwrap())
                .collect();
            best[0] = 0;
            return best;
        })
        .collect();

    // combine players: dp[mask] = cheapest cost to pick mask using the players seen so far
    let mut dp = per_player[0].clone();
    for best in &per_player[1..] {
        let mut new_dp = vec![usize::MAX; full];
        for mask in 0..full {
            // iterate over all submasks assigned to the new player
            let mut sub = mask;
            loop {
                let (a, b) = (dp[mask ^ sub], best[sub]);
                if a != usize::MAX && b != usize::MAX {
                    new_dp[mask] = new_dp[mask].min(a + b);
                }
                if sub == 0 {
                    break;
                }
                sub = (sub - 1) & mask;
            }
        }
        dp = new_dp;
    }
    return dp[full - 1];
}

#[test]
fn check_repo_map() {
    let out = run(&["check", &repo_map("map_s50_p3_m15.txt")]);
    assert!(out.status.success());
    let s = stdout(&out);
    assert!(s.contains("map parsable: true"), "{s}");
    assert!(s.contains("map accessible: true"), "{s}");
}

#[test]
fn check_reports_inaccessible_map() {
    let map = temp_file("P X -\n- X M\n- X -\n");
    let s = stdout(&run(&["check", map.to_str().unwrap()]));
    assert!(s.contains("map accessible: false"), "{s}");
}

#[test]
fn check_accepts_valid_solution() {
    let map = temp_file("P - M\n- X -\nM - P\n");
    assert!(check(&map, "P0:rr\nP1:ll\n").contains("solution passed: true"));
    // one player picking everything is fine too
    assert!(check(&map, "P0:rrddll\nP1:\n").contains("solution passed: true"));
}

#[test]
fn check_rejects_wrong_solutions() {
    let map = temp_file("P - M\n- X -\nM - P\n");
    // walks into a wall
    assert!(check(&map, "P0:rd\nP1:ll\n").contains("solution passed: false"));
    // leaves the map
    assert!(check(&map, "P0:u\nP1:ll\n").contains("solution passed: false"));
    // misses a mushroom
    assert!(check(&map, "P0:rr\nP1:\n").contains("solution passed: false"));
    // more players than the map has
    assert!(check(&map, "P0:rr\nP1:ll\nP2:\n").contains("solution passed: false"));
}

#[test]
fn check_rejects_malformed_solutions() {
    let map = temp_file("P - M\n- X -\nM - P\n");
    assert!(check(&map, "P0 rr\n").contains("solution parsable: false"));
    assert!(check(&map, "P0:rx\n").contains("solution parsable: false"));
}

#[test]
fn invalid_maps_fail() {
    for bad in [
        "P - Q\n- - -\n- - M\n",
        "P - M\n- -\n- - -\n",
        "P - M -\n- - - -\n",
    ] {
        let map = temp_file(bad);
        let out = run(&["check", map.to_str().unwrap()]);
        assert!(!out.status.success(), "map should be rejected:\n{bad}");
    }
}

#[test]
fn solve_repo_map_passes_check() {
    let map = PathBuf::from(repo_map("map_s100_p3_m5.txt"));
    let solution = solve(&map, false);
    assert_eq!(solution.lines().count(), 3);
    assert!(check(&map, &solution).contains("solution passed: true"));
}

#[test]
fn generated_maps_are_solved_optimally() {
    for (size, walls, players, mushrooms) in
        [(6, 10, 1, 5), (8, 20, 2, 6), (8, 20, 3, 7), (10, 35, 3, 8), (10, 35, 5, 7)]
    {
        for _ in 0..5 {
            let text = generate(size, walls, players, mushrooms);
            let map = temp_file(&text);
            let solution = solve(&map, false);
            assert!(
                check(&map, &solution).contains("solution passed: true"),
                "invalid solution\nmap:\n{text}\nsolution:\n{solution}"
            );
            let optimum = optimal_steps(&text);
            assert_eq!(
                total_steps(&solution),
                optimum,
                "not optimal\nmap:\n{text}\nsolution:\n{solution}"
            );

            let fast = solve(&map, true);
            assert!(
                check(&map, &fast).contains("solution passed: true"),
                "invalid --fast solution\nmap:\n{text}\nsolution:\n{fast}"
            );
            assert!(total_steps(&fast) >= optimum);
        }
    }
}

#[test]
fn repo_maps_have_known_optimal_costs() {
    for (name, steps) in [
        ("map_s100_p3_m5", 691),
        ("map_s100_p3_m6", 616),
        ("map_s100_p3_m7", 414),
        ("map_s100_p3_m8", 599),
        ("map_s100_p3_m9", 802),
        ("map_s100_p3_m10", 923),
        ("map_s100_p3_m11", 617),
        ("map_s100_p3_m12", 1064),
        ("map_s100_p3_m13", 1198),
        ("map_s100_p3_m14", 1205),
        ("map_s100_p3_m15", 1094),
        ("map_s100_p3_m16", 1288),
        ("map_s100_p3_m17", 1412),
        ("map_s100_p3_m18", 1123),
        ("map_s100_p3_m19", 706),
        ("map_s100_p3_m20", 1269),
        ("map_s50_p3_m15", 597),
    ] {
        let map = PathBuf::from(repo_map(&format!("{name}.txt")));
        let solution = solve(&map, false);
        assert!(check(&map, &solution).contains("solution passed: true"), "{name}");
        assert_eq!(total_steps(&solution), steps, "{name}");
    }
}

#[test]
fn solve_map_split_by_walls() {
    // each player can reach only the mushrooms on its side
    let map = temp_file("P - X M\nM - X -\n- - X -\n- - X P\n");
    let solution = solve(&map, false);
    assert!(check(&map, &solution).contains("solution passed: true"));
    assert_eq!(total_steps(&solution), 1 + 3);
}

#[test]
fn solve_fails_on_unreachable_mushroom() {
    let map = temp_file("P - X M\n- - X -\n- - X -\n- - X -\n");
    for fast in [false, true] {
        let mut args = vec!["solve", map.to_str().unwrap()];
        if fast {
            args.push("--fast");
        }
        let out = run(&args);
        assert!(!out.status.success());
        assert!(stdout(&out).is_empty());
    }
}

#[test]
fn solve_large_map_with_heuristics() {
    let text = generate(60, 1200, 10, 150);
    let map = temp_file(&text);
    let solution = solve(&map, true);
    assert_eq!(solution.lines().count(), 10);
    assert!(check(&map, &solution).contains("solution passed: true"));
}

#[test]
fn solve_exactly_beyond_subset_tables() {
    for _ in 0..3 {
        let text = generate(40, 500, 3, 35);
        let map = temp_file(&text);
        let exact = solve(&map, false);
        assert!(check(&map, &exact).contains("solution passed: true"));
        let fast = solve(&map, true);
        assert!(total_steps(&exact) <= total_steps(&fast));
    }
}

#[test]
fn generate_save_writes_parsable_map() {
    let path =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("gen-save-{}.txt", std::process::id()));
    let out = run(&[
        "gen",
        "-s",
        "8",
        "-w",
        "15",
        "-p",
        "2",
        "-m",
        "4",
        "--save",
        path.to_str().unwrap(),
    ]);
    assert!(out.status.success());
    let saved = std::fs::read_to_string(&path).unwrap();
    assert_eq!(saved.trim_end(), stdout(&out).trim_end());
    let s = stdout(&run(&["check", path.to_str().unwrap()]));
    assert!(s.contains("map accessible: true"), "{s}");
}

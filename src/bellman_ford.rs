/// Bellman-Ford shortest path implementation using BigInt for distances.
/// This avoids floating-point precision issues and integer overflow with very large negative distances.

use crate::graph::Graph;
use num_bigint::BigInt;

/// Find the shortest path from source to target in the graph.
/// Returns the sequence of node indices along the path.
pub fn shortest_path(graph: &Graph, source_idx: usize, target_idx: usize) -> Option<Vec<usize>> {
    let n = graph.nodes.len();
    if n == 0 {
        return None;
    }

    // Use BigInt for distances to avoid overflow
    let mut dist: Vec<BigInt> = vec![BigInt::from(i64::MAX); n];
    let mut prev = vec![None; n];
    dist[source_idx] = BigInt::from(0);

    // Relax edges up to n-1 times
    for _ in 0..n - 1 {
        let mut updated = false;
        for u in 0..n {
            if dist[u] == BigInt::from(i64::MAX) {
                continue;
            }
            for (v, weight) in &graph.edges[u] {
                let new_dist = &dist[u] + weight;
                if new_dist < dist[*v] {
                    dist[*v] = new_dist;
                    prev[*v] = Some(u);
                    updated = true;
                }
            }
        }
        if !updated {
            break;
        }
    }

    if dist[target_idx] == BigInt::from(i64::MAX) {
        return None;
    }

    // Reconstruct path
    let mut path = Vec::new();
    let mut curr = Some(target_idx);
    while let Some(u) = curr {
        path.push(u);
        curr = prev[u];
    }
    path.reverse();
    Some(path)
}

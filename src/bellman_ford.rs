//! Shortest path implementation.
//! Uses topological-order relaxation for DAGs, falls back to Bellman-Ford
//! if cycles are detected (from strand-switch edges).

use crate::graph::Graph;
use num_bigint::BigInt;

/// Find the shortest path from source to target in the graph.
/// Returns the sequence of node indices along the path.
pub fn shortest_path(graph: &Graph, source_idx: usize, target_idx: usize) -> Option<Vec<usize>> {
    let n = graph.nodes.len();
    if n == 0 {
        return None;
    }

    // Try topological-order relaxation first (O(V + E)).
    // Nodes are already sorted by position due to BTreeMap insertion order.
    // If any edge goes backward in the node ordering, we have a potential cycle
    // and must fall back to Bellman-Ford.
    let has_backward_edge = graph.edges.iter().enumerate().any(|(u, edges)| {
        edges.iter().any(|(v, _)| *v < u)
    });

    if !has_backward_edge {
        topological_relaxation(graph, source_idx, target_idx)
    } else {
        bellman_ford(graph, source_idx, target_idx)
    }
}

/// O(V + E) shortest path on a DAG. Nodes must be in topological order.
fn topological_relaxation(
    graph: &Graph,
    source_idx: usize,
    target_idx: usize,
) -> Option<Vec<usize>> {
    let n = graph.nodes.len();
    let mut dist: Vec<Option<BigInt>> = vec![None; n];
    let mut prev = vec![None; n];
    dist[source_idx] = Some(BigInt::from(0));

    for u in 0..n {
        let du = match &dist[u] {
            Some(d) => d.clone(),
            None => continue,
        };
        for (v, weight) in &graph.edges[u] {
            let new_dist = &du + weight;
            if dist[*v].as_ref().is_none_or(|dv| &new_dist < dv) {
                dist[*v] = Some(new_dist);
                prev[*v] = Some(u);
            }
        }
    }

    dist[target_idx].as_ref()?;

    let mut path = Vec::new();
    let mut curr = Some(target_idx);
    while let Some(u) = curr {
        path.push(u);
        curr = prev[u];
    }
    path.reverse();
    Some(path)
}

/// Standard Bellman-Ford O(V·E) — used when cycles may exist.
fn bellman_ford(graph: &Graph, source_idx: usize, target_idx: usize) -> Option<Vec<usize>> {
    let n = graph.nodes.len();
    let mut dist: Vec<Option<BigInt>> = vec![None; n];
    let mut prev = vec![None; n];
    dist[source_idx] = Some(BigInt::from(0));

    for _ in 0..n - 1 {
        let mut updated = false;
        for u in 0..n {
            let du = match &dist[u] {
                Some(d) => d.clone(),
                None => continue,
            };
            for (v, weight) in &graph.edges[u] {
                let new_dist = &du + weight;
                if dist[*v].as_ref().is_none_or(|dv| &new_dist < dv) {
                    dist[*v] = Some(new_dist);
                    prev[*v] = Some(u);
                    updated = true;
                }
            }
        }
        if !updated {
            break;
        }
    }

    dist[target_idx].as_ref()?;

    let mut path = Vec::new();
    let mut curr = Some(target_idx);
    while let Some(u) = curr {
        path.push(u);
        curr = prev[u];
    }
    path.reverse();
    Some(path)
}

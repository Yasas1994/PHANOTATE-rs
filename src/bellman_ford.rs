//! Shortest path implementation.
//! Uses topological-order relaxation for DAGs, falls back to Bellman-Ford
//! if cycles are detected (from strand-switch edges).

use crate::graph::{Graph, Node};
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
            if dist[*v].as_ref().map_or(true, |dv| &new_dist < dv) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple linear graph: source -> a -> b -> target
    fn linear_graph() -> (Graph, usize, usize) {
        let mut graph = Graph::new();
        let source = Node::new("source", "source", 0, 0);
        let a = Node::new("CDS", "start", 1, 10);
        let b = Node::new("CDS", "stop", 1, 30);
        let target = Node::new("target", "target", 0, 100);

        let s_idx = graph.add_node(source);
        let a_idx = graph.add_node(a);
        let b_idx = graph.add_node(b);
        let t_idx = graph.add_node(target);

        graph.edges[s_idx].push((a_idx, BigInt::from(5)));
        graph.edges[a_idx].push((b_idx, BigInt::from(3)));
        graph.edges[b_idx].push((t_idx, BigInt::from(2)));

        (graph, s_idx, t_idx)
    }

    /// Build a graph with two paths where the shorter total weight wins.
    fn branched_graph() -> (Graph, usize, usize) {
        let mut graph = Graph::new();
        let source = Node::new("source", "source", 0, 0);
        let a = Node::new("CDS", "start", 1, 10);
        let b = Node::new("CDS", "stop", 1, 20);
        let c = Node::new("CDS", "start", 1, 15);
        let d = Node::new("CDS", "stop", 1, 25);
        let target = Node::new("target", "target", 0, 100);

        let s_idx = graph.add_node(source);
        let a_idx = graph.add_node(a);
        let b_idx = graph.add_node(b);
        let c_idx = graph.add_node(c);
        let d_idx = graph.add_node(d);
        let t_idx = graph.add_node(target);

        // Path 1: source -> a -> b -> target (total weight 10)
        graph.edges[s_idx].push((a_idx, BigInt::from(4)));
        graph.edges[a_idx].push((b_idx, BigInt::from(3)));
        graph.edges[b_idx].push((t_idx, BigInt::from(3)));

        // Path 2: source -> c -> d -> target (total weight 6)
        graph.edges[s_idx].push((c_idx, BigInt::from(1)));
        graph.edges[c_idx].push((d_idx, BigInt::from(2)));
        graph.edges[d_idx].push((t_idx, BigInt::from(3)));

        (graph, s_idx, t_idx)
    }

    #[test]
    fn test_shortest_path_linear() {
        let (graph, s, t) = linear_graph();
        let path = shortest_path(&graph, s, t);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], s);
        assert_eq!(path[path.len() - 1], t);
    }

    #[test]
    fn test_shortest_path_branched() {
        let (graph, s, t) = branched_graph();
        let path = shortest_path(&graph, s, t);
        assert!(path.is_some());
        let path = path.unwrap();

        // The shortest path should go through c and d (lower total weight)
        let c_idx = graph.node_to_idx[&Node::new("CDS", "start", 1, 15)];
        let d_idx = graph.node_to_idx[&Node::new("CDS", "stop", 1, 25)];
        assert!(path.contains(&c_idx));
        assert!(path.contains(&d_idx));
    }

    #[test]
    fn test_shortest_path_empty_graph() {
        let graph = Graph::new();
        let path = shortest_path(&graph, 0, 0);
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_unreachable() {
        let mut graph = Graph::new();
        let source = Node::new("source", "source", 0, 0);
        let target = Node::new("target", "target", 0, 100);
        let s_idx = graph.add_node(source);
        let t_idx = graph.add_node(target);

        // No edges connecting source to target
        let path = shortest_path(&graph, s_idx, t_idx);
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_same_source_target() {
        let mut graph = Graph::new();
        let source = Node::new("source", "source", 0, 0);
        let s_idx = graph.add_node(source);

        let path = shortest_path(&graph, s_idx, s_idx);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![s_idx]);
    }

    #[test]
    fn test_topological_relaxation_used_for_dag() {
        // A DAG has no backward edges, so topological relaxation should be used
        let (graph, s, t) = linear_graph();
        let path = shortest_path(&graph, s, t);
        assert!(path.is_some());
    }

    #[test]
    fn test_bellman_ford_with_backward_edge() {
        // Graph with a backward edge (cycle) forces Bellman-Ford fallback
        let mut graph = Graph::new();
        let source = Node::new("source", "source", 0, 0);
        let a = Node::new("CDS", "start", 1, 10);
        let b = Node::new("CDS", "stop", 1, 30);
        let target = Node::new("target", "target", 0, 100);

        let s_idx = graph.add_node(source);
        let a_idx = graph.add_node(a);
        let b_idx = graph.add_node(b);
        let t_idx = graph.add_node(target);

        // Forward edges
        graph.edges[s_idx].push((a_idx, BigInt::from(5)));
        graph.edges[a_idx].push((b_idx, BigInt::from(3)));
        graph.edges[b_idx].push((t_idx, BigInt::from(2)));

        // Backward edge (creates a cycle)
        graph.edges[b_idx].push((a_idx, BigInt::from(10)));

        let path = shortest_path(&graph, s_idx, t_idx);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path[0], s_idx);
        assert_eq!(path[path.len() - 1], t_idx);
    }
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
                if dist[*v].as_ref().map_or(true, |dv| &new_dist < dv) {
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

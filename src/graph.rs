use crate::orf::Orf;
use crate::weights::{score_gap, score_overlap};
use num_bigint::BigInt;

/// Convert a f64 weight (already scaled by 1000 and truncated) to BigInt.
/// This avoids i64 overflow for very large negative weights.
fn f64_to_bigint_weight(w: f64) -> BigInt {
    let truncated = w.trunc();
    // Handle values outside i64 range by converting through string
    if truncated <= i64::MIN as f64 || truncated >= i64::MAX as f64 {
        BigInt::parse_bytes(format!("{:.0}", truncated).as_bytes(), 10)
            .unwrap_or_else(|| BigInt::from(0))
    } else {
        BigInt::from(truncated as i64)
    }
}

/// A node in the graph: a start or stop codon at a specific position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Node {
    pub gene: &'static str, // "CDS", "tRNA", "source", "target"
    pub node_type: &'static str, // "start", "stop", "source", "target"
    pub frame: i8,
    pub position: usize, // 1-based position on forward strand
}

impl Node {
    pub fn new(gene: &'static str, node_type: &'static str, frame: i8, position: usize) -> Self {
        Node {
            gene,
            node_type,
            frame,
            position,
        }
    }
}

/// An edge in the directed graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub source: Node,
    pub target: Node,
    pub weight: BigInt,
}

/// The graph structure: adjacency list mapping source node -> list of (target, weight).
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Vec<(usize, BigInt)>>, // edges[from_idx] = [(to_idx, weight), ...]
    pub node_to_idx: std::collections::BTreeMap<Node, usize>, // BTreeMap for deterministic ordering
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_to_idx: std::collections::BTreeMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) -> usize {
        if let Some(&idx) = self.node_to_idx.get(&node) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.edges.push(Vec::new());
        self.node_to_idx.insert(node, idx);
        idx
    }

    pub fn add_edge(&mut self, edge: Edge) {
        let src_idx = self.add_node(edge.source);
        let tgt_idx = self.add_node(edge.target);
        self.edges[src_idx].push((tgt_idx, edge.weight));
    }

    /// Build the graph from ORFs.
    pub fn from_orfs(
        orfs: &[Orf],
        contig_length: usize,
        pgap: f64,
    ) -> (Self, Vec<usize>) {
        let mut graph = Graph::new();

        // other_end maps position -> counterpart position in same ORF
        // Like Python's my_orfs.other_end, for each stop we keep:
        // - minimum start for forward ORFs
        // - maximum start for reverse ORFs
        let mut other_end: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        // stop_to_orfs maps stop position -> list of ORFs with that stop
        let mut stop_to_orfs: std::collections::BTreeMap<usize, Vec<&Orf>> =
            std::collections::BTreeMap::new();

        for orf in orfs {
            // Update other_end for stop -> "best" start
            if let Some(existing_start) = other_end.get(&orf.stop) {
                if orf.frame > 0 && orf.start < *existing_start {
                    other_end.insert(orf.stop, orf.start);
                } else if orf.frame < 0 && orf.start > *existing_start {
                    other_end.insert(orf.stop, orf.start);
                }
            } else {
                other_end.insert(orf.stop, orf.start);
            }
            // Always map start -> stop
            other_end.insert(orf.start, orf.stop);
            // Build stop_to_orfs
            stop_to_orfs.entry(orf.stop).or_default().push(orf);
        }

        // Add ORF edges
        for orf in orfs {
            let (source, target) = if orf.frame > 0 {
                (
                    Node::new("CDS", "start", orf.frame, orf.start),
                    Node::new("CDS", "stop", orf.frame, orf.stop),
                )
            } else {
                (
                    Node::new("CDS", "stop", orf.frame, orf.stop),
                    Node::new("CDS", "start", orf.frame, orf.start),
                )
            };
            let w = (orf.weight * 1000.0).trunc();
            graph.add_edge(Edge {
                source,
                target,
                weight: f64_to_bigint_weight(w),
            });
        }

        // Check for long noncoding regions that would break the path
        let mut bases: Vec<Option<usize>> = vec![None; contig_length];
        for orf in orfs {
            let mi = orf.start.min(orf.stop);
            let ma = orf.start.max(orf.stop);
            for n in mi..=ma.min(contig_length.saturating_sub(1)) {
                if n > 0 {
                    bases[n - 1] = Some(n);
                }
            }
        }

        let mut last = 0usize;
        for base in &bases {
            if let Some(b) = base {
                if *b > last && *b - last > 500 {
                    let node_list: Vec<Node> = graph.nodes.clone();
                    for right_node in &node_list {
                        let r = right_node.position;
                        for left_node in &node_list {
                            let l = left_node.position;
                            if last + 1 >= l && l > last.saturating_sub(500)
                                && b - 1 <= r && r < b + 500
                            {
                                if left_node.frame * right_node.frame > 0 {
                                    if left_node.node_type == "stop"
                                        && right_node.node_type == "start"
                                        && left_node.frame > 0
                                    {
                                        let score = score_gap((r as isize) - (l as isize) - 3, "same", pgap);
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                                        graph.add_edge(Edge {
                                            source: *left_node,
                                            target: *right_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    } else if left_node.node_type == "start"
                                        && right_node.node_type == "stop"
                                        && left_node.frame < 0
                                    {
                                        let score = score_gap((r as isize) - (l as isize) - 3, "same", pgap);
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                                        graph.add_edge(Edge {
                                            source: *left_node,
                                            target: *right_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    }
                                } else {
                                    if left_node.node_type == "stop"
                                        && right_node.node_type == "stop"
                                        && left_node.frame > 0
                                    {
                                        let score = score_gap((r as isize) - (l as isize) - 3, "diff", pgap);
                                        graph.add_edge(Edge {
                                            source: *left_node,
                                            target: *right_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    } else if left_node.node_type == "start"
                                        && right_node.node_type == "start"
                                        && left_node.frame < 0
                                    {
                                        let score = score_gap((r as isize) - (l as isize) - 3, "diff", pgap);
                                        graph.add_edge(Edge {
                                            source: *left_node,
                                            target: *right_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                last = *b;
            }
        }

        // Connect the open reading frames to each other
        let node_list: Vec<Node> = graph.nodes.clone();
        for right_node in &node_list {
            let r = right_node.position;
            let r_other = other_end.get(&r).copied();

            for left_node in &node_list {
                let l = left_node.position;
                if l >= r {
                    continue;
                }
                let gap = r - l;
                if gap >= 500 {
                    continue;
                }

                let l_other = other_end.get(&l).copied();

                // Get pstop for flanking ORFs
                // Python: if(l in my_orfs and my_orfs.other_end[l] in my_orfs[l]):
                //             o1 = my_orfs.get_orf(my_orfs.other_end[l], l).pstop
                //         elif(l in my_orfs):
                //             o1 = my_orfs.get_orf(l, my_orfs.other_end[l]).pstop
                //         else:
                //             o1 = pgap
                let o1 = if let Some(orfs_at_l) = stop_to_orfs.get(&l) {
                    if let Some(other) = l_other {
                        orfs_at_l.iter()
                            .find(|o| o.start == other)
                            .map(|o| o.pstop)
                            .unwrap_or(pgap)
                    } else {
                        pgap
                    }
                } else {
                    pgap
                };
                let o2 = if let Some(orfs_at_r) = stop_to_orfs.get(&r) {
                    if let Some(other) = r_other {
                        orfs_at_r.iter()
                            .find(|o| o.start == other)
                            .map(|o| o.pstop)
                            .unwrap_or(pgap)
                    } else {
                        pgap
                    }
                } else {
                    pgap
                };
                let pstop = (o1 + o2) / 2.0;

                let same_strand = left_node.frame * right_node.frame > 0;

                if same_strand {
                    if left_node.node_type == "stop" && right_node.node_type == "start" {
                        if left_node.frame > 0 {
                            let score = score_gap((r as isize) - (l as isize) - 3, "same", pgap);
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        } else {
                            // Different frames on same strand -> possible overlap
                            if left_node.frame != right_node.frame {
                                if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                    if r < lo && ro < l {
                                        let score = score_overlap(r as isize - l as isize + 3, "same", pstop);
                                        graph.add_edge(Edge {
                                            source: *right_node,
                                            target: *left_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    }
                                }
                            }
                        }
                    } else if left_node.node_type == "start" && right_node.node_type == "stop" {
                        if left_node.frame > 0 {
                            if left_node.frame != right_node.frame {
                                if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                    if r < lo && ro < l {
                                        let score = score_overlap(r as isize - l as isize + 3, "same", pstop);
                                        graph.add_edge(Edge {
                                            source: *right_node,
                                            target: *left_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    }
                                }
                            }
                        } else {
                            let score = score_gap((r as isize) - (l as isize) - 3, "same", pgap);
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                                        if (l == 20853 && r == 21029) || (l == 20853 && r == 20815) || (l == 20962 && r == 21029) {
                                        }
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        }
                    }
                } else {
                    // Different strands
                    if left_node.node_type == "stop" && right_node.node_type == "stop" {
                        if right_node.frame > 0 {
                            if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                if ro + 3 < l && r < lo {
                                    let score = score_overlap(r as isize - l as isize + 3, "diff", pstop);
                                    graph.add_edge(Edge {
                                        source: *right_node,
                                        target: *left_node,
                                        weight: f64_to_bigint_weight(score * 1000.0),
                                    });
                                }
                            }
                        } else {
                            let score = score_gap((r as isize) - (l as isize) - 3, "diff", pgap);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        }
                    } else if left_node.node_type == "start" && right_node.node_type == "start" {
                        if right_node.frame > 0 && r - l > 2 {
                            let score = score_gap((r as isize) - (l as isize) - 3, "diff", pgap);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        } else if right_node.frame < 0 {
                            if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                if ro < l && r < lo {
                                    let score = score_overlap(r as isize - l as isize + 3, "diff", pstop);
                                    graph.add_edge(Edge {
                                        source: *right_node,
                                        target: *left_node,
                                        weight: f64_to_bigint_weight(score * 1000.0),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Source and target nodes
        let source = Node::new("source", "source", 0, 0);
        let target = Node::new("target", "target", 0, contig_length + 1);
        graph.add_node(source);
        graph.add_node(target);

        let source_idx = graph.node_to_idx[&source];
        let target_idx = graph.node_to_idx[&target];

        for node in &node_list {
            if node.position <= 2000 {
                if (node.node_type == "start" && node.frame > 0)
                    || (node.node_type == "stop" && node.frame < 0)
                {
                    let score = score_gap(node.position as isize, "same", pgap);
                    graph.add_edge(Edge {
                        source,
                        target: *node,
                        weight: f64_to_bigint_weight(score * 1000.0),
                    });
                }
            }
            if contig_length >= node.position && contig_length - node.position <= 2000 {
                if (node.node_type == "start" && node.frame < 0)
                    || (node.node_type == "stop" && node.frame > 0)
                {
                    let score = score_gap((contig_length - node.position) as isize, "same", pgap);
                    graph.add_edge(Edge {
                        source: *node,
                        target,
                        weight: f64_to_bigint_weight(score * 1000.0),
                    });
                }
            }
        }

        (graph, vec![source_idx, target_idx])
    }
}

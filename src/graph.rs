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
    pub gene: &'static str,      // "CDS", "tRNA", "source", "target"
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

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
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

    /// Export the graph to Graphviz DOT format.
    ///
    /// If `path` is supplied, nodes and edges on that path are highlighted in
    /// red with thicker pens. `endpoints` are the source/target node indices.
    pub fn to_dot(&self, path: Option<&[usize]>, endpoints: &[usize]) -> String {
        use std::collections::HashSet;

        let path_nodes: HashSet<usize> = path
            .map(|p| p.iter().copied().collect())
            .unwrap_or_default();
        let path_edges: HashSet<(usize, usize)> = path
            .map(|p| p.windows(2).map(|w| (w[0], w[1])).collect())
            .unwrap_or_default();
        let endpoint_set: HashSet<usize> = endpoints.iter().copied().collect();

        let mut out = String::from("digraph phanotate {\n");
        out.push_str("  rankdir=LR;\n");
        out.push_str("  node [fontname=\"Helvetica\"];\n");

        for (i, node) in self.nodes.iter().enumerate() {
            let label = format!(
                "{}\\npos {}\\nframe {}",
                node.node_type, node.position, node.frame
            );
            let shape = match node.node_type {
                "start" => "ellipse",
                "stop" => "box",
                _ => "diamond",
            };
            let (fillcolor, style) = if endpoint_set.contains(&i) {
                ("lightblue", "filled")
            } else if path_nodes.contains(&i) {
                ("lightyellow", "filled")
            } else {
                ("white", "filled")
            };
            let color = if path_nodes.contains(&i) {
                "red"
            } else {
                "black"
            };
            out.push_str(&format!(
                "  n{} [label=\"{}\", shape={}, color={}, fillcolor={}, style={}];\n",
                i, label, shape, color, fillcolor, style
            ));
        }

        for (src, targets) in self.edges.iter().enumerate() {
            for (tgt, weight) in targets {
                let on_path = path_edges.contains(&(src, *tgt));
                let color = if on_path { "red" } else { "gray" };
                let penwidth = if on_path { "3" } else { "1" };
                // Truncate huge BigInt weights for readability.
                let wstr = weight.to_string();
                let label = if wstr.len() > 10 {
                    format!("{:.3e}", weight.to_string().parse::<f64>().unwrap_or(0.0))
                } else {
                    wstr
                };
                out.push_str(&format!(
                    "  n{} -> n{} [label=\"{}\", color={}, penwidth={}];\n",
                    src, tgt, label, color, penwidth
                ));
            }
        }

        out.push_str("}\n");
        out
    }

    /// Build the graph from ORFs.
    ///
    /// `gap_scale` multiplies gap edge weights and `overlap_scale` multiplies
    /// overlap edge weights. In the learned `--model` path these are used to
    /// calibrate penalty magnitudes against the ORF score; the default
    /// heuristic path uses `1.0` for both.
    pub fn from_orfs(
        orfs: &[Orf],
        contig_length: usize,
        pgap: f64,
        gap_scale: f64,
        overlap_scale: f64,
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
                if (orf.frame > 0 && orf.start < *existing_start)
                    || (orf.frame < 0 && orf.start > *existing_start)
                {
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
        for b in bases.iter().flatten() {
            if *b > last && *b - last > 500 {
                let node_list: Vec<Node> = graph.nodes.clone();
                for right_node in &node_list {
                    let r = right_node.position;
                    for left_node in &node_list {
                        let l = left_node.position;
                        if last + 1 >= l
                            && l > last.saturating_sub(500)
                            && b - 1 <= r
                            && r < b + 500
                        {
                            if left_node.frame * right_node.frame > 0 {
                                if (left_node.node_type == "stop"
                                    && right_node.node_type == "start"
                                    && left_node.frame > 0)
                                    || (left_node.node_type == "start"
                                        && right_node.node_type == "stop"
                                        && left_node.frame < 0)
                                {
                                    let score = score_gap(
                                        (r as isize) - (l as isize) - 3,
                                        "same",
                                        pgap,
                                        gap_scale,
                                    );
                                    graph.add_edge(Edge {
                                        source: *left_node,
                                        target: *right_node,
                                        weight: f64_to_bigint_weight(score * 1000.0),
                                    });
                                }
                            } else if (left_node.node_type == "stop"
                                && right_node.node_type == "stop"
                                && left_node.frame > 0)
                                || (left_node.node_type == "start"
                                    && right_node.node_type == "start"
                                    && left_node.frame < 0)
                            {
                                let score = score_gap(
                                    (r as isize) - (l as isize) - 3,
                                    "diff",
                                    pgap,
                                    gap_scale,
                                );
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
            last = *b;
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
                        orfs_at_l
                            .iter()
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
                        orfs_at_r
                            .iter()
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
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "same", pgap, gap_scale);
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
                                        let score = score_overlap(
                                            r as isize - l as isize + 3,
                                            "same",
                                            pstop,
                                            overlap_scale,
                                        );
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
                                        let score = score_overlap(
                                            r as isize - l as isize + 3,
                                            "same",
                                            pstop,
                                            overlap_scale,
                                        );
                                        graph.add_edge(Edge {
                                            source: *right_node,
                                            target: *left_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    }
                                }
                            }
                        } else {
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "same", pgap, gap_scale);
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
                                    let score = score_overlap(
                                        r as isize - l as isize + 3,
                                        "diff",
                                        pstop,
                                        overlap_scale,
                                    );
                                    graph.add_edge(Edge {
                                        source: *right_node,
                                        target: *left_node,
                                        weight: f64_to_bigint_weight(score * 1000.0),
                                    });
                                }
                            }
                        } else {
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "diff", pgap, gap_scale);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        }
                    } else if left_node.node_type == "start" && right_node.node_type == "start" {
                        if right_node.frame > 0 && r - l > 2 {
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "diff", pgap, gap_scale);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        } else if right_node.frame < 0 {
                            if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                if ro < l && r < lo {
                                    let score = score_overlap(
                                        r as isize - l as isize + 3,
                                        "diff",
                                        pstop,
                                        overlap_scale,
                                    );
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
            if node.position <= 2000
                && ((node.node_type == "start" && node.frame > 0)
                    || (node.node_type == "stop" && node.frame < 0))
            {
                let score = score_gap(node.position as isize, "same", pgap, gap_scale);
                graph.add_edge(Edge {
                    source,
                    target: *node,
                    weight: f64_to_bigint_weight(score * 1000.0),
                });
            }
            if contig_length >= node.position
                && contig_length - node.position <= 2000
                && ((node.node_type == "start" && node.frame < 0)
                    || (node.node_type == "stop" && node.frame > 0))
            {
                let score = score_gap(
                    (contig_length - node.position) as isize,
                    "same",
                    pgap,
                    gap_scale,
                );
                graph.add_edge(Edge {
                    source: *node,
                    target,
                    weight: f64_to_bigint_weight(score * 1000.0),
                });
            }
        }

        (graph, vec![source_idx, target_idx])
    }

    /// Build the graph for a circular genome.
    ///
    /// `orfs` must be ORFs discovered on the doubled sequence (`seq || seq`).
    /// `original_len` is the length of the original (non-doubled) genome. The
    /// source is at position 0 and the target is at `original_len + 1`, i.e.
    /// just after the origin, so the path traverses the circle exactly once.
    /// ORFs whose start lies in the second copy (`start > original_len`) are
    /// ignored; ORFs that wrap the origin (`start <= original_len` and
    /// `stop > original_len`) end past the target and connect to it with a
    /// short wrap-around gap.
    pub fn from_orfs_circular(
        orfs: &[Orf],
        original_len: usize,
        pgap: f64,
        gap_scale: f64,
        overlap_scale: f64,
    ) -> (Self, Vec<usize>) {
        let mut graph = Graph::new();

        // Only keep ORFs that start in the first copy. Duplicates from the
        // second copy and impossible wrapping ORFs are dropped.
        let orfs: Vec<&Orf> = orfs.iter().filter(|o| o.start <= original_len).collect();

        let mut other_end: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        let mut stop_to_orfs: std::collections::BTreeMap<usize, Vec<&Orf>> =
            std::collections::BTreeMap::new();

        for orf in &orfs {
            if let Some(existing_start) = other_end.get(&orf.stop) {
                if (orf.frame > 0 && orf.start < *existing_start)
                    || (orf.frame < 0 && orf.start > *existing_start)
                {
                    other_end.insert(orf.stop, orf.start);
                }
            } else {
                other_end.insert(orf.stop, orf.start);
            }
            other_end.insert(orf.start, orf.stop);
            stop_to_orfs.entry(orf.stop).or_default().push(orf);
        }

        // Add ORF edges using doubled-sequence positions.
        for orf in &orfs {
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

        // Mark bases covered in the original genome. Wrapping ORFs cover the
        // tail of the genome and the start of the genome.
        let mut bases: Vec<Option<usize>> = vec![None; original_len];
        for orf in &orfs {
            let mi = orf.start.min(orf.stop);
            let ma = orf.start.max(orf.stop);
            if ma <= original_len {
                // Non-wrapping ORF.
                for n in mi..=ma.min(original_len.saturating_sub(1)) {
                    if n > 0 {
                        bases[n - 1] = Some(n);
                    }
                }
            } else if mi <= original_len {
                // Wrapping ORF: mark [mi, original_len] and [1, ma - original_len].
                for n in mi..=original_len {
                    if n > 0 {
                        bases[n - 1] = Some(n);
                    }
                }
                let wrap_end = ma - original_len;
                for n in 1..=wrap_end.min(original_len) {
                    bases[n - 1] = Some(n);
                }
            }
        }

        // Long noncoding regions (same logic as linear, but within original_len).
        let mut last = 0usize;
        for b in bases.iter().flatten() {
            if *b > last && *b - last > 500 {
                let node_list: Vec<Node> = graph.nodes.clone();
                for right_node in &node_list {
                    let r = right_node.position;
                    for left_node in &node_list {
                        let l = left_node.position;
                        if last + 1 >= l
                            && l > last.saturating_sub(500)
                            && b - 1 <= r
                            && r < b + 500
                        {
                            if left_node.frame * right_node.frame > 0 {
                                if (left_node.node_type == "stop"
                                    && right_node.node_type == "start"
                                    && left_node.frame > 0)
                                    || (left_node.node_type == "start"
                                        && right_node.node_type == "stop"
                                        && left_node.frame < 0)
                                {
                                    let score = score_gap(
                                        (r as isize) - (l as isize) - 3,
                                        "same",
                                        pgap,
                                        gap_scale,
                                    );
                                    graph.add_edge(Edge {
                                        source: *left_node,
                                        target: *right_node,
                                        weight: f64_to_bigint_weight(score * 1000.0),
                                    });
                                }
                            } else if (left_node.node_type == "stop"
                                && right_node.node_type == "stop"
                                && left_node.frame > 0)
                                || (left_node.node_type == "start"
                                    && right_node.node_type == "start"
                                    && left_node.frame < 0)
                            {
                                let score = score_gap(
                                    (r as isize) - (l as isize) - 3,
                                    "diff",
                                    pgap,
                                    gap_scale,
                                );
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
            last = *b;
        }

        // Inter-ORF edges (same logic as linear; positions remain doubled).
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

                let o1 = if let Some(orfs_at_l) = stop_to_orfs.get(&l) {
                    if let Some(other) = l_other {
                        orfs_at_l
                            .iter()
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
                        orfs_at_r
                            .iter()
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
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "same", pgap, gap_scale);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        } else {
                            if left_node.frame != right_node.frame {
                                if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                    if r < lo && ro < l {
                                        let score = score_overlap(
                                            r as isize - l as isize + 3,
                                            "same",
                                            pstop,
                                            overlap_scale,
                                        );
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
                                        let score = score_overlap(
                                            r as isize - l as isize + 3,
                                            "same",
                                            pstop,
                                            overlap_scale,
                                        );
                                        graph.add_edge(Edge {
                                            source: *right_node,
                                            target: *left_node,
                                            weight: f64_to_bigint_weight(score * 1000.0),
                                        });
                                    }
                                }
                            }
                        } else {
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "same", pgap, gap_scale);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        }
                    }
                } else {
                    if left_node.node_type == "stop" && right_node.node_type == "stop" {
                        if right_node.frame > 0 {
                            if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                if ro + 3 < l && r < lo {
                                    let score = score_overlap(
                                        r as isize - l as isize + 3,
                                        "diff",
                                        pstop,
                                        overlap_scale,
                                    );
                                    graph.add_edge(Edge {
                                        source: *right_node,
                                        target: *left_node,
                                        weight: f64_to_bigint_weight(score * 1000.0),
                                    });
                                }
                            }
                        } else {
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "diff", pgap, gap_scale);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        }
                    } else if left_node.node_type == "start" && right_node.node_type == "start" {
                        if right_node.frame > 0 && r - l > 2 {
                            let score =
                                score_gap((r as isize) - (l as isize) - 3, "diff", pgap, gap_scale);
                            graph.add_edge(Edge {
                                source: *left_node,
                                target: *right_node,
                                weight: f64_to_bigint_weight(score * 1000.0),
                            });
                        } else if right_node.frame < 0 {
                            if let (Some(lo), Some(ro)) = (l_other, r_other) {
                                if ro < l && r < lo {
                                    let score = score_overlap(
                                        r as isize - l as isize + 3,
                                        "diff",
                                        pstop,
                                        overlap_scale,
                                    );
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

        // Source and target for one traversal of the circle.
        let source = Node::new("source", "source", 0, 0);
        let target = Node::new("target", "target", 0, original_len + 1);
        graph.add_node(source);
        graph.add_node(target);

        let source_idx = graph.node_to_idx[&source];
        let target_idx = graph.node_to_idx[&target];

        for node in &node_list {
            // Source to start nodes near the beginning of the circle.
            if node.position <= 2000
                && ((node.node_type == "start" && node.frame > 0)
                    || (node.node_type == "stop" && node.frame < 0))
            {
                let score = score_gap(node.position as isize, "same", pgap, gap_scale);
                graph.add_edge(Edge {
                    source,
                    target: *node,
                    weight: f64_to_bigint_weight(score * 1000.0),
                });
            }

            // Stop nodes to target. Non-wrapping ORFs end before the origin;
            // wrapping ORFs end just past the origin (position > original_len).
            if node.node_type == "stop" && node.frame > 0 {
                let dist = if node.position <= original_len {
                    (original_len + 1 - node.position) as isize
                } else {
                    // Wrapping ORF: the sequence from the origin back to the stop
                    // is part of the gene itself, not an intergenic tail, so use
                    // a minimal closing penalty.
                    0
                };
                if dist <= 2000 {
                    let score = score_gap(dist, "same", pgap, gap_scale);
                    graph.add_edge(Edge {
                        source: *node,
                        target,
                        weight: f64_to_bigint_weight(score * 1000.0),
                    });
                }
            } else if node.node_type == "start" && node.frame < 0 {
                // Reverse-strand ORFs: the "end" of the gene is the stop node,
                // but for connecting to target we use the start node position.
                let dist = if node.position <= original_len {
                    (original_len + 1 - node.position) as isize
                } else {
                    // Wrapping reverse ORF: no intergenic tail past the origin.
                    0
                };
                if dist <= 2000 {
                    let score = score_gap(dist, "same", pgap, gap_scale);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_new() {
        let node = Node::new("CDS", "start", 1, 100);
        assert_eq!(node.gene, "CDS");
        assert_eq!(node.node_type, "start");
        assert_eq!(node.frame, 1);
        assert_eq!(node.position, 100);
    }

    #[test]
    fn test_graph_new() {
        let graph = Graph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
        assert!(graph.node_to_idx.is_empty());
    }

    #[test]
    fn test_graph_add_node() {
        let mut graph = Graph::new();
        let node = Node::new("CDS", "start", 1, 100);
        let idx = graph.add_node(node);
        assert_eq!(idx, 0);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.edges.len(), 1);
        assert!(graph.node_to_idx.contains_key(&node));
    }

    #[test]
    fn test_graph_add_duplicate_node() {
        let mut graph = Graph::new();
        let node = Node::new("CDS", "start", 1, 100);
        let idx1 = graph.add_node(node);
        let idx2 = graph.add_node(node);
        assert_eq!(idx1, idx2);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_graph_add_edge() {
        let mut graph = Graph::new();
        let source = Node::new("CDS", "start", 1, 100);
        let target = Node::new("CDS", "stop", 1, 200);

        graph.add_edge(Edge {
            source,
            target,
            weight: BigInt::from(-42),
        });

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 2);

        let s_idx = graph.node_to_idx[&source];
        assert_eq!(graph.edges[s_idx].len(), 1);
        assert_eq!(graph.edges[s_idx][0].1, BigInt::from(-42));
    }

    #[test]
    fn test_graph_add_multiple_edges() {
        let mut graph = Graph::new();
        let a = Node::new("CDS", "start", 1, 10);
        let b = Node::new("CDS", "stop", 1, 20);
        let c = Node::new("CDS", "start", 2, 30);

        let a_idx = graph.add_node(a);
        let b_idx = graph.add_node(b);
        let c_idx = graph.add_node(c);

        graph.edges[a_idx].push((b_idx, BigInt::from(1)));
        graph.edges[a_idx].push((c_idx, BigInt::from(2)));

        assert_eq!(graph.edges[a_idx].len(), 2);
    }

    #[test]
    fn test_node_equality() {
        let a = Node::new("CDS", "start", 1, 100);
        let b = Node::new("CDS", "start", 1, 100);
        let c = Node::new("CDS", "stop", 1, 100);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_node_ordering() {
        let a = Node::new("CDS", "start", 1, 100);
        let b = Node::new("CDS", "start", 1, 200);
        assert!(a < b);
    }

    #[test]
    fn test_graph_from_orfs_empty() {
        let orfs: Vec<Orf> = vec![];
        let (graph, endpoints) = Graph::from_orfs(&orfs, 100, 0.05, 1.0, 1.0);
        assert_eq!(graph.nodes.len(), 2); // source + target
        assert_eq!(endpoints.len(), 2);
    }

    #[test]
    fn test_graph_from_orfs_single() {
        let hold = 1.0;
        let orfs = vec![Orf {
            start: 1,
            stop: 30,
            frame: 1,
            seq: b"atg".to_vec(),
            rbs_score: 10,
            pstop: 0.05,
            sd_rbs_score: 1.0,
            non_sd_rbs_score: 1.0,
            rbs_motif: None,
            hold,
            coding_potential: 1.0 / hold,
            weight: -1.0,
            cscore: 0.0,
            cai: 0.0,
            gc1: 0.0,
            gc2: 0.0,
            gc3: 0.0,
            overlap_upstream_length: 0.0,
            overlap_upstream_same_strand: 0.0,
            overlap_downstream_length: 0.0,
            overlap_downstream_same_strand: 0.0,
            stop_sharing_count: 0.0,
            gc_skew: 0.0,
            truncation_penalty: 0.0,
            upstream_pwm_score: 0.0,
            rbs_spacer: 0.0,
            best_alt_pwm_score: 0.0,
            pwm_ratio: 1.0,
            start_rank: 1.0,
            num_alt_starts: 1.0,
            start_codon_log_freq: 0.0,
            wraps_origin: false,
        }];
        let (graph, endpoints) = Graph::from_orfs(&orfs, 100, 0.05, 1.0, 1.0);
        assert!(graph.nodes.len() >= 2);
        assert_eq!(endpoints.len(), 2);
    }

    #[test]
    fn test_graph_default() {
        let graph: Graph = Default::default();
        assert!(graph.nodes.is_empty());
    }

    #[test]
    fn test_from_orfs_circular_basic() {
        let original_len = 100;
        let orfs = vec![
            Orf {
                start: 10,
                stop: 30,
                frame: 1,
                seq: b"atg".to_vec(),
                weight: -1.0,
                wraps_origin: false,
                ..Default::default()
            },
            Orf {
                start: 80,
                stop: 110,
                frame: 1,
                seq: b"atg".to_vec(),
                weight: -1.0,
                wraps_origin: false,
                ..Default::default()
            },
        ];

        let (graph, endpoints) = Graph::from_orfs_circular(&orfs, original_len, 0.05, 1.0, 1.0);
        assert_eq!(endpoints.len(), 2);

        let source = Node::new("source", "source", 0, 0);
        let target = Node::new("target", "target", 0, original_len + 1);
        assert!(graph.node_to_idx.contains_key(&source));
        assert!(graph.node_to_idx.contains_key(&target));

        // The non-wrapping ORF has normal start/stop nodes.
        let start10 = Node::new("CDS", "start", 1, 10);
        let stop30 = Node::new("CDS", "stop", 1, 30);
        assert!(graph.node_to_idx.contains_key(&start10));
        assert!(graph.node_to_idx.contains_key(&stop30));

        // The wrapping ORF's stop node (past the origin) connects to target.
        let stop110 = Node::new("CDS", "stop", 1, 110);
        let stop_idx = graph.node_to_idx[&stop110];
        let target_idx = graph.node_to_idx[&target];
        assert!(graph.edges[stop_idx].iter().any(|(t, _)| *t == target_idx));
    }
}

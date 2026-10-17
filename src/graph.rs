//! Typed Temporal Knowledge Graph & Multi-Hop Spreading Activation Engine

use std::collections::HashMap;

/// Semantic relationship type connecting two memory nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EdgeKind {
    RelatesTo = 1,
    Causes = 2,
    Supersedes = 3,
    PartOf = 4,
    DependsOn = 5,
}

impl EdgeKind {
    pub fn from_u8(val: u8) -> Self {
        match val {
            2 => Self::Causes,
            3 => Self::Supersedes,
            4 => Self::PartOf,
            5 => Self::DependsOn,
            _ => Self::RelatesTo,
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "causes" | "caused" | "triggers" => Self::Causes,
            "supersedes" | "replaces" | "updates" => Self::Supersedes,
            "part_of" | "partof" | "belongs_to" => Self::PartOf,
            "depends_on" | "dependson" | "requires" => Self::DependsOn,
            _ => Self::RelatesTo,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RelatesTo => "RELATES_TO",
            Self::Causes => "CAUSES",
            Self::Supersedes => "SUPERSEDES",
            Self::PartOf => "PART_OF",
            Self::DependsOn => "DEPENDS_ON",
        }
    }

    /// Propagation multiplier during spreading activation.
    #[inline]
    pub fn propagation_factor(&self) -> f32 {
        match self {
            Self::Causes => 0.90,
            Self::DependsOn => 0.85,
            Self::PartOf => 0.80,
            Self::Supersedes => 0.95,
            Self::RelatesTo => 0.70,
        }
    }
}

/// Directed weighted edge in the temporal knowledge graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphEdge {
    pub source_id: u64,
    pub target_id: u64,
    pub weight: f32,
    pub kind: EdgeKind,
}

/// Bidirectional adjacency list over memory IDs for 1-hop and 2-hop spreading activation.
#[derive(Clone, Debug, Default)]
pub struct TemporalGraph {
    pub edges: Vec<GraphEdge>,
    outgoing: HashMap<u64, Vec<usize>>,
    incoming: HashMap<u64, Vec<usize>>,
    superseded_by: HashMap<u64, u64>,
}

impl TemporalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_edge(&mut self, source_id: u64, target_id: u64, weight: f32, kind: EdgeKind) {
        let idx = self.edges.len();
        let clamped_w = weight.clamp(0.0, 1.0);
        self.edges.push(GraphEdge {
            source_id,
            target_id,
            weight: clamped_w,
            kind,
        });
        self.outgoing.entry(source_id).or_default().push(idx);
        self.incoming.entry(target_id).or_default().push(idx);

        // If `source_id` SUPERSEDES `target_id`, record that `target_id` is superseded by `source_id`
        if kind == EdgeKind::Supersedes {
            self.superseded_by.insert(target_id, source_id);
        }
    }

    /// Returns `true` if `node_id` has been superseded by a newer memory node.
    #[inline]
    pub fn is_superseded(&self, node_id: u64) -> bool {
        self.superseded_by.contains_key(&node_id)
    }

    /// Computes 2-hop spreading activation boost in `[0.0, 1.0]` from initial seed scores `(node_id -> seed_score)`.
    /// Also transfers activation from superseded older nodes to the newer nodes that replaced them.
    pub fn spreading_activation(&self, seeds: &[(u64, f32)], damping: f32) -> HashMap<u64, f32> {
        let mut boost: HashMap<u64, f32> = HashMap::new();
        if self.edges.is_empty() || seeds.is_empty() {
            return boost;
        }

        // Hop 1 propagation
        let mut hop1_nodes: Vec<(u64, f32)> = Vec::new();
        for &(seed_id, seed_score) in seeds {
            if seed_score <= 0.05 {
                continue;
            }
            if let Some(out_indices) = self.outgoing.get(&seed_id) {
                for &e_idx in out_indices {
                    let e = &self.edges[e_idx];
                    let delta = seed_score * e.weight * e.kind.propagation_factor() * damping;
                    let entry = boost.entry(e.target_id).or_insert(0.0);
                    if delta > *entry {
                        *entry = delta;
                        hop1_nodes.push((e.target_id, delta));
                    }
                }
            }
            if let Some(in_indices) = self.incoming.get(&seed_id) {
                for &e_idx in in_indices {
                    let e = &self.edges[e_idx];
                    // If `source_id` supersedes `seed_id`, strongly boost `source_id`
                    let dir_scale = if e.kind == EdgeKind::Supersedes {
                        1.0
                    } else {
                        0.65
                    };
                    let delta =
                        seed_score * e.weight * e.kind.propagation_factor() * damping * dir_scale;
                    let entry = boost.entry(e.source_id).or_insert(0.0);
                    if delta > *entry {
                        *entry = delta;
                        hop1_nodes.push((e.source_id, delta));
                    }
                }
            }
        }

        // Hop 2 propagation (attenuated by `damping^2`)
        let hop2_damping = damping * 0.5;
        for (h1_id, h1_score) in hop1_nodes {
            if let Some(out_indices) = self.outgoing.get(&h1_id) {
                for &e_idx in out_indices {
                    let e = &self.edges[e_idx];
                    let delta = h1_score * e.weight * e.kind.propagation_factor() * hop2_damping;
                    let entry = boost.entry(e.target_id).or_insert(0.0);
                    if delta > *entry {
                        *entry = delta.min(1.0);
                    }
                }
            }
        }

        for val in boost.values_mut() {
            *val = val.clamp(0.0, 1.0);
        }
        boost
    }

    /// Returns all incident edges (`outgoing` and `incoming`) for a given `node_id`.
    pub fn neighbors_of(&self, node_id: u64) -> Vec<GraphEdge> {
        let mut res = Vec::new();
        if let Some(out) = self.outgoing.get(&node_id) {
            for &idx in out {
                res.push(self.edges[idx]);
            }
        }
        if let Some(inc) = self.incoming.get(&node_id) {
            for &idx in inc {
                res.push(self.edges[idx]);
            }
        }
        res
    }
}

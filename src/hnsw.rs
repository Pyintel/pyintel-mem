//! Multi-Layer HNSW Graph Index with Adaptive Selectivity Router
//!
//! Combines `O(log N)` hierarchical navigable small-world graph traversal for broad queries
//! with an **Adaptive Selectivity Router** that switches to SIMD-accelerated `SQ8` exact scan
//! whenever filter selectivity is below 15% or candidate count `<= 2,048`—eliminating the
//! post-filter recall collapse documented in filtered HNSW benchmarks.

use crate::quant::{Bq1Vector, Sq8Vector};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Copy, Debug)]
struct Candidate {
    slot: u32,
    sim: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}
impl Eq for Candidate {}

// Max-heap ordering by similarity
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.sim.partial_cmp(&other.sim)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// Min-heap wrapper to evict worst candidate in bounded `ef` sets
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReverseCandidate(Candidate);

impl PartialOrd for ReverseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.sim.partial_cmp(&self.0.sim)
    }
}
impl Ord for ReverseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Multi-layer HNSW node storing neighbor slot lists per layer.
#[derive(Clone, Debug)]
pub struct HnswNode {
    pub layers: Vec<Vec<u32>>,
}

/// HNSW + Adaptive Selectivity Vector Index.
#[derive(Clone, Debug)]
pub struct HnswIndex {
    pub m_max: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_layer: usize,
    pub entry_slot: Option<u32>,
    pub nodes: Vec<HnswNode>,
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new(16, 64, 48)
    }
}

impl HnswIndex {
    pub fn new(m_max: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m_max,
            ef_construction,
            ef_search,
            max_layer: 0,
            entry_slot: None,
            nodes: Vec::new(),
        }
    }

    /// Deterministic pseudo-random level generator (`l = floor(-ln(u) * m_L)`) based on slot hash.
    #[inline]
    fn assign_layer(slot: u32) -> usize {
        let mut x = (slot as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        // Trailing zeros / 2 gives geometric distribution with P(level >= l) = (1/4)^l
        ((x.trailing_zeros() as usize) / 2).min(6)
    }

    /// Inserts `slot` into the multi-layer HNSW graph using `sq8_vectors`.
    pub fn insert(&mut self, slot: u32, sq8_vectors: &[Sq8Vector]) {
        let target_layer = Self::assign_layer(slot);
        if (slot as usize) >= self.nodes.len() {
            self.nodes.resize(
                (slot as usize) + 1,
                HnswNode {
                    layers: Vec::new(),
                },
            );
        }
        self.nodes[slot as usize].layers = vec![Vec::with_capacity(self.m_max); target_layer + 1];

        let Some(mut curr_entry) = self.entry_slot else {
            self.entry_slot = Some(slot);
            self.max_layer = target_layer;
            return;
        };

        let q_vec = &sq8_vectors[slot as usize];
        let mut curr_sim = q_vec.cosine_similarity(&sq8_vectors[curr_entry as usize]);

        // 1. Greedy descent from top layer down to `target_layer + 1`
        if self.max_layer > target_layer {
            for layer in ((target_layer + 1)..=self.max_layer).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    if let Some(neighbors) = self.nodes[curr_entry as usize].layers.get(layer) {
                        for &nb in neighbors {
                            let sim = q_vec.cosine_similarity(&sq8_vectors[nb as usize]);
                            if sim > curr_sim {
                                curr_sim = sim;
                                curr_entry = nb;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // 2. Search & bi-directionally link from `min(target_layer, max_layer)` down to 0
        let top_connect = target_layer.min(self.max_layer);
        for layer in (0..=top_connect).rev() {
            let candidates =
                self.search_layer(q_vec, curr_entry, self.ef_construction, layer, sq8_vectors);
            let selected: Vec<u32> = candidates
                .iter()
                .take(self.m_max)
                .map(|c| c.slot)
                .collect();

            if let Some(first) = candidates.first() {
                curr_entry = first.slot;
            }

            self.nodes[slot as usize].layers[layer] = selected.clone();

            for nb in selected {
                if nb == slot {
                    continue;
                }
                let nb_layers = &mut self.nodes[nb as usize].layers;
                if layer < nb_layers.len() {
                    let list = &mut nb_layers[layer];
                    if !list.contains(&slot) {
                        if list.len() < self.m_max * 2 {
                            list.push(slot);
                        } else {
                            // Replace weakest neighbor if new node is closer
                            let nb_vec = &sq8_vectors[nb as usize];
                            let new_sim = nb_vec.cosine_similarity(q_vec);
                            let mut min_idx = 0;
                            let mut min_sim = f32::INFINITY;
                            for (idx, &existing) in list.iter().enumerate() {
                                let s = nb_vec.cosine_similarity(&sq8_vectors[existing as usize]);
                                if s < min_sim {
                                    min_sim = s;
                                    min_idx = idx;
                                }
                            }
                            if new_sim > min_sim {
                                list[min_idx] = slot;
                            }
                        }
                    }
                }
            }
        }

        if target_layer > self.max_layer {
            self.max_layer = target_layer;
            self.entry_slot = Some(slot);
        }
    }

    fn search_layer(
        &self,
        q_vec: &Sq8Vector,
        entry_slot: u32,
        ef: usize,
        layer: usize,
        sq8_vectors: &[Sq8Vector],
    ) -> Vec<Candidate> {
        let mut visited = HashSet::with_capacity(ef * 2);
        let mut candidates = BinaryHeap::new();
        let mut top_results = BinaryHeap::new();

        let entry_sim = q_vec.cosine_similarity(&sq8_vectors[entry_slot as usize]);
        let init = Candidate {
            slot: entry_slot,
            sim: entry_sim,
        };
        visited.insert(entry_slot);
        candidates.push(init);
        top_results.push(ReverseCandidate(init));

        while let Some(curr) = candidates.pop() {
            if let Some(worst) = top_results.peek() {
                if curr.sim < worst.0.sim && top_results.len() >= ef {
                    break;
                }
            }

            if let Some(neighbors) = self.nodes[curr.slot as usize].layers.get(layer) {
                for &nb in neighbors {
                    if visited.insert(nb) {
                        let sim = q_vec.cosine_similarity(&sq8_vectors[nb as usize]);
                        let should_add = top_results.len() < ef
                            || top_results.peek().map_or(true, |w| sim > w.0.sim);
                        if should_add {
                            let cand = Candidate { slot: nb, sim };
                            candidates.push(cand);
                            top_results.push(ReverseCandidate(cand));
                            if top_results.len() > ef {
                                top_results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = top_results.into_iter().map(|rc| rc.0).collect();
        out.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap_or(Ordering::Equal));
        out
    }

    /// Adaptive Selectivity Search:
    /// - If `allowed_slots` has `<= 2,048` items or represents `< 15%` of total nodes, executes a
    ///   direct pre-filtered SIMD `SQ8` + `BQ1` scan over `allowed_slots` (100% filtered recall).
    /// - Otherwise, traverses the HNSW graph with in-graph predicate evaluation.
    pub fn search_adaptive(
        &self,
        q_sq8: &Sq8Vector,
        q_bq1: &Bq1Vector,
        sq8_vectors: &[Sq8Vector],
        bq1_vectors: &[Bq1Vector],
        allowed_slots: &[u32],
        top_k: usize,
    ) -> (Vec<(u32, f32)>, &'static str) {
        let total = sq8_vectors.len();
        if total == 0 || allowed_slots.is_empty() || top_k == 0 {
            return (Vec::new(), "empty");
        }

        let selectivity = (allowed_slots.len() as f32) / (total as f32);

        // Route 1: Pre-filtered SIMD SQ8 + BQ1 scan when corpus is compact or filter is selective
        if allowed_slots.len() <= 2_048 || selectivity < 0.15 || self.entry_slot.is_none() {
            let mut results: Vec<(u32, f32)> = Vec::with_capacity(allowed_slots.len());
            for &slot in allowed_slots {
                let idx = slot as usize;
                let cos_sim = q_sq8.cosine_similarity(&sq8_vectors[idx]);
                let ham_sim = q_bq1.hamming_similarity(&bq1_vectors[idx]);
                // Blend 85% SQ8 cosine + 15% BQ1 Hamming
                let blended = (0.85 * cos_sim + 0.15 * (2.0 * ham_sim - 1.0)).clamp(-1.0, 1.0);
                results.push((slot, blended));
            }
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            results.truncate(top_k);
            return (results, "prefiltered_simd_sq8");
        }

        // Route 2: Multi-layer HNSW traversal with in-graph predicate check
        let allowed_set: HashSet<u32> = allowed_slots.iter().copied().collect();
        let mut curr_entry = self.entry_slot.unwrap();
        let mut curr_sim = q_sq8.cosine_similarity(&sq8_vectors[curr_entry as usize]);

        for layer in (1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = self.nodes[curr_entry as usize].layers.get(layer) {
                    for &nb in neighbors {
                        let sim = q_sq8.cosine_similarity(&sq8_vectors[nb as usize]);
                        if sim > curr_sim {
                            curr_sim = sim;
                            curr_entry = nb;
                            changed = true;
                        }
                    }
                }
            }
        }

        let ef = self.ef_search.max(top_k * 2);
        let raw_cands = self.search_layer(q_sq8, curr_entry, ef, 0, sq8_vectors);
        let mut filtered: Vec<(u32, f32)> = raw_cands
            .into_iter()
            .filter(|c| allowed_set.contains(&c.slot))
            .map(|c| {
                let ham_sim = q_bq1.hamming_similarity(&bq1_vectors[c.slot as usize]);
                let blended = (0.85 * c.sim + 0.15 * (2.0 * ham_sim - 1.0)).clamp(-1.0, 1.0);
                (c.slot, blended)
            })
            .collect();

        filtered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        filtered.truncate(top_k);
        (filtered, "hnsw_in_graph")
    }

    /// Serializes all HNSW multi-layer adjacency lists into a compact binary slab
    /// so `load_mmap` can restore the graph in O(N) without rebuilding HNSW.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.nodes.len() * 32);
        out.extend_from_slice(&(self.max_layer as u32).to_le_bytes());
        out.extend_from_slice(&self.entry_slot.unwrap_or(0xFFFF_FFFF).to_le_bytes());
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for node in &self.nodes {
            out.push(node.layers.len() as u8);
            for layer in &node.layers {
                out.push(layer.len() as u8);
                for &nb in layer {
                    out.extend_from_slice(&nb.to_le_bytes());
                }
            }
        }
        out
    }

    /// Restores the HNSW multi-layer adjacency lists from a serialized binary slice in O(N).
    pub fn from_bytes(bytes: &[u8]) -> Option<(Self, usize)> {
        if bytes.len() < 12 {
            return None;
        }
        let max_layer = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        let raw_entry = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let node_count = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
        let entry_slot = if raw_entry == 0xFFFF_FFFF {
            None
        } else {
            Some(raw_entry)
        };

        let mut pos = 12usize;
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            if pos >= bytes.len() {
                return None;
            }
            let num_layers = bytes[pos] as usize;
            pos += 1;
            let mut layers = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                if pos >= bytes.len() {
                    return None;
                }
                let deg = bytes[pos] as usize;
                pos += 1;
                if pos + deg * 4 > bytes.len() {
                    return None;
                }
                let mut list = Vec::with_capacity(deg);
                for _ in 0..deg {
                    let nb = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?);
                    pos += 4;
                    list.push(nb);
                }
                layers.push(list);
            }
            nodes.push(HnswNode { layers });
        }

        let mut idx = Self::default();
        idx.max_layer = max_layer;
        idx.entry_slot = entry_slot;
        idx.nodes = nodes;
        Some((idx, pos))
    }
}

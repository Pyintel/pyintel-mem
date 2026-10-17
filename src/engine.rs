//! Core Hybrid Memory Engine (`PyintelMem`)
//!
//! Unifies:
//! 1. `SQ8` (Symmetric `int8`) + `BQ1` (1-bit Hamming) Vector Quantization
//! 2. Multi-Layer HNSW Graph + Adaptive Selectivity Router
//! 3. Okapi BM25 Lexical Inverted Index
//! 4. Typed Temporal Knowledge Graph with 2-Hop Spreading Activation
//! 5. Ebbinghaus Spaced-Repetition Exponential Decay

use crate::bm25::Bm25Index;
use crate::format::{
    write_atomic, MmapView, PmemHeader, EDGE_RECORD_SIZE, FLAG_BM25, FLAG_BQ1, FLAG_GRAPH,
    FLAG_HNSW_CSR, FLAG_SQ8, HEADER_SIZE, NODE_FLAG_PINNED, NODE_FLAG_TOMBSTONE, NODE_RECORD_SIZE,
    PMEM_VERSION,
};
use crate::graph::{EdgeKind, GraphEdge, TemporalGraph};
use crate::hnsw::HnswIndex;
use crate::quant::{deterministic_text_embedding, fnv1a_32, Bq1Vector, Sq8Vector};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Stored memory node inside `PyintelMem`.
#[derive(Clone, Debug)]
pub struct MemoryRecord {
    pub id: u64,
    pub namespace: String,
    pub namespace_hash: u32,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u32,
    pub importance: f32,
    pub flags: u32,
}

impl MemoryRecord {
    #[inline]
    pub fn is_tombstoned(&self) -> bool {
        (self.flags & NODE_FLAG_TOMBSTONE) != 0
    }

    #[inline]
    pub fn is_pinned(&self) -> bool {
        (self.flags & NODE_FLAG_PINNED) != 0
    }
}

/// Query configuration for hybrid 4-signal retrieval.
#[derive(Clone, Debug)]
pub struct QueryOptions {
    pub top_k: usize,
    pub namespace: Option<String>,
    pub required_tag: Option<String>,
    pub query_vector: Option<Vec<f32>>,
    pub weight_vector: f32,
    pub weight_bm25: f32,
    pub weight_graph: f32,
    pub now_timestamp: Option<u64>,
    pub reinforce_on_recall: bool,
    pub include_superseded: bool,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            namespace: None,
            required_tag: None,
            query_vector: None,
            weight_vector: 0.45,
            weight_bm25: 0.35,
            weight_graph: 0.20,
            now_timestamp: None,
            reinforce_on_recall: true,
            include_superseded: false,
        }
    }
}

/// Detailed score breakdown returned for each recalled memory.
#[derive(Clone, Debug)]
pub struct ScoredMemory {
    pub id: u64,
    pub namespace: String,
    pub content: String,
    pub tags: Vec<String>,
    pub final_score: f32,
    pub vector_score: f32,
    pub bm25_score: f32,
    pub graph_boost: f32,
    pub temporal_decay: f32,
    pub importance: f32,
    pub access_count: u32,
    pub superseded: bool,
}

/// Diagnostics and telemetry for a query execution.
#[derive(Clone, Debug)]
pub struct QueryReport {
    pub hits: Vec<ScoredMemory>,
    pub route_used: &'static str,
    pub candidate_count: usize,
    pub total_active_nodes: usize,
    pub latency_micros: f64,
}

/// Primary in-process and `.pmem`-backed hybrid memory store.
pub struct PyintelMem {
    pub dim: usize,
    pub half_life_secs: f32,
    pub records: Vec<MemoryRecord>,
    pub id_to_slot: HashMap<u64, u32>,
    pub sq8_vectors: Vec<Sq8Vector>,
    pub bq1_vectors: Vec<Bq1Vector>,
    pub hnsw: HnswIndex,
    pub bm25: Bm25Index,
    pub graph: TemporalGraph,
    pub next_id: u64,
    pub last_mmap_native: bool,
}

impl PyintelMem {
    /// Creates a new memory store with embedding dimension `dim` (default e.g. `128` or `384`)
    /// and default Ebbinghaus half-life `half_life_secs` (default `604,800` = 7 days).
    pub fn new(dim: usize, half_life_secs: f32) -> Self {
        Self {
            dim: dim.max(8),
            half_life_secs: half_life_secs.max(1.0),
            records: Vec::new(),
            id_to_slot: HashMap::new(),
            sq8_vectors: Vec::new(),
            bq1_vectors: Vec::new(),
            hnsw: HnswIndex::default(),
            bm25: Bm25Index::new(),
            graph: TemporalGraph::new(),
            next_id: 1,
            last_mmap_native: false,
        }
    }

    #[inline]
    pub fn current_unix_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(1_792_193_400)
    }

    /// Inserts or updates a memory node. If `embedding` is `None`, computes a deterministic
    /// subword + bigram CountSketch embedding of dimension `self.dim`.
    pub fn store(
        &mut self,
        id: Option<u64>,
        namespace: &str,
        content: &str,
        tags: Vec<String>,
        embedding: Option<&[f32]>,
        importance: f32,
        pinned: bool,
        timestamp: Option<u64>,
    ) -> u64 {
        let now = timestamp.unwrap_or_else(Self::current_unix_secs);
        let ns_clean = if namespace.trim().is_empty() {
            "default"
        } else {
            namespace.trim()
        };
        let ns_hash = fnv1a_32(ns_clean.as_bytes());

        // Smart Auto-Consolidation: if no explicit ID was forced and an active memory in the same
        // namespace already has identical content, reinforce it in-place rather than duplicating.
        if id.is_none() {
            let trimmed_new = content.trim();
            for rec in &mut self.records {
                if !rec.is_tombstoned()
                    && rec.namespace_hash == ns_hash
                    && rec.content.trim().eq_ignore_ascii_case(trimmed_new)
                {
                    rec.access_count = rec.access_count.saturating_add(1);
                    rec.last_accessed = now;
                    rec.importance = rec.importance.max(importance.clamp(0.1, 10.0));
                    if pinned {
                        rec.flags |= NODE_FLAG_PINNED;
                    }
                    for t in tags {
                        if !rec.tags.iter().any(|existing| existing.eq_ignore_ascii_case(&t)) {
                            rec.tags.push(t);
                        }
                    }
                    return rec.id;
                }
            }
        }

        let assigned_id = id.unwrap_or_else(|| {
            let v = self.next_id;
            self.next_id += 1;
            v
        });
        if assigned_id >= self.next_id {
            self.next_id = assigned_id + 1;
        }

        // Prepare dense vector (external embedding or deterministic n-gram CountSketch)
        let owned_vec;
        let vec_slice: &[f32] = match embedding {
            Some(v) if v.len() == self.dim => v,
            Some(v) => {
                let mut resized = vec![0.0f32; self.dim];
                let copy_len = v.len().min(self.dim);
                resized[..copy_len].copy_from_slice(&v[..copy_len]);
                owned_vec = resized;
                &owned_vec
            }
            None => {
                let mut combined = String::with_capacity(content.len() + 32);
                combined.push_str(content);
                for t in &tags {
                    combined.push(' ');
                    combined.push_str(t);
                }
                owned_vec = deterministic_text_embedding(&combined, self.dim);
                &owned_vec
            }
        };

        let sq8 = Sq8Vector::quantize(vec_slice);
        let bq1 = Bq1Vector::quantize(vec_slice);

        // If ID already exists, mark old slot as tombstoned and create an update slot
        if let Some(&old_slot) = self.id_to_slot.get(&assigned_id) {
            self.records[old_slot as usize].flags |= NODE_FLAG_TOMBSTONE;
        }

        let slot = self.records.len() as u32;
        let mut flags = 0u32;
        if pinned {
            flags |= NODE_FLAG_PINNED;
        }

        let mut indexable_text = String::with_capacity(content.len() + 32);
        indexable_text.push_str(content);
        for t in &tags {
            indexable_text.push(' ');
            indexable_text.push_str(t);
        }

        self.records.push(MemoryRecord {
            id: assigned_id,
            namespace: ns_clean.to_string(),
            namespace_hash: ns_hash,
            content: content.to_string(),
            tags,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            importance: importance.clamp(0.1, 10.0),
            flags,
        });
        self.id_to_slot.insert(assigned_id, slot);
        self.sq8_vectors.push(sq8);
        self.bq1_vectors.push(bq1);
        self.hnsw.insert(slot, &self.sq8_vectors);
        self.bm25.insert(slot, &indexable_text);

        assigned_id
    }

    /// Creates a directed typed edge in the temporal knowledge graph.
    pub fn link(&mut self, source_id: u64, target_id: u64, kind: EdgeKind, weight: f32) {
        self.graph.add_edge(source_id, target_id, weight, kind);
    }

    /// Marks a memory node as tombstoned (soft-deleted).
    pub fn forget(&mut self, id: u64) -> bool {
        if let Some(&slot) = self.id_to_slot.get(&id) {
            self.records[slot as usize].flags |= NODE_FLAG_TOMBSTONE;
            true
        } else {
            false
        }
    }

    /// Computes Ebbinghaus spaced-repetition retention in `[0.05, 1.0]`:
    /// `R(t) = 2^(-delta_t / (half_life * (1 + 0.45 * ln(1 + access_count))))`
    #[inline]
    pub fn compute_temporal_decay(&self, rec: &MemoryRecord, now_secs: u64) -> f32 {
        if rec.is_pinned() {
            return 1.0;
        }
        let dt = now_secs.saturating_sub(rec.last_accessed) as f32;
        if dt <= 0.0 {
            return 1.0;
        }
        let reinforcement = 1.0 + 0.45 * ((1 + rec.access_count) as f32).ln();
        let effective_half_life = (self.half_life_secs * reinforcement).max(1.0);
        let exponent = -(dt / effective_half_life);
        exponent.exp2().clamp(0.05, 1.0)
    }

    /// Executes a 4-signal hybrid query:
    /// 1. Pre-filtering by namespace & tag bitmap
    /// 2. Adaptive Vector Search (`SQ8` + `BQ1` via SIMD scan or HNSW)
    /// 3. Okapi BM25 Lexical Search
    /// 4. 2-Hop Temporal Graph Spreading Activation + Ebbinghaus Decay
    pub fn recall(&mut self, query_text: &str, opts: QueryOptions) -> QueryReport {
        let t0 = Instant::now();
        let now = opts.now_timestamp.unwrap_or_else(Self::current_unix_secs);

        // 1. Build allowed candidate slots bitmap
        let ns_filter_hash = opts
            .namespace
            .as_ref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| fnv1a_32(s.trim().as_bytes()));
        let req_tag_lower = opts
            .required_tag
            .as_ref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_lowercase());

        let mut total_active = 0usize;
        let mut allowed_slots: Vec<u32> = Vec::with_capacity(self.records.len());

        for (slot_idx, rec) in self.records.iter().enumerate() {
            if rec.is_tombstoned() {
                continue;
            }
            if !opts.include_superseded && self.graph.is_superseded(rec.id) {
                continue;
            }
            total_active += 1;

            if let Some(nh) = ns_filter_hash {
                if rec.namespace_hash != nh {
                    continue;
                }
            }
            if let Some(ref tag) = req_tag_lower {
                if !rec.tags.iter().any(|t| t.to_lowercase() == *tag) {
                    continue;
                }
            }
            allowed_slots.push(slot_idx as u32);
        }

        if allowed_slots.is_empty() {
            return QueryReport {
                hits: Vec::new(),
                route_used: "empty",
                candidate_count: 0,
                total_active_nodes: total_active,
                latency_micros: t0.elapsed().as_secs_f64() * 1_000_000.0,
            };
        }

        // 2. Vector similarity search via Adaptive Selectivity Router
        let owned_qvec;
        let qvec_slice: &[f32] = match opts.query_vector {
            Some(ref v) if v.len() == self.dim => v.as_slice(),
            _ => {
                owned_qvec = deterministic_text_embedding(query_text, self.dim);
                &owned_qvec
            }
        };
        let q_sq8 = Sq8Vector::quantize(qvec_slice);
        let q_bq1 = Bq1Vector::quantize(qvec_slice);

        let fetch_k = (opts.top_k * 8).max(32).min(allowed_slots.len());
        let (vec_hits, route_used) = self.hnsw.search_adaptive(
            &q_sq8,
            &q_bq1,
            &self.sq8_vectors,
            &self.bq1_vectors,
            &allowed_slots,
            fetch_k,
        );

        let mut vec_score_map: HashMap<u32, f32> = HashMap::with_capacity(vec_hits.len());
        for (slot, sim) in vec_hits {
            // Normalize cosine [-1, 1] into [0, 1] positive relevance
            vec_score_map.insert(slot, sim.max(0.0));
        }

        // 3. Okapi BM25 lexical scores
        let bm25_map = self.bm25.score_query(query_text);

        // 4. Identify top seed nodes for 2-hop Temporal Knowledge Graph spreading activation
        let mut candidate_slots: std::collections::HashSet<u32> =
            std::collections::HashSet::with_capacity(vec_score_map.len() + bm25_map.len() + 32);
        for &slot in vec_score_map.keys() {
            candidate_slots.insert(slot);
        }
        let allowed_set: std::collections::HashSet<u32> =
            allowed_slots.iter().copied().collect();
        for &slot in bm25_map.keys() {
            if allowed_set.contains(&slot) {
                candidate_slots.insert(slot);
            }
        }

        let mut seed_pairs: Vec<(u64, f32)> = Vec::with_capacity(candidate_slots.len());
        for &slot in &candidate_slots {
            let v_s = vec_score_map.get(&slot).copied().unwrap_or(0.0);
            let b_s = bm25_map.get(&slot).copied().unwrap_or(0.0);
            let combined_seed = opts.weight_vector * v_s + opts.weight_bm25 * b_s;
            if combined_seed > 0.10 {
                let node_id = self.records[slot as usize].id;
                seed_pairs.push((node_id, combined_seed));
            }
        }
        // Also check superseded nodes so queries matching an old fact activate the newer superseding fact
        for (slot_idx, rec) in self.records.iter().enumerate() {
            if !rec.is_tombstoned() && self.graph.is_superseded(rec.id) {
                let slot = slot_idx as u32;
                let b_s = bm25_map.get(&slot).copied().unwrap_or(0.0);
                let v_s = q_sq8
                    .cosine_similarity(&self.sq8_vectors[slot_idx])
                    .max(0.0);
                let seed = opts.weight_vector * v_s + opts.weight_bm25 * b_s;
                if seed > 0.15 {
                    seed_pairs.push((rec.id, seed));
                }
            }
        }

        let graph_boost_map = self.graph.spreading_activation(&seed_pairs, 0.85);
        for &boosted_id in graph_boost_map.keys() {
            if let Some(&slot) = self.id_to_slot.get(&boosted_id) {
                if allowed_set.contains(&slot) {
                    candidate_slots.insert(slot);
                }
            }
        }

        // 5. Final 4-Signal Hybrid Scoring over candidate set (zero string allocations during ranking)
        struct RawScore {
            slot: u32,
            final_score: f32,
            v_s: f32,
            b_s: f32,
            g_s: f32,
            decay: f32,
        }

        let mut raw_ranked: Vec<RawScore> = Vec::with_capacity(candidate_slots.len());
        for slot in candidate_slots {
            let rec = &self.records[slot as usize];
            let v_s = vec_score_map.get(&slot).copied().unwrap_or_else(|| {
                q_sq8
                    .cosine_similarity(&self.sq8_vectors[slot as usize])
                    .max(0.0)
            });
            let b_s = bm25_map.get(&slot).copied().unwrap_or(0.0);
            let g_s = graph_boost_map.get(&rec.id).copied().unwrap_or(0.0);
            let decay = self.compute_temporal_decay(rec, now);

            let base_relevance =
                opts.weight_vector * v_s + opts.weight_bm25 * b_s + opts.weight_graph * g_s;
            // Multi-Signal Agreement Synergy: boosts nodes corroborated by both semantic vector and lexical/graph signals
            let synergy = 1.0 + 0.25 * (v_s * b_s).sqrt() + 0.15 * g_s.min(1.0);
            let final_score = base_relevance * synergy * decay * rec.importance;

            raw_ranked.push(RawScore {
                slot,
                final_score,
                v_s,
                b_s,
                g_s,
                decay,
            });
        }

        raw_ranked.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(Ordering::Equal)
        });
        raw_ranked.truncate(opts.top_k);

        // 6. Materialize top_k ScoredMemory structs & apply spaced-repetition reinforcement
        let mut hits: Vec<ScoredMemory> = Vec::with_capacity(raw_ranked.len());
        for item in raw_ranked {
            let rec = &mut self.records[item.slot as usize];
            if opts.reinforce_on_recall {
                rec.access_count = rec.access_count.saturating_add(1);
                rec.last_accessed = now;
            }
            let rec_immut = &self.records[item.slot as usize];
            hits.push(ScoredMemory {
                id: rec_immut.id,
                namespace: rec_immut.namespace.clone(),
                content: rec_immut.content.clone(),
                tags: rec_immut.tags.clone(),
                final_score: item.final_score,
                vector_score: item.v_s,
                bm25_score: item.b_s,
                graph_boost: item.g_s,
                temporal_decay: item.decay,
                importance: rec_immut.importance,
                access_count: rec_immut.access_count,
                superseded: self.graph.is_superseded(rec_immut.id),
            });
        }

        let latency_micros = t0.elapsed().as_secs_f64() * 1_000_000.0;

        QueryReport {
            hits,
            route_used,
            candidate_count: allowed_slots.len(),
            total_active_nodes: total_active,
            latency_micros,
        }
    }

    /// Serializes the entire memory store into the canonical `.pmem` single-file binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let node_count = self.records.len() as u32;
        let edge_count = self.graph.edges.len() as u32;
        let dim = self.dim as u32;
        let bq1_words = (self.dim + 63) / 64;

        let nodes_offset = HEADER_SIZE as u64;
        let sq8_offset = nodes_offset + (node_count as u64) * (NODE_RECORD_SIZE as u64);
        let sq8_stride = 4 + self.dim;
        // Align BQ1 offset to 8 bytes
        let raw_bq1_offset = sq8_offset + (node_count as u64) * (sq8_stride as u64);
        let bq1_offset = (raw_bq1_offset + 7) & !7;
        let edges_offset = bq1_offset + (node_count as u64) * ((bq1_words * 8) as u64);
        let arena_offset = edges_offset + (edge_count as u64) * (EDGE_RECORD_SIZE as u64);

        // Build UTF-8 payload arena first to know offsets
        let mut arena: Vec<u8> = Vec::new();
        let mut payload_spans: Vec<(u32, u32)> = Vec::with_capacity(self.records.len());

        for rec in &self.records {
            let start = arena.len() as u32;
            let ns_bytes = rec.namespace.as_bytes();
            let content_bytes = rec.content.as_bytes();
            let tags_joined = rec.tags.join("\x1F");
            let tags_bytes = tags_joined.as_bytes();

            arena.extend_from_slice(&(ns_bytes.len() as u16).to_le_bytes());
            arena.extend_from_slice(ns_bytes);
            arena.extend_from_slice(&(tags_bytes.len() as u16).to_le_bytes());
            arena.extend_from_slice(tags_bytes);
            arena.extend_from_slice(&(content_bytes.len() as u32).to_le_bytes());
            arena.extend_from_slice(content_bytes);

            let len = (arena.len() as u32) - start;
            payload_spans.push((start, len));
        }

        let hnsw_bytes = self.hnsw.to_bytes();
        let total_size = (arena_offset as usize) + arena.len() + hnsw_bytes.len();
        let mut buf = vec![0u8; total_size];

        let header = PmemHeader {
            version: PMEM_VERSION,
            flags: FLAG_SQ8 | FLAG_BQ1 | FLAG_BM25 | FLAG_GRAPH | FLAG_HNSW_CSR,
            dim,
            node_count,
            edge_count,
            hnsw_entry: self.hnsw.entry_slot.unwrap_or(0xFFFF_FFFF),
            nodes_offset,
            sq8_offset,
            bq1_offset,
            edges_offset,
            half_life_secs: self.half_life_secs,
            header_crc32: 0,
        };
        buf[0..HEADER_SIZE].copy_from_slice(&header.encode());

        // Write fixed-stride 48B node records
        for (i, rec) in self.records.iter().enumerate() {
            let off = (nodes_offset as usize) + i * NODE_RECORD_SIZE;
            let (p_off, p_len) = payload_spans[i];
            buf[off..off + 8].copy_from_slice(&rec.id.to_le_bytes());
            buf[off + 8..off + 16].copy_from_slice(&rec.created_at.to_le_bytes());
            buf[off + 16..off + 24].copy_from_slice(&rec.last_accessed.to_le_bytes());
            buf[off + 24..off + 28].copy_from_slice(&rec.access_count.to_le_bytes());
            buf[off + 28..off + 32].copy_from_slice(&rec.importance.to_le_bytes());
            buf[off + 32..off + 36].copy_from_slice(&rec.flags.to_le_bytes());
            buf[off + 36..off + 40].copy_from_slice(&rec.namespace_hash.to_le_bytes());
            buf[off + 40..off + 44].copy_from_slice(&p_off.to_le_bytes());
            buf[off + 44..off + 48].copy_from_slice(&p_len.to_le_bytes());
        }

        // Write SQ8 quantized vectors
        for (i, sq8) in self.sq8_vectors.iter().enumerate() {
            let off = (sq8_offset as usize) + i * sq8_stride;
            buf[off..off + 4].copy_from_slice(&sq8.scale.to_le_bytes());
            for (j, &code) in sq8.codes.iter().enumerate() {
                buf[off + 4 + j] = code as u8;
            }
        }

        // Write BQ1 1-bit packed vectors
        for (i, bq1) in self.bq1_vectors.iter().enumerate() {
            let off = (bq1_offset as usize) + i * (bq1_words * 8);
            for (w_idx, &word) in bq1.words.iter().enumerate() {
                let w_off = off + w_idx * 8;
                buf[w_off..w_off + 8].copy_from_slice(&word.to_le_bytes());
            }
        }

        // Write 24B Temporal Graph Edges
        for (i, edge) in self.graph.edges.iter().enumerate() {
            let off = (edges_offset as usize) + i * EDGE_RECORD_SIZE;
            buf[off..off + 8].copy_from_slice(&edge.source_id.to_le_bytes());
            buf[off + 8..off + 16].copy_from_slice(&edge.target_id.to_le_bytes());
            buf[off + 16..off + 20].copy_from_slice(&edge.weight.to_le_bytes());
            buf[off + 20] = edge.kind as u8;
        }

        // Write UTF-8 Payload Arena + HNSW CSR Adjacency Slab
        let a_off = arena_offset as usize;
        buf[a_off..a_off + arena.len()].copy_from_slice(&arena);
        let h_off = a_off + arena.len();
        buf[h_off..h_off + hnsw_bytes.len()].copy_from_slice(&hnsw_bytes);

        buf
    }

    /// Saves the `.pmem` binary image to `path`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let bytes = self.to_bytes();
        write_atomic(path, &bytes)
    }

    /// Loads a `.pmem` file via OS virtual memory mapping (`MmapView`).
    pub fn load_mmap(path: &Path) -> Result<Self, String> {
        let mmap = MmapView::open(path).map_err(|e| format!("Failed to open .pmem file: {}", e))?;
        let is_native = mmap.is_native_mmap();
        let mut engine = Self::from_bytes(mmap.as_slice())?;
        engine.last_mmap_native = is_native;
        Ok(engine)
    }

    /// Decodes a `.pmem` binary image from a byte slice (e.g. from `MmapView`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let header = PmemHeader::decode(bytes)?;
        let dim = header.dim as usize;
        let node_count = header.node_count as usize;
        let edge_count = header.edge_count as usize;
        let bq1_words = (dim + 63) / 64;
        let sq8_stride = 4 + dim;
        let arena_offset =
            (header.edges_offset as usize) + edge_count * EDGE_RECORD_SIZE;

        if bytes.len() < arena_offset {
            return Err("Truncated .pmem file: smaller than section offsets".to_string());
        }

        let mut engine = Self::new(dim, header.half_life_secs);
        let has_hnsw_csr = (header.flags & FLAG_HNSW_CSR) != 0;
        let mut max_payload_end = 0usize;

        for i in 0..node_count {
            let n_off = (header.nodes_offset as usize) + i * NODE_RECORD_SIZE;
            let id = u64::from_le_bytes(bytes[n_off..n_off + 8].try_into().unwrap());
            let created_at =
                u64::from_le_bytes(bytes[n_off + 8..n_off + 16].try_into().unwrap());
            let last_accessed =
                u64::from_le_bytes(bytes[n_off + 16..n_off + 24].try_into().unwrap());
            let access_count =
                u32::from_le_bytes(bytes[n_off + 24..n_off + 28].try_into().unwrap());
            let importance =
                f32::from_le_bytes(bytes[n_off + 28..n_off + 32].try_into().unwrap());
            let flags = u32::from_le_bytes(bytes[n_off + 32..n_off + 36].try_into().unwrap());
            let namespace_hash =
                u32::from_le_bytes(bytes[n_off + 36..n_off + 40].try_into().unwrap());
            let payload_offset =
                u32::from_le_bytes(bytes[n_off + 40..n_off + 44].try_into().unwrap()) as usize;
            let payload_len =
                u32::from_le_bytes(bytes[n_off + 44..n_off + 48].try_into().unwrap()) as usize;

            if payload_offset + payload_len > max_payload_end {
                max_payload_end = payload_offset + payload_len;
            }

            let p_base = arena_offset + payload_offset;
            let ns_len = u16::from_le_bytes(bytes[p_base..p_base + 2].try_into().unwrap()) as usize;
            let ns_str = String::from_utf8_lossy(&bytes[p_base + 2..p_base + 2 + ns_len]).into_owned();

            let t_base = p_base + 2 + ns_len;
            let tags_len = u16::from_le_bytes(bytes[t_base..t_base + 2].try_into().unwrap()) as usize;
            let tags_str = String::from_utf8_lossy(&bytes[t_base + 2..t_base + 2 + tags_len]);
            let tags: Vec<String> = if tags_str.is_empty() {
                Vec::new()
            } else {
                tags_str.split('\x1F').map(|s| s.to_string()).collect()
            };

            let c_base = t_base + 2 + tags_len;
            let content_len =
                u32::from_le_bytes(bytes[c_base..c_base + 4].try_into().unwrap()) as usize;
            let content =
                String::from_utf8_lossy(&bytes[c_base + 4..c_base + 4 + content_len]).into_owned();

            // Decode SQ8
            let s_off = (header.sq8_offset as usize) + i * sq8_stride;
            let scale = f32::from_le_bytes(bytes[s_off..s_off + 4].try_into().unwrap());
            let mut codes = Vec::with_capacity(dim);
            for j in 0..dim {
                codes.push(bytes[s_off + 4 + j] as i8);
            }
            let sq8 = Sq8Vector::from_raw(scale, codes);

            // Decode BQ1
            let b_off = (header.bq1_offset as usize) + i * (bq1_words * 8);
            let mut words = Vec::with_capacity(bq1_words);
            for w in 0..bq1_words {
                let w_off = b_off + w * 8;
                words.push(u64::from_le_bytes(bytes[w_off..w_off + 8].try_into().unwrap()));
            }
            let bq1 = Bq1Vector { dim, words };

            let slot = i as u32;
            let mut indexable = content.clone();
            for t in &tags {
                indexable.push(' ');
                indexable.push_str(t);
            }

            engine.records.push(MemoryRecord {
                id,
                namespace: ns_str,
                namespace_hash,
                content,
                tags,
                created_at,
                last_accessed,
                access_count,
                importance,
                flags,
            });
            if (flags & NODE_FLAG_TOMBSTONE) == 0 {
                engine.id_to_slot.insert(id, slot);
            }
            if id >= engine.next_id {
                engine.next_id = id + 1;
            }
            engine.sq8_vectors.push(sq8);
            engine.bq1_vectors.push(bq1);
            if !has_hnsw_csr {
                engine.hnsw.insert(slot, &engine.sq8_vectors);
            }
            engine.bm25.insert(slot, &indexable);
        }

        if has_hnsw_csr {
            let h_off = arena_offset + max_payload_end;
            if let Some((restored_hnsw, _)) = HnswIndex::from_bytes(&bytes[h_off..]) {
                engine.hnsw = restored_hnsw;
            } else {
                for slot in 0..(node_count as u32) {
                    engine.hnsw.insert(slot, &engine.sq8_vectors);
                }
            }
        }

        // Decode Graph Edges
        for e_idx in 0..edge_count {
            let e_off = (header.edges_offset as usize) + e_idx * EDGE_RECORD_SIZE;
            let source_id = u64::from_le_bytes(bytes[e_off..e_off + 8].try_into().unwrap());
            let target_id = u64::from_le_bytes(bytes[e_off + 8..e_off + 16].try_into().unwrap());
            let weight = f32::from_le_bytes(bytes[e_off + 16..e_off + 20].try_into().unwrap());
            let kind = EdgeKind::from_u8(bytes[e_off + 20]);
            engine.graph.add_edge(source_id, target_id, weight, kind);
        }

        Ok(engine)
    }

    /// Returns all edges in the temporal knowledge graph.
    pub fn edges(&self) -> &[GraphEdge] {
        &self.graph.edges
    }
}

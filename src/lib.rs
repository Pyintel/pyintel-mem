//! # `pyintel-mem` (`pmem`)
//!
//! Sub-millisecond, single-file binary vector (`SQ8`/`BQ1` HNSW), Okapi BM25 lexical,
//! and temporal knowledge graph memory engine for local AI agents.

pub mod bm25;
pub mod engine;
pub mod format;
pub mod graph;
pub mod hnsw;
pub mod quant;

pub use engine::{MemoryRecord, PyintelMem, QueryOptions, QueryReport, ScoredMemory};
pub use format::{MmapView, PmemHeader, PMEM_MAGIC, PMEM_VERSION};
pub use graph::{EdgeKind, GraphEdge, TemporalGraph};
pub use quant::{deterministic_text_embedding, Bq1Vector, Sq8Vector};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_sq8_and_bq1_quantization_accuracy() {
        let v1 = deterministic_text_embedding(
            "Pyintel Lux 14-byte binary telemetry frame over ESP-NOW",
            128,
        );
        let v2 = deterministic_text_embedding(
            "Pyintel Lux binary telemetry over ESP-NOW wireless mesh",
            128,
        );
        let v3 = deterministic_text_embedding(
            "Multivariable calculus partial derivatives and tangent planes",
            128,
        );

        let sq1 = Sq8Vector::quantize(&v1);
        let sq2 = Sq8Vector::quantize(&v2);
        let sq3 = Sq8Vector::quantize(&v3);

        let sim_close = sq1.cosine_similarity(&sq2);
        let sim_far = sq1.cosine_similarity(&sq3);

        assert!(
            sim_close > 0.65,
            "Expected high SQ8 similarity for related texts, got {}",
            sim_close
        );
        assert!(
            sim_close > sim_far + 0.35,
            "Expected related similarity ({}) >> unrelated ({})",
            sim_close,
            sim_far
        );

        let bq1 = Bq1Vector::quantize(&v1);
        let bq2 = Bq1Vector::quantize(&v2);
        let bq3 = Bq1Vector::quantize(&v3);
        assert!(bq1.hamming_similarity(&bq2) > bq1.hamming_similarity(&bq3));
    }

    #[test]
    fn test_hybrid_4_signal_recall_and_superseding_graph() {
        let mut mem = PyintelMem::new(128, 86_400.0);
        let t_base = 1_790_000_000u64;

        let id_old_port = mem.store(
            None,
            "apex",
            "Apex Studio UDP ingest listener runs on port 3000",
            vec!["network".into(), "udp".into()],
            None,
            1.0,
            false,
            Some(t_base),
        );

        let id_new_port = mem.store(
            None,
            "apex",
            "Apex Studio UDP ingest listener migrated to port 4141 (0x4C 0x58)",
            vec!["network".into(), "udp".into()],
            None,
            1.2,
            false,
            Some(t_base + 3600),
        );

        let id_dependent = mem.store(
            None,
            "apex",
            "Firewall rule must allow inbound UDP traffic on the Apex Studio ingest port",
            vec!["firewall".into(), "security".into()],
            None,
            1.1,
            false,
            Some(t_base + 3600),
        );

        // Link: id_new_port SUPERSEDES id_old_port, and id_dependent DEPENDS_ON id_new_port
        mem.link(id_new_port, id_old_port, EdgeKind::Supersedes, 1.0);
        mem.link(id_new_port, id_dependent, EdgeKind::Causes, 0.95);

        let report = mem.recall(
            "What UDP port does Apex Studio ingest listener use?",
            QueryOptions {
                top_k: 3,
                namespace: Some("apex".into()),
                now_timestamp: Some(t_base + 7200),
                ..Default::default()
            },
        );

        assert!(!report.hits.is_empty());
        // Superseded node (id_old_port) must be excluded by default, and id_new_port must be #1
        assert_eq!(report.hits[0].id, id_new_port);
        assert!(report.hits.iter().all(|h| h.id != id_old_port));
        // Dependent firewall rule should receive positive graph_boost via CAUSES edge!
        let dep_hit = report
            .hits
            .iter()
            .find(|h| h.id == id_dependent)
            .expect("Expected graph-linked firewall memory to be recalled");
        assert!(
            dep_hit.graph_boost > 0.0,
            "Expected positive 1-hop graph boost, got {}",
            dep_hit.graph_boost
        );
    }

    #[test]
    fn test_ebbinghaus_decay_and_spaced_repetition() {
        let mut mem = PyintelMem::new(64, 3600.0); // 1-hour half-life
        let t0 = 1_790_000_000u64;

        let id_stale = mem.store(
            None,
            "notes",
            "Sensor calibration offset is +0.12 rad on joint alpha",
            vec!["calibration".into()],
            None,
            1.0,
            false,
            Some(t0),
        );
        let id_fresh = mem.store(
            None,
            "notes",
            "Sensor calibration offset is +0.15 rad on joint alpha",
            vec!["calibration".into()],
            None,
            1.0,
            false,
            Some(t0 + 7200), // 2 hours newer
        );

        let rep = mem.recall(
            "Sensor calibration offset joint alpha",
            QueryOptions {
                top_k: 2,
                now_timestamp: Some(t0 + 7200),
                reinforce_on_recall: false,
                ..Default::default()
            },
        );

        assert_eq!(rep.hits[0].id, id_fresh);
        assert_eq!(rep.hits[1].id, id_stale);
        assert!(rep.hits[0].temporal_decay > rep.hits[1].temporal_decay);
    }

    #[test]
    fn test_pmem_binary_roundtrip_and_native_mmap() {
        let mut mem = PyintelMem::new(128, 604_800.0);
        let id1 = mem.store(
            None,
            "robotics",
            "ELISA Lab AutoVLA achieved 0.2742m mean ADE across 100 benchmark frames",
            vec!["elisa".into(), "vla".into()],
            None,
            2.0,
            true,
            Some(1_790_000_100),
        );
        let id2 = mem.store(
            None,
            "hardware",
            "Pyintel Lux uses magic sync bytes 0x4C 0x58 and CRC-16/CCITT-FALSE",
            vec!["lux".into(), "embedded".into()],
            None,
            1.5,
            false,
            Some(1_790_000_200),
        );
        mem.link(id1, id2, EdgeKind::RelatesTo, 0.8);

        let tmp_path: PathBuf = std::env::temp_dir().join("pyintel_mem_roundtrip_test.pmem");
        mem.save(&tmp_path).expect("Failed to save .pmem file");

        let mut loaded =
            PyintelMem::load_mmap(&tmp_path).expect("Failed to load .pmem via OS mmap");
        #[cfg(target_os = "windows")]
        assert!(
            loaded.last_mmap_native,
            "Expected native Windows MapViewOfFile memory mapping"
        );

        assert_eq!(loaded.dim, 128);
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.edges().len(), 1);

        let rep = loaded.recall(
            "0x4C 0x58 CRC-16",
            QueryOptions {
                now_timestamp: Some(1_790_000_300),
                ..Default::default()
            },
        );
        assert_eq!(rep.hits[0].id, id2);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_smart_typo_recovery_camelcase_and_consolidation() {
        let mut mem = PyintelMem::new(128, 604_800.0);
        let id1 = mem.store(
            None,
            "research",
            "AutoVLA benchmark score achieved 0.2742m Mean ADE on ELISA Lab dataset",
            vec!["vla".into()],
            None,
            1.5,
            false,
            Some(1_790_000_000),
        );
        // Storing the exact same fact in the same namespace should auto-consolidate into id1
        let id_dup = mem.store(
            None,
            "research",
            "AutoVLA benchmark score achieved 0.2742m Mean ADE on ELISA Lab dataset",
            vec!["robotics".into()],
            None,
            2.0,
            false,
            Some(1_790_000_100),
        );
        assert_eq!(id1, id_dup);
        assert_eq!(mem.records.len(), 1);
        assert_eq!(mem.records[0].access_count, 2);
        assert_eq!(mem.records[0].tags.len(), 2);

        // Query with typos ("autvla", "scre") and conversational filler ("what is the")
        let rep = mem.recall(
            "what is the autvla ADE scre?",
            QueryOptions {
                top_k: 3,
                ..Default::default()
            },
        );
        assert!(!rep.hits.is_empty());
        assert_eq!(rep.hits[0].id, id1);
        assert!(
            rep.hits[0].bm25_score > 0.4,
            "Expected fuzzy BM25 typo recovery for 'autvla'/'scre', got {}",
            rep.hits[0].bm25_score
        );
    }

    #[test]
    fn test_forget_and_tombstone_lifecycle() {
        let mut mem = PyintelMem::new(128, 604_800.0);
        let id1 = mem.store(
            None,
            "security",
            "Temporary staging token: sk-temp-999888777",
            vec!["auth".into()],
            None,
            1.0,
            false,
            Some(1_790_000_000),
        );
        let id2 = mem.store(
            None,
            "security",
            "Production API gateway is active on port 20128",
            vec!["gateway".into()],
            None,
            1.5,
            false,
            Some(1_790_000_010),
        );

        // Both are active initially
        let rep1 = mem.recall("staging token sk-temp", QueryOptions::default());
        assert_eq!(rep1.hits[0].id, id1);

        // Forget id1 by ID
        assert!(mem.forget(id1));
        assert!(!mem.forget(id1), "Second forget should return false");

        // Subsequent recall should NEVER surface id1
        let rep2 = mem.recall("staging token sk-temp", QueryOptions::default());
        assert!(rep2.hits.iter().all(|h| h.id != id1));

        // Forget id2 by query
        let forgotten = mem.forget_query("production api gateway");
        assert!(forgotten.is_some());
        let (forgotten_id, _) = forgotten.unwrap();
        assert_eq!(forgotten_id, id2);

        // Active node count is now 0
        let active = mem.records.iter().filter(|r| !r.is_tombstoned()).count();
        assert_eq!(active, 0);

        // Save to disk and reload: tombstones must persist across sessions
        let tmp_path = PathBuf::from("target").join("test_tombstone_persist.pmem");
        let _ = std::fs::remove_file(&tmp_path);
        mem.save(&tmp_path).expect("save pmem");

        let mut loaded = PyintelMem::load_mmap(&tmp_path).expect("load pmem");
        let loaded_active = loaded.records.iter().filter(|r| !r.is_tombstoned()).count();
        assert_eq!(loaded_active, 0);

        let rep3 = loaded.recall("staging token", QueryOptions::default());
        assert!(rep3.hits.is_empty());

        let _ = std::fs::remove_file(&tmp_path);
    }
}


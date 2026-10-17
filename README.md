# Pyintel Mem (`pyintel-mem`)

> Single-file, memory-mapped (`.pmem`) binary vector (`SQ8`/`BQ1` HNSW), Okapi BM25 lexical, and temporal knowledge graph memory engine for local AI agents.

**Short-form:** `pmem` | **Magic Bytes:** `0x50 0x4D` (`'P'` `'M'`) | **File Extension:** `.pmem` | **License:** MIT

---

## Why `pyintel-mem`?

Most AI agent memory libraries (Chroma, Qdrant, LangChain vector wrappers) require heavy Python/Docker runtimes, HTTP/JSON serialization, and rely on flat vector-only search that misses exact identifiers, loses causal relationships, and suffers from recall collapse under selective metadata filters.

`pyintel-mem` compiles to a single **zero-dependency Rust binary (`pmem`)** and library crate with **Python**, **TypeScript**, and **Model Context Protocol (MCP)** interfaces:

| Feature | Cloud / Container Vector DBs | `pyintel-mem` (`.pmem`) |
|---|---|---|
| **Storage Layout** | Multi-directory RocksDB / SQLite + Parquet | Single portable `.pmem` binary file (`0x50 0x4D` `'PM'`) |
| **OS Memory Mapping** | Heavy heap deserialization | Native OS `mmap` (`CreateFileMappingW` / `MapViewOfFile` on Windows) |
| **Vector Quantization** | 32-bit `float32` (`4D` bytes) | Symmetric `int8` (`SQ8`) + 1-bit Binary (`BQ1` hardware `POPCNT`) |
| **Retrieval Signals** | Dense vector only | **4-Signal Hybrid**: `SQ8`/`BQ1` Vector + Okapi BM25 + 2-Hop Graph Boost + Ebbinghaus Decay |
| **Filtered Recall** | Naive post-filtering (loses recall at `<5%` selectivity) | **Adaptive Selectivity Router**: switches to pre-filtered SIMD `SQ8` scan below 15% selectivity |
| **Memory Lifecycle** | Static text chunks | Spaced-repetition exponential decay + `SUPERSEDES` graph edge pruning |

---

## Empirical Benchmark (`pmem bench --nodes 2500 --dim 128`)

Measured on Windows (`x86_64-pc-windows-gnu`, `rustc 1.99.1 --release`, LTO enabled):

| Metric | Measured Result |
|---|---|
| **Nodes & Graph Edges Indexed** | `2,500` nodes (`128-D`) + `833` directed graph edges |
| **`.pmem` Single-File Size** | `775,412 bytes` (`757.24 KB` total — **`310.2 bytes/node`** including vectors, edges, and UTF-8 text) |
| **Index Build Throughput** | `262.05 ms` total (`104.8 µs/node`) |
| **`.pmem` Atomic Save Time** | `40.08 ms` |
| **Native OS `mmap` Verified** | `true` (`MapViewOfFile` zero-copy virtual memory view + CRC-32 header verification) |
| **Selective Filtered Query (`5%` selectivity)** | **`342.6 µs` (`0.34 ms`) p50** / `679.0 µs` p99 (`prefiltered_simd_sq8`, 100% filtered recall) |
| **Broad Unfiltered Query (`2,500` nodes)** | **`863.2 µs` (`0.86 ms`) p50** / `1,197.0 µs` p99 (`hnsw_in_graph` + BM25 + 2-hop graph activation) |

> **Note on Embeddings:** By default, when raw `f32` embedding vectors are not passed explicitly by the caller, `pyintel-mem` uses a fast deterministic CountSketch n-gram/bigram projection (`deterministic_text_embedding`) paired with Okapi BM25. For neural semantic embeddings, callers can pass dense `f32` vectors from `google/embeddinggemma-2` or any ONNX/local embedder directly to `PyintelMem::store()`.

---

## Repository Structure

```
pyintel-mem/
├── Cargo.toml                   # Zero-dependency Rust crate & CLI manifest
├── spec/
│   └── pmem-format.md           # Canonical 64-byte header & .pmem binary format specification
├── src/
│   ├── lib.rs                   # Public Rust library API & unit test suite
│   ├── format.rs                # .pmem binary codec, IEEE CRC-32, and Windows/POSIX OS mmap
│   ├── quant.rs                 # SQ8 (symmetric int8) & BQ1 (1-bit POPCNT) quantization kernels
│   ├── hnsw.rs                  # Multi-layer HNSW graph + Adaptive Selectivity Router
│   ├── bm25.rs                  # Okapi BM25 lexical inverted index for exact identifiers/code
│   ├── graph.rs                 # Typed temporal knowledge graph & 2-hop spreading activation
│   ├── engine.rs                # Unified 4-signal hybrid engine & spaced-repetition decay
│   └── bin/
│       └── pmem.rs              # CLI, benchmark runner & JSON-RPC 2.0 MCP stdio server
└── sdk/
    ├── python/                  # Zero-dependency Python SDK (`pyintel_mem`)
    └── ts/                      # TypeScript / Node.js SDK (`@pyintel/mem`)
```

---

## Quickstart

### 1. Build & Run CLI
```powershell
cargo test
cargo build --release

# Initialize a .pmem store
.\target\release\pmem.exe init agent.pmem --dim 128

# Store memories
.\target\release\pmem.exe store agent.pmem --ns apex --tags network,udp --text "Apex Studio UDP listener migrated to port 4141 (0x4C 0x58)"
.\target\release\pmem.exe store agent.pmem --ns apex --tags firewall --text "Inbound UDP firewall rule must allow the Apex Studio ingest port"

# Link causal/temporal graph edges
.\target\release\pmem.exe link agent.pmem --src 1 --dst 2 --kind CAUSES --weight 0.95

# Hybrid 4-signal recall
.\target\release\pmem.exe recall agent.pmem "What UDP port does Apex Studio use?"
```

### 2. Mount as an MCP Server (Antigravity / Apex Arc / Claude Code)
```json
{
  "mcpServers": {
    "pyintel-mem": {
      "command": "P:/Projects/Pyintel/Pyintel/pyintel-mem/target/release/pmem.exe",
      "args": ["mcp", "P:/Projects/Pyintel/Pyintel/pyintel-mem/agent.pmem"]
    }
  }
}
```
Exposes `mem_store`, `mem_recall`, `mem_link`, and `mem_stats` over JSON-RPC 2.0 stdio.

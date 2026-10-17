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
| **Memory Lifecycle** | Static text chunks | Spaced-repetition exponential decay + `SUPERSEDES` graph edge pruning + automatic duplicate consolidation |
| **Typo & Code Resilience** | Misses typos or CamelCase identifiers | **Damerau-Levenshtein $\le 1$** fuzzy typo recovery + CamelCase/alphanumeric boundary splitting (`AutoVLA`, `RP2350`, `0x4C58`) + gapped character skip-grams |
| **Federated Search** | Single database instance | **`pmem global`** searches across every tracked `.pmem` store on disk in `< 100 µs` |
| **Binary Footprint** | Hundreds of MBs (Docker / Python venv) | **`536.5 KB`** standalone `pmem.exe` binary (`0` external Rust crates) |

---

## Empirical Benchmark (`pmem bench --nodes 2500 --dim 128`)

Measured on Windows (`x86_64-pc-windows-gnu`, `rustc 1.99.1 --release`, LTO + strip + `panic="abort"`):

| Metric | Measured Result |
|---|---|
| **Standalone Binary Size (`pmem.exe`)** | **`549,376 bytes` (`536.5 KB`)** — zero external crate dependencies |
| **Nodes & Graph Edges Indexed** | `2,500` nodes (`128-D`) + `833` directed graph edges |
| **`.pmem` Single-File Size** | `1,114,136 bytes` (`1,088.02 KB` total — **`445.7 bytes/node`** including vectors, serialized HNSW layers, edges, and UTF-8 text) |
| **Single-Store Query (`16`–`21` nodes)** | **`31.0 µs – 52.9 µs`** (`native_mmap = true`) |
| **Federated `pmem global` (3 `.pmem` stores)** | **`84.0 µs` query** (`1.00 ms` total including OS `mmap` across all 3 stores) |
| **Selective Filtered Query (`5%` selectivity, `2,500` nodes)** | **`332.2 µs` – `417.6 µs` p50** (`prefiltered_simd_sq8`, 100% filtered recall) |
| **Broad Unfiltered Query (`2,500` nodes)** | **`611.5 µs` – `637.0 µs` p50** (`hnsw_in_graph` + BM25 + 2-hop graph activation) |

> **Note on Embeddings:** By default, when raw `f32` embedding vectors are not passed explicitly by the caller, `pyintel-mem` uses a fast deterministic CountSketch n-gram/skip-gram/bigram projection (`deterministic_text_embedding`) paired with `u64` hash-keyed Okapi BM25. For neural semantic embeddings, callers can pass dense `f32` vectors from `google/embeddinggemma-2` or any ONNX/local embedder directly to `PyintelMem::store()`.

---

## Repository Structure

```
pyintel-mem/
├── Cargo.toml                   # Zero-dependency Rust crate & CLI manifest (536.5 KB release binary)
├── spec/
│   └── pmem-format.md           # Canonical 64-byte header & .pmem binary format specification
├── src/
│   ├── lib.rs                   # Public Rust library API & unit test suite (5/5 passing)
│   ├── format.rs                # .pmem binary codec, IEEE CRC-32, and Windows/POSIX OS mmap
│   ├── quant.rs                 # SQ8 (symmetric int8), BQ1 (1-bit POPCNT) & skip-gram CountSketch
│   ├── hnsw.rs                  # Multi-layer HNSW graph + Adaptive Selectivity Router + CSR codec
│   ├── bm25.rs                  # Hash-keyed Okapi BM25 + CamelCase splitter + edit-dist<=1 typo recovery
│   ├── graph.rs                 # Typed temporal knowledge graph & 2-hop spreading activation
│   ├── engine.rs                # Unified 4-signal hybrid engine, synergy & auto-consolidation
│   └── bin/
│       └── pmem.rs              # CLI, federated `pmem global` registry, bench & MCP stdio server
└── sdk/
    ├── python/                  # Zero-dependency Python SDK (`pyintel_mem`)
    └── ts/                      # TypeScript / Node.js SDK (`@pyintel/mem`)
```

---

## Quickstart

### 1. Install & Run Shorthand CLI (`pmem`)
```powershell
cargo install --path .

# Store memories (defaults to workspace.pmem or $PMEM_FILE when path is omitted)
pmem store "Apex Studio UDP listener migrated to port 4141 (0x4C 0x58)" --ns apex --tags network,udp --importance 2.0
pmem store "Inbound UDP firewall rule must allow the Apex Studio ingest port" --ns apex --tags firewall

# Link causal/temporal graph edges
pmem link --src 1 --dst 2 --kind CAUSES --weight 0.95

# Hybrid 4-signal recall (resilient to typos like 'scre' -> 'score')
pmem recall "what is the autoVLA ADE scre?"

# Delete / forget outdated memories by ID or natural query (tombstone flag 0x01)
pmem forget 1
pmem forget "outdated firewall rule"

# Federated global search across EVERY tracked .pmem store on disk
pmem global list
pmem global "what is the autoVLA ADE scre?"
```

### 2. Mount as an MCP Server (Antigravity / Apex Arc / Claude Code)
```json
{
  "mcpServers": {
    "pyintel-mem": {
      "command": "pmem",
      "args": ["mcp", "P:/Projects/Pyintel/Pyintel/pyintel-mem/workspace.pmem"]
    }
  }
}
```
Exposes `mem_store`, `mem_recall`, `mem_link`, `mem_forget` (or `mem_delete`), and `mem_stats` over JSON-RPC 2.0 stdio.

### 3. Harvest Active Chat Memory into `.pmem` (Atomic Chunker)
To migrate an active coding session or chatbot conversation into persistent `.pmem` storage without creating large monolithic text dumps, paste the **Agent Memory Harvester Prompt** ([`https://pyintel.cc/pmem-harvest.prompt`](https://pyintel.cc/pmem-harvest.prompt)) into Cursor, Antigravity, Claude, or ChatGPT:
```powershell
# Or pipe directly in CLI
irm https://pyintel.cc/pmem-harvest.prompt | claude -p
```
The agent will automatically:
1. Deconstruct the entire conversation into **atomic 1–2 sentence facts** (20–50 words each).
2. Classify facts into namespaces (`architecture`, `config`, `auth`, `preference`, `decision`).
3. Pin non-negotiable rules against Ebbinghaus decay.
4. Link `SUPERSEDES`, `CAUSES`, and `DEPENDS_ON` edges across the knowledge graph.


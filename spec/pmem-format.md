# Pyintel Mem (`.pmem`) Binary File Specification

**Version:** `0.1.0`  
**Magic Bytes:** `0x50 0x4D` (`'P'` `'M'`)  
**Endianness:** Little-Endian (`LE`)  
**Alignment:** 64-byte header, 8-byte aligned sections for zero-copy `mmap`

---

## 1. File Header (64 Bytes)

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| `0x00` | 2B | `[u8; 2]` | `magic` | `0x50 0x4D` (ASCII `'P'`, `'M'`) |
| `0x02` | 2B | `u16` | `version` | Format version (`0x0001`) |
| `0x04` | 4B | `u32` | `flags` | Bit 0: `SQ8`, Bit 1: `BQ1`, Bit 2: `BM25`, Bit 3: `GRAPH` |
| `0x08` | 4B | `u32` | `dim` | Embedding vector dimensionality $D$ |
| `0x0C` | 4B | `u32` | `node_count` | Number of memory records stored |
| `0x10` | 4B | `u32` | `edge_count` | Number of directed temporal graph edges |
| `0x14` | 4B | `u32` | `hnsw_entry` | Entry point node ID (`0xFFFF_FFFF` if empty) |
| `0x18` | 8B | `u64` | `nodes_offset` | Byte offset to Node Metadata Records |
| `0x20` | 8B | `u64` | `sq8_offset` | Byte offset to `SQ8` quantized vectors ($N \times (4 + D)$ bytes) |
| `0x28` | 8B | `u64` | `bq1_offset` | Byte offset to `BQ1` 1-bit packed vectors ($N \times \lceil D/64 \rceil \times 8$ bytes) |
| `0x30` | 8B | `u64` | `edges_offset` | Byte offset to Temporal Graph Edge Records ($E \times 24$ bytes) |
| `0x38` | 4B | `f32` | `half_life_secs` | Default Ebbinghaus half-life $T_{1/2}$ in seconds |
| `0x3C` | 4B | `u32` | `header_crc32` | IEEE CRC-32 checksum over bytes `0x00..0x3B` |

---

## 2. Fixed-Stride Node Record (48 Bytes per Node)

Located at `nodes_offset`, indexed `0 .. node_count - 1`:

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| `0x00` | 8B | `u64` | `id` | Unique external/internal memory ID |
| `0x08` | 8B | `u64` | `created_at` | Creation Unix timestamp (seconds) |
| `0x10` | 8B | `u64` | `last_accessed` | Last recall/reinforcement Unix timestamp (seconds) |
| `0x18` | 4B | `u32` | `access_count` | Spaced-repetition reinforcement counter |
| `0x1C` | 4B | `f32` | `importance` | Base priority weight $\in [0.0, 10.0]$ (default `1.0`) |
| `0x20` | 4B | `u32` | `flags` | Bit 0: `TOMBSTONE`, Bit 1: `PINNED` (exempt from decay) |
| `0x24` | 4B | `u32` | `namespace_hash`| FNV-1a / XXH32 hash of namespace string for O(1) pre-filtering |
| `0x28` | 4B | `u32` | `payload_offset`| Offset into Payload Arena for UTF-8 text & tags |
| `0x2C` | 4B | `u32` | `payload_len` | Byte length of UTF-8 payload record |

---

## 3. Quantized Vector Slab (`SQ8` & `BQ1`)

### A. `SQ8` Section (`sq8_offset`)
Each node $i \in [0, N)$ occupies $4 + D$ contiguous bytes:
* `scale` (`f32`, 4 bytes): $s_i = \max_j |x_{i,j}| / 127.0$
* `codes` (`[i8; D]`, $D$ bytes): $q_{i,j} = \text{round}(x_{i,j} / s_i) \in [-127, 127]$

Dot product between two `SQ8` vectors is computed via integer multiply-accumulate:
$$\langle x_a, x_b \rangle \approx s_a s_b \sum_{j=0}^{D-1} \text{i32}(q_{a,j}) \cdot \text{i32}(q_{b,j})$$

### B. `BQ1` Section (`bq1_offset`)
Each node $i \in [0, N)$ occupies $W = \lceil D / 64 \rceil$ `u64` words ($8W$ bytes):
* Bit $b$ of word $w$ is `1` iff $x_{i, 64w + b} > 0.0$.
* Hamming similarity in $[0, 1]$ is evaluated in $W$ CPU instructions:
$$\text{Sim}_{\text{BQ1}}(a, b) = 1.0 - \frac{1}{D} \sum_{w=0}^{W-1} \text{popcnt}(B_{a,w} \oplus B_{b,w})$$

---

## 4. Temporal Graph Edge Record (24 Bytes per Edge)

Located at `edges_offset`:

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| `0x00` | 8B | `u64` | `source_id` | Source memory node ID |
| `0x08` | 8B | `u64` | `target_id` | Target memory node ID |
| `0x10` | 4B | `f32` | `weight` | Edge strength $\in [0.0, 1.0]$ |
| `0x14` | 1B | `u8` | `kind` | `1=RelatesTo`, `2=Causes`, `3=Supersedes`, `4=PartOf`, `5=DependsOn` |
| `0x15` | 3B | `[u8;3]`| `reserved` | Zero-padded alignment |

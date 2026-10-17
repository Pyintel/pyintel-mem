---
name: pyintel-mem
description: >-
  Sub-millisecond single-file binary vector (SQ8/BQ1 HNSW), BM25 lexical,
  and 2-hop temporal knowledge graph memory for local AI agents (.pmem).
  Use when storing persistent facts, recalling contextual decisions, linking causal/temporal
  relationships (CAUSES, SUPERSEDES), resolving contradictions, or forgetting outdated memories.
---

# Pyintel Mem (`.pmem`) Agent Skill

Pyintel Mem is a zero-dependency, single-file binary memory engine operating over native OS memory mapping (`mmap`). It combines symmetric 8-bit scalar quantization (`SQ8`), 1-bit hardware `POPCNT` binary signatures (`BQ1`), hash-keyed Okapi BM25+ with typo recovery, and a 2-hop temporal knowledge graph inside a single `.pmem` file with a 64-byte header (`0x50 0x4D`).

## When to Invoke
- **Store Fact:** After solving a difficult problem, discovering project configuration (ports, endpoints, auth tokens), finalizing architectural decisions, or learning user preferences.
- **Recall Context:** At the start of a debugging session, before picking a library or database, or when user asks questions about prior work or project facts.
- **Link Relationships:** When a new fact causes (`CAUSES`), supersedes (`SUPERSEDES`), or depends on (`DEPENDS_ON`) an existing fact.
- **Forget / Delete:** When an old fact is deprecated, credentials rotate, or erroneous assumptions must be purged ($O(1)$ tombstoning).

## Standard MCP Tools Reference (`pyintel-mem`)
When connected via Model Context Protocol, the following tools are available:

1. **`mem_store`**:
   - `content` (string, required): Fact or knowledge snippet to store.
   - `namespace` (string, optional, default `"default"`): Category/subsystem (e.g. `"apex"`, `"network"`, `"research"`).
   - `tags` (array of strings, optional): Indexed filter tags (e.g. `["udp", "firewall"]`).
   - `importance` (float, optional, default `1.0`): Priority multiplier between `0.1` and `10.0`.
   - `pin` (boolean, optional, default `false`): If true, memory never decays under Ebbinghaus forgetting curves.

2. **`mem_recall`**:
   - `query` (string, required): Natural language search query or code symbol (e.g. `"what port does apex ingest on"`). Typo-tolerant up to Damerau-Levenshtein <= 1 (`"scre"` -> `"score"`).
   - `namespace` (string, optional): Filter by namespace.
   - `tag` (string, optional): Filter by tag.
   - `top_k` (integer, optional, default `5`): Maximum scored hits to return.

3. **`mem_link`**:
   - `source_id` (integer, required): Source memory node ID.
   - `target_id` (integer, required): Target memory node ID.
   - `kind` (string, optional, default `"RELATES_TO"`): One of `CAUSES`, `SUPERSEDES`, `RELATES_TO`, `PART_OF`, `DEPENDS_ON`.
     - *Tip:* When node B replaces node A, link `mem_link(source_id=B, target_id=A, kind="SUPERSEDES")` to automatically suppress the outdated node during recall.
   - `weight` (float, optional, default `0.9`): Edge confidence weight in `[0.0, 1.0]`.

4. **`mem_forget`** (alias: **`mem_delete`**):
   - `id` (integer, optional): Numeric memory node ID to tombstone.
   - `query` (string, optional): Natural language search query of the memory to find and tombstone automatically.

5. **`mem_stats`**:
   - Returns active nodes count, edges count, embedding dimensions, file size, and geometry.

---

## Shorthand CLI Commands (`pmem`)

```powershell
# Store fact with namespace and tags
pmem store "Apex Studio UDP listener migrated to port 4141 (0x4C58)" --ns apex --tags network,udp --importance 2.0

# Recall memory (resilient to typos and CamelCase splits)
pmem recall "what is the apex udp port?"

# Link causal / superseding graph edges
pmem link --src 2 --dst 1 --kind SUPERSEDES --weight 1.0

# Delete / forget outdated memory by ID or search query
pmem forget 1
pmem forget "UDP port 3000"

# Federated cross-store search across EVERY tracked .pmem file on disk in <100 µs
pmem global list
pmem global "what is the autoVLA ADE scre?"

# Benchmark engine performance locally
pmem bench --nodes 2500 --dim 128

# Install this skill into your active agent directory
pmem skill install
```

---

## Best Practices for AI Agents

1. **Namespace Hygiene:** Use consistent namespaces (e.g. `config`, `architecture`, `auth`, `hardware`, `tasks`).
2. **Auto-Consolidation:** Identical or near-identical facts stored in the same namespace automatically reinforce `access_count` rather than creating duplicate nodes.
3. **Superseding Over Deleting:** If a fact evolved (e.g., port changed from 3000 to 4141), link with `SUPERSEDES` so the historical lineage is preserved while suppressing the old fact from recall. If a fact was purely wrong or private, use `pmem forget` / `mem_forget`.

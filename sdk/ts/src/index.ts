/**
 * `@pyintel/mem` — TypeScript / Node.js SDK for Pyintel Mem (`.pmem`)
 * Supports zero-copy 64-byte binary header parsing via `DataView` and native `pmem` execution.
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export const PMEM_MAGIC = 0x4d50; // Little-endian 'P' (0x50) 'M' (0x4D)

export interface PmemHeader {
  version: number;
  flags: number;
  dim: number;
  nodeCount: number;
  edgeCount: number;
  hnswEntry: number;
  nodesOffset: bigint;
  sq8Offset: bigint;
  bq1Offset: bigint;
  edgesOffset: bigint;
  halfLifeSecs: number;
  headerCrc32: number;
}

export interface ScoredHit {
  id: number;
  namespace: string;
  content: string;
  tags: string[];
  final_score: number;
  vector_score: number;
  bm25_score: number;
  graph_boost: number;
  temporal_decay: number;
  importance: number;
  access_count: number;
}

export interface ForgetResult {
  status: string;
  id: number;
  active_nodes: number;
  content: string;
  file: string;
}

export interface RecallReport {
  route: string;
  latency_micros: number;
  candidate_count: number;
  total_active_nodes: number;
  native_mmap: boolean;
  hits: ScoredHit[];
}

export function decodePmemHeader(buffer: ArrayBuffer | Uint8Array): PmemHeader {
  const view =
    buffer instanceof Uint8Array
      ? new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)
      : new DataView(buffer);

  if (view.byteLength < 64) {
    throw new Error(`Invalid .pmem buffer: length ${view.byteLength} < 64`);
  }
  const magic = view.getUint16(0, true);
  if (magic !== PMEM_MAGIC) {
    throw new Error(`Invalid .pmem magic: 0x${magic.toString(16)}`);
  }

  return {
    version: view.getUint16(2, true),
    flags: view.getUint32(4, true),
    dim: view.getUint32(8, true),
    nodeCount: view.getUint32(12, true),
    edgeCount: view.getUint32(16, true),
    hnswEntry: view.getUint32(20, true),
    nodesOffset: view.getBigUint64(24, true),
    sq8Offset: view.getBigUint64(32, true),
    bq1Offset: view.getBigUint64(40, true),
    edgesOffset: view.getBigUint64(48, true),
    halfLifeSecs: view.getFloat32(56, true),
    headerCrc32: view.getUint32(60, true),
  };
}

export class PyintelMem {
  readonly dbPath: string;
  readonly binPath: string;

  constructor(dbPath: string, binPath = "pmem") {
    this.dbPath = resolve(dbPath);
    this.binPath = binPath;
  }

  inspectHeader(): PmemHeader {
    const raw = readFileSync(this.dbPath);
    return decodePmemHeader(raw.subarray(0, 64));
  }

  store(
    text: string,
    opts?: { namespace?: string; tags?: string[]; importance?: number; pin?: boolean }
  ): { status: string; id: number; namespace: string; nodes: number } {
    const args = ["store", this.dbPath, "--text", text];
    if (opts?.namespace) args.push("--ns", opts.namespace);
    if (opts?.tags?.length) args.push("--tags", opts.tags.join(","));
    if (opts?.importance !== undefined) args.push("--importance", String(opts.importance));
    if (opts?.pin) args.push("--pin");

    const out = execFileSync(this.binPath, args, { encoding: "utf-8" });
    return JSON.parse(out.trim());
  }

  recall(
    query: string,
    opts?: { topK?: number; namespace?: string; tag?: string }
  ): RecallReport {
    const args = ["recall", this.dbPath, query, "--json"];
    if (opts?.topK !== undefined) args.push("--top-k", String(opts.topK));
    if (opts?.namespace) args.push("--ns", opts.namespace);
    if (opts?.tag) args.push("--tag", opts.tag);

    const out = execFileSync(this.binPath, args, { encoding: "utf-8" });
    return JSON.parse(out.trim());
  }

  forget(idOrQuery: number | string): ForgetResult {
    const args = ["forget", this.dbPath, String(idOrQuery), "--json"];
    const out = execFileSync(this.binPath, args, { encoding: "utf-8" });
    return JSON.parse(out.trim());
  }

  delete(idOrQuery: number | string): ForgetResult {
    return this.forget(idOrQuery);
  }
}

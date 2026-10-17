"""
Seed & Live Interactive Verification of `workspace.pmem` using `pyintel-mem`
Populates real-world memories across Pyintel, Apex, Athera, ELISA Lab, SECS HPC,
Fall 2026 Coursework, and Admin, wires a typed Temporal Knowledge Graph, and runs
live multi-hop, lexical/identifier, and superseding recall tests.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk" / "python"))

from pyintel_mem import PyintelMemClient

DB_PATH = ROOT / "workspace.pmem"
BIN_PATH = ROOT / "target" / "release" / "pmem.exe"


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    mem = PyintelMemClient(DB_PATH, BIN_PATH)
    mem.init(dim=128, half_life_secs=604800.0)

    # 1. Pyintel / Lux / Apex / Athera Ecosystem Memories
    m_lux_spec = mem.store(
        "Pyintel Lux uses a 14-byte zero-copy binary header with magic bytes 0x4C 0x58 ('LX'), uint16 sequence number, uint16 symbol ID, uint32 microsecond timestamp, and CRC-16/CCITT-FALSE",
        namespace="pyintel",
        tags=["lux", "protocol", "binary"],
        importance=2.2,
        pin=True,
    )["id"]

    m_lux_swarm = mem.store(
        "Pyintel Lux 7-node multi-bearer hardware swarm bridges 4 ESP32 nodes (ESP-NOW Ch 1) and 3 Nordic nRF nodes (BBC micro:bit + 2 Adafruit CLUEs) over BLE Channel 37 (2402 MHz)",
        namespace="pyintel",
        tags=["lux", "swarm", "ble", "espnow"],
        importance=1.9,
    )["id"]

    m_lux_coexist = mem.store(
        "ESP32 shares a single 2.4 GHz antenna between Wi-Fi/ESP-NOW and BLE; time-slicing BLE scan to 25% duty cycle (interval=160, window=40) prevents ESP-NOW packet drops",
        namespace="pyintel",
        tags=["lux", "esp32", "coexistence", "rf"],
        importance=1.8,
    )["id"]

    m_apex_old = mem.store(
        "LEGACY: Apex Studio backend prototype ingested simulated telemetry on port 3000",
        namespace="apex",
        tags=["studio", "legacy"],
        importance=1.0,
    )["id"]

    m_apex_new = mem.store(
        "Apex Studio Phase 1 backend ingests live hardware Pyintel Lux frames on UDP port 4141 (0x4C58) and USB-Serial COM9 with 20/20 unit tests passing",
        namespace="apex",
        tags=["studio", "udp", "serial", "com9"],
        importance=2.0,
        pin=True,
    )["id"]

    m_athera_mc = mem.store(
        "Athera Motion Core is a custom RP2350 PCB under Athera Controls with CAN-FD transceiver, MOSFET actuation channels, Qwiic I2C, and lux-emb firmware",
        namespace="athera",
        tags=["hardware", "rp2350", "canfd"],
        importance=1.7,
    )["id"]

    m_pmem = mem.store(
        "Pyintel Mem (.pmem, magic 0x50 0x4D 'PM') is a single-file Rust binary vector (SQ8/BQ1 HNSW), Okapi BM25, and 2-hop temporal knowledge graph memory store with Windows MapViewOfFile mmap",
        namespace="pyintel",
        tags=["pmem", "memory", "rust", "hnsw"],
        importance=2.2,
        pin=True,
    )["id"]

    # 2. ELISA Lab & SECS HPC Servers
    m_elisa_autovla = mem.store(
        "ELISA Lab (Dr. Jun Chen): AutoVLA with intent conditioning and symmetry augmentation achieved 0.2742m Mean ADE, 0.2101m Median ADE, 98% Intent Accuracy, and 94% Steering Accuracy across 100 benchmark frames",
        namespace="research",
        tags=["elisa", "autovla", "vlm", "robotics"],
        importance=2.3,
        pin=True,
    )["id"]

    m_elisa_thor = mem.store(
        "High-parameter VLM fine-tuning (Qwen2-VL-7B and Pixtral-12B in BF16) executed on NVIDIA Thor with 122 GB unified VRAM using Dynamic Epochs Contract Engine",
        namespace="research",
        tags=["elisa", "thor", "qwen2-vl", "pixtral"],
        importance=1.9,
    )["id"]

    m_secs_yoko = mem.store(
        "Oakland University SECS HPC server yoko.secs.oakland.edu has 48 CPU threads (Dual Xeon E5-2670 v3), 96 GB RAM, and Dual NVIDIA Tesla K80 GPUs (~23 GB VRAM, 2688 CUDA cores); Gaga was retired",
        namespace="infra",
        tags=["secs", "hpc", "yoko", "gpu"],
        importance=1.8,
        pin=True,
    )["id"]

    m_secs_lennon = mem.store(
        "Oakland University SECS HPC server lennon.secs.oakland.edu has 48 CPU threads, 96 GB RAM, and 1536 CUDA cores; used for Pyintel Vantage Gemma-4-E2B + EmbeddingGemma-2 CPU float32 integration checks",
        namespace="infra",
        tags=["secs", "hpc", "lennon", "vantage"],
        importance=1.7,
    )["id"]

    # 3. Fall 2026 Coursework & Admin
    m_egr2400 = mem.store(
        "EGR 2400 (Intro to ECE, Dr. Khalid Mirza, EC 254): Midterm Exam 1 is Tuesday, October 20, 2026 (3:30-5:17 PM, 20% of grade) covering digital logic, K-maps, Verilog HDL, muxes, adders, and flip-flops",
        namespace="coursework",
        tags=["egr2400", "exam", "digital-logic"],
        importance=2.5,
        pin=True,
    )["id"]

    m_eng1510 = mem.store(
        "ENG 1510 (Composition I, Noreen Mitchell, OCC D2L): Final Reflection Essay (600-1200 words MLA 9) is due Tuesday, October 20, 2026 at 11:59 PM and completes all course requirements",
        namespace="coursework",
        tags=["eng1510", "essay", "deadline"],
        importance=2.4,
        pin=True,
    )["id"]

    m_phy1510 = mem.store(
        "PHY 1510 (Intro Physics I, Dr. Jyothi Raman, HH 190): WebAssign HW 5 (extended) and HW 7 due Sunday, October 18, 2026 at 11:59 PM; Exam 1 score was 42.5/50 (85%) + extra credit (~96.8% course average)",
        namespace="coursework",
        tags=["phy1510", "webassign", "physics"],
        importance=2.1,
    )["id"]

    m_uscis = mem.store(
        "USCIS Form I-539 / I-907 F-1 Change of Status (Receipt IOE9265528770, SEVIS N0037994637): Premium Processing clock started October 8, 2026 with 30-day decision window targeting November 7, 2026",
        namespace="admin",
        tags=["uscis", "visa", "f1", "i539"],
        importance=2.5,
        pin=True,
    )["id"]

    m_nasa = mem.store(
        "NASA Space Apps Challenge 2026 (Nov 14-15, 2026): Team AresPath AI building Interplanetary Survival Guide Martian Map for astronaut EVA 3D terrain traversability and slope hazard routing",
        namespace="hackathon",
        tags=["nasa", "arespath", "mars"],
        importance=1.8,
    )["id"]

    # Wire Typed Temporal Knowledge Graph Edges
    mem.link(m_apex_new, m_apex_old, kind="SUPERSEDES", weight=1.0)
    mem.link(m_lux_swarm, m_lux_coexist, kind="DEPENDS_ON", weight=0.95)
    mem.link(m_lux_swarm, m_lux_spec, kind="PART_OF", weight=0.90)
    mem.link(m_apex_new, m_lux_spec, kind="DEPENDS_ON", weight=0.90)
    mem.link(m_athera_mc, m_lux_spec, kind="DEPENDS_ON", weight=0.85)
    mem.link(m_elisa_autovla, m_elisa_thor, kind="CAUSES", weight=0.92)
    mem.link(m_elisa_autovla, m_secs_yoko, kind="RELATES_TO", weight=0.80)

    hdr = mem.inspect_header()
    stats = mem.stats()
    print("=" * 74)
    print(f"SEEDED WORKSPACE MEMORY: {DB_PATH}")
    print(
        f"Header Magic: 'PM' (v{hdr.version}) | Dim: {hdr.dim} | Active Nodes: {stats['active_nodes']}/{stats['total_nodes']} "
        f"(1 Superseded) | Graph Edges: {stats['edges']} | Size: {stats['size_bytes']} bytes ({stats['size_bytes']/1024:.2f} KB)"
    )
    print("=" * 74)

    queries = [
        ("Query 1 — Superseding Fact Check (Port 3000 vs Port 4141)", "What port does Apex Studio backend ingest telemetry on?", None),
        ("Query 2 — Multi-Hop Hardware Swarm + Coexistence", "How does the 7-node BLE and ESP-NOW swarm avoid antenna drops?", "pyintel"),
        ("Query 3 — Exact Receipt / Identifier Lookup (BM25 + SQ8)", "IOE9265528770 N0037994637 F-1 premium processing window", None),
        ("Query 4 — ELISA Lab Research & GPU Hardware", "AutoVLA Mean ADE trajectory benchmark and VLM fine-tuning", "research"),
        ("Query 5 — Immediate Next Week Academic Deadlines", "What exams and essays are due on Tuesday October 20?", "coursework"),
    ]

    for title, q, ns in queries:
        rep = mem.recall(q, top_k=3, namespace=ns)
        print(f"\n>>> {title}")
        print(f"    Q: \"{q}\" (ns={ns or 'ALL'})")
        print(
            f"    Latency: {rep['latency_micros']:.1f} µs | Route: {rep['route']} | "
            f"Candidates: {rep['candidate_count']}/{rep['total_active_nodes']} | Native mmap: {rep['native_mmap']}"
        )
        for rank, h in enumerate(rep["hits"], 1):
            print(
                f"      #{rank} [id={h['id']:<2}] score={h['final_score']:.4f} "
                f"(vec={h['vector_score']:.3f} bm25={h['bm25_score']:.3f} graph={h['graph_boost']:.3f} imp={h['importance']:.1f}) "
                f"[{h['namespace']}]"
            )
            print(f"         -> {h['content']}")

    # Also test MCP JSON-RPC tool call (`tools/call` -> `mem_recall`) over stdio!
    print("\n" + "=" * 74)
    print("LIVE MCP SERVER STDIO TOOL CALL (`tools/call` -> `mem_recall`):")
    mcp_payload = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        + "\n"
        + json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "mem_recall",
                    "arguments": {
                        "query": "Which SECS server replaced Gaga and what GPUs does it have?",
                        "top_k": 2,
                    },
                },
            }
        )
        + "\n"
    )
    proc = subprocess.run(
        [str(BIN_PATH), "mcp", str(DB_PATH)],
        input=mcp_payload,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    mcp_resp = json.loads(lines[-1])
    inner_json = json.loads(mcp_resp["result"]["content"][0]["text"])
    print(f"MCP Response Latency: {inner_json['latency_micros']} µs | Top Hit: {inner_json['hits'][0]['content']}")
    print("=" * 74)


if __name__ == "__main__":
    main()

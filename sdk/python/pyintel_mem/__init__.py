"""
Pyintel Mem (`pyintel-mem`) — Python SDK
Zero-dependency Python interface to the `.pmem` binary vector & temporal graph engine.
Supports direct binary header inspection (`0x50 0x4D` 'PM') and subprocess/CLI bridge to `pmem`.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PMEM_MAGIC = b"PM"
HEADER_STRUCT = struct.Struct("<2sHIIIIIQQQQfI")


@dataclass
class PmemHeaderInfo:
    version: int
    flags: int
    dim: int
    node_count: int
    edge_count: int
    hnsw_entry: int
    nodes_offset: int
    sq8_offset: int
    bq1_offset: int
    edges_offset: int
    half_life_secs: float
    header_crc32: int


class PyintelMemClient:
    """High-speed Python wrapper for `pmem` (.pmem binary stores)."""

    def __init__(self, db_path: str | Path, binary_path: Optional[str | Path] = None) -> None:
        self.db_path = Path(db_path).resolve()
        if binary_path:
            self.bin = str(Path(binary_path).resolve())
        else:
            repo_release = Path(__file__).resolve().parents[3] / "target" / "release" / "pmem.exe"
            if repo_release.exists():
                self.bin = str(repo_release)
            else:
                self.bin = shutil.which("pmem") or str(repo_release)

    def inspect_header(self) -> PmemHeaderInfo:
        """Reads the 64-byte `.pmem` header directly via zero-copy file slice."""
        with open(self.db_path, "rb") as f:
            raw = f.read(64)
        if len(raw) < 64:
            raise ValueError(f"File too small ({len(raw)} < 64 bytes)")
        unpacked = HEADER_STRUCT.unpack(raw)
        if unpacked[0] != PMEM_MAGIC:
            raise ValueError(f"Invalid magic bytes: {unpacked[0]!r}")
        return PmemHeaderInfo(*unpacked[1:])

    def init(self, dim: int = 128, half_life_secs: float = 604800.0) -> None:
        subprocess.run(
            [self.bin, "init", str(self.db_path), "--dim", str(dim), "--half-life", str(half_life_secs)],
            check=True,
            capture_output=True,
            text=True,
        )

    def store(
        self,
        text: str,
        namespace: str = "default",
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
        pin: bool = False,
    ) -> Dict[str, Any]:
        cmd = [
            self.bin,
            "store",
            str(self.db_path),
            "--text",
            text,
            "--ns",
            namespace,
            "--importance",
            str(importance),
        ]
        if tags:
            cmd.extend(["--tags", ",".join(tags)])
        if pin:
            cmd.append("--pin")
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(res.stdout.strip())

    def link(
        self,
        source_id: int,
        target_id: int,
        kind: str = "RELATES_TO",
        weight: float = 0.9,
    ) -> Dict[str, Any]:
        cmd = [
            self.bin,
            "link",
            str(self.db_path),
            "--src",
            str(source_id),
            "--dst",
            str(target_id),
            "--kind",
            kind,
            "--weight",
            str(weight),
        ]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(res.stdout.strip())

    def recall(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        cmd = [
            self.bin,
            "recall",
            str(self.db_path),
            query,
            "--top-k",
            str(top_k),
            "--json",
        ]
        if namespace:
            cmd.extend(["--ns", namespace])
        if tag:
            cmd.extend(["--tag", tag])
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(res.stdout.strip())

    def forget(self, id_or_query: int | str) -> Dict[str, Any]:
        """Tombstones/deletes a memory by node ID or best-matching search query."""
        cmd = [self.bin, "forget", str(self.db_path), str(id_or_query), "--json"]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(res.stdout.strip())

    def delete(self, id_or_query: int | str) -> Dict[str, Any]:
        """Alias for forget()."""
        return self.forget(id_or_query)

    def stats(self) -> Dict[str, Any]:
        res = subprocess.run(
            [self.bin, "stats", str(self.db_path), "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(res.stdout.strip())

//! `pmem` — Pyintel Mem Standalone CLI, Empirical Benchmark Runner & MCP Stdio Server

use pyintel_mem::{EdgeKind, PyintelMem, QueryOptions};
use std::env;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" || args[0] == "help" {
        print_usage();
        return;
    }

    let sub = args[0].as_str();
    let rest = &args[1..];

    let res = match sub {
        "init" => cmd_init(rest),
        "store" => cmd_store(rest),
        "link" => cmd_link(rest),
        "recall" => cmd_recall(rest),
        "stats" => cmd_stats(rest),
        "bench" => cmd_bench(rest),
        "mcp" => cmd_mcp(rest),
        _ => Err(format!("Unknown subcommand '{}'. Run `pmem --help`.", sub)),
    };

    if let Err(e) = res {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    println!(
        "\
Pyintel Mem (`pmem`) v0.1.0 — Sub-Millisecond Binary Vector & Temporal Graph Memory
Magic: 0x50 0x4D ('PM') | Quantization: SQ8 + BQ1 | Hybrid: Vector + BM25 + Graph + Decay

USAGE:
  pmem recall [<file.pmem>] \"<query>\" [--top-k 5] [--ns <namespace>] [--tag <tag>] [--json]
  pmem store  [<file.pmem>] \"<fact>\" [--ns <namespace>] [--tags a,b] [--importance 1.0] [--pin]
  pmem link   [<file.pmem>] --src <id> --dst <id> [--kind CAUSES|SUPERSEDES|RELATES_TO|PART_OF|DEPENDS_ON] [--weight 0.9]
  pmem stats  [<file.pmem>] [--json]
  pmem init   [<file.pmem>] [--dim 128] [--half-life 604800]
  pmem bench  [--nodes 2000] [--dim 128]
  pmem mcp    [<file.pmem>]
"
    );
}

fn default_db_path() -> PathBuf {
    if let Ok(env_path) = env::var("PMEM_FILE") {
        if !env_path.trim().is_empty() {
            return PathBuf::from(env_path);
        }
    }
    let workspace_primary = PathBuf::from(r"P:\Projects\Pyintel\Pyintel\pyintel-mem\workspace.pmem");
    if workspace_primary.exists()
        || Path::new(r"P:\Projects\Pyintel\Pyintel\pyintel-mem").exists()
    {
        return workspace_primary;
    }
    if let Ok(home) = env::var("USERPROFILE").or_else(|_| env::var("HOME")) {
        return PathBuf::from(home).join(".pmem").join("workspace.pmem");
    }
    PathBuf::from("workspace.pmem")
}

/// If `args[0]` looks like an explicit `.pmem` path, consumes it; otherwise uses `default_db_path()`.
fn resolve_db_and_args(args: &[String]) -> (PathBuf, &[String]) {
    if let Some(first) = args.first() {
        let lower = first.to_ascii_lowercase();
        if !first.starts_with('-') && (lower.ends_with(".pmem") || lower.ends_with(".db")) {
            return (PathBuf::from(first), &args[1..]);
        }
    }
    (default_db_path(), args)
}

fn load_or_create(path: &Path, default_dim: usize, default_hl: f32) -> Result<PyintelMem, String> {
    if path.exists() {
        PyintelMem::load_mmap(path)
    } else {
        Ok(PyintelMem::new(default_dim, default_hl))
    }
}

fn cmd_init(args: &[String]) -> Result<(), String> {
    let (path, rem) = resolve_db_and_args(args);
    let mut dim = 128usize;
    let mut half_life = 604_800.0f32;

    let mut i = 0;
    while i < rem.len() {
        match rem[i].as_str() {
            "--dim" if i + 1 < rem.len() => {
                dim = rem[i + 1].parse().map_err(|_| "Invalid --dim")?;
                i += 2;
            }
            "--half-life" if i + 1 < rem.len() => {
                half_life = rem[i + 1].parse().map_err(|_| "Invalid --half-life")?;
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mem = PyintelMem::new(dim, half_life);
    mem.save(&path).map_err(|e| e.to_string())?;
    println!(
        "Initialized .pmem store at '{}' (dim={}, half_life={}s, header=64B)",
        path.display(),
        dim,
        half_life
    );
    Ok(())
}

fn cmd_store(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "Usage: pmem store [<file.pmem>] \"<fact>\" [--ns default] [--tags a,b] [--importance 1.0] [--pin]"
                .into(),
        );
    }
    let (path, rem) = resolve_db_and_args(args);
    let mut text = String::new();
    let mut ns = String::from("default");
    let mut tags: Vec<String> = Vec::new();
    let mut importance = 1.0f32;
    let mut pin = false;
    let mut custom_id: Option<u64> = None;

    let mut i = 0;
    while i < rem.len() {
        match rem[i].as_str() {
            "--text" | "-t" if i + 1 < rem.len() => {
                text = rem[i + 1].clone();
                i += 2;
            }
            "--ns" | "--namespace" if i + 1 < rem.len() => {
                ns = rem[i + 1].clone();
                i += 2;
            }
            "--tags" if i + 1 < rem.len() => {
                tags = rem[i + 1]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                i += 2;
            }
            "--importance" if i + 1 < rem.len() => {
                importance = rem[i + 1].parse().map_err(|_| "Invalid --importance")?;
                i += 2;
            }
            "--id" if i + 1 < rem.len() => {
                custom_id = Some(rem[i + 1].parse().map_err(|_| "Invalid --id")?);
                i += 2;
            }
            "--pin" => {
                pin = true;
                i += 1;
            }
            other if !other.starts_with('-') && text.is_empty() => {
                text = other.to_string();
                i += 1;
            }
            _ => i += 1,
        }
    }

    if text.is_empty() {
        return Err("Missing fact text. Pass `pmem store \"<fact>\"` or `--text \"<fact>\"`.".into());
    }

    let mut mem = load_or_create(&path, 128, 604_800.0)?;
    let id = mem.store(custom_id, &ns, &text, tags, None, importance, pin, None);
    mem.save(&path).map_err(|e| e.to_string())?;
    println!(
        "{{\"status\":\"ok\",\"id\":{},\"namespace\":\"{}\",\"nodes\":{}}}",
        id,
        escape_json(&ns),
        mem.records.len()
    );
    Ok(())
}

fn cmd_link(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "Usage: pmem link [<file.pmem>] --src <id> --dst <id> [--kind CAUSES|SUPERSEDES|RELATES_TO|PART_OF|DEPENDS_ON] [--weight 0.9]"
                .into(),
        );
    }
    let (path, rem) = resolve_db_and_args(args);
    let mut src: Option<u64> = None;
    let mut dst: Option<u64> = None;
    let mut kind = EdgeKind::RelatesTo;
    let mut weight = 0.9f32;

    let mut i = 0;
    while i < rem.len() {
        match rem[i].as_str() {
            "--src" if i + 1 < rem.len() => {
                src = Some(rem[i + 1].parse().map_err(|_| "Invalid --src")?);
                i += 2;
            }
            "--dst" if i + 1 < rem.len() => {
                dst = Some(rem[i + 1].parse().map_err(|_| "Invalid --dst")?);
                i += 2;
            }
            "--kind" if i + 1 < rem.len() => {
                kind = EdgeKind::from_str(&rem[i + 1]);
                i += 2;
            }
            "--weight" if i + 1 < rem.len() => {
                weight = rem[i + 1].parse().map_err(|_| "Invalid --weight")?;
                i += 2;
            }
            _ => i += 1,
        }
    }

    let s = src.ok_or("Missing --src <id>")?;
    let d = dst.ok_or("Missing --dst <id>")?;

    let mut mem = PyintelMem::load_mmap(&path)?;
    mem.link(s, d, kind, weight);
    mem.save(&path).map_err(|e| e.to_string())?;
    println!(
        "{{\"status\":\"ok\",\"src\":{},\"dst\":{},\"kind\":\"{}\",\"weight\":{:.2},\"edges\":{}}}",
        s,
        d,
        kind.as_str(),
        weight,
        mem.edges().len()
    );
    Ok(())
}

fn cmd_recall(args: &[String]) -> Result<(), String> {
    let (path, rem) = resolve_db_and_args(args);
    if rem.is_empty() {
        return Err(
            "Usage: pmem recall [<file.pmem>] \"<query>\" [--top-k 5] [--ns <namespace>] [--tag <tag>] [--json]"
                .into(),
        );
    }
    let query = &rem[0];
    let mut opts = QueryOptions::default();
    let mut as_json = false;

    let mut i = 1;
    while i < rem.len() {
        match rem[i].as_str() {
            "--top-k" | "-k" if i + 1 < rem.len() => {
                opts.top_k = rem[i + 1].parse().map_err(|_| "Invalid --top-k")?;
                i += 2;
            }
            "--ns" | "--namespace" if i + 1 < rem.len() => {
                opts.namespace = Some(rem[i + 1].clone());
                i += 2;
            }
            "--tag" if i + 1 < rem.len() => {
                opts.required_tag = Some(rem[i + 1].clone());
                i += 2;
            }
            "--include-superseded" => {
                opts.include_superseded = true;
                i += 1;
            }
            "--json" => {
                as_json = true;
                i += 1;
            }
            _ => i += 1,
        }
    }

    let mut mem = PyintelMem::load_mmap(&path)?;
    let mmap_native = mem.last_mmap_native;
    let report = mem.recall(query, opts.clone());
    if opts.reinforce_on_recall && !report.hits.is_empty() {
        let _ = mem.save(&path);
    }

    if as_json {
        println!("{}", format_report_json(&report, mmap_native));
    } else {
        println!(
            "=== Pyintel Mem Recall ({:.1} µs | route: {} | candidates: {}/{} | native_mmap: {}) ===",
            report.latency_micros,
            report.route_used,
            report.candidate_count,
            report.total_active_nodes,
            mmap_native
        );
        for (rank, h) in report.hits.iter().enumerate() {
            println!(
                "  #{:<2} [id={}] (score={:.4} | vec={:.3} bm25={:.3} graph={:.3} decay={:.3} imp={:.1}) [{}]",
                rank + 1,
                h.id,
                h.final_score,
                h.vector_score,
                h.bm25_score,
                h.graph_boost,
                h.temporal_decay,
                h.importance,
                h.namespace
            );
            println!("      {}", h.content);
            if !h.tags.is_empty() {
                println!("      tags: {:?}", h.tags);
            }
        }
    }
    Ok(())
}

fn cmd_stats(args: &[String]) -> Result<(), String> {
    let (path, _rem) = resolve_db_and_args(args);
    let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let mem = PyintelMem::load_mmap(&path)?;
    let active = mem.records.iter().filter(|r| !r.is_tombstoned()).count();

    println!(
        "{{\"file\":\"{}\",\"size_bytes\":{},\"dim\":{},\"total_nodes\":{},\"active_nodes\":{},\"edges\":{},\"hnsw_max_layer\":{},\"native_mmap\":{}}}",
        escape_json(&path.display().to_string()),
        file_size,
        mem.dim,
        mem.records.len(),
        active,
        mem.edges().len(),
        mem.hnsw.max_layer,
        mem.last_mmap_native
    );
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let mut num_nodes = 2_500usize;
    let mut dim = 128usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--nodes" if i + 1 < args.len() => {
                num_nodes = args[i + 1].parse().map_err(|_| "Invalid --nodes")?;
                i += 2;
            }
            "--dim" if i + 1 < args.len() => {
                dim = args[i + 1].parse().map_err(|_| "Invalid --dim")?;
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!(
        "Running Pyintel Mem Empirical Benchmark (nodes={}, dim={})...",
        num_nodes, dim
    );

    let t_build = Instant::now();
    let mut mem = PyintelMem::new(dim, 604_800.0);
    let domains = ["pyintel", "apex", "athera", "robotics", "coursework"];

    for idx in 0..num_nodes {
        let ns = domains[idx % domains.len()];
        let text = match idx % 5 {
            0 => format!(
                "Pyintel Lux node #{} streaming 14-byte binary frame 0x4C58 over ESP-NOW channel {}",
                idx,
                (idx % 11) + 1
            ),
            1 => format!(
                "Apex Studio automation rule #{} triggers GPIO pin {} when temperature exceeds {}C",
                idx,
                idx % 40,
                25 + (idx % 30)
            ),
            2 => format!(
                "Athera Motion Core RP2350 controller #{} CAN-FD bus bitrate {} kbps",
                idx,
                500 + (idx % 4) * 250
            ),
            3 => format!(
                "ELISA Lab VLM trajectory benchmark frame #{} achieved ADE {:.4} meters",
                idx,
                0.20 + (idx as f32 * 0.0001)
            ),
            _ => format!(
                "EGR 2400 digital logic FullAdder circuit verification test vector #{}",
                idx
            ),
        };
        let tag = format!("batch_{}", idx % 20);
        let id = mem.store(None, ns, &text, vec![ns.to_string(), tag], None, 1.0, false, None);
        if id > 1 && idx % 3 == 0 {
            mem.link(id, id - 1, EdgeKind::RelatesTo, 0.85);
        }
    }
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    let tmp_path = env::temp_dir().join("pyintel_mem_bench.pmem");
    let t_save = Instant::now();
    mem.save(&tmp_path).map_err(|e| e.to_string())?;
    let save_ms = t_save.elapsed().as_secs_f64() * 1000.0;
    let file_size = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);

    let t_mmap = Instant::now();
    let mut loaded = PyintelMem::load_mmap(&tmp_path)?;
    let mmap_ms = t_mmap.elapsed().as_secs_f64() * 1000.0;

    // 1. Broad query (HNSW in-graph route across all 2,500 nodes)
    let mut broad_latencies = Vec::with_capacity(50);
    let mut broad_route = "";
    for _ in 0..50 {
        let rep = loaded.recall(
            "Pyintel Lux 14-byte binary frame 0x4C58 over ESP-NOW",
            QueryOptions {
                top_k: 5,
                reinforce_on_recall: false,
                ..Default::default()
            },
        );
        broad_route = rep.route_used;
        broad_latencies.push(rep.latency_micros);
    }
    broad_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_broad_us = broad_latencies[broad_latencies.len() / 2];
    let p99_broad_us = broad_latencies[(broad_latencies.len() * 99) / 100];

    // 2. Selective filtered query (5% selectivity via tag filter -> Adaptive Selectivity Router switches to prefiltered_simd_sq8)
    let mut sel_latencies = Vec::with_capacity(50);
    let mut sel_route = "";
    for _ in 0..50 {
        let rep = loaded.recall(
            "Athera Motion Core RP2350 CAN-FD bitrate",
            QueryOptions {
                top_k: 5,
                required_tag: Some("batch_2".into()),
                reinforce_on_recall: false,
                ..Default::default()
            },
        );
        sel_route = rep.route_used;
        sel_latencies.push(rep.latency_micros);
    }
    sel_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_sel_us = sel_latencies[sel_latencies.len() / 2];
    let p99_sel_us = sel_latencies[(sel_latencies.len() * 99) / 100];

    let _ = std::fs::remove_file(&tmp_path);

    println!("----------------------------------------------------------------");
    println!("  Nodes Indexed           : {} (dim={})", num_nodes, dim);
    println!("  Graph Edges             : {}", loaded.edges().len());
    println!("  .pmem Binary File Size  : {} bytes ({:.2} KB, {:.1} B/node)", file_size, file_size as f64 / 1024.0, file_size as f64 / num_nodes as f64);
    println!("  Index Build Time        : {:.2} ms ({:.1} µs/node)", build_ms, (build_ms * 1000.0) / num_nodes as f64);
    println!("  .pmem Atomic Save Time  : {:.2} ms", save_ms);
    println!("  OS mmap Load + Rehydrate: {:.2} ms (native_mmap={})", mmap_ms, loaded.last_mmap_native);
    println!("  Broad Query (route={:<20}): p50 = {:.1} µs | p99 = {:.1} µs", broad_route, p50_broad_us, p99_broad_us);
    println!("  Selective 5% (route={:<19}): p50 = {:.1} µs | p99 = {:.1} µs", sel_route, p50_sel_us, p99_sel_us);
    println!("----------------------------------------------------------------");
    Ok(())
}

/// JSON-RPC 2.0 Model Context Protocol (MCP) Server over stdio (`pmem mcp <file.pmem>`)
fn cmd_mcp(args: &[String]) -> Result<(), String> {
    let path = if args.is_empty() {
        PathBuf::from("agent_memory.pmem")
    } else {
        PathBuf::from(&args[0])
    };

    let mut mem = load_or_create(&path, 128, 604_800.0)?;
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line_res in stdin.lock().lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req_id = extract_json_raw_field(trimmed, "id").unwrap_or_else(|| "null".to_string());
        let method = extract_json_str(trimmed, "method").unwrap_or_default();

        if method == "notifications/initialized" || method.starts_with("notifications/") {
            continue;
        }

        let response = match method.as_str() {
            "initialize" => format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{{\"tools\":{{}}}},\"serverInfo\":{{\"name\":\"pyintel-mem\",\"version\":\"0.1.0\"}}}}}}",
                req_id
            ),
            "tools/list" => format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"tools\":[{}, {}, {}, {}]}}}}",
                req_id,
                r#"{"name":"mem_store","description":"Store a fact or episodic memory into the persistent .pmem binary vector & temporal graph store","inputSchema":{"type":"object","properties":{"content":{"type":"string"},"namespace":{"type":"string"},"tags":{"type":"string","description":"Comma-separated tags"},"importance":{"type":"number"}},"required":["content"]}}"#,
                r#"{"name":"mem_recall","description":"Recall relevant memories using 4-signal hybrid scoring (SQ8/BQ1 vector + Okapi BM25 + 2-hop graph boost + Ebbinghaus temporal decay)","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"namespace":{"type":"string"},"tag":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}}"#,
                r#"{"name":"mem_link","description":"Create a typed directed edge (CAUSES, SUPERSEDES, RELATES_TO, PART_OF, DEPENDS_ON) between two memory node IDs","inputSchema":{"type":"object","properties":{"source_id":{"type":"integer"},"target_id":{"type":"integer"},"kind":{"type":"string"},"weight":{"type":"number"}},"required":["source_id","target_id"]}}"#,
                r#"{"name":"mem_stats","description":"Return live telemetry and node/edge counts for the active .pmem store","inputSchema":{"type":"object","properties":{}}}"#
            ),
            "tools/call" => {
                let tool_name = extract_json_str(trimmed, "name").unwrap_or_default();
                let tool_output = match tool_name.as_str() {
                    "mem_store" => {
                        let content = extract_json_str(trimmed, "content").unwrap_or_default();
                        let ns = extract_json_str(trimmed, "namespace")
                            .unwrap_or_else(|| "default".to_string());
                        let tags_raw = extract_json_str(trimmed, "tags").unwrap_or_default();
                        let tags: Vec<String> = tags_raw
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        let imp = extract_json_num(trimmed, "importance").unwrap_or(1.0) as f32;
                        let id = mem.store(None, &ns, &content, tags, None, imp, false, None);
                        let _ = mem.save(&path);
                        format!(
                            "{{\"status\":\"stored\",\"id\":{},\"namespace\":\"{}\",\"total_nodes\":{}}}",
                            id,
                            escape_json(&ns),
                            mem.records.len()
                        )
                    }
                    "mem_recall" => {
                        let query = extract_json_str(trimmed, "query").unwrap_or_default();
                        let ns = extract_json_str(trimmed, "namespace");
                        let tag = extract_json_str(trimmed, "tag");
                        let top_k = extract_json_num(trimmed, "top_k").unwrap_or(5.0) as usize;
                        let rep = mem.recall(
                            &query,
                            QueryOptions {
                                top_k: top_k.max(1),
                                namespace: ns,
                                required_tag: tag,
                                ..Default::default()
                            },
                        );
                        let _ = mem.save(&path);
                        format_report_json(&rep, mem.last_mmap_native)
                    }
                    "mem_link" => {
                        let src = extract_json_num(trimmed, "source_id").unwrap_or(0.0) as u64;
                        let dst = extract_json_num(trimmed, "target_id").unwrap_or(0.0) as u64;
                        let kind_s = extract_json_str(trimmed, "kind")
                            .unwrap_or_else(|| "RELATES_TO".to_string());
                        let weight = extract_json_num(trimmed, "weight").unwrap_or(0.9) as f32;
                        let kind = EdgeKind::from_str(&kind_s);
                        mem.link(src, dst, kind, weight);
                        let _ = mem.save(&path);
                        format!(
                            "{{\"status\":\"linked\",\"source_id\":{},\"target_id\":{},\"kind\":\"{}\",\"edges\":{}}}",
                            src,
                            dst,
                            kind.as_str(),
                            mem.edges().len()
                        )
                    }
                    "mem_stats" => {
                        let active = mem.records.iter().filter(|r| !r.is_tombstoned()).count();
                        format!(
                            "{{\"dim\":{},\"total_nodes\":{},\"active_nodes\":{},\"edges\":{},\"native_mmap\":{}}}",
                            mem.dim,
                            mem.records.len(),
                            active,
                            mem.edges().len(),
                            mem.last_mmap_native
                        )
                    }
                    _ => format!("{{\"error\":\"Unknown tool {}\"}}", escape_json(&tool_name)),
                };

                format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}]}}}}",
                    req_id,
                    escape_json(&tool_output)
                )
            }
            _ => format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{},\"error\":{{\"code\":-32601,\"message\":\"Method not found\"}}}}",
                req_id
            ),
        };

        let _ = writeln!(stdout, "{}", response);
        let _ = stdout.flush();
    }

    Ok(())
}

fn format_report_json(rep: &pyintel_mem::QueryReport, native_mmap: bool) -> String {
    let mut hits_json = Vec::with_capacity(rep.hits.len());
    for h in &rep.hits {
        let tags_str: Vec<String> = h
            .tags
            .iter()
            .map(|t| format!("\"{}\"", escape_json(t)))
            .collect();
        hits_json.push(format!(
            "{{\"id\":{},\"namespace\":\"{}\",\"content\":\"{}\",\"tags\":[{}],\"final_score\":{:.4},\"vector_score\":{:.4},\"bm25_score\":{:.4},\"graph_boost\":{:.4},\"temporal_decay\":{:.4},\"importance\":{:.2},\"access_count\":{}}}",
            h.id,
            escape_json(&h.namespace),
            escape_json(&h.content),
            tags_str.join(","),
            h.final_score,
            h.vector_score,
            h.bm25_score,
            h.graph_boost,
            h.temporal_decay,
            h.importance,
            h.access_count
        ));
    }
    format!(
        "{{\"route\":\"{}\",\"latency_micros\":{:.2},\"candidate_count\":{},\"total_active_nodes\":{},\"native_mmap\":{},\"hits\":[{}]}}",
        rep.route_used,
        rep.latency_micros,
        rep.candidate_count,
        rep.total_active_nodes,
        native_mmap,
        hits_json.join(",")
    )
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let mut search_from = 0;
    while let Some(rel_idx) = json[search_from..].find(&pat) {
        let idx = search_from + rel_idx + pat.len();
        let rest = json[idx..].trim_start();
        if let Some(after_colon) = rest.strip_prefix(':') {
            let val_start = after_colon.trim_start();
            if let Some(after_quote) = val_start.strip_prefix('"') {
                let mut out = String::new();
                let mut chars = after_quote.chars();
                while let Some(c) = chars.next() {
                    if c == '\\' {
                        if let Some(esc) = chars.next() {
                            match esc {
                                'n' => out.push('\n'),
                                'r' => out.push('\r'),
                                't' => out.push('\t'),
                                other => out.push(other),
                            }
                        }
                    } else if c == '"' {
                        return Some(out);
                    } else {
                        out.push(c);
                    }
                }
            }
        }
        search_from = idx;
    }
    None
}

fn extract_json_num(json: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\"", key);
    let idx = json.find(&pat)? + pat.len();
    let after_colon = json[idx..].trim_start().strip_prefix(':')?.trim_start();
    let end = after_colon
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(after_colon.len());
    after_colon[..end].parse().ok()
}

fn extract_json_raw_field(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let idx = json.find(&pat)? + pat.len();
    let after_colon = json[idx..].trim_start().strip_prefix(':')?.trim_start();
    if after_colon.starts_with('"') {
        let s = extract_json_str(json, key)?;
        return Some(format!("\"{}\"", escape_json(&s)));
    }
    let end = after_colon
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(after_colon.len());
    Some(after_colon[..end].to_string())
}

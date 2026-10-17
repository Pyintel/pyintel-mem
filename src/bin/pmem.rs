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
        "forget" | "delete" | "rm" | "remove" => cmd_forget(rest),
        "global" => cmd_global(rest),
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
Pyintel Mem (`pmem`) v0.1.2 — Sub-Millisecond Binary Vector & Temporal Graph Memory
Magic: 0x50 0x4D ('PM') | Quantization: SQ8 + BQ1 | Hybrid: Vector + BM25 + Graph + Decay

USAGE:
  pmem recall [<file.pmem>] \"<query>\" [--top-k 5] [--ns <namespace>] [--tag <tag>] [--json]
  pmem forget [<file.pmem>] <id | \"query\"> [--json]
  pmem delete [<file.pmem>] <id | \"query\"> [--json]
  pmem global \"<query>\" [--top-k 5] [--ns <namespace>] [--tag <tag>] [--json]
  pmem global list | scan [<dir>] | add <file.pmem> | prune
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

fn global_registry_path() -> PathBuf {
    if let Ok(reg_env) = env::var("PMEM_REGISTRY") {
        if !reg_env.trim().is_empty() {
            return PathBuf::from(reg_env);
        }
    }
    let primary_dir = Path::new(r"P:\Projects\Pyintel\Pyintel\pyintel-mem");
    if primary_dir.exists() {
        return primary_dir.join("pmem_registry.txt");
    }
    if let Ok(home) = env::var("USERPROFILE").or_else(|_| env::var("HOME")) {
        let dir = PathBuf::from(home).join(".pmem");
        let _ = std::fs::create_dir_all(&dir);
        return dir.join("pmem_registry.txt");
    }
    PathBuf::from("pmem_registry.txt")
}

fn normalize_store_path(path: &Path) -> String {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let s = abs
        .canonicalize()
        .unwrap_or(abs)
        .display()
        .to_string();
    // Strip Windows extended-length prefix `\\?\` for clean display & comparison
    s.strip_prefix(r"\\?\").unwrap_or(&s).to_string()
}

/// Automatically records a `.pmem` store path into the global registry so `pmem global` tracks it.
fn register_store(path: &Path) {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if file_name.contains("pyintel_mem_bench") || file_name.starts_with("tmp") {
        return;
    }
    let norm = normalize_store_path(path);
    let reg_path = global_registry_path();
    let mut existing = load_registered_stores();
    if !existing
        .iter()
        .any(|p| p.to_string_lossy().eq_ignore_ascii_case(&norm))
    {
        existing.push(PathBuf::from(&norm));
        save_registered_stores(&reg_path, &existing);
    }
}

fn load_registered_stores() -> Vec<PathBuf> {
    let reg_path = global_registry_path();
    let mut stores: Vec<PathBuf> = Vec::new();
    if let Ok(content) = std::fs::read_to_string(&reg_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                let pb = PathBuf::from(trimmed);
                if !stores
                    .iter()
                    .any(|p| p.to_string_lossy().eq_ignore_ascii_case(trimmed))
                {
                    stores.push(pb);
                }
            }
        }
    }
    // Always ensure default workspace.pmem is included if it exists
    let def = default_db_path();
    if def.exists() {
        let norm = normalize_store_path(&def);
        if !stores
            .iter()
            .any(|p| p.to_string_lossy().eq_ignore_ascii_case(&norm))
        {
            stores.push(PathBuf::from(&norm));
            save_registered_stores(&reg_path, &stores);
        }
    }
    stores
}

fn save_registered_stores(reg_path: &Path, stores: &[PathBuf]) {
    if let Some(parent) = reg_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut body = String::new();
    for s in stores {
        body.push_str(&s.display().to_string());
        body.push('\n');
    }
    let _ = std::fs::write(reg_path, body);
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
        let mem = PyintelMem::load_mmap(path)?;
        register_store(path);
        Ok(mem)
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
    register_store(&path);
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
    register_store(&path);
    println!(
        "{{\"status\":\"ok\",\"id\":{},\"namespace\":\"{}\",\"nodes\":{}}}",
        id,
        escape_json(&ns),
        mem.records.len()
    );
    Ok(())
}

fn cmd_forget(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("Usage: pmem forget [<file.pmem>] <id | \"query\"> [--json]".into());
    }
    let (path, rem) = resolve_db_and_args(args);
    if rem.is_empty() {
        return Err("Specify memory node ID or search query to forget. Usage: pmem forget <id | \"query\">".into());
    }

    let mut target_id: Option<u64> = None;
    let mut target_query: Option<String> = None;
    let mut as_json = false;

    let mut i = 0;
    while i < rem.len() {
        match rem[i].as_str() {
            "--id" if i + 1 < rem.len() => {
                target_id = Some(rem[i + 1].parse().map_err(|_| "Invalid --id integer")?);
                i += 2;
            }
            "--query" | "-q" if i + 1 < rem.len() => {
                target_query = Some(rem[i + 1].clone());
                i += 2;
            }
            "--json" => {
                as_json = true;
                i += 1;
            }
            s if target_id.is_none() && target_query.is_none() && !s.starts_with('-') => {
                if let Ok(num) = s.parse::<u64>() {
                    target_id = Some(num);
                } else {
                    target_query = Some(s.to_string());
                }
                i += 1;
            }
            _ => i += 1,
        }
    }

    let mut mem = load_or_create(&path, 128, 604_800.0)?;

    let (forgotten_id, content) = if let Some(id) = target_id {
        let content = mem.records.iter().find(|r| r.id == id && !r.is_tombstoned()).map(|r| r.content.clone());
        if mem.forget(id) {
            (id, content.unwrap_or_default())
        } else {
            return Err(format!("Memory node #{} not found or already forgotten in '{}'.", id, path.display()));
        }
    } else if let Some(query) = target_query {
        if let Some((id, text)) = mem.forget_query(&query) {
            (id, text)
        } else {
            return Err(format!("No active memories matched query '{}' to forget.", query));
        }
    } else {
        return Err("Specify node ID or search query to forget: `pmem forget <id | \"query\">`".into());
    };

    mem.save(&path).map_err(|e| e.to_string())?;
    let active = mem.records.iter().filter(|r| !r.is_tombstoned()).count();

    if as_json {
        println!(
            "{{\"status\":\"forgotten\",\"id\":{},\"active_nodes\":{},\"content\":\"{}\",\"file\":\"{}\"}}",
            forgotten_id,
            active,
            escape_json(&content),
            path.display()
        );
    } else {
        let preview = if content.chars().count() > 60 {
            format!("{}...", content.chars().take(57).collect::<String>())
        } else {
            content
        };
        println!(
            "[Pyintel-Mem] Forgot memory node #{}: \"{}\". Active memories: {}.",
            forgotten_id,
            preview,
            active
        );
    }
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
    register_store(&path);
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
    register_store(&path);
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

fn scan_dir_for_pmem(dir: &Path, found: &mut Vec<PathBuf>, depth: usize) {
    if depth > 6 {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            scan_dir_for_pmem(&p, found, depth + 1);
        } else if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("pmem") {
                found.push(p);
            }
        }
    }
}

/// Federated global command: searches or manages every `.pmem` store tracked in the global registry.
fn cmd_global(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "Usage:\n  pmem global \"<query>\" [--top-k 5] [--ns <namespace>] [--tag <tag>] [--json]\n  pmem global list\n  pmem global scan [<dir>]\n  pmem global add <file.pmem>\n  pmem global prune"
                .into(),
        );
    }

    match args[0].as_str() {
        "list" | "--list" | "status" | "stats" => {
            let stores = load_registered_stores();
            println!(
                "=== Pyintel Mem Global Registry ({} tracked stores | registry: {}) ===",
                stores.len(),
                global_registry_path().display()
            );
            for (idx, sp) in stores.iter().enumerate() {
                if sp.exists() {
                    let sz = std::fs::metadata(sp).map(|m| m.len()).unwrap_or(0);
                    match PyintelMem::load_mmap(sp) {
                        Ok(mem) => {
                            let active = mem.records.iter().filter(|r| !r.is_tombstoned()).count();
                            println!(
                                "  [{}] {} ({} B | active_nodes={} | edges={} | dim={})",
                                idx + 1,
                                sp.display(),
                                sz,
                                active,
                                mem.edges().len(),
                                mem.dim
                            );
                        }
                        Err(e) => {
                            println!("  [{}] {} (ERROR: {})", idx + 1, sp.display(), e);
                        }
                    }
                } else {
                    println!("  [{}] {} (MISSING on disk — run `pmem global prune`)", idx + 1, sp.display());
                }
            }
            return Ok(());
        }
        "add" | "register" => {
            if args.len() < 2 {
                return Err("Usage: pmem global add <file.pmem>".into());
            }
            let p = PathBuf::from(&args[1]);
            if !p.exists() {
                return Err(format!("Store '{}' does not exist", p.display()));
            }
            register_store(&p);
            println!("Registered '{}' in global pmem registry.", normalize_store_path(&p));
            return Ok(());
        }
        "scan" => {
            let root = if args.len() >= 2 {
                PathBuf::from(&args[1])
            } else {
                env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
            };
            let mut found = Vec::new();
            scan_dir_for_pmem(&root, &mut found, 0);
            for f in &found {
                register_store(f);
            }
            let total = load_registered_stores().len();
            println!(
                "Scanned '{}' -> discovered {} .pmem files ({} total in global registry).",
                root.display(),
                found.len(),
                total
            );
            return Ok(());
        }
        "prune" => {
            let reg_path = global_registry_path();
            let before = load_registered_stores();
            let kept: Vec<PathBuf> = before.into_iter().filter(|p| p.exists()).collect();
            save_registered_stores(&reg_path, &kept);
            println!("Pruned global registry -> {} active .pmem stores remain.", kept.len());
            return Ok(());
        }
        _ => {}
    }

    // Allow either `pmem global "<query>"` or `pmem global recall "<query>"`
    let rem = if args[0] == "recall" || args[0] == "search" {
        if args.len() < 2 {
            return Err("Usage: pmem global \"<query>\" [--top-k 5] [--ns <ns>] [--tag <tag>] [--json]".into());
        }
        &args[1..]
    } else {
        args
    };

    let query = &rem[0];
    let mut opts = QueryOptions {
        reinforce_on_recall: false,
        ..Default::default()
    };
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
            "--reinforce" => {
                opts.reinforce_on_recall = true;
                i += 1;
            }
            "--json" => {
                as_json = true;
                i += 1;
            }
            _ => i += 1,
        }
    }

    let stores = load_registered_stores();
    let t0 = Instant::now();
    let mut global_hits: Vec<(String, String, pyintel_mem::ScoredMemory)> = Vec::new();
    let mut stores_searched = 0usize;
    let mut total_nodes = 0usize;
    let mut total_candidates = 0usize;
    let mut sum_query_us = 0.0f64;

    for sp in &stores {
        if !sp.exists() {
            continue;
        }
        if let Ok(mut mem) = PyintelMem::load_mmap(sp) {
            stores_searched += 1;
            let rep = mem.recall(query, opts.clone());
            sum_query_us += rep.latency_micros;
            total_nodes += rep.total_active_nodes;
            total_candidates += rep.candidate_count;
            if opts.reinforce_on_recall && !rep.hits.is_empty() {
                let _ = mem.save(sp);
            }
            let short_name = sp
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("store.pmem")
                .to_string();
            let full_path = sp.display().to_string();
            for h in rep.hits {
                global_hits.push((short_name.clone(), full_path.clone(), h));
            }
        }
    }

    global_hits.sort_by(|a, b| {
        b.2.final_score
            .partial_cmp(&a.2.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    global_hits.truncate(opts.top_k.max(1));
    let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

    if as_json {
        let mut items = Vec::with_capacity(global_hits.len());
        for (short_store, full_store, h) in &global_hits {
            let tags_str: Vec<String> = h
                .tags
                .iter()
                .map(|t| format!("\"{}\"", escape_json(t)))
                .collect();
            items.push(format!(
                "{{\"store\":\"{}\",\"store_path\":\"{}\",\"id\":{},\"namespace\":\"{}\",\"content\":\"{}\",\"tags\":[{}],\"final_score\":{:.4},\"vector_score\":{:.4},\"bm25_score\":{:.4},\"graph_boost\":{:.4},\"temporal_decay\":{:.4},\"importance\":{:.2}}}",
                escape_json(short_store),
                escape_json(full_store),
                h.id,
                escape_json(&h.namespace),
                escape_json(&h.content),
                tags_str.join(","),
                h.final_score,
                h.vector_score,
                h.bm25_score,
                h.graph_boost,
                h.temporal_decay,
                h.importance
            ));
        }
        println!(
            "{{\"stores_searched\":{},\"total_active_nodes\":{},\"candidate_count\":{},\"query_micros\":{:.2},\"total_mmap_micros\":{:.2},\"hits\":[{}]}}",
            stores_searched,
            total_nodes,
            total_candidates,
            sum_query_us,
            elapsed_us,
            items.join(",")
        );
    } else {
        println!(
            "=== Pyintel Mem Global Recall (query: {:.1} µs | total mmap: {:.1} µs | stores: {} | candidates: {}/{}) ===",
            sum_query_us, elapsed_us, stores_searched, total_candidates, total_nodes
        );
        for (rank, (short_store, _full_store, h)) in global_hits.iter().enumerate() {
            println!(
                "  #{:<2} [{} :: {} #{}] (score={:.4} | vec={:.3} bm25={:.3} graph={:.3} decay={:.3} imp={:.1})",
                rank + 1,
                short_store,
                h.namespace,
                h.id,
                h.final_score,
                h.vector_score,
                h.bm25_score,
                h.graph_boost,
                h.temporal_decay,
                h.importance
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
    register_store(&path);
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
                "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"tools\":[{}, {}, {}, {}, {}, {}]}}}}",
                req_id,
                r#"{"name":"mem_store","description":"Store a fact or episodic memory into the persistent .pmem binary vector & temporal graph store","inputSchema":{"type":"object","properties":{"content":{"type":"string"},"namespace":{"type":"string"},"tags":{"type":"string","description":"Comma-separated tags"},"importance":{"type":"number"}},"required":["content"]}}"#,
                r#"{"name":"mem_recall","description":"Recall relevant memories using 4-signal hybrid scoring (SQ8/BQ1 vector + Okapi BM25 + 2-hop graph boost + Ebbinghaus temporal decay)","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"namespace":{"type":"string"},"tag":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}}"#,
                r#"{"name":"mem_forget","description":"Delete or tombstone an outdated or unwanted memory by node ID or search query from the active .pmem store","inputSchema":{"type":"object","properties":{"id":{"type":"integer","description":"Memory node ID to forget"},"query":{"type":"string","description":"Search query to find and forget the top matching memory if ID is unknown"}}}}"#,
                r#"{"name":"mem_delete","description":"Alias for mem_forget. Delete or tombstone an outdated or unwanted memory by node ID or search query","inputSchema":{"type":"object","properties":{"id":{"type":"integer","description":"Memory node ID to delete"},"query":{"type":"string","description":"Search query to find and delete the top matching memory if ID is unknown"}}}}"#,
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
                    "mem_forget" | "mem_delete" => {
                        let target_id = extract_json_num(trimmed, "id").map(|n| n as u64);
                        let query = extract_json_str(trimmed, "query");

                        let result = if let Some(id) = target_id {
                            let content = mem.records.iter().find(|r| r.id == id && !r.is_tombstoned()).map(|r| r.content.clone());
                            if mem.forget(id) {
                                Some((id, content.unwrap_or_default()))
                            } else {
                                None
                            }
                        } else if let Some(ref q) = query {
                            mem.forget_query(q)
                        } else {
                            None
                        };

                        if let Some((id, text)) = result {
                            let _ = mem.save(&path);
                            let active = mem.records.iter().filter(|r| !r.is_tombstoned()).count();
                            format!(
                                "{{\"status\":\"forgotten\",\"id\":{},\"content\":\"{}\",\"active_nodes\":{}}}",
                                id,
                                escape_json(&text),
                                active
                            )
                        } else {
                            format!(
                                "{{\"status\":\"not_found\",\"message\":\"No active memory matched ID or query to forget\"}}"
                            )
                        }
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

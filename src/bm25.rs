//! Smart Okapi BM25+ Lexical Index with:
//! - `u64` FNV-1a Hash-Keyed Postings (minimal heap overhead)
//! - CamelCase & Alphanumeric Boundary Splitting (`AutoVLA` -> `autovla`, `auto`, `vla`; `RP2350` -> `rp2350`, `rp`, `2350`)
//! - Conversational Stop-Word Stripping (so natural questions don't dilute IDF denominator)
//! - Morphological Stemming (`calibration`/`calibrated` -> `calibrat`)
//! - Damerau-Levenshtein $\le 1$ Typo Recovery (`scre` -> `score`, `midtrm` -> `midterm`)

use crate::quant::fnv1a_64;
use std::collections::HashMap;

/// Inverted index entry mapping `(slot_index, term_frequency)`.
#[derive(Clone, Copy, Debug)]
pub struct Posting {
    pub slot: u32,
    pub tf: u16,
}

/// Okapi BM25 Lexical Index (`k1 = 1.2`, `b = 0.75`).
#[derive(Clone, Debug, Default)]
pub struct Bm25Index {
    postings: HashMap<u64, Vec<Posting>>,
    /// Compact unique vocabulary `(term_hash, term_str)` for $O(|V|)$ 1-edit fuzzy typo recovery on misses.
    vocab: Vec<(u64, String)>,
    doc_lengths: Vec<u32>,
    total_tokens: u64,
    active_docs: usize,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_stopword(w: &str) -> bool {
        matches!(
            w,
            "the"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "what"
                | "when"
                | "where"
                | "which"
                | "who"
                | "whom"
                | "whose"
                | "why"
                | "how"
                | "can"
                | "could"
                | "would"
                | "should"
                | "will"
                | "shall"
                | "may"
                | "might"
                | "must"
                | "do"
                | "does"
                | "did"
                | "have"
                | "has"
                | "had"
                | "having"
                | "a"
                | "an"
                | "and"
                | "or"
                | "but"
                | "if"
                | "then"
                | "else"
                | "for"
                | "of"
                | "to"
                | "in"
                | "on"
                | "at"
                | "by"
                | "with"
                | "from"
                | "into"
                | "about"
                | "as"
                | "this"
                | "that"
                | "these"
                | "those"
                | "my"
                | "your"
                | "our"
                | "their"
                | "its"
                | "me"
                | "you"
                | "us"
                | "them"
                | "tell"
                | "show"
                | "give"
                | "find"
                | "get"
        )
    }

    /// Lightweight morphological stemmer preserving technical precision.
    pub fn stem_token(w: &str) -> String {
        let len = w.len();
        if len <= 3 || w.bytes().any(|b| b.is_ascii_digit()) {
            return w.to_string();
        }
        if let Some(s) = w.strip_suffix("ation") {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
        if let Some(s) = w.strip_suffix("ments") {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
        if let Some(s) = w.strip_suffix("ment") {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
        if let Some(s) = w.strip_suffix("ings") {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
        if let Some(s) = w.strip_suffix("ing") {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
        if let Some(s) = w.strip_suffix("ies") {
            if s.len() >= 2 {
                return format!("{}y", s);
            }
        }
        if let Some(s) = w.strip_suffix("ed") {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
        if let Some(s) = w.strip_suffix("es") {
            if s.len() >= 3 && (s.ends_with('s') || s.ends_with('x') || s.ends_with('z') || s.ends_with("ch") || s.ends_with("sh")) {
                return s.to_string();
            }
        }
        if let Some(s) = w.strip_suffix('s') {
            if s.len() >= 3 && !s.ends_with('s') && !s.ends_with('u') {
                return s.to_string();
            }
        }
        w.to_string()
    }

    /// Splits CamelCase (`AutoVLA` -> `Auto`, `VLA`) and letter-digit boundaries (`RP2350` -> `RP`, `2350`).
    fn split_camel_and_digits(raw: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let chars: Vec<char> = raw.chars().collect();
        if chars.len() < 2 {
            return parts;
        }
        let mut start = 0usize;
        for i in 1..chars.len() {
            let prev = chars[i - 1];
            let curr = chars[i];
            let next_is_lower = i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase();

            let boundary = (prev.is_ascii_lowercase() && curr.is_ascii_uppercase())
                || (prev.is_ascii_uppercase() && curr.is_ascii_uppercase() && next_is_lower)
                || (prev.is_ascii_alphabetic() && curr.is_ascii_digit() && !(start == 0 && i == 1 && prev == '0' && (curr == 'x' || curr == 'X')))
                || (prev.is_ascii_digit() && curr.is_ascii_alphabetic() && !(i == 1 && prev == '0' && (curr == 'x' || curr == 'X')));

            if boundary {
                let seg: String = chars[start..i].iter().collect();
                let low = seg.to_ascii_lowercase();
                if low.len() >= 2 {
                    parts.push(low);
                }
                start = i;
            }
        }
        if start > 0 && start < chars.len() {
            let seg: String = chars[start..].iter().collect();
            let low = seg.to_ascii_lowercase();
            if low.len() >= 2 {
                parts.push(low);
            }
        }
        parts
    }

    /// Smart tokenizer: preserves full identifiers (`0x4C58`, `AutoVLA`, `RP2350`), splits delimiters
    /// and CamelCase/digit boundaries, strips conversational stop-words, and emits morphological stems.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut out = Vec::new();

        for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.') {
            let trimmed = raw.trim_matches(|c: char| c == '_' || c == '-' || c == '.');
            if trimmed.len() < 2 {
                continue;
            }
            let lower = trimmed.to_ascii_lowercase();
            if Self::is_stopword(&lower) {
                continue;
            }

            let stem = Self::stem_token(&lower);
            out.push(lower.clone());
            if stem != lower && stem.len() >= 3 {
                out.push(stem);
            }

            // Delimiter sub-tokens
            if trimmed.contains('_') || trimmed.contains('-') || trimmed.contains('.') {
                for part in trimmed.split(|c: char| c == '_' || c == '-' || c == '.') {
                    if part.len() >= 2 {
                        let p_low = part.to_ascii_lowercase();
                        if !Self::is_stopword(&p_low) {
                            out.push(p_low.clone());
                            for cp in Self::split_camel_and_digits(part) {
                                if !Self::is_stopword(&cp) {
                                    out.push(cp);
                                }
                            }
                        }
                    }
                }
            } else {
                // CamelCase / alphanumeric sub-tokens (e.g., AutoVLA -> auto, vla; RP2350 -> rp, 2350)
                for cp in Self::split_camel_and_digits(trimmed) {
                    if !Self::is_stopword(&cp) {
                        out.push(cp);
                    }
                }
            }
        }
        out
    }

    #[inline]
    fn term_hash(term: &str) -> u64 {
        fnv1a_64(term.as_bytes(), 0x424D_3235_5F50_4D45)
    }

    /// Checks whether two ASCII strings are within Damerau-Levenshtein distance 1
    /// (1 insertion, 1 deletion, 1 substitution, or 1 adjacent transposition).
    pub fn is_edit_dist_1(a: &str, b: &str) -> bool {
        let ab = a.as_bytes();
        let bb = b.as_bytes();
        let la = ab.len();
        let lb = bb.len();
        if la < 3 || lb < 3 || la.abs_diff(lb) > 1 {
            return false;
        }
        // Must share either first or last character to avoid noisy false positives on short acronyms
        if ab[0] != bb[0] && ab[la - 1] != bb[lb - 1] {
            return false;
        }

        if la == lb {
            let mut diffs = Vec::with_capacity(3);
            for i in 0..la {
                if ab[i] != bb[i] {
                    diffs.push(i);
                    if diffs.len() > 2 {
                        return false;
                    }
                }
            }
            if diffs.len() == 1 {
                return true;
            }
            if diffs.len() == 2 {
                let i = diffs[0];
                let j = diffs[1];
                return j == i + 1 && ab[i] == bb[j] && ab[j] == bb[i];
            }
            false
        } else {
            // Deletion / Insertion (la + 1 == lb or lb + 1 == la)
            let (shorter, longer) = if la < lb { (ab, bb) } else { (bb, ab) };
            let mut i = 0;
            let mut j = 0;
            let mut skipped = false;
            while i < shorter.len() && j < longer.len() {
                if shorter[i] == longer[j] {
                    i += 1;
                    j += 1;
                } else if !skipped {
                    skipped = true;
                    j += 1;
                } else {
                    return false;
                }
            }
            true
        }
    }

    /// Indexes a document at `slot`.
    pub fn insert(&mut self, slot: u32, text: &str) {
        let tokens = Self::tokenize(text);
        let len = tokens.len() as u32;

        if (slot as usize) >= self.doc_lengths.len() {
            self.doc_lengths.resize((slot as usize) + 1, 0);
        }
        self.doc_lengths[slot as usize] = len;
        self.total_tokens += len as u64;
        self.active_docs += 1;

        let mut tf_map: HashMap<u64, u16> = HashMap::with_capacity(tokens.len());
        for tok in tokens {
            let h = Self::term_hash(&tok);
            let entry = tf_map.entry(h).or_insert(0);
            if *entry == 0 && !self.postings.contains_key(&h) {
                self.vocab.push((h, tok));
            }
            *entry = entry.saturating_add(1);
        }

        for (h, tf) in tf_map {
            self.postings
                .entry(h)
                .or_default()
                .push(Posting { slot, tf });
        }
    }

    /// Computes normalized BM25 scores in `[0.0, 1.0]` for all matching slots, with automatic
    /// Damerau-Levenshtein $\le 1$ typo recovery when an exact query token misses.
    pub fn score_query(&self, query: &str) -> HashMap<u32, f32> {
        let mut raw_scores: HashMap<u32, f32> = HashMap::new();
        if self.active_docs == 0 {
            return raw_scores;
        }

        let q_tokens = Self::tokenize(query);
        if q_tokens.is_empty() {
            return raw_scores;
        }

        let avgdl = (self.total_tokens as f32 / self.active_docs as f32).max(1.0);
        let n_docs = self.active_docs as f32;
        let k1 = 1.2f32;
        let b = 0.75f32;

        let mut max_possible = 0.0f32;

        for term in &q_tokens {
            let h = Self::term_hash(term);
            if let Some(plist) = self.postings.get(&h) {
                let df = plist.len() as f32;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
                max_possible += idf * (k1 + 1.0);

                for post in plist {
                    let dl = self
                        .doc_lengths
                        .get(post.slot as usize)
                        .copied()
                        .unwrap_or(1) as f32;
                    let tf = post.tf as f32;
                    let numer = tf * (k1 + 1.0);
                    let denom = tf + k1 * (1.0 - b + b * (dl / avgdl));
                    *raw_scores.entry(post.slot).or_insert(0.0) += idf * (numer / denom);
                }
            } else {
                // Fuzzy 1-edit typo recovery across vocabulary (e.g., "scre" -> "score")
                let mut matched_fuzzy = false;
                for (vh, vword) in &self.vocab {
                    if Self::is_edit_dist_1(term, vword) {
                        if let Some(plist) = self.postings.get(vh) {
                            matched_fuzzy = true;
                            let df = plist.len() as f32;
                            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln() * 0.78;
                            max_possible += idf * (k1 + 1.0);
                            for post in plist {
                                let dl = self
                                    .doc_lengths
                                    .get(post.slot as usize)
                                    .copied()
                                    .unwrap_or(1) as f32;
                                let tf = post.tf as f32;
                                let numer = tf * (k1 + 1.0);
                                let denom = tf + k1 * (1.0 - b + b * (dl / avgdl));
                                *raw_scores.entry(post.slot).or_insert(0.0) += idf * (numer / denom);
                            }
                            break;
                        }
                    }
                }
                if !matched_fuzzy {
                    let idf = ((n_docs + 0.5) / 0.5f32 + 1.0).ln();
                    max_possible += idf * 0.35;
                }
            }
        }

        if max_possible > 1e-6 {
            let mut max_observed = 0.0f32;
            for &s in raw_scores.values() {
                if s > max_observed {
                    max_observed = s;
                }
            }
            let norm_denom = max_observed.max(max_possible * 0.35).max(1e-6);
            for val in raw_scores.values_mut() {
                *val = (*val / norm_denom).clamp(0.0, 1.0);
            }
        }

        raw_scores
    }
}


//! Okapi BM25 Lexical Inverted Index for Exact Identifier, Code & Keyword Retrieval

use std::collections::HashMap;

/// Inverted index entry mapping `(slot_index, term_frequency)`.
#[derive(Clone, Debug)]
pub struct Posting {
    pub slot: u32,
    pub tf: u16,
}

/// Okapi BM25 Lexical Index (`k1 = 1.2`, `b = 0.75`).
#[derive(Clone, Debug, Default)]
pub struct Bm25Index {
    postings: HashMap<String, Vec<Posting>>,
    doc_lengths: Vec<u32>,
    total_tokens: u64,
    active_docs: usize,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenizes text keeping alphanumeric characters, underscores, hyphens, and dots
    /// so technical identifiers (`0x4C58`, `ERR_403`, `lux_mesh.h`) remain searchable both
    /// as whole tokens and split sub-tokens.
    pub fn tokenize(text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        let mut out = Vec::new();

        for raw in lower.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.') {
            let trimmed = raw.trim_matches(|c: char| c == '_' || c == '-' || c == '.');
            if trimmed.len() >= 2 {
                out.push(trimmed.to_string());
                if trimmed.contains('_') || trimmed.contains('-') || trimmed.contains('.') {
                    for part in trimmed.split(|c: char| c == '_' || c == '-' || c == '.') {
                        if part.len() >= 2 {
                            out.push(part.to_string());
                        }
                    }
                }
            }
        }
        out
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

        let mut tf_map: HashMap<String, u16> = HashMap::new();
        for tok in tokens {
            let count = tf_map.entry(tok).or_insert(0);
            *count = count.saturating_add(1);
        }

        for (term, tf) in tf_map {
            self.postings
                .entry(term)
                .or_default()
                .push(Posting { slot, tf });
        }
    }

    /// Computes normalized BM25 scores in `[0.0, 1.0]` for all matching slots.
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
            if let Some(plist) = self.postings.get(term) {
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
                // Unseen query term contributes to theoretical max so partial matches scale accurately
                let idf = ((n_docs + 0.5) / 0.5f32 + 1.0).ln();
                max_possible += idf * 0.5;
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

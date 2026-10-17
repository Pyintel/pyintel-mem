//! Multi-Tier Vector Quantization Kernels (`SQ8` Scalar int8 & `BQ1` 1-Bit Binary POPCNT)
//!
//! Provides deterministic, zero-allocation inner-product and Hamming similarity routines
//! optimized for auto-vectorization (SIMD) and sub-millisecond candidate scoring.

/// Symmetric scalar int8 quantized vector (`SQ8`).
/// Each `f32` component `x_i` is mapped to `q_i = round(x_i / scale) in [-127, 127]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sq8Vector {
    pub scale: f32,
    pub norm_sq: i32,
    pub codes: Vec<i8>,
}

impl Sq8Vector {
    /// Quantizes an `f32` slice into symmetric `i8` codes (`[-127, 127]`).
    /// Normalizes the input vector to unit L2 length before quantization so dot products
    /// directly approximate cosine similarity.
    pub fn quantize(vec: &[f32]) -> Self {
        let mut l2_sq = 0.0f32;
        for &v in vec {
            l2_sq += v * v;
        }
        let inv_norm = if l2_sq > 1e-12 { 1.0 / l2_sq.sqrt() } else { 0.0 };

        let mut max_abs = 0.0f32;
        for &v in vec {
            let abs_v = (v * inv_norm).abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let scale = if max_abs > 1e-12 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = if max_abs > 1e-12 { 127.0 / max_abs } else { 0.0 };

        let mut codes = Vec::with_capacity(vec.len());
        let mut norm_sq: i32 = 0;

        for &v in vec {
            let normalized = v * inv_norm;
            let q = (normalized * inv_scale).round().clamp(-127.0, 127.0) as i8;
            let qi = q as i32;
            norm_sq += qi * qi;
            codes.push(q);
        }

        Self {
            scale,
            norm_sq,
            codes,
        }
    }

    /// Reconstructs an `Sq8Vector` from raw serialized `scale` and `i8` codes.
    pub fn from_raw(scale: f32, codes: Vec<i8>) -> Self {
        let mut norm_sq: i32 = 0;
        for &c in &codes {
            let ci = c as i32;
            norm_sq += ci * ci;
        }
        Self {
            scale,
            norm_sq,
            codes,
        }
    }

    /// Computes cosine similarity in `[-1.0, 1.0]` between two `SQ8` vectors using
    /// unrolled 8-lane integer multiply-accumulate (auto-vectorizes to AVX2/NEON).
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        Self::cosine_raw(&self.codes, self.norm_sq, &other.codes, other.norm_sq)
    }

    /// Zero-copy cosine similarity operating directly on borrowed `[i8]` slices (e.g. from `mmap`).
    #[inline]
    pub fn cosine_raw(a: &[i8], norm_sq_a: i32, b: &[i8], norm_sq_b: i32) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        if norm_sq_a == 0 || norm_sq_b == 0 {
            return 0.0;
        }

        let n = a.len();
        let chunks = n / 8;
        let mut acc0: i32 = 0;
        let mut acc1: i32 = 0;
        let mut acc2: i32 = 0;
        let mut acc3: i32 = 0;

        for c in 0..chunks {
            let base = c * 8;
            acc0 += (a[base] as i32) * (b[base] as i32)
                + (a[base + 1] as i32) * (b[base + 1] as i32);
            acc1 += (a[base + 2] as i32) * (b[base + 2] as i32)
                + (a[base + 3] as i32) * (b[base + 3] as i32);
            acc2 += (a[base + 4] as i32) * (b[base + 4] as i32)
                + (a[base + 5] as i32) * (b[base + 5] as i32);
            acc3 += (a[base + 6] as i32) * (b[base + 6] as i32)
                + (a[base + 7] as i32) * (b[base + 7] as i32);
        }

        let mut dot = acc0 + acc1 + acc2 + acc3;
        for i in (chunks * 8)..n {
            dot += (a[i] as i32) * (b[i] as i32);
        }

        let denom = ((norm_sq_a as f64) * (norm_sq_b as f64)).sqrt() as f32;
        if denom > 0.0 {
            (dot as f32 / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

/// 1-Bit Binary Quantized Vector (`BQ1`) packed into 64-bit words for hardware `POPCNT` Hamming search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bq1Vector {
    pub dim: usize,
    pub words: Vec<u64>,
}

impl Bq1Vector {
    /// Packs the sign bits of `vec` (centered around its mean) into `u64` words.
    pub fn quantize(vec: &[f32]) -> Self {
        let dim = vec.len();
        let num_words = (dim + 63) / 64;
        let mut words = vec![0u64; num_words];
        if dim == 0 {
            return Self { dim, words };
        }

        let mean: f32 = vec.iter().copied().sum::<f32>() / (dim as f32);
        for (i, &v) in vec.iter().enumerate() {
            if v > mean {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self { dim, words }
    }

    /// Computes normalized Hamming similarity in `[0.0, 1.0]` using hardware `count_ones` (`POPCNT`).
    #[inline]
    pub fn hamming_similarity(&self, other: &Self) -> f32 {
        Self::hamming_raw(&self.words, &other.words, self.dim)
    }

    /// Zero-copy Hamming similarity directly over `[u64]` word slices.
    #[inline]
    pub fn hamming_raw(a: &[u64], b: &[u64], dim: usize) -> f32 {
        if dim == 0 {
            return 0.0;
        }
        let mut xor_bits: u32 = 0;
        for (&wa, &wb) in a.iter().zip(b.iter()) {
            xor_bits += (wa ^ wb).count_ones();
        }
        let dist = (xor_bits as usize).min(dim);
        1.0 - (dist as f32) / (dim as f32)
    }
}

/// Deterministic feature-hashing text embedder (`HashEmbedder`) for zero-dependency
/// character n-gram + word token semantic vectors when an external model is not attached.
/// Produces L2-normalized dense `f32` vectors of arbitrary dimension `D` using signed
/// Murmur/FNV projections over word unigrams, bigrams, and character trigrams.
pub fn deterministic_text_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    if dim == 0 || text.is_empty() {
        return vec;
    }

    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|s| !s.is_empty())
        .collect();

    // 1. Word unigrams (weight 2.0)
    for &tok in &tokens {
        accumulate_token(&mut vec, tok.as_bytes(), 0x9E37_79B9, 2.0);
        // Character trigrams inside word (weight 0.65) for subword / typo resilience
        let bytes = tok.as_bytes();
        if bytes.len() >= 3 {
            for win in bytes.windows(3) {
                accumulate_token(&mut vec, win, 0x85EB_CA6B, 0.65);
            }
        }
    }

    // 2. Word bigrams (weight 1.25) for phrase order sensitivity
    for pair in tokens.windows(2) {
        let mut buf = Vec::with_capacity(pair[0].len() + 1 + pair[1].len());
        buf.extend_from_slice(pair[0].as_bytes());
        buf.push(b'_');
        buf.extend_from_slice(pair[1].as_bytes());
        accumulate_token(&mut vec, &buf, 0xC2B2_AE35, 1.25);
    }

    // L2 normalize
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

#[inline]
fn accumulate_token(vec: &mut [f32], bytes: &[u8], seed: u64, weight: f32) {
    let dim = vec.len();
    // Dual projection (2 indices + 2 signs per feature) to minimize hash collisions (CountSketch)
    let h1 = fnv1a_64(bytes, seed);
    let h2 = fnv1a_64(bytes, seed ^ 0x517C_C1B7_2722_0A95);

    let idx1 = (h1 as usize) % dim;
    let sign1 = if (h1 >> 63) == 0 { weight } else { -weight };
    vec[idx1] += sign1;

    let idx2 = (h2 as usize) % dim;
    let sign2 = if (h2 >> 63) == 0 { weight * 0.5 } else { -weight * 0.5 };
    vec[idx2] += sign2;
}

#[inline]
pub fn fnv1a_64(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64 ^ seed;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

#[inline]
pub fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut hash = 0x811C_9DC5u32;
    for &b in bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

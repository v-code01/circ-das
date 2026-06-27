//! Coded-Merkle-tree data-availability sampling (DAS) and its HONEST soundness bound.
//!
//! # What this is
//! A light client cannot download a whole block. Instead it (1) trusts a short
//! commitment to the `n` ENCODED symbols and (2) queries a few random symbols with
//! Merkle openings. If enough symbols are provably present, the GLOBAL erasure
//! decoder (`BlockCirculant::recover` / `Rs2d::recover`) can reconstruct the block,
//! so the data is available. A malicious block producer tries to publish a block
//! that is NOT recoverable while evading the sampler.
//!
//! # The commitment: a coded Merkle tree
//! We build a Merkle tree whose `n` leaves are the encoded symbols. The root is the
//! (short) commitment; a symbol's opening is its authentication path. `verify_opening`
//! recomputes the root from a claimed (index, symbol) and rejects any tamper. The tree
//! is built over the CODED symbols (hence "coded Merkle tree"): the code's minimum
//! distance `d` is what turns "a few random present symbols" into a global-availability
//! guarantee.
//!
//! # The soundness bound (derived from the DEPLOYED decoder's real threshold)
//! The decoder deployed in this crate is the GLOBAL Gauss-Jordan decoder. It recovers
//! EVERY erasure pattern of size `<= d-1` and FAILS on a minimum-weight codeword support
//! of size exactly `d` (proven exhaustively + at the boundary in
//! `tests/invariant_erasure.rs`). So the real, deployed recovery threshold is `d-1`:
//!
//!   * A block is AVAILABLE iff `> n-d` symbols are present (`<= d-1` missing).
//!   * To make a block UNAVAILABLE the producer must WITHHOLD an unrecoverable set of
//!     size `w >= d`.
//!
//! A sampler drawing `s` symbol indices independently and uniformly detects a withheld
//! set of size `w` with probability `>= 1 - ((n-w)/n)^s` (equality for i.i.d. sampling;
//! sampling WITHOUT replacement only raises detection, so the bound is conservative
//! there too). The WORST-CASE / TARGETED producer MINIMIZES `w` by withholding the
//! SMALLEST unrecoverable set — a minimum-weight codeword support of size `w = d`
//! (`grs::min_weight_support`). Withholding fewer is recoverable; withholding a larger or
//! scattered set only makes detection EASIER. Hence, against the worst case,
//!
//!   detection >= 1 - ((n-d)/n)^s,   miss (soundness error) <= ((n-d)/n)^s.
//!
//! For a `2^-lambda` soundness error the client needs
//!
//!   s >= lambda*ln2 / ln(n/(n-d))   ~=   lambda*ln2*(n/d)   (small d/n),
//!
//! i.e. `s` scales as `1/(d/n)`. A code with LARGER relative distance `d/n` therefore
//! needs FEWER samples for the same soundness. Block-circulant's `d/n` advantage over
//! 2D-RS at high rate (see `bench_results/dn.md`, crossover `R ~ 0.67`) becomes a
//! FEWER-SAMPLES advantage here (see `bench_results/das.md`).
//!
//! # Honesty
//! The bound is stated at the GLOBAL decoder's ACTUAL threshold `d-1` — not an idealized
//! value and not an iterative-decoder threshold (there is NO iterative decoder deployed;
//! one would be suboptimal, with stopping sets below `d-1`, and would only SHRINK the
//! advantage, never inflate it). The Monte-Carlo test withholds the worst-case
//! min-weight support, not random symbols (random withholding inflates detection and is
//! the tautology trap).
//!
//! The Merkle digest here is a compact, self-contained 64-bit avalanche hash
//! (FNV-1a + splitmix finaliser) chosen to keep the crate dependency-free; it
//! demonstrates the commitment STRUCTURE and binds symbol+position for tamper
//! detection. A production deployment substitutes a collision-resistant hash
//! (SHA-256 / BLAKE3). The Merkle construction and the soundness argument are
//! hash-agnostic.

/// A Merkle digest. 64 bits is ample for the commitment DEMONSTRATION and for the
/// tamper-detection tests; production would widen this to a 256-bit CR hash.
pub type Digest = u64;

// --- Self-contained avalanche hash (no external crypto dependency). ------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over a byte slice.
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// splitmix64 finaliser — strong avalanche so single-bit input changes scatter.
#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Leaf digest binds BOTH the coordinate index and the symbol byte (domain tag 0x00),
/// so a permuted or altered symbol fails verification at its position.
#[inline]
fn leaf_digest(index: usize, symbol: u8) -> Digest {
    let mut buf = [0u8; 10];
    buf[0] = 0x00; // leaf domain tag
    buf[1..9].copy_from_slice(&(index as u64).to_le_bytes());
    buf[9] = symbol;
    splitmix(fnv1a(&buf))
}

/// Internal node digest of an ordered pair (domain tag 0x01) — distinct tag from leaves
/// gives domain separation (a leaf digest can never be confused with an internal one).
#[inline]
fn node_digest(left: Digest, right: Digest) -> Digest {
    let mut buf = [0u8; 17];
    buf[0] = 0x01; // internal domain tag
    buf[1..9].copy_from_slice(&left.to_le_bytes());
    buf[9..17].copy_from_slice(&right.to_le_bytes());
    splitmix(fnv1a(&buf))
}

/// One step of a Merkle opening: a sibling digest and which side it sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofStep {
    /// The sibling's digest.
    pub sibling: Digest,
    /// `true` if the sibling is the RIGHT child (so our running hash is the left one).
    pub sibling_is_right: bool,
}

/// A coded Merkle tree over the `n` encoded symbols.
///
/// Odd levels carry the last (unpaired) node up UNCHANGED — this keeps openings a clean
/// list of real siblings with no synthetic-duplicate ambiguity, and needs no power-of-two
/// padding.
pub struct MerkleTree {
    /// `levels[0]` = leaf digests; `levels.last()` = `[root]`.
    levels: Vec<Vec<Digest>>,
}

impl MerkleTree {
    /// Commit to `symbols` (the `n` encoded coordinates). Panics on empty input.
    pub fn commit(symbols: &[u8]) -> Self {
        assert!(!symbols.is_empty(), "cannot commit to an empty symbol vector");
        let leaves: Vec<Digest> = symbols
            .iter()
            .enumerate()
            .map(|(i, &s)| leaf_digest(i, s))
            .collect();
        let mut levels = vec![leaves];
        while levels.last().expect("non-empty").len() > 1 {
            let cur = levels.last().expect("non-empty");
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut i = 0;
            while i < cur.len() {
                if i + 1 < cur.len() {
                    next.push(node_digest(cur[i], cur[i + 1]));
                    i += 2;
                } else {
                    next.push(cur[i]); // carry the odd tail node up unchanged
                    i += 1;
                }
            }
            levels.push(next);
        }
        MerkleTree { levels }
    }

    /// Number of committed symbols (leaves).
    #[inline]
    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    /// Always false (commit rejects empty input); present to satisfy clippy/idiom.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.levels[0].is_empty()
    }

    /// The commitment: the Merkle root.
    #[inline]
    pub fn root(&self) -> Digest {
        *self.levels.last().expect("non-empty tree").last().expect("non-empty level")
    }

    /// Authentication path proving that `symbols[index]` sits under `root()`.
    pub fn open(&self, index: usize) -> Vec<ProofStep> {
        assert!(index < self.len(), "opening index {index} out of range {}", self.len());
        let mut path = Vec::new();
        let mut idx = index;
        for level in 0..self.levels.len() - 1 {
            let cur = &self.levels[level];
            if idx.is_multiple_of(2) {
                if idx + 1 < cur.len() {
                    path.push(ProofStep { sibling: cur[idx + 1], sibling_is_right: true });
                }
                // else: this node is the carried-up odd tail — no sibling at this level.
            } else {
                path.push(ProofStep { sibling: cur[idx - 1], sibling_is_right: false });
            }
            idx /= 2;
        }
        path
    }
}

/// Verify a Merkle opening against a commitment `root`.
///
/// Recomputes the root from `(index, symbol)` and `path`; returns `true` iff it matches.
/// Rejects a tampered symbol, a wrong index, or a doctored path (all change the recomputed
/// root with overwhelming probability). Pure function of its inputs — no tree needed.
pub fn verify_opening(root: Digest, index: usize, symbol: u8, path: &[ProofStep]) -> bool {
    let mut h = leaf_digest(index, symbol);
    for step in path {
        h = if step.sibling_is_right {
            node_digest(h, step.sibling)
        } else {
            node_digest(step.sibling, h)
        };
    }
    h == root
}

// --- Sampling + soundness arithmetic ------------------------------------------

/// A tiny, self-contained, reproducible PRNG (LCG, top bits) for the sampler — avoids a
/// `rand` dependency and keeps Monte-Carlo trials byte-reproducible from a seed.
pub struct SampleRng(u64);

impl SampleRng {
    /// Seed the stream.
    #[inline]
    pub fn new(seed: u64) -> Self {
        SampleRng(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A uniform index in `0..n`.
    #[inline]
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// Draw `s` symbol indices INDEPENDENTLY and uniformly from `0..n` (i.i.d., with
/// replacement) — the standard light-client DAS model, for which the detection bound
/// `1 - ((n-w)/n)^s` is EXACT rather than merely a bound. (Sampling distinct indices
/// only increases detection, so every result here is conservative for that variant.)
pub fn sample(rng: &mut SampleRng, n: usize, s: usize) -> Vec<usize> {
    (0..s).map(|_| rng.index(n)).collect()
}

/// A block is AVAILABLE (globally recoverable) iff strictly MORE than `n-d` symbols are
/// present, i.e. at most `d-1` are missing — the deployed global decoder's exact
/// threshold. `present_count` counts distinct present coordinates.
#[inline]
pub fn is_available(n: usize, present_count: usize, d: usize) -> bool {
    present_count > n - d
}

/// Detection probability of a size-`w` withheld set by `s` i.i.d. samples over `n`
/// symbols: `1 - ((n-w)/n)^s`. `w == 0` gives 0 (nothing to detect); `w >= n` gives 1.
pub fn detection_probability(n: usize, w: usize, s: usize) -> f64 {
    assert!(n > 0, "n must be positive");
    if w == 0 {
        return 0.0;
    }
    if w >= n {
        return 1.0;
    }
    let miss = ((n - w) as f64 / n as f64).powi(s as i32);
    1.0 - miss
}

/// Smallest sample count `s` that drives the WORST-CASE soundness error below
/// `2^-lambda`, i.e. the least `s` with `((n-d)/n)^s <= 2^-lambda`, against a producer
/// withholding the minimum unrecoverable set of size `w = d`.
///
/// Closed form: `s = ceil( lambda*ln2 / ln(n/(n-d)) )`. If `d >= n` the whole block is
/// withheld and a single sample always detects (`s = 1`).
pub fn required_samples(n: usize, d: usize, lambda: u32) -> usize {
    assert!(n > 0 && d > 0, "n and d must be positive");
    if d >= n {
        return 1;
    }
    let ln2 = std::f64::consts::LN_2;
    let denom = (n as f64 / (n - d) as f64).ln(); // = -ln((n-d)/n) > 0
    let s = (lambda as f64 * ln2 / denom).ceil();
    s as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols(n: usize) -> Vec<u8> {
        (0..n as u32).map(|x| ((x.wrapping_mul(2_654_435_761)) >> 13) as u8 ^ 0x5a).collect()
    }

    // --- Merkle commitment --------------------------------------------------

    #[test]
    fn opening_verifies_for_every_index_various_sizes() {
        // Includes odd, even, prime, and power-of-two leaf counts to exercise the
        // odd-tail carry-up path at multiple levels.
        for &n in &[1usize, 2, 3, 5, 8, 13, 16, 31, 64, 100] {
            let syms = symbols(n);
            let tree = MerkleTree::commit(&syms);
            assert_eq!(tree.len(), n);
            let root = tree.root();
            for (i, &sym) in syms.iter().enumerate() {
                let path = tree.open(i);
                assert!(
                    verify_opening(root, i, sym, &path),
                    "valid opening failed to verify: n={n}, i={i}"
                );
            }
        }
    }

    #[test]
    fn tampered_symbol_is_rejected() {
        let n = 37;
        let syms = symbols(n);
        let tree = MerkleTree::commit(&syms);
        let root = tree.root();
        for (i, &sym) in syms.iter().enumerate() {
            let path = tree.open(i);
            // Flip the symbol to any other value: opening must be rejected.
            let bad = sym ^ 0xff;
            assert_ne!(bad, sym);
            assert!(
                !verify_opening(root, i, bad, &path),
                "tampered symbol at i={i} was accepted — commitment not binding"
            );
        }
    }

    #[test]
    fn wrong_index_and_doctored_path_are_rejected() {
        let n = 40;
        let syms = symbols(n);
        let tree = MerkleTree::commit(&syms);
        let root = tree.root();
        // Correct symbol, WRONG index (claim position 0 with symbol from position 7).
        let path7 = tree.open(7);
        assert!(!verify_opening(root, 0, syms[7], &path7));
        // Correct (index, symbol) but a doctored sibling in the path.
        let mut path5 = tree.open(5);
        assert!(!path5.is_empty());
        path5[0].sibling ^= 1;
        assert!(!verify_opening(root, 5, syms[5], &path5));
        // Flipping the side bit must also break verification.
        let mut path5b = tree.open(5);
        path5b[0].sibling_is_right = !path5b[0].sibling_is_right;
        assert!(!verify_opening(root, 5, syms[5], &path5b));
    }

    #[test]
    fn distinct_symbols_give_distinct_roots() {
        let a = MerkleTree::commit(&symbols(64)).root();
        let mut s = symbols(64);
        s[17] ^= 0x01; // one-bit change anywhere must move the root
        let b = MerkleTree::commit(&s).root();
        assert_ne!(a, b, "root failed to bind a one-bit symbol change");
    }

    // --- Soundness arithmetic ----------------------------------------------

    #[test]
    fn availability_predicate_matches_threshold() {
        // n=24, d=5 => available iff present > 19 iff missing <= 4 = d-1.
        let (n, d) = (24usize, 5usize);
        assert!(is_available(n, 20, d)); // 4 missing -> recoverable
        assert!(!is_available(n, 19, d)); // 5 missing -> boundary, NOT recoverable
        assert!(!is_available(n, 10, d));
        assert!(is_available(n, n, d)); // nothing missing
    }

    #[test]
    fn detection_probability_is_monotone_and_bounded() {
        let (n, w) = (1024usize, 15usize);
        assert_eq!(detection_probability(n, 0, 100), 0.0);
        assert_eq!(detection_probability(n, n, 1), 1.0);
        let mut prev = 0.0;
        for s in 0..2000 {
            let p = detection_probability(n, w, s);
            assert!((0.0..=1.0).contains(&p));
            assert!(p >= prev - 1e-12, "detection must be nondecreasing in s");
            prev = p;
        }
    }

    #[test]
    fn required_samples_is_the_exact_threshold() {
        // s must be the SMALLEST value with miss <= 2^-lambda; check s works and s-1 fails.
        let lambda = 80u32;
        let target = (2.0f64).powi(-(lambda as i32));
        for &(n, d) in &[(1136usize, 15usize), (1156, 9), (1072, 7), (1089, 4), (24, 5)] {
            let s = required_samples(n, d, lambda);
            let miss = |s: usize| ((n - d) as f64 / n as f64).powi(s as i32);
            assert!(miss(s) <= target * (1.0 + 1e-9), "s={s} does not reach 2^-{lambda} for (n={n},d={d})");
            assert!(s >= 1);
            if s > 1 {
                assert!(miss(s - 1) > target, "s={s} is not minimal for (n={n},d={d})");
            }
        }
    }

    #[test]
    fn fewer_samples_track_larger_d_over_n() {
        // At R=0.90 the disclosed operating point: BC [1136,1024,15], 2D-RS [1156,1024,9].
        // BC has the larger d/n, so it needs strictly FEWER samples for the same lambda.
        let s_bc = required_samples(1136, 15, 80);
        let s_rs = required_samples(1156, 9, 80);
        assert!(s_bc < s_rs, "BC {s_bc} should need fewer samples than 2D-RS {s_rs}");
    }

    #[test]
    fn empirical_detection_matches_formula_on_a_toy_set() {
        // Monte-Carlo the SAMPLER itself against an arbitrary size-w withheld set and
        // confirm empirical detection tracks 1-((n-w)/n)^s. (The code-coupled WORST-CASE
        // min-weight-support adversary lives in tests/das_soundness.rs.)
        let (n, w) = (24usize, 5usize);
        let withheld: std::collections::BTreeSet<usize> = (0..w).collect();
        for &s in &[3usize, 5, 10] {
            let trials = 40_000;
            let mut rng = SampleRng::new(0xda5_u64 ^ ((s as u64) << 20));
            let mut detected = 0;
            for _ in 0..trials {
                let q = sample(&mut rng, n, s);
                if q.iter().any(|i| withheld.contains(i)) {
                    detected += 1;
                }
            }
            let emp = detected as f64 / trials as f64;
            let theory = detection_probability(n, w, s);
            assert!((emp - theory).abs() < 0.01, "s={s}: empirical {emp:.4} vs theory {theory:.4}");
        }
    }
}

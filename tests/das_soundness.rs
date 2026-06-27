//! DAS SOUNDNESS against the WORST-CASE (targeted) data-availability adversary.
//!
//! This is the load-bearing honesty test for GROUP 3. Everything is coupled to the
//! REAL deployed pieces:
//!   * the REAL encoders (`BlockCirculant::encode`, `Rs2d::encode`),
//!   * the REAL global Gauss-Jordan decoder (`*::recover`) — the ONLY decoder in the
//!     crate, whose exact recovery threshold is `d-1`,
//!   * the REAL coded-Merkle-tree commitment + sampler (`das`).
//!
//! WORST-CASE adversary (NOT random). A malicious block producer wants an UNRECOVERABLE
//! block that evades sampling. The CHEAPEST unrecoverable set is a minimum-weight
//! codeword support of size EXACTLY `d = grs::min_weight_support(...)`:
//!   * withhold `< d`  => recoverable (block is available) — no point,
//!   * withhold the min-weight support (`= d`) => genuinely unrecoverable (decoder
//!     returns `None`) AND smallest, so hardest to catch,
//!   * withhold a larger / scattered set => only EASIER to detect.
//!
//! Random withholding would inflate detection (the tautology trap); we do not do it.
//!
//! We assert three things per code:
//!   (A) the min-weight support of size `d` is genuinely UNRECOVERABLE by the deployed
//!       decoder, and dropping ONE of its symbols (size `d-1`) IS recoverable
//!       (== original) and is flagged AVAILABLE — the threshold is exactly `d-1`;
//!   (B) Monte-Carlo: sampling that worst-case block, empirical detection over many
//!       trials matches the predicted `1 - ((n-d)/n)^s` within tolerance;
//!   (C) the soundness bound holds: at `s = required_samples(n, d, lambda)` the
//!       empirical MISS rate is at/below `2^-lambda` for a small lambda we can measure.

use circ_das::block_circulant::BlockCirculant;
use circ_das::das::{
    detection_probability, is_available, required_samples, sample, verify_opening, MerkleTree,
    SampleRng,
};
use circ_das::gf256::Gf256;
use circ_das::grs::min_weight_support;
use circ_das::rs2d::Rs2d;
use std::collections::BTreeSet;

/// Deterministic pseudo-message of `k` nonzero-ish bytes.
fn message(k: usize) -> Vec<u8> {
    (0..k as u32).map(|x| ((x.wrapping_mul(2_654_435_761)) >> 13) as u8 | 1).collect()
}

/// Deployed global decoder: `Some(recovered)` iff the erased set is recoverable.
type Decoder = Box<dyn Fn(&[u8], &[usize]) -> Option<Vec<u8>>>;

/// One code instance packaged for the shared soundness driver.
struct Instance {
    label: &'static str,
    n: usize,
    d: usize,
    /// The encoded codeword (what an honest producer publishes).
    cw: Vec<u8>,
    /// The worst-case (min-weight codeword support) withheld set, size == d.
    worst: Vec<usize>,
    /// Decoder closure: `Some(recovered)` iff the erased set is recoverable.
    recover: Decoder,
}

fn bc_instance(f: &'static Gf256) -> Instance {
    let bc = BlockCirculant::new(f, 6, 2, 2); // n=24, d=2*rho+1=5
    let n = bc.n();
    let d = bc.d_formula();
    let cw = bc.encode(f, &message(bc.k()));
    let worst = min_weight_support(f, &bc.parity_check(f), n);
    Instance {
        label: "block-circulant (mu=6,omega=2,rho=2)",
        n,
        d,
        cw,
        worst,
        recover: Box::new(move |dmg: &[u8], erased: &[usize]| bc.recover(f, dmg, erased)),
    }
}

fn rs2d_instance(f: &'static Gf256) -> Instance {
    let code = Rs2d::new(f, 4, 2, 4, 2); // n=16, d=9
    let n = code.n();
    let d = code.d_formula();
    let msg: Vec<u8> = (0..code.k() as u8).map(|x| x.wrapping_mul(5).wrapping_add(1)).collect();
    let cw = code.encode(f, &msg);
    let worst = min_weight_support(f, &code.parity_check(), n);
    Instance {
        label: "2D-RS (4,2)x(4,2)",
        n,
        d,
        cw,
        worst,
        recover: Box::new(move |dmg: &[u8], erased: &[usize]| code.recover(f, dmg, erased)),
    }
}

/// Damage a codeword by zeroing the withheld positions (the producer publishes nothing
/// there; the sampler cannot open them).
fn withhold(cw: &[u8], set: &[usize]) -> Vec<u8> {
    let mut dmg = cw.to_vec();
    for &e in set {
        dmg[e] = 0x00;
    }
    dmg
}

// ===========================================================================
// (A) The worst case is EXACTLY the threshold: size-d unrecoverable,
//     size-(d-1) recoverable & flagged available.
// ===========================================================================

fn assert_threshold(inst: &Instance) {
    assert_eq!(inst.worst.len(), inst.d, "{}: worst set must have size d", inst.label);

    // size d (min-weight support): genuinely UNRECOVERABLE by the deployed decoder.
    let dmg_d = withhold(&inst.cw, &inst.worst);
    assert!(
        (inst.recover)(&dmg_d, &inst.worst).is_none(),
        "{}: size-d min-weight support MUST be unrecoverable (else soundness is a lie)",
        inst.label
    );
    // and it is correctly NOT available (present = n-d).
    assert!(
        !is_available(inst.n, inst.n - inst.d, inst.d),
        "{}: block with d symbols withheld must be flagged unavailable",
        inst.label
    );

    // size d-1 (drop one symbol from the support): RECOVERABLE == original, and AVAILABLE.
    let smaller: Vec<usize> = inst.worst[..inst.d - 1].to_vec();
    let dmg_d1 = withhold(&inst.cw, &smaller);
    let rec = (inst.recover)(&dmg_d1, &smaller)
        .unwrap_or_else(|| panic!("{}: size-(d-1) withholding MUST recover (available)", inst.label));
    assert_eq!(rec, inst.cw, "{}: size-(d-1) recovery must equal the original block", inst.label);
    assert!(
        is_available(inst.n, inst.n - (inst.d - 1), inst.d),
        "{}: recoverable block must be flagged available (not falsely unavailable)",
        inst.label
    );
}

#[test]
fn worst_case_is_exactly_the_global_decoder_threshold() {
    let f: &'static Gf256 = Box::leak(Box::new(Gf256::new()));
    assert_threshold(&bc_instance(f));
    assert_threshold(&rs2d_instance(f));
}

// ===========================================================================
// (B) Monte-Carlo: sampling the WORST-CASE block, empirical detection tracks
//     the predicted 1 - ((n-d)/n)^s.
// ===========================================================================

fn empirical_detection(inst: &Instance, s: usize, trials: usize, seed: u64) -> f64 {
    // Honest producer commits to the FULL codeword; the withheld positions are simply
    // never served. A sample DETECTS unavailability iff it lands on a withheld index
    // (the producer cannot supply a valid opening for a symbol it is withholding).
    let withheld: BTreeSet<usize> = inst.worst.iter().copied().collect();
    let mut rng = SampleRng::new(seed);
    let mut detected = 0usize;
    for _ in 0..trials {
        let q = sample(&mut rng, inst.n, s);
        if q.iter().any(|i| withheld.contains(i)) {
            detected += 1;
        }
    }
    detected as f64 / trials as f64
}

fn assert_monte_carlo_matches_formula(inst: &Instance) {
    for (k, &s) in [3usize, 5, 8, 12].iter().enumerate() {
        let trials = 60_000;
        let emp = empirical_detection(inst, s, trials, 0xC0DE_00A5 ^ ((k as u64) << 24));
        let theory = detection_probability(inst.n, inst.d, s);
        assert!(
            (emp - theory).abs() < 0.01,
            "{}: s={s} empirical detection {emp:.4} vs worst-case theory {theory:.4} (n={}, d={})",
            inst.label,
            inst.n,
            inst.d
        );
    }
}

#[test]
fn worst_case_detection_matches_predicted_rate() {
    let f: &'static Gf256 = Box::leak(Box::new(Gf256::new()));
    assert_monte_carlo_matches_formula(&bc_instance(f));
    assert_monte_carlo_matches_formula(&rs2d_instance(f));
}

// ===========================================================================
// (C) The soundness BOUND holds against the worst case: at
//     s = required_samples(n, d, lambda) the measured MISS rate <= 2^-lambda.
//     lambda is kept small enough to measure with a feasible trial count.
// ===========================================================================

fn assert_soundness_bound(inst: &Instance) {
    // Pick lambda so 2^-lambda is measurable: expected misses = trials * 2^-lambda >= ~a few.
    let lambda = 10u32; // target miss <= 2^-10 ~= 9.77e-4
    let s = required_samples(inst.n, inst.d, lambda);
    let target = (2.0f64).powi(-(lambda as i32));
    // Predicted worst-case miss at this s must already be <= target (definition of s).
    let predicted_miss = 1.0 - detection_probability(inst.n, inst.d, s);
    assert!(
        predicted_miss <= target * (1.0 + 1e-9),
        "{}: required_samples gave s={s} whose predicted miss {predicted_miss:.3e} > 2^-{lambda}",
        inst.label
    );
    // Empirical miss over many trials must be <= a small multiple of target (Monte-Carlo
    // slack): use 3x target as the ceiling — with ~2e6 trials the estimator std is tiny.
    let trials = 2_000_000usize;
    let emp_detect = empirical_detection(inst, s, trials, 0x5011_D001_u64);
    let emp_miss = 1.0 - emp_detect;
    assert!(
        emp_miss <= 3.0 * target,
        "{}: s={s} empirical miss {emp_miss:.3e} exceeds 3x 2^-{lambda} ({:.3e}) — soundness bound violated",
        inst.label,
        3.0 * target
    );
}

#[test]
fn soundness_bound_holds_against_worst_case() {
    let f: &'static Gf256 = Box::leak(Box::new(Gf256::new()));
    assert_soundness_bound(&bc_instance(f));
    assert_soundness_bound(&rs2d_instance(f));
}

// ===========================================================================
// (D) End-to-end: honest producer -> commit -> sampler -> verify openings.
//     Every served sample of an AVAILABLE block verifies against the root.
// ===========================================================================

#[test]
fn honest_available_block_passes_sampling_end_to_end() {
    let f = Gf256::new();
    let bc = BlockCirculant::new(&f, 8, 6, 4); // larger instance, n=80
    let cw = bc.encode(&f, &message(bc.k()));
    let tree = MerkleTree::commit(&cw);
    let root = tree.root();
    let n = bc.n();
    let mut rng = SampleRng::new(0x0000_A1A1_AB1E);
    // Sample s symbols; the honest producer serves each with a valid opening.
    let queried = sample(&mut rng, n, 64);
    for &i in &queried {
        let path = tree.open(i);
        assert!(
            verify_opening(root, i, cw[i], &path),
            "honest available block: sample at {i} failed to verify"
        );
    }
    // Availability predicate agrees the block is available (nothing withheld).
    assert!(is_available(n, n, bc.d_formula()));
}

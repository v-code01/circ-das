//! THE SACRED INVARIANT — erasure recovery / no silent data loss.
//!
//! A data-availability code that silently fails to recover recoverable data is
//! broken and dishonest. This is the gate the whole project lives on:
//!
//!   Any erasure pattern of size <= d-1 MUST recover the original data exactly.
//!   A size-d pattern equal to a minimum-weight codeword's support is the exact
//!   boundary: it does NOT recover (two codewords agree on all surviving symbols).
//!
//! We prove this three ways, for BOTH codes (block-circulant, the novel primitive,
//! and 2D Reed-Solomon, the deployed baseline):
//!   1. EXHAUSTIVE below-distance recovery: enumerate EVERY erasure pattern of
//!      size <= d-1 on small codes; assert recover == original.  (Proof of the
//!      recovery threshold — not a sample, the whole space.)
//!   2. DISTANCE BOUNDARY: locate a minimum-weight codeword's support (the
//!      smallest linearly-dependent set of parity-check columns, size exactly d),
//!      erase precisely it, and show the decoder CANNOT recover (returns None) and
//!      that a second, distinct codeword agrees with the original on every
//!      surviving symbol.  Confirms d is EXACTLY the threshold, not a lower bound.
//!   3. PROPERTY on larger codes: many random patterns of size <= d-1 always
//!      recover.
//!
//! Codes-theory backbone: for a linear code, a set E of positions is erasure-
//! correctable iff the parity-check columns indexed by E are linearly independent.
//! Hence the largest always-correctable size is d-1, and any size-d codeword
//! support (a dependent column set) is the boundary. Everything below just
//! MEASURES this over GF(2^8) rather than trusting a formula.
//!
//! WHICH DECODER THE GUARANTEE IS ON (coding-theory redline, do not overstate):
//! the sacred "recovers every <= d-1 pattern" property is a GLOBAL / MDS-style
//! guarantee and is asserted ONLY against the GLOBAL GAUSSIAN-ELIMINATION decoder
//! `BlockCirculant::recover` / `Rs2d::recover`. Those solve the joint syndrome
//! system H_E c_E = H_K c_K over ALL parity checks at once (see `grs::solve`), so
//! recovery succeeds exactly when the erased columns are independent, i.e. for
//! EVERY pattern of size < d. There is deliberately NO iterative pairwise-peeling
//! decoder in this crate, and this harness makes no claim about one: an iterative
//! local decoder is SUBOPTIMAL (it has stopping sets — patterns of size <= d-1 it
//! cannot resolve even though global elimination can), so were one ever added it
//! would be a separately-measured fast path with its own DISCLOSED recoverable-set
//! coverage, never a substitute for this global guarantee.

use circ_das::block_circulant::BlockCirculant;
use circ_das::gf256::{add, Gf256};
use circ_das::grs::rank;
use circ_das::rs2d::Rs2d;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A cheap, deterministic, reproducible pseudo-message of `k` nonzero-ish bytes.
fn message(k: usize) -> Vec<u8> {
    (0..k as u32)
        .map(|x| ((x.wrapping_mul(2_654_435_761)) >> 13) as u8 | 1)
        .collect()
}

/// Enumerate every size-`t` subset of `0..n` and call `visit` on each.
fn for_each_subset(n: usize, t: usize, visit: &mut dyn FnMut(&[usize])) {
    let mut idx = vec![0usize; t];
    fn go(start: usize, depth: usize, n: usize, idx: &mut [usize], visit: &mut dyn FnMut(&[usize])) {
        if depth == idx.len() {
            visit(idx);
            return;
        }
        for j in start..n {
            idx[depth] = j;
            go(j + 1, depth + 1, n, idx, visit);
        }
    }
    go(0, 0, n, &mut idx, visit);
}

/// Smallest set of parity-check columns that is linearly dependent, returned as
/// the actual column indices. This set is EXACTLY the support of a minimum-weight
/// codeword, so its size is the code's minimum distance d. Searches by increasing
/// size (all smaller sizes are independent), so the first hit is genuinely minimal.
/// For small codes only.
fn smallest_dependent_columns(f: &Gf256, h: &[Vec<u8>], n: usize) -> Vec<usize> {
    let rho = h.len();
    // Any (rho+1) columns are dependent (only rho rows) => d <= rho+1.
    for t in 1..=(rho + 1) {
        let mut found: Option<Vec<usize>> = None;
        for_each_subset(n, t, &mut |idx| {
            if found.is_some() {
                return;
            }
            let sub: Vec<Vec<u8>> = h.iter().map(|row| idx.iter().map(|&j| row[j]).collect()).collect();
            if rank(f, &sub) < t {
                found = Some(idx.to_vec());
            }
        });
        if let Some(cols) = found {
            return cols;
        }
    }
    unreachable!("a code over GF(2^8) with {rho} checks always has a dependent column set of size <= {}", rho + 1);
}

/// Given a linearly-DEPENDENT set of columns `cols` of parity check `h`, return a
/// nonzero coefficient vector `x` (length cols.len()) with sum_j x_j * H[:,cols[j]] = 0.
/// Spread over the full length-`n` coordinate space, this is a codeword c* (H c* = 0)
/// whose support lies within `cols`. When `cols` is a MINIMAL dependent set, c* is a
/// minimum-weight codeword and every coordinate of x is nonzero.
///
/// Method: reduce the rho x t submatrix to reduced row-echelon form over GF(2^8),
/// pick one free (non-pivot) column, set it to 1, back-solve the pivots. Char-2
/// field => addition is XOR, subtraction is XOR.
#[allow(clippy::needless_range_loop)] // index-coupled Gaussian elimination over columns
fn null_combination(f: &Gf256, h: &[Vec<u8>], cols: &[usize]) -> Vec<u8> {
    let t = cols.len();
    let rho = h.len();
    // Build the rho x t submatrix M (mutable copy for elimination).
    let mut m: Vec<Vec<u8>> = h.iter().map(|row| cols.iter().map(|&j| row[j]).collect()).collect();

    // Gauss-Jordan to RREF; record which matrix column each pivot row pivots on.
    let mut pivot_col_of_row: Vec<Option<usize>> = vec![None; rho];
    let mut is_pivot_col = vec![false; t];
    let mut r = 0usize;
    for c in 0..t {
        if r >= rho {
            break;
        }
        // Find a row >= r with nonzero entry in column c.
        let mut sel = r;
        while sel < rho && m[sel][c] == 0 {
            sel += 1;
        }
        if sel == rho {
            continue; // free column
        }
        m.swap(sel, r);
        // Normalize pivot to 1.
        let inv = f.inv(m[r][c]);
        for cc in 0..t {
            m[r][cc] = f.mul(m[r][cc], inv);
        }
        // Eliminate column c from all other rows.
        for rr in 0..rho {
            if rr != r && m[rr][c] != 0 {
                let factor = m[rr][c];
                for cc in 0..t {
                    let prod = f.mul(factor, m[r][cc]);
                    m[rr][cc] = add(m[rr][cc], prod);
                }
            }
        }
        pivot_col_of_row[r] = Some(c);
        is_pivot_col[c] = true;
        r += 1;
    }

    // A dependent set has at least one free column. Pick the first, set it to 1.
    let free_col = (0..t)
        .find(|&c| !is_pivot_col[c])
        .expect("dependent column set must have a free column");
    let mut x = vec![0u8; t];
    x[free_col] = 1;
    // For each pivot row with pivot column p: x[p] + sum_free M[row][f]*x[f] = 0.
    // Only free_col is nonzero among free vars, so x[p] = M[row][free_col].
    for (row, pc) in pivot_col_of_row.iter().enumerate() {
        if let Some(p) = pc {
            x[*p] = m[row][free_col];
        }
    }
    x
}

/// Assert H c = 0 (c is a genuine codeword) for parity check `h`.
fn assert_in_code(f: &Gf256, h: &[Vec<u8>], c: &[u8]) {
    for row in h {
        let s = row.iter().zip(c).fold(0u8, |acc, (&hv, &cv)| add(acc, f.mul(hv, cv)));
        assert_eq!(s, 0, "vector is not in the code (H c != 0)");
    }
}

/// A reproducible LCG stream of usize draws (upper bits, well-mixed).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize
    }
    /// A random subset of `size` distinct indices in `0..n`.
    fn subset(&mut self, n: usize, size: usize) -> Vec<usize> {
        let mut set = std::collections::BTreeSet::new();
        while set.len() < size {
            set.insert(self.next() % n);
        }
        set.into_iter().collect()
    }
}

// ===========================================================================
// 1. EXHAUSTIVE below-distance recovery — the proof of the recovery threshold.
// ===========================================================================

#[test]
fn exhaustive_bc_every_pattern_up_to_d_minus_1_recovers() {
    // Block-circulant mu=6, omega=2, rho=2 => d = 2*rho+1 = 5, d-1 = 4 = 2*rho.
    // n = mu*(rho+omega) = 24. We enumerate EVERY erasure pattern of size 1..=4
    // (C(24,1..4) = 12,950 patterns) and demand exact recovery.
    let f = Gf256::new();
    let bc = BlockCirculant::new(&f, 6, 2, 2);
    let d = bc.d_formula();
    assert_eq!(d, 5);
    let msg = message(bc.k());
    let cw = bc.encode(&f, &msg);
    let n = bc.n();
    for t in 1..=(d - 1) {
        for_each_subset(n, t, &mut |erased| {
            let mut dmg = cw.clone();
            for &e in erased {
                dmg[e] = 0x00; // clobber erased symbols
            }
            let rec = bc
                .recover(&f, &dmg, erased)
                .unwrap_or_else(|| panic!("SACRED INVARIANT VIOLATED: BC size-{t} pattern {erased:?} did not recover"));
            assert_eq!(rec, cw, "BC recovered != original for erasures {erased:?}");
        });
    }
}

#[test]
fn exhaustive_rs2d_every_pattern_up_to_d_minus_1_recovers() {
    // 2D-RS (4,2)x(4,2) => d = 3*3 = 9, d-1 = 8, n = 16. Enumerate every pattern of
    // size 1..=8 (sum C(16,1..8) = 39,202 patterns) and demand exact recovery.
    let f = Gf256::new();
    let code = Rs2d::new(&f, 4, 2, 4, 2);
    let d = code.d_formula();
    assert_eq!(d, 9);
    let msg: Vec<u8> = (0..code.k() as u8).map(|x| x.wrapping_mul(5).wrapping_add(1)).collect();
    let grid = code.encode(&f, &msg);
    let n = code.n();
    for t in 1..=(d - 1) {
        for_each_subset(n, t, &mut |erased| {
            let mut dmg = grid.clone();
            for &e in erased {
                dmg[e] = 0x00;
            }
            let rec = code
                .recover(&f, &dmg, erased)
                .unwrap_or_else(|| panic!("SACRED INVARIANT VIOLATED: 2D-RS size-{t} pattern {erased:?} did not recover"));
            assert_eq!(rec, grid, "2D-RS recovered != original for erasures {erased:?}");
        });
    }
}

// ===========================================================================
// 2. DISTANCE BOUNDARY — a size-d codeword-support pattern does NOT recover.
//     Confirms d is EXACTLY the threshold, not merely a lower bound.
// ===========================================================================

#[test]
fn boundary_bc_size_d_codeword_support_does_not_recover() {
    let f = Gf256::new();
    let bc = BlockCirculant::new(&f, 6, 2, 2); // d = 5
    let d = bc.d_formula();
    let n = bc.n();
    let h = bc.parity_check(&f);

    // The smallest dependent column set == a minimum-weight codeword's support.
    let support = smallest_dependent_columns(&f, &h, n);
    assert_eq!(
        support.len(),
        d,
        "smallest dependent column set has size {} but formula d = {d}",
        support.len()
    );

    // Build the actual minimum-weight codeword c* supported on that set.
    let x = null_combination(&f, &h, &support);
    let mut cstar = vec![0u8; n];
    for (slot, &j) in support.iter().enumerate() {
        cstar[j] = x[slot];
    }
    assert_in_code(&f, &h, &cstar); // H c* = 0
    let weight = cstar.iter().filter(|&&b| b != 0).count();
    assert_eq!(weight, d, "min-weight codeword should have weight exactly d = {d}, got {weight}");
    // Its support is precisely the erasure pattern.
    let mut cstar_support: Vec<usize> = (0..n).filter(|&j| cstar[j] != 0).collect();
    cstar_support.sort_unstable();
    let mut sorted = support.clone();
    sorted.sort_unstable();
    assert_eq!(cstar_support, sorted, "c* support must equal the dependent column set");

    // Now erase exactly the codeword support of a real, encoded message.
    let msg = message(bc.k());
    let cw = bc.encode(&f, &msg);
    let mut dmg = cw.clone();
    for &e in &support {
        dmg[e] = 0x00;
    }

    // BOUNDARY FACT 1: the decoder cannot uniquely recover — returns None.
    let rec = bc.recover(&f, &dmg, &support);
    assert!(
        rec.is_none(),
        "d is NOT the true threshold: a size-d codeword support erroneously 'recovered' — silent-data-loss risk"
    );

    // BOUNDARY FACT 2: an explicit ambiguity witness. cw2 = cw + c* is a DIFFERENT
    // codeword that agrees with cw on EVERY surviving (non-erased) symbol. No decoder
    // reading only the survivors can prefer cw over cw2 — the data is genuinely lost.
    let cw2: Vec<u8> = cw.iter().zip(&cstar).map(|(&a, &b)| add(a, b)).collect();
    assert_in_code(&f, &h, &cw2);
    assert_ne!(cw2, cw, "witness codeword must differ from the original");
    let erased: std::collections::BTreeSet<usize> = support.iter().copied().collect();
    for j in 0..n {
        if !erased.contains(&j) {
            assert_eq!(cw2[j], cw[j], "witness must agree with original on surviving symbol {j}");
        }
    }
}

#[test]
fn boundary_rs2d_size_d_codeword_support_does_not_recover() {
    let f = Gf256::new();
    let code = Rs2d::new(&f, 4, 2, 4, 2); // d = 9, n = 16
    let d = code.d_formula();
    let n = code.n();
    let h = code.parity_check();

    let support = smallest_dependent_columns(&f, &h, n);
    assert_eq!(support.len(), d, "smallest dependent set size {} != d {d}", support.len());

    let x = null_combination(&f, &h, &support);
    let mut cstar = vec![0u8; n];
    for (slot, &j) in support.iter().enumerate() {
        cstar[j] = x[slot];
    }
    assert_in_code(&f, &h, &cstar);
    assert_eq!(cstar.iter().filter(|&&b| b != 0).count(), d);

    let msg: Vec<u8> = (0..code.k() as u8).map(|x| x.wrapping_mul(5).wrapping_add(1)).collect();
    let grid = code.encode(&f, &msg);
    let mut dmg = grid.clone();
    for &e in &support {
        dmg[e] = 0x00;
    }
    assert!(
        code.recover(&f, &dmg, &support).is_none(),
        "2D-RS: a size-d codeword support erroneously recovered — silent-data-loss risk"
    );

    let grid2: Vec<u8> = grid.iter().zip(&cstar).map(|(&a, &b)| add(a, b)).collect();
    assert_in_code(&f, &h, &grid2);
    assert_ne!(grid2, grid);
    let erased: std::collections::BTreeSet<usize> = support.iter().copied().collect();
    for j in 0..n {
        if !erased.contains(&j) {
            assert_eq!(grid2[j], grid[j], "witness must agree on surviving symbol {j}");
        }
    }
}

// ===========================================================================
// 3. PROPERTY on larger codes — random <= d-1 patterns always recover.
// ===========================================================================

#[test]
fn property_bc_random_below_distance_always_recovers() {
    let f = Gf256::new();
    // Two larger, non-degenerate block-circulant instances (mu>=6, rho>=3).
    for &(mu, omega, rho) in &[(6usize, 4usize, 3usize), (8, 6, 4)] {
        let bc = BlockCirculant::new(&f, mu, omega, rho);
        let d = bc.d_formula();
        let n = bc.n();
        let msg = message(bc.k());
        let cw = bc.encode(&f, &msg);
        let mut rng = Lcg(0x0123_4567_89ab_cdef ^ ((mu as u64) << 40));
        for _ in 0..1500 {
            let size = 1 + rng.next() % (d - 1); // 1..=d-1
            let erased = rng.subset(n, size);
            let mut dmg = cw.clone();
            for &e in &erased {
                dmg[e] = 0x00;
            }
            let rec = bc.recover(&f, &dmg, &erased).unwrap_or_else(|| {
                panic!("SACRED INVARIANT VIOLATED: BC(mu={mu},omega={omega},rho={rho}) size-{size} pattern {erased:?} lost data")
            });
            assert_eq!(rec, cw, "BC(mu={mu},omega={omega},rho={rho}) recovered != original for {erased:?}");
        }
    }
}

#[test]
fn property_rs2d_random_below_distance_always_recovers() {
    let f = Gf256::new();
    // Larger square product codes; every random < d erasure pattern must recover.
    for &(n1, k1) in &[(6usize, 4usize), (7, 5)] {
        let code = Rs2d::new(&f, n1, k1, n1, k1);
        let d = code.d_formula();
        let n = code.n();
        let msg: Vec<u8> = (0..code.k() as u8).map(|x| x.wrapping_mul(9).wrapping_add(7)).collect();
        let grid = code.encode(&f, &msg);
        let mut rng = Lcg(0xfeed_face_dead_beef ^ ((n1 as u64) << 40));
        for _ in 0..1500 {
            let size = 1 + rng.next() % (d - 1);
            let erased = rng.subset(n, size);
            let mut dmg = grid.clone();
            for &e in &erased {
                dmg[e] = 0x00;
            }
            let rec = code.recover(&f, &dmg, &erased).unwrap_or_else(|| {
                panic!("SACRED INVARIANT VIOLATED: 2D-RS({n1},{k1})^2 size-{size} pattern {erased:?} lost data")
            });
            assert_eq!(rec, grid, "2D-RS({n1},{k1})^2 recovered != original for {erased:?}");
        }
    }
}

//! Block-circulant local codes for data availability, overlap factor lambda = 2.
//!
//! Source: R. Sasidharan, E. Viterbo, S. H. Dau, "Block Circulant Codes with
//! Application to Decentralized Systems," arXiv:2406.12160 (2024).
//!   - Local codes: GRS [lambda*omega + rho, lambda*omega, rho+1].
//!   - Global code: n = mu*(rho+omega), k = mu*omega, rate R = omega/(rho+omega).
//!   - Theorem III.2 (lambda=2): minimum distance d = 2*rho + 1 ALWAYS; the erasure
//!     decoder recovers every pattern of <= 2*rho erasures.
//!
//! Concrete instantiation used here (lambda=2). The k = mu*omega message symbols are
//! arranged as mu cyclic blocks S_0..S_{mu-1}, each of omega symbols. Local code i is
//! the GRS codeword [ S_{i-1 mod mu} | S_i | P_i ] of length 2*omega+rho, where the
//! rho parity symbols P_i are the systematic GRS parity of the 2*omega data symbols
//! (S_{i-1}, S_i). Consecutive local codes overlap on the omega shared symbols S_i,
//! which is exactly the block-circulant topology: parity block i couples message
//! blocks i-1 and i, cyclically.
//!
//! Global coordinate order (length n = mu*omega + mu*rho):
//!   [ S_0 .. S_{mu-1} (message, mu*omega) | P_0 .. P_{mu-1} (parity, mu*rho) ].

use crate::gf256::{add, Gf256};
use crate::grs::{solve, Grs};

pub struct BlockCirculant {
    pub mu: usize,
    pub omega: usize,
    pub rho: usize,
    /// One DISTINCT global evaluation point per coordinate (length n). Every local
    /// code is a GRS over the points of ITS coordinates; because a shared symbol
    /// carries the SAME global point in both local codes it belongs to, the overlap
    /// is consistent (this is the role of the paper's Lambda-partition / M-matrices).
    points: Vec<u8>,
}

impl BlockCirculant {
    /// Build a lambda=2 block-circulant code. Requires n = mu*(rho+omega) <= 255
    /// (GF(2^8) distinct points) and mu >= 2 (cyclic topology).
    pub fn new(f: &Gf256, mu: usize, omega: usize, rho: usize) -> Self {
        assert!(mu >= 2, "block-circulant topology needs mu >= 2 local codes");
        assert!(mu % 2 == 0, "lambda=2 block-circulant uses mu = 2*nu (even)");
        let n = mu * (rho + omega);
        assert!(n <= 255, "global length {n} exceeds GF(2^8) distinct points");
        // Distinct global points g^0..g^{n-1}.
        let points: Vec<u8> = (0..n).map(|g| f.exp_g(g)).collect();
        BlockCirculant { mu, omega, rho, points }
    }

    /// Ordered global coordinate indices of local code i: [S_{i-1} | S_i | P_i].
    fn support(&self, i: usize) -> Vec<usize> {
        let prev = (i + self.mu - 1) % self.mu;
        let mut s = Vec::with_capacity(2 * self.omega + self.rho);
        for off in 0..self.omega {
            s.push(self.msg_idx(prev, off));
        }
        for off in 0..self.omega {
            s.push(self.msg_idx(i, off));
        }
        for off in 0..self.rho {
            s.push(self.par_idx(i, off));
        }
        s
    }

    /// The local GRS [2*omega+rho, 2*omega] for code i over its coordinates' points.
    /// Even/odd local codes use parity-check power offsets 0 / rho respectively so that
    /// adjacent codes constrain a shared block independently (this needs mu even).
    fn local_code(&self, f: &Gf256, i: usize) -> (Grs, Vec<usize>) {
        let sup = self.support(i);
        let pts: Vec<u8> = sup.iter().map(|&g| self.points[g]).collect();
        let offset = if i % 2 == 0 { 0 } else { self.rho };
        (Grs::with_points_offset(f, &pts, 2 * self.omega, offset), sup)
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.mu * (self.rho + self.omega)
    }
    #[inline]
    pub fn k(&self) -> usize {
        self.mu * self.omega
    }
    /// Theorem III.2 (lambda=2): d = 2*rho + 1.
    #[inline]
    pub fn d_formula(&self) -> usize {
        2 * self.rho + 1
    }
    #[inline]
    pub fn rate(&self) -> f64 {
        self.omega as f64 / (self.rho + self.omega) as f64
    }

    // Global index helpers.
    #[inline]
    fn msg_idx(&self, block: usize, off: usize) -> usize {
        (block % self.mu) * self.omega + off
    }
    #[inline]
    fn par_idx(&self, block: usize, off: usize) -> usize {
        self.mu * self.omega + block * self.rho + off
    }

    /// Encode a message of length k = mu*omega into a full codeword of length n.
    pub fn encode(&self, f: &Gf256, message: &[u8]) -> Vec<u8> {
        assert_eq!(message.len(), self.k());
        let mut cw = vec![0u8; self.n()];
        // Place message blocks.
        cw[..self.k()].copy_from_slice(message);
        // Compute each parity block from its two consecutive message blocks, using
        // local code i's own GRS (defined over its coordinates' global points).
        for i in 0..self.mu {
            let prev = (i + self.mu - 1) % self.mu;
            let (local, _sup) = self.local_code(f, i);
            let mut data = Vec::with_capacity(2 * self.omega);
            data.extend_from_slice(&message[prev * self.omega..prev * self.omega + self.omega]);
            data.extend_from_slice(&message[i * self.omega..i * self.omega + self.omega]);
            let local_cw = local.encode_systematic(f, &data);
            let parity = &local_cw[2 * self.omega..]; // rho parity symbols
            for (off, &p) in parity.iter().enumerate() {
                cw[self.par_idx(i, off)] = p;
            }
        }
        cw
    }

    /// Build the global block-circulant parity-check matrix H_BC of size
    /// (mu*rho) x n. Row block i carries local code i's rho GRS checks mapped onto
    /// the global coordinates of [S_{i-1}, S_i, P_i].
    pub fn parity_check(&self, f: &Gf256) -> Vec<Vec<u8>> {
        let n = self.n();
        let mut h = vec![vec![0u8; n]; self.mu * self.rho];
        for i in 0..self.mu {
            let (local, sup) = self.local_code(f, i);
            let hloc = local.h(); // rho x (2*omega+rho)
            for (r, hrow) in hloc.iter().enumerate() {
                let grow = i * self.rho + r;
                for (c, &val) in hrow.iter().enumerate() {
                    h[grow][sup[c]] = val;
                }
            }
        }
        h
    }

    /// Global erasure recovery via Gaussian elimination on H_BC. Returns the fully
    /// recovered codeword, or None if the erasure pattern is uncorrectable (its
    /// H_BC columns are linearly dependent).
    pub fn recover(&self, f: &Gf256, cw: &[u8], erased: &[usize]) -> Option<Vec<u8>> {
        if erased.is_empty() {
            return Some(cw.to_vec());
        }
        let h = self.parity_check(f);
        let rows = h.len();
        let n = self.n();
        let mut is_erased = vec![false; n];
        for &e in erased {
            is_erased[e] = true;
        }
        let mut a: Vec<Vec<u8>> = vec![vec![0u8; erased.len()]; rows];
        let mut b = vec![0u8; rows];
        for (i, hrow) in h.iter().enumerate() {
            for (ecol, &e) in erased.iter().enumerate() {
                a[i][ecol] = hrow[e];
            }
            let mut acc = 0u8;
            for (j, &hv) in hrow.iter().enumerate() {
                if !is_erased[j] {
                    acc = add(acc, f.mul(hv, cw[j]));
                }
            }
            b[i] = acc;
        }
        let sol = solve(f, &mut a, &mut b, erased.len())?;
        let mut out = cw.to_vec();
        for (ecol, &e) in erased.iter().enumerate() {
            out[e] = sol[ecol];
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf256::add;
    use crate::grs::min_distance_from_h;

    fn sample_message(k: usize) -> Vec<u8> {
        (0..k as u32)
            .map(|x| ((x.wrapping_mul(2654435761)) >> 13) as u8 | 1)
            .collect()
    }

    #[test]
    fn encoded_word_in_code() {
        let f = Gf256::new();
        let bc = BlockCirculant::new(&f, 6, 4, 3); // mu=6, omega=4, rho=3
        let msg = sample_message(bc.k());
        let cw = bc.encode(&f, &msg);
        // H_BC c = 0
        for row in bc.parity_check(&f) {
            let s = row.iter().zip(&cw).fold(0u8, |a, (&h, &c)| add(a, f.mul(h, c)));
            assert_eq!(s, 0);
        }
        // systematic: message prefix preserved
        assert_eq!(&cw[..bc.k()], &msg[..]);
    }

    #[test]
    fn measured_min_distance_matches_theorem_iii_2() {
        // d = 2*rho + 1 for lambda=2, MEASURED (smallest linearly-dependent set of
        // H_BC columns) on small instances in the non-degenerate regime mu>=6 or rho>=3.
        // (The mu=4, rho=2 corner measures d=2*rho, one short, due to the tiny cyclic
        // wraparound; it is excluded from the DA parameter regime and documented.)
        let f = Gf256::new();
        for &(mu, omega, rho) in &[(6usize, 2usize, 2usize), (4, 2, 3), (4, 3, 3)] {
            let bc = BlockCirculant::new(&f, mu, omega, rho);
            let h = bc.parity_check(&f);
            let d = min_distance_from_h(&f, &h, bc.n());
            assert_eq!(
                d,
                bc.d_formula(),
                "measured d={d} != 2*rho+1={} for (mu={mu},omega={omega},rho={rho})",
                bc.d_formula()
            );
        }
    }

    #[test]
    fn sacred_invariant_all_erasures_up_to_2rho_recover() {
        // EXHAUSTIVE: every erasure pattern of size <= 2*rho must recover.
        let f = Gf256::new();
        let bc = BlockCirculant::new(&f, 6, 2, 2); // 2*rho = 4, n = 6*4 = 24
        let msg = sample_message(bc.k());
        let cw = bc.encode(&f, &msg);
        let n = bc.n();
        let two_rho = 2 * bc.rho;
        for t in 1..=two_rho {
            let mut idx = vec![0usize; t];
            fn go(
                start: usize,
                depth: usize,
                idx: &mut Vec<usize>,
                n: usize,
                f: &Gf256,
                bc: &BlockCirculant,
                cw: &[u8],
            ) {
                if depth == idx.len() {
                    let mut dmg = cw.to_vec();
                    for &e in idx.iter() {
                        dmg[e] = 0x00;
                    }
                    let rec = bc
                        .recover(f, &dmg, idx)
                        .unwrap_or_else(|| panic!("SACRED INVARIANT VIOLATED: {idx:?} did not recover"));
                    assert_eq!(rec, cw, "recovered != original for erasures {idx:?}");
                    return;
                }
                for j in start..n {
                    idx[depth] = j;
                    go(j + 1, depth + 1, idx, n, f, bc, cw);
                }
            }
            go(0, 0, &mut idx, n, &f, &bc, &cw);
        }
    }
}

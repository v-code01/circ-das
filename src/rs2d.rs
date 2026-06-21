//! 2D (product) Reed-Solomon code -- the real, deployed DA baseline.
//!
//! This is the code Celestia / the fraud-and-data-availability-proofs scheme
//! (Al-Bassam, Sonnino, Buterin, arXiv:1809.09044) uses: a k1 x k2 message grid is
//! RS-encoded along rows (row code C1 = RS[n1,k1]) and columns (col code C2 =
//! RS[n2,k2]) to an n2 x n1 grid. The product code C1 (x) C2 has:
//!   n = n1*n2,   k = k1*k2,   d = d1*d2 = (n1-k1+1)*(n2-k2+1).
//! As a linear code it corrects every erasure pattern of size <= d-1 (columns of its
//! parity check are independent below the minimum distance), exactly like the
//! block-circulant code -- so d/n is an apples-to-apples comparison.
//!
//! We use SQUARE product codes (n1=n2, k1=k2), as deployed 2D-RS DA schemes do.

use crate::gf256::{add, Gf256};
use crate::grs::{solve, Grs};

pub struct Rs2d {
    pub n1: usize,
    pub k1: usize,
    pub n2: usize,
    pub k2: usize,
    row_code: Grs, // C1 : RS[n1, k1]
    col_code: Grs, // C2 : RS[n2, k2]
}

impl Rs2d {
    pub fn new(f: &Gf256, n1: usize, k1: usize, n2: usize, k2: usize) -> Self {
        Rs2d {
            n1,
            k1,
            n2,
            k2,
            row_code: Grs::new(f, n1, k1),
            col_code: Grs::new(f, n2, k2),
        }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n1 * self.n2
    }
    #[inline]
    pub fn k(&self) -> usize {
        self.k1 * self.k2
    }
    /// Product-code minimum distance d = d1*d2 = (n1-k1+1)*(n2-k2+1).
    #[inline]
    pub fn d_formula(&self) -> usize {
        (self.n1 - self.k1 + 1) * (self.n2 - self.k2 + 1)
    }
    #[inline]
    pub fn rate(&self) -> f64 {
        self.k() as f64 / self.n() as f64
    }

    #[inline]
    fn idx(&self, r: usize, c: usize) -> usize {
        r * self.n1 + c // n2 x n1 grid, row-major
    }

    /// Systematic encode of a k2 x k1 message (row-major, length k1*k2) into the full
    /// n2 x n1 codeword grid (row-major, length n1*n2).
    pub fn encode(&self, f: &Gf256, message: &[u8]) -> Vec<u8> {
        assert_eq!(message.len(), self.k());
        // Stage 1: RS-encode each of the k2 message rows to width n1.
        let mut rows_encoded = vec![0u8; self.k2 * self.n1];
        for r in 0..self.k2 {
            let data = &message[r * self.k1..r * self.k1 + self.k1];
            let cw = self.row_code.encode_systematic(f, data);
            rows_encoded[r * self.n1..r * self.n1 + self.n1].copy_from_slice(&cw);
        }
        // Stage 2: RS-encode each of the n1 columns (currently height k2) to height n2.
        let mut grid = vec![0u8; self.n()];
        for c in 0..self.n1 {
            let col: Vec<u8> = (0..self.k2).map(|r| rows_encoded[r * self.n1 + c]).collect();
            let cw = self.col_code.encode_systematic(f, &col);
            for (r, &v) in cw.iter().enumerate() {
                grid[self.idx(r, c)] = v;
            }
        }
        grid
    }

    /// Global product-code parity-check matrix, size (n - k) x n... actually the
    /// natural (redundant) check set: n2 row-checks of C1 plus n1 col-checks of C2.
    /// Rows: for each grid row r in 0..n2, (n1-k1) checks H1 across that row; for each
    /// grid col c in 0..n1, (n2-k2) checks H2 down that column.
    pub fn parity_check(&self) -> Vec<Vec<u8>> {
        let n = self.n();
        let h1 = self.row_code.h(); // (n1-k1) x n1
        let h2 = self.col_code.h(); // (n2-k2) x n2
        let mut h: Vec<Vec<u8>> = Vec::new();
        // Row constraints: each grid row must be a C1 codeword.
        for r in 0..self.n2 {
            for hr in h1 {
                let mut row = vec![0u8; n];
                for c in 0..self.n1 {
                    row[self.idx(r, c)] = hr[c];
                }
                h.push(row);
            }
        }
        // Column constraints: each grid column must be a C2 codeword.
        for c in 0..self.n1 {
            for hc in h2 {
                let mut row = vec![0u8; n];
                for r in 0..self.n2 {
                    row[self.idx(r, c)] = hc[r];
                }
                h.push(row);
            }
        }
        h
    }

    /// Erasure recovery via Gaussian elimination on the product parity check.
    pub fn recover(&self, f: &Gf256, cw: &[u8], erased: &[usize]) -> Option<Vec<u8>> {
        if erased.is_empty() {
            return Some(cw.to_vec());
        }
        let h = self.parity_check();
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
    use crate::grs::min_distance_from_h;

    #[test]
    fn encoded_grid_in_product_code() {
        let f = Gf256::new();
        let code = Rs2d::new(&f, 6, 4, 6, 4); // [36,16], d=9
        let msg: Vec<u8> = (0..code.k() as u8).map(|x| x.wrapping_mul(5).wrapping_add(1)).collect();
        let grid = code.encode(&f, &msg);
        for row in code.parity_check() {
            let s = row.iter().zip(&grid).fold(0u8, |a, (&h, &c)| add(a, f.mul(h, c)));
            assert_eq!(s, 0);
        }
        // systematic top-left k2 x k1 block equals the message
        for r in 0..code.k2 {
            for c in 0..code.k1 {
                assert_eq!(grid[code.idx(r, c)], msg[r * code.k1 + c]);
            }
        }
    }

    #[test]
    fn measured_min_distance_is_d1_times_d2() {
        // RS[4,2] x RS[4,2]: d1=d2=3, d=9, n=16.
        let f = Gf256::new();
        let code = Rs2d::new(&f, 4, 2, 4, 2);
        let d = min_distance_from_h(&f, &code.parity_check(), code.n());
        assert_eq!(d, code.d_formula());
        assert_eq!(d, 9);
    }

    #[test]
    fn erasure_roundtrip_random_below_distance() {
        // Random erasure patterns of size d-1 must always recover (d-1 < d).
        let f = Gf256::new();
        let code = Rs2d::new(&f, 5, 3, 5, 3); // [25,9], d=9
        let msg: Vec<u8> = (0..code.k() as u8).map(|x| x.wrapping_mul(9).wrapping_add(7)).collect();
        let grid = code.encode(&f, &msg);
        let d = code.d_formula();
        let n = code.n();
        // deterministic LCG for reproducible "random" patterns
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for _ in 0..200 {
            // choose d-1 distinct positions
            let mut chosen = std::collections::BTreeSet::new();
            while chosen.len() < d - 1 {
                chosen.insert(next() % n);
            }
            let erased: Vec<usize> = chosen.into_iter().collect();
            let mut dmg = grid.clone();
            for &e in &erased {
                dmg[e] = 0;
            }
            let rec = code
                .recover(&f, &dmg, &erased)
                .expect("all <= d-1 erasures must recover");
            assert_eq!(rec, grid);
        }
    }
}

//! Generalized Reed-Solomon (GRS) local codes and GF(2^8) linear algebra.
//!
//! A GRS [n0, k0, rho+1] code (rho = n0 - k0) is defined here by a parity-check
//! matrix H that is a rho x n0 Vandermonde over n0 distinct evaluation points:
//!   H[i][j] = alpha_j^i,   i in 0..rho,  j in 0..n0.
//! A vector c is a codeword iff H c = 0. Because every rho columns of a Vandermonde
//! over distinct points are linearly independent, the code is MDS with minimum
//! distance exactly rho+1, and it corrects ANY erasure pattern of size <= rho.
//! (Sasidharan/Viterbo/Dau use GRS local codes; the block-circulant global code is
//! assembled from these in `block_circulant.rs`.)
//!
//! This module also provides the GF(2^8) Gaussian-elimination routines that both the
//! erasure decoder and the empirical minimum-distance measurement rely on.

use crate::gf256::{add, Gf256};

/// Solve a linear system A x = b over GF(2^8) by Gauss-Jordan elimination.
///
/// `a` is `rows` x `cols` row-major. `b` has length `rows`. Returns:
///   - `Some(x)` (length `cols`) if the system is consistent AND has a unique
///     solution (rank == cols). Free variables / inconsistency return `None`.
///
/// This is exactly the condition we need for erasure recovery: a unique fill-in of
/// the erased symbols exists iff the erased-column submatrix has full column rank.
#[allow(clippy::needless_range_loop)] // index-coupled dual-array Gaussian elimination
pub fn solve(f: &Gf256, a: &mut [Vec<u8>], b: &mut [u8], cols: usize) -> Option<Vec<u8>> {
    let rows = a.len();
    let mut pivot_row = 0usize;
    let mut where_pivot = vec![usize::MAX; cols]; // which row pivots each column
    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        // Find a row at/below pivot_row with a nonzero entry in this column.
        let mut sel = pivot_row;
        while sel < rows && a[sel][col] == 0 {
            sel += 1;
        }
        if sel == rows {
            continue; // no pivot in this column
        }
        a.swap(sel, pivot_row);
        b.swap(sel, pivot_row);
        // Normalize pivot row so the pivot becomes 1.
        let inv = f.inv(a[pivot_row][col]);
        for c in col..cols {
            a[pivot_row][c] = f.mul(a[pivot_row][c], inv);
        }
        b[pivot_row] = f.mul(b[pivot_row], inv);
        // Eliminate this column from every other row.
        for r in 0..rows {
            if r != pivot_row && a[r][col] != 0 {
                let factor = a[r][col];
                for c in col..cols {
                    let t = f.mul(factor, a[pivot_row][c]);
                    a[r][c] = add(a[r][c], t);
                }
                b[r] = add(b[r], f.mul(factor, b[pivot_row]));
            }
        }
        where_pivot[col] = pivot_row;
        pivot_row += 1;
    }
    // Every column must have a pivot for a unique solution.
    if where_pivot.contains(&usize::MAX) {
        return None;
    }
    // Consistency: rows beyond rank must have b == 0.
    for r in pivot_row..rows {
        if b[r] != 0 {
            return None;
        }
    }
    let mut x = vec![0u8; cols];
    for (col, &w) in where_pivot.iter().enumerate() {
        x[col] = b[w];
    }
    Some(x)
}

/// Rank of a `rows` x `cols` GF(2^8) matrix (row-major), by Gaussian elimination.
#[allow(clippy::needless_range_loop)] // index-coupled dual-array Gaussian elimination
pub fn rank(f: &Gf256, mat: &[Vec<u8>]) -> usize {
    let mut a: Vec<Vec<u8>> = mat.to_vec();
    let rows = a.len();
    if rows == 0 {
        return 0;
    }
    let cols = a[0].len();
    let mut r = 0usize;
    for col in 0..cols {
        if r >= rows {
            break;
        }
        let mut sel = r;
        while sel < rows && a[sel][col] == 0 {
            sel += 1;
        }
        if sel == rows {
            continue;
        }
        a.swap(sel, r);
        let inv = f.inv(a[r][col]);
        for c in col..cols {
            a[r][c] = f.mul(a[r][c], inv);
        }
        for rr in 0..rows {
            if rr != r && a[rr][col] != 0 {
                let factor = a[rr][col];
                for c in col..cols {
                    let t = f.mul(factor, a[r][c]);
                    a[rr][c] = add(a[rr][c], t);
                }
            }
        }
        r += 1;
    }
    r
}

/// A GRS/RS code specified by a rho x n0 Vandermonde parity-check over `points`.
/// Convention: positions `0..k0` are DATA, positions `k0..n0` are PARITY.
pub struct Grs {
    pub n0: usize,
    pub k0: usize,
    pub rho: usize,
    /// H is rho x n0: H[i][j] = points[j]^i.
    h: Vec<Vec<u8>>,
}

impl Grs {
    /// Build a GRS code over the default `n0` distinct evaluation points g^0..g^{n0-1}.
    pub fn new(f: &Gf256, n0: usize, k0: usize) -> Self {
        let points: Vec<u8> = (0..n0).map(|j| f.exp_g(j)).collect();
        Grs::with_points(f, &points, k0)
    }

    /// Build a GRS code over DISTINCT points with parity-check powers 0..rho-1.
    pub fn with_points(f: &Gf256, points: &[u8], k0: usize) -> Self {
        Grs::with_points_offset(f, points, k0, 0)
    }

    /// Build a GRS code over DISTINCT points whose parity-check rows use the SHIFTED
    /// power window [offset .. offset+rho-1]:  H[r][j] = points[j]^(offset + r).
    ///
    /// This is a genuine GRS code (equivalently, column multipliers m_j = points[j]^offset
    /// times a standard Vandermonde), hence still MDS [n0,k0,rho+1]. The offset is the
    /// mechanism (playing the role of the paper's diagonal M matrices) that makes two
    /// overlapping local codes impose INDEPENDENT constraints on a shared block: with
    /// offsets 0 and rho, a shared block of omega symbols sees the full power window
    /// 0..2*rho-1, so the combined 2*rho constraints have rank omega and kill the
    /// spurious low-weight codewords that a plain RS (offset always 0) would admit.
    pub fn with_points_offset(f: &Gf256, points: &[u8], k0: usize, offset: usize) -> Self {
        let n0 = points.len();
        assert!(n0 <= 255, "GF(2^8) supports codes of length <= 255");
        assert!(k0 <= n0);
        for a in 0..n0 {
            for b in (a + 1)..n0 {
                assert!(points[a] != points[b], "GRS evaluation points must be distinct");
            }
        }
        let rho = n0 - k0;
        let mut h = vec![vec![0u8; n0]; rho];
        for (r, row) in h.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                // points[j]^(offset + r)
                let mut acc = 1u8;
                for _ in 0..(offset + r) {
                    acc = f.mul(acc, points[j]);
                }
                *cell = acc;
            }
        }
        Grs { n0, k0, rho, h }
    }

    /// Reference to the full parity-check matrix (rho x n0).
    pub fn h(&self) -> &[Vec<u8>] {
        &self.h
    }

    /// The j-th column of H (length rho) — one code coordinate's parity signature.
    pub fn column(&self, j: usize) -> Vec<u8> {
        self.h.iter().map(|row| row[j]).collect()
    }

    /// Systematic encode: given k0 data symbols (positions 0..k0), return the full
    /// codeword of length n0 with the rho parity symbols filled in (positions k0..n0).
    pub fn encode_systematic(&self, f: &Gf256, data: &[u8]) -> Vec<u8> {
        assert_eq!(data.len(), self.k0);
        // Parity positions are the erasures to solve for: H c = 0 => H_P p = H_D d.
        let erased: Vec<usize> = (self.k0..self.n0).collect();
        let mut cw = vec![0u8; self.n0];
        cw[..self.k0].copy_from_slice(data);
        self.recover(f, &cw, &erased)
            .expect("systematic parity is always solvable (rho independent columns)")
    }

    /// Erasure recovery: given a codeword `cw` (erased positions may hold any value)
    /// and the set of `erased` positions, return the full recovered codeword, or
    /// `None` if this erasure pattern is not correctable (|erased|>rho or dependent).
    ///
    /// Solves H_E c_E = H_K c_K where K = known positions (char 2 => no sign).
    pub fn recover(&self, f: &Gf256, cw: &[u8], erased: &[usize]) -> Option<Vec<u8>> {
        if erased.is_empty() {
            return Some(cw.to_vec());
        }
        let mut is_erased = vec![false; self.n0];
        for &e in erased {
            is_erased[e] = true;
        }
        // Build A = H_E (rho x |E|), b = sum over known j of H[:,j]*cw[j].
        let mut a: Vec<Vec<u8>> = vec![vec![0u8; erased.len()]; self.rho];
        let mut b = vec![0u8; self.rho];
        for i in 0..self.rho {
            for (ecol, &e) in erased.iter().enumerate() {
                a[i][ecol] = self.h[i][e];
            }
            let mut acc = 0u8;
            for j in 0..self.n0 {
                if !is_erased[j] {
                    acc = add(acc, f.mul(self.h[i][j], cw[j]));
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

/// Empirical minimum distance of a linear code from its parity-check matrix H:
/// d = smallest number of columns of H that are linearly dependent.
/// (Equivalently d-1 = largest t such that every t columns are independent = the
/// erasure-correction threshold.) Enumerates column subsets by increasing size;
/// intended for SMALL codes only (n0 up to ~30).
pub fn min_distance_from_h(f: &Gf256, h: &[Vec<u8>], n0: usize) -> usize {
    let rho = h.len();
    // Any (rho+1) columns are dependent (rho rows) => d <= rho+1 always.
    let upper = rho + 1;
    for t in 1..=upper {
        let mut idx = vec![0usize; t];
        if subset_has_dependent(f, h, n0, &mut idx, 0, 0) {
            return t;
        }
    }
    upper
}

fn subset_has_dependent(
    f: &Gf256,
    h: &[Vec<u8>],
    n0: usize,
    idx: &mut Vec<usize>,
    start: usize,
    depth: usize,
) -> bool {
    if depth == idx.len() {
        // Build rho x t submatrix; dependent iff rank < t.
        let t = idx.len();
        let sub: Vec<Vec<u8>> = h
            .iter()
            .map(|row| idx.iter().map(|&j| row[j]).collect())
            .collect();
        return rank(f, &sub) < t;
    }
    for j in start..n0 {
        idx[depth] = j;
        if subset_has_dependent(f, h, n0, idx, j + 1, depth + 1) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rs_roundtrip_all_erasure_patterns_up_to_rho() {
        // Small RS [12, 8, 5]: rho=4, must correct EVERY erasure pattern of size<=4.
        let f = Gf256::new();
        let (n0, k0) = (12usize, 8usize);
        let code = Grs::new(&f, n0, k0);
        let data: Vec<u8> = (0..k0 as u8).map(|x| x.wrapping_mul(7).wrapping_add(3)).collect();
        let cw = code.encode_systematic(&f, &data);
        // Codeword satisfies H c = 0.
        for row in code.h() {
            let s = row
                .iter()
                .zip(&cw)
                .fold(0u8, |acc, (&h, &c)| add(acc, f.mul(h, c)));
            assert_eq!(s, 0, "encoded word not in code");
        }
        // Exhaustive: every subset of size t<=rho recovers.
        let rho = code.rho;
        for t in 1..=rho {
            let mut idx = vec![0usize; t];
            fn go(
                start: usize,
                depth: usize,
                idx: &mut Vec<usize>,
                n0: usize,
                f: &Gf256,
                code: &Grs,
                cw: &[u8],
            ) {
                if depth == idx.len() {
                    let mut damaged = cw.to_vec();
                    for &e in idx.iter() {
                        damaged[e] = 0xAB; // clobber
                    }
                    let rec = code.recover(f, &damaged, idx).expect("must recover");
                    assert_eq!(rec, cw, "mismatch on erasures {idx:?}");
                    return;
                }
                for j in start..n0 {
                    idx[depth] = j;
                    go(j + 1, depth + 1, idx, n0, f, code, cw);
                }
            }
            go(0, 0, &mut idx, n0, &f, &code, &cw);
        }
    }

    #[test]
    fn rs_min_distance_is_rho_plus_one() {
        // Empirical: smallest linearly-dependent set of H-columns == d == rho+1.
        let f = Gf256::new();
        let code = Grs::new(&f, 10, 6); // rho=4 => d=5
        let d = crate::grs::min_distance_from_h(&f, code.h(), code.n0);
        assert_eq!(d, code.rho + 1);
    }
}

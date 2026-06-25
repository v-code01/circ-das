//! NEON GF(2^8) block-circulant shard encoder, with a tuned scalar reference.
//!
//! The block-circulant encode is, per parity symbol, a GF(2^8) mul-accumulate over the
//! `2*omega` data symbols of its local code: `P_r = XOR_c gen[r][c] * D_c`. In a real
//! data-availability deployment each "symbol" is not one field element but a SHARD of
//! many bytes, so the encode is a stack of `dst[] ^= const * src[]` GF(2^8)
//! mul-accumulates over byte vectors — exactly the kernel that ARM NEON can vectorize.
//!
//! ## The split-nibble TBL technique (`vqtbl1q_u8`)
//! GF(2^8) multiply by a fixed constant `c` is an F2-linear map on the 8-bit input, so it
//! distributes over the XOR decomposition of a byte into its low and high nibble:
//!   `x = (x_hi << 4) ^ x_lo`  =>  `c*x = c*(x_hi<<4) ^ c*x_lo`.
//! Precompute two 16-byte tables for `c`:  `lo[i] = c*i`, `hi[i] = c*(i<<4)`, i in 0..16.
//! Then for a 16-byte vector `v`:
//!   `c*v = vqtbl1q_u8(lo_tbl, v & 0x0f) ^ vqtbl1q_u8(hi_tbl, v >> 4)`
//! — two 16-way in-register table lookups plus an XOR, 16 bytes per iteration, with no
//! memory gather beyond the two 16-byte tables held in vector registers.
//!
//! ## Honesty (prove-or-demote)
//! `encode_neon` and `encode_scalar` perform IDENTICAL work on the SAME block-circulant
//! generator matrices — the only difference is scalar-per-byte vs 16-bytes/instruction.
//! That is the apples-to-apples gate. The `reed-solomon-simd` crate is measured separately
//! as a well-tuned NEON GF(2^8) *reference point* for a DIFFERENT code (standard RS); it is
//! never presented as a head-to-head on identical computation. See `bin/throughput.rs`.

use crate::block_circulant::BlockCirculant;
use crate::gf256::Gf256;

/// Split-nibble product tables for one fixed field constant `c`.
///
/// `lo[i] = c * i`        for i in 0..16 (the low-nibble contribution)
/// `hi[i] = c * (i << 4)` for i in 0..16 (the high-nibble contribution)
///
/// so that `c*x = hi[x>>4] ^ lo[x & 0x0f]` for every byte `x`.
#[derive(Clone, Copy)]
struct MulTable {
    lo: [u8; 16],
    hi: [u8; 16],
}

impl MulTable {
    #[inline]
    fn new(f: &Gf256, c: u8) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for i in 0u8..16 {
            lo[i as usize] = f.mul(c, i);
            hi[i as usize] = f.mul(c, i << 4);
        }
        MulTable { lo, hi }
    }
}

/// Tuned scalar `dst[] ^= c * src[]` over GF(2^8).
///
/// The tuned scalar path (as used by e.g. klauspost/reedsolomon's non-SIMD backend)
/// builds one full 256-entry product table for the constant, then does a single table
/// lookup plus XOR per byte — branch-free, one load per byte. This is a REAL tuned
/// baseline, not a bit-serial strawman. `src` and `dst` must have equal length.
#[inline]
fn mul_add_scalar(f: &Gf256, c: u8, src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    let mut table = [0u8; 256];
    for (x, t) in table.iter_mut().enumerate() {
        *t = f.mul(c, x as u8);
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= table[s as usize];
    }
}

/// NEON `dst[] ^= c * src[]` over GF(2^8) via the split-nibble TBL technique.
///
/// # Safety
/// Requires the `neon` target feature (guaranteed present on every `aarch64` target: NEON
/// is mandatory in ARMv8-A). Caller must guarantee `src.len() == dst.len()`; only indices
/// `0..len` of each are accessed, all in bounds, so the raw loads/stores are sound.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_add_neon(t: &MulTable, src: &[u8], dst: &mut [u8]) {
    use core::arch::aarch64::{
        uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vqtbl1q_u8, vshrq_n_u8, vst1q_u8,
    };
    debug_assert_eq!(src.len(), dst.len());
    // SAFETY: the two tables are exactly 16 bytes, matching a full uint8x16_t load.
    let lo_tbl: uint8x16_t = vld1q_u8(t.lo.as_ptr());
    let hi_tbl: uint8x16_t = vld1q_u8(t.hi.as_ptr());
    let mask = vdupq_n_u8(0x0f);
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut j = 0usize;
    // Main vectorized body: 16 bytes / iteration.
    while j + 16 <= n {
        // SAFETY: j+16 <= n and dst.len()==n, so [j, j+16) is in bounds for both.
        let v = vld1q_u8(sp.add(j));
        let lo = vandq_u8(v, mask); // low nibble of each byte
        let hi = vshrq_n_u8::<4>(v); // high nibble of each byte (logical >> 4)
        // vqtbl1q_u8 returns 0 for indices >= 16; nibbles are always 0..15, so every
        // lookup hits a real table entry.
        let prod = veorq_u8(vqtbl1q_u8(lo_tbl, lo), vqtbl1q_u8(hi_tbl, hi));
        let acc = vld1q_u8(dp.add(j));
        vst1q_u8(dp.add(j), veorq_u8(acc, prod));
        j += 16;
    }
    // Scalar tail for the final < 16 bytes, using the same split tables.
    while j < n {
        // SAFETY: j < n and both slices have length n.
        let s = *src.get_unchecked(j);
        let p = t.lo[(s & 0x0f) as usize] ^ t.hi[(s >> 4) as usize];
        *dst.get_unchecked_mut(j) ^= p;
        j += 1;
    }
}

/// A block-circulant SHARD encoder: encodes `mu*omega` data shards into `mu*rho` parity
/// shards, each shard `shard_len` bytes. Scalar and NEON paths do identical work.
pub struct BcEncoder {
    pub mu: usize,
    pub omega: usize,
    pub rho: usize,
    /// `gen[i]` is the `rho x (2*omega)` systematic generator of local code `i`.
    gen: Vec<Vec<Vec<u8>>>,
}

impl BcEncoder {
    /// Build from a constructed block-circulant code (captures its generator matrices).
    pub fn new(f: &Gf256, bc: &BlockCirculant) -> Self {
        BcEncoder {
            mu: bc.mu,
            omega: bc.omega,
            rho: bc.rho,
            gen: bc.local_generators(f),
        }
    }

    /// Number of input data shards (`k = mu*omega`).
    #[inline]
    pub fn k_shards(&self) -> usize {
        self.mu * self.omega
    }

    /// Number of output parity shards (`mu*rho`).
    #[inline]
    pub fn parity_shards(&self) -> usize {
        self.mu * self.rho
    }

    /// Global index of the data shard feeding column `c` of local code `i`.
    /// Columns `0..omega` are block `i-1`; columns `omega..2*omega` are block `i`.
    #[inline]
    fn src_shard(&self, i: usize, c: usize) -> usize {
        let prev = (i + self.mu - 1) % self.mu;
        if c < self.omega {
            prev * self.omega + c
        } else {
            i * self.omega + (c - self.omega)
        }
    }

    /// Tuned-scalar encode. `data` = `k_shards()*shard_len` bytes (shard-major),
    /// `parity` = `parity_shards()*shard_len` bytes (overwritten).
    pub fn encode_scalar(&self, f: &Gf256, data: &[u8], shard_len: usize, parity: &mut [u8]) {
        assert_eq!(data.len(), self.k_shards() * shard_len);
        assert_eq!(parity.len(), self.parity_shards() * shard_len);
        for out in parity.iter_mut() {
            *out = 0;
        }
        let two_omega = 2 * self.omega;
        for i in 0..self.mu {
            for r in 0..self.rho {
                let dst_off = (i * self.rho + r) * shard_len;
                for c in 0..two_omega {
                    let coef = self.gen[i][r][c];
                    if coef == 0 {
                        continue;
                    }
                    let src_off = self.src_shard(i, c) * shard_len;
                    // Disjoint slices (data vs parity are separate buffers).
                    mul_add_scalar(
                        f,
                        coef,
                        &data[src_off..src_off + shard_len],
                        &mut parity[dst_off..dst_off + shard_len],
                    );
                }
            }
        }
    }

    /// NEON encode — identical work to `encode_scalar`, 16 bytes/instruction.
    /// On non-aarch64 targets this delegates to the scalar path (portable fallback).
    #[cfg(target_arch = "aarch64")]
    pub fn encode_neon(&self, f: &Gf256, data: &[u8], shard_len: usize, parity: &mut [u8]) {
        assert_eq!(data.len(), self.k_shards() * shard_len);
        assert_eq!(parity.len(), self.parity_shards() * shard_len);
        for out in parity.iter_mut() {
            *out = 0;
        }
        let two_omega = 2 * self.omega;
        for i in 0..self.mu {
            for r in 0..self.rho {
                let dst_off = (i * self.rho + r) * shard_len;
                for c in 0..two_omega {
                    let coef = self.gen[i][r][c];
                    if coef == 0 {
                        continue;
                    }
                    let src_off = self.src_shard(i, c) * shard_len;
                    let table = MulTable::new(f, coef);
                    // SAFETY: neon is guaranteed on aarch64; the two slices are equal-length
                    // (`shard_len`) disjoint subslices of separate buffers.
                    unsafe {
                        mul_add_neon(
                            &table,
                            &data[src_off..src_off + shard_len],
                            &mut parity[dst_off..dst_off + shard_len],
                        );
                    }
                }
            }
        }
    }

    /// Portable fallback: on non-aarch64 `encode_neon` is the scalar path.
    #[cfg(not(target_arch = "aarch64"))]
    pub fn encode_neon(&self, f: &Gf256, data: &[u8], shard_len: usize, parity: &mut [u8]) {
        self.encode_scalar(f, data, shard_len, parity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Deterministic pseudo-random byte stream for reproducible fixtures.
    fn fill(seed: u64, n: usize) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                // xorshift64
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn split_table_mul_matches_scalar_field_mul() {
        // The split-nibble decomposition must equal the field multiply for every (c, x).
        let f = Gf256::new();
        for c in 0u16..=255 {
            let t = MulTable::new(&f, c as u8);
            for x in 0u16..=255 {
                let expect = f.mul(c as u8, x as u8);
                let got = t.lo[(x as u8 & 0x0f) as usize] ^ t.hi[(x as u8 >> 4) as usize];
                assert_eq!(got, expect, "split table wrong for c={c} x={x}");
            }
        }
    }

    #[test]
    fn shard_l1_matches_reference_encode() {
        // With shard_len = 1, the shard encoder must reproduce BlockCirculant::encode's
        // parity bytes exactly (proves the extracted generator == the tested encode path).
        let f = Gf256::new();
        for &(mu, omega, rho) in &[(6usize, 4usize, 3usize), (6, 2, 2), (8, 3, 2)] {
            let bc = BlockCirculant::new(&f, mu, omega, rho);
            let enc = BcEncoder::new(&f, &bc);
            let msg = fill(0xC0FFEE ^ (mu as u64), bc.k());
            let full = bc.encode(&f, &msg);
            let reference_parity = &full[bc.k()..]; // mu*rho parity bytes, global order
            let mut parity = vec![0u8; enc.parity_shards()];
            enc.encode_scalar(&f, &msg, 1, &mut parity);
            assert_eq!(
                parity, reference_parity,
                "shard(L=1) parity != reference encode for (mu={mu},omega={omega},rho={rho})"
            );
        }
    }

    #[test]
    fn neon_equals_scalar_fixed_params() {
        // The non-negotiable gate on real DA-shaped params and non-trivial shard sizes,
        // including sizes that exercise the vector tail (not a multiple of 16).
        let f = Gf256::new();
        let params = [(6usize, 8usize, 4usize), (6, 16, 2), (4, 6, 3), (8, 4, 2)];
        for &(mu, omega, rho) in &params {
            let bc = BlockCirculant::new(&f, mu, omega, rho);
            let enc = BcEncoder::new(&f, &bc);
            for &shard_len in &[1usize, 15, 16, 17, 100, 4096, 4099] {
                let data = fill(0xABCDEF ^ (shard_len as u64), enc.k_shards() * shard_len);
                let mut ps = vec![0u8; enc.parity_shards() * shard_len];
                let mut pn = vec![0u8; enc.parity_shards() * shard_len];
                enc.encode_scalar(&f, &data, shard_len, &mut ps);
                enc.encode_neon(&f, &data, shard_len, &mut pn);
                assert_eq!(
                    ps, pn,
                    "NEON != scalar for (mu={mu},omega={omega},rho={rho}) L={shard_len}"
                );
            }
        }
    }

    proptest! {
        // Byte-identical NEON == scalar over random inputs, params, and shard sizes.
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn neon_equals_scalar_random(
            omega in 2usize..12,
            rho in 1usize..6,
            mu_half in 1usize..4,
            shard_len in 1usize..600,
            seed in any::<u64>(),
        ) {
            let mu = 2 * mu_half + 2; // even, >= 4
            let f = Gf256::new();
            // Keep local length within GF(2^8): 2*omega + rho <= 255 (always true here).
            let bc = BlockCirculant::new(&f, mu, omega, rho);
            let enc = BcEncoder::new(&f, &bc);
            let data = fill(seed, enc.k_shards() * shard_len);
            let mut ps = vec![0u8; enc.parity_shards() * shard_len];
            let mut pn = vec![0u8; enc.parity_shards() * shard_len];
            enc.encode_scalar(&f, &data, shard_len, &mut ps);
            enc.encode_neon(&f, &data, shard_len, &mut pn);
            prop_assert_eq!(ps, pn);
        }
    }
}

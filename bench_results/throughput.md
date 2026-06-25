# circ-das — NEON GF(2^8) block-circulant encoder throughput (honest)

Machine: Apple M4 (aarch64, ARMv8 NEON mandatory). Rust release (`opt-level=3`, `lto`).
Emitted by `cargo run --release --example throughput`; this file is the committed artifact.
Metric for all rows: **input-data throughput** `MB/s = (data bytes) / encode wall time`,
`1 MB = 1e6 B`; best-of-N (peak) and median over 300 timed iterations (30 warmup),
`shard_len = 16384 B`.

Honesty condition 2 (spec): NEON is measured against a REAL well-tuned baseline, never a
bit-serial strawman. Two baselines are reported with distinct, clearly-labeled roles.

## PRIMARY — the prove-or-demote gate (apples-to-apples)

NEON encode vs **our own tuned scalar** encode on the **SAME block-circulant code**:
identical generator matrices, identical `dst[] ^= const * src[]` GF(2^8) mul-accumulate
structure, identical shard layout. The ONLY difference is the inner multiply:

- **Scalar (tuned, not a strawman):** one full 256-entry product table per constant, then
  a single table lookup + XOR per byte, branch-free (the klauspost/reedsolomon non-SIMD
  technique). This is the honest best-effort scalar, not bit-serial.
- **NEON:** split-nibble `vqtbl1q_u8` — `c*x = tbl(lo, x & 0x0f) ^ tbl(hi, x >> 4)`,
  16 bytes per instruction, two in-register 16-byte tables, no memory gather.

Real, GF(2^8)-constructible block-circulant operating points. The encode KERNEL is
size-independent (cost/byte = `2*omega` mul-adds per parity shard x `mu*rho` parity
shards), so these small constructible codes measure the same per-byte kernel that runs at
the k=1024 DA sizes.

| BC (mu,omega,rho) | rate R | k shards | parity shards | scalar MB/s (peak/med) | NEON MB/s (peak/med) | **NEON/scalar** |
|---|---|---|---|---|---|---|
| (6, 8, 4)  | 0.667 | 48 | 24 | 527 / 524   | 6942 / 6627   | **13.18x** |
| (8, 8, 2)  | 0.800 | 64 | 16 | 1054 / 1049 | 13767 / 13336 | **13.07x** |
| (6, 16, 2) | 0.889 | 96 | 12 | 1060 / 1053 | 13919 / 13241 | **13.13x** |

**Verdict: PASS.** NEON beats our own tuned scalar by **~13.1x** on identical
block-circulant work, stable across rate. The split-nibble TBL kernel is the win: it
replaces one dependent table load per byte with 16-lane in-register lookups. (Absolute
MB/s scales with rate because higher rate = fewer parity shards = fewer mul-adds per input
byte, so more input bytes clear per unit work; the NEON/scalar ratio is kernel-bound and
rate-independent, as expected.)

Correctness gate behind every number: `neon.rs` proves `encode_neon == encode_scalar`
byte-identical over proptest (200 cases, random params/data/shard-length incl. vector
tails) + fixed DA params, and the scalar encode itself is proven == `BlockCirculant::encode`
at shard_len=1. A fast-but-wrong encoder is worthless; this one is bit-exact.

## REFERENCE POINT — NOT a head-to-head

`reed-solomon-simd` v3.1.0 (Leopard-RS): a well-tuned SIMD Reed-Solomon encoder on this
same arm64 machine, given its best shot (working space reused via `reset()` each encode).

| RS (orig, recovery) | shard_len | input MB/s (peak/med) |
|---|---|---|
| (48, 24) | 16384 | 6332 / 6200 |
| (96, 12) | 16384 | 9887 / 9841 |

**This is a ballpark reference, explicitly NOT "we beat reed-solomon-simd."** It encodes a
DIFFERENT code (standard Reed-Solomon, not block-circulant local codes), with a DIFFERENT
algorithm (O(n log n) FFT, not systematic mul-accumulate), over a DIFFERENT field
(GF(2^16), not GF(2^8)). The numbers are not comparable as identical computation. Its only
purpose: confirm our NEON GF(2^8) kernel lands in the **right neighborhood** for a tuned
SIMD erasure encoder on M4 (single-digit GB/s), i.e. we are not accidentally slow. It does.
Our per-byte GF(2^8) TBL kernel and its FFT-over-GF(2^16) kernel have different
work-per-byte and different redundancy scaling; reading these two tables against each
other as a race would be exactly the baseline dishonesty this project forbids.

## Bottom line

The apples-to-apples, prove-or-demote gate **PASSES**: the NEON split-nibble `vqtbl1q_u8`
encoder is ~13.1x our own tuned scalar on identical block-circulant work, bit-for-bit
correct. The `reed-solomon-simd` reference confirms we are in the right performance
neighborhood for tuned SIMD erasure coding on this machine. The d/n headline finding
(`bench_results/dn.md`) stands independently and does not depend on this systems layer.

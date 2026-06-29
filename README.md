# circ-das

First implementation and honest measurement of block-circulant local codes
(arXiv:2406.12160, Sasidharan/Viterbo/Dau, 2024) for blockchain data availability.
A NEON GF(2^8) encoder, a coded-Merkle DAS sampler, a global Gaussian erasure decoder.
Rust, single binary, measured on Apple M4.

The sacred invariant is no data loss: the global decoder recovers every erasure pattern
up to the code's true minimum distance, proven exhaustively.

## What this is, stated at its real level

A rigorous, genuinely novel primitive. It contains the first measured
block-circulant-vs-2D-Reed-Solomon data-availability distance curve, and the first
implementation of the 2024 block-circulant code family for a DA setting. It is not a
proven breakthrough and does not claim to be one. Every finding below is falsifiable,
disclosed in full range, and backed by code you can run.

Baseline throughout is 2D Reed-Solomon, the deployed DA scheme (Celestia, arXiv:1809.09044).
Every result is at matched dimension k = 1024, block-circulant lambda = 2, mu = 16.

## Finding 1: relative minimum distance (d/n)

Block-circulant beats 2D-RS relative minimum distance in the high-rate regime.

| rate R | BC d/n | 2D-RS d/n | winner | BC / 2D-RS |
|---|---|---|---|---|
| 0.500 | 0.0630 | 0.0968 | 2D-RS | 0.65x |
| 0.667 | 0.0423 | 0.0421 | BC | 1.01x |
| 0.800 | 0.0258 | 0.0193 | BC | 1.34x |
| 0.900 | 0.0132 | 0.0078 | BC | 1.70x |
| 0.950 | 0.0065 | 0.0037 | BC | 1.78x |

The win reaches 1.70x at R = 0.90 and 1.78x at R = 0.95, and reproduces the paper's
1.35x reference point. Full 7-point curve in `bench_results/spike.md` and `bench_results/dn.md`.

Scope, disclosed:

- Crossover is at R ~ 0.67. Below it 2D-RS wins. This is a high-rate result, not a universal one.
- The win is mu-dependent. It holds iff mu < mu*(R) = 2(1 + sqrt R) / (1 - sqrt R), a
  threshold that rises steeply with rate (mu*(0.9) ~ 76). Measured at the disclosed mu = 16.
  A larger mu at the same rate can erase the advantage.
- Only lambda = 2 is implemented. `d = 2*rho + 1` is the lambda = 2 theorem.
- Distances are MDS-exact formulas. The k = 1024 sizes would require GF(2^16), so both
  formulas are confirmed empirically on small GF(2^8) instances by direct minimum-distance
  search, then applied at DA sizes. No field is fabricated.

## Finding 2: NEON GF(2^8) encoder throughput

The split-nibble TBL GF(2^8) encoder runs ~13.1x our own tuned scalar on identical
block-circulant work.

| BC (mu,omega,rho) | rate R | scalar MB/s | NEON MB/s | NEON / scalar |
|---|---|---|---|---|
| (6, 8, 4)  | 0.667 | 527  | 6942  | 13.18x |
| (8, 8, 2)  | 0.800 | 1054 | 13767 | 13.07x |
| (6, 16, 2) | 0.889 | 1060 | 13919 | 13.13x |

This is the prove-or-demote gate: NEON `vqtbl1q_u8` split-nibble multiply versus a real
tuned scalar (full 256-entry product table, one lookup plus XOR per byte, the
klauspost/reedsolomon non-SIMD technique), same generator matrices, same shard layout,
same mul-accumulate structure. The only difference is the inner multiply. Not a bit-serial
strawman. Both paths are proven byte-identical (`encode_neon == encode_scalar` over
proptest plus fixed DA params). Full numbers in `bench_results/throughput.md`.

`reed-solomon-simd` (v3.1.0, Leopard-RS) is a reference point only, not a head-to-head.
It is a different code, a different algorithm (O(n log n) FFT), and a different field
(GF(2^16), not GF(2^8)). Its only job is to confirm the NEON kernel lands in the right
neighborhood for tuned SIMD erasure coding on M4. Reading the two as a race is exactly
the baseline dishonesty this project forbids.

Scope caveat, stated plainly. The NEON encoder is GF(2^8). The DA operating point
(mu = 16, omega = 64) needs GF(2^16). So throughput is measured on GF(2^8)-constructible
block-circulant codes, and the per-byte ratio is argued to transfer because the encode
kernel is size-independent (cost per byte is `2*omega` mul-adds per parity shard times
`mu*rho` parity shards, independent of shard length). Finding 1 (DA sizes) and Finding 2
(GF(2^8) throughput) are measured at different scales and are not conflated.

## Finding 3: DAS sample count

Block-circulant needs fewer DAS samples than 2D-RS for the same 2^-80 soundness at high rate.

| rate R | BC samples | 2D-RS samples | fewer | 2D-RS / BC |
|---|---|---|---|---|
| 0.667 | 1283 | 1290  | BC | 1.01x |
| 0.900 | 4172 | 7095  | BC | 1.70x |
| 0.950 | 8465 | 15070 | BC | 1.78x |

At R = 0.90 block-circulant needs 4172 samples versus 2D-RS 7095, which is 1.70x fewer.
At R = 0.95 the ratio is 1.78x fewer. Because required samples scale as `1/(d/n)`, the
sample-count ratio tracks the distance advantage exactly, and crosses over at the same
R ~ 0.67. Full table in `bench_results/das.md`.

Two honesty conditions:

- The bound is derived from the deployed global decoder's true threshold d, not an
  idealized value and not an iterative-decoder threshold. There is no iterative decoder,
  so there is no MDS-vs-iterative conflation.
- The result is tested against the worst-case adversary: withhold the size-d min-weight
  codeword support (`grs::min_weight_support`), not a random set. Random withholding would
  inflate detection and report fewer samples for both codes, which is a tautology, not a
  result. Self-audit: at R = 0.90 the tabulated 4172 samples give worst-case miss
  8.246e-25 <= 2^-80.

Caveat. The Merkle digest is a self-contained demo hash. Production would use BLAKE3 or
SHA-256. The commitment structure and the soundness argument are hash-agnostic, so the
sample counts are unaffected.

## The sacred invariant: no data loss

Every d above is real recovery, not an assumption. The global Gauss-Jordan decoder
(`BlockCirculant::recover`, `Rs2d::recover`) recovers every erasure pattern of size
<= d-1. Proof, in `tests/invariant_erasure.rs`:

- Exhaustive recovery over 52,152 patterns: all 12,950 block-circulant patterns of size
  <= d-1 on the (mu=6,omega=2,rho=2) instance, plus all 39,202 2D-RS patterns of size
  <= d-1 on the (4,2)x(4,2) instance. Every one recovers byte-identical.
- Distance boundary: a size-d min-weight codeword support genuinely does NOT recover,
  confirming d is the exact threshold and not merely a lower bound.

Erasure-correctability of a set E holds iff the parity-check columns indexed by E are
linearly independent, so d is the smallest number of linearly dependent columns. Both MDS
formulas are confirmed against this direct measurement on small GF(2^8) instances before
being applied at the k = 1024 DA sizes.

## Status and limitations

- Single-thread throughout.
- lambda = 2 only. General lambda is out of scope.
- NEON encoder is GF(2^8); DA operating sizes need GF(2^16). Different scales, disclosed.
- Merkle digest is a demo hash, not production BLAKE3 or SHA-256.
- The d/n and sample-count wins are high-rate only (R >~ 0.67) and mu-dependent (mu = 16).
  Below the crossover, 2D-RS wins, disclosed in the full curves.

## Reproduce

```
cargo test                       # 36 tests: invariants, distances, soundness, NEON equality
cargo run --release --bin spike        # d/n vs rate -> bench_results/spike.md
cargo run --release --bin das_bench    # DAS samples vs rate -> bench_results/das.md
cargo run --release --example throughput   # NEON vs scalar -> bench_results/throughput.md
```

## License

MIT. See `LICENSE`.

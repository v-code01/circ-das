//! Honest throughput benchmark for the block-circulant NEON GF(2^8) encoder.
//!
//! PRIMARY (apples-to-apples, prove-or-demote gate): NEON encode vs OUR OWN tuned scalar
//! encode on the SAME block-circulant code — identical generator matrices, identical
//! mul-accumulate work, only the inner GF(2^8) multiply differs (16 bytes/instruction TBL
//! vs a 256-entry per-constant table). This ratio is the legitimate NEON number.
//!
//! REFERENCE POINT (NOT a head-to-head): `reed-solomon-simd` (Leopard-RS, GF(2^16),
//! FFT-based, O(n log n)) as a well-tuned SIMD RS encoder on this same arm64 machine — a
//! ballpark "are we in the right neighborhood" number. It encodes a DIFFERENT code with a
//! DIFFERENT algorithm and field, so it is explicitly a reference, never "we beat it".
//!
//! Throughput metric (all three): input-data MB/s = (data bytes) / encode wall time,
//! where 1 MB = 1e6 bytes. Best-of-N (peak) plus median over N timed iterations.

use circ_das::block_circulant::BlockCirculant;
use circ_das::gf256::Gf256;
use circ_das::neon::BcEncoder;
use reed_solomon_simd::ReedSolomonEncoder;
use std::hint::black_box;
use std::time::Instant;

const SHARD_LEN: usize = 16 * 1024; // 16 KiB per shard (mult. of 64, fits rs-simd)
const ITERS: usize = 300;
const WARMUP: usize = 30;

/// xorshift64 fill for reproducible data.
fn fill(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

/// Run `f` ITERS times (after WARMUP), return (peak_MBps, median_MBps) over `data_bytes`.
fn measure(data_bytes: usize, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times[0];
    let median = times[ITERS / 2];
    let mbps = |secs: f64| (data_bytes as f64) / secs / 1.0e6;
    (mbps(best), mbps(median))
}

fn main() {
    let f = Gf256::new();
    println!("# circ-das NEON GF(2^8) encoder throughput (Apple M4, aarch64/NEON)");
    println!("shard_len = {SHARD_LEN} B, iters = {ITERS} (warmup {WARMUP}), MB = 1e6 B\n");

    // Real, GF(2^8)-constructible block-circulant operating points (the encode KERNEL is
    // size-independent: cost/byte = 2*omega mul-adds per parity shard, mu*rho parity shards).
    let params = [
        (6usize, 8usize, 4usize),  // rate 0.667 (crossover point)
        (8, 8, 2),                 // rate 0.800
        (6, 16, 2),                // rate 0.889 (high-rate DA sweet spot)
    ];

    println!("## PRIMARY: NEON vs our own tuned scalar (SAME BC code, identical work)\n");
    println!("| BC (mu,omega,rho) | rate | k shards | parity shards | scalar MB/s (peak/med) | NEON MB/s (peak/med) | NEON/scalar |");
    println!("|---|---|---|---|---|---|---|");
    for &(mu, omega, rho) in &params {
        let bc = BlockCirculant::new(&f, mu, omega, rho);
        let enc = BcEncoder::new(&f, &bc);
        let data = fill(0x5EED ^ (mu * 131 + omega * 17 + rho) as u64, enc.k_shards() * SHARD_LEN);
        let data_bytes = enc.k_shards() * SHARD_LEN;
        let mut par = vec![0u8; enc.parity_shards() * SHARD_LEN];

        let (s_peak, s_med) = measure(data_bytes, || {
            enc.encode_scalar(&f, black_box(&data), SHARD_LEN, black_box(&mut par));
        });
        let (n_peak, n_med) = measure(data_bytes, || {
            enc.encode_neon(&f, black_box(&data), SHARD_LEN, black_box(&mut par));
        });
        let rate = omega as f64 / (rho + omega) as f64;
        println!(
            "| ({mu},{omega},{rho}) | {:.3} | {} | {} | {:.0} / {:.0} | {:.0} / {:.0} | {:.2}x |",
            rate,
            enc.k_shards(),
            enc.parity_shards(),
            s_peak, s_med, n_peak, n_med,
            n_peak / s_peak
        );
    }

    // REFERENCE POINT: reed-solomon-simd (Leopard-RS, GF(2^16), FFT). Different code,
    // different algorithm, different field. Sized to a comparable data shape (original
    // = a BC data-shard count, recovery = a BC parity-shard count) for a ballpark only.
    println!("\n## REFERENCE POINT (NOT head-to-head): reed-solomon-simd (Leopard, GF(2^16), FFT)\n");
    println!("Different code (standard RS), different algorithm (O(n log n) FFT), different");
    println!("field (GF(2^16)). Ballpark 'well-tuned SIMD RS on this machine' only.\n");
    println!("| RS (orig, recovery) | shard_len | input MB/s (peak/med) |");
    println!("|---|---|---|");
    for &(orig, recov) in &[(48usize, 24usize), (96usize, 12usize)] {
        if !ReedSolomonEncoder::supports(orig, recov) {
            println!("| ({orig},{recov}) | {SHARD_LEN} | unsupported config |");
            continue;
        }
        let shards: Vec<Vec<u8>> = (0..orig)
            .map(|i| fill(0xA11CE ^ i as u64, SHARD_LEN))
            .collect();
        let data_bytes = orig * SHARD_LEN;
        // Give rs-simd its best shot: allocate once, reuse working space via reset()
        // each encode so the reference number is its true per-encode throughput (feed
        // shards + FFT encode), not penalized by repeated allocation.
        let mut enc = ReedSolomonEncoder::new(orig, recov, SHARD_LEN).unwrap();
        let (peak, med) = measure(data_bytes, || {
            enc.reset(orig, recov, SHARD_LEN).unwrap();
            for s in &shards {
                enc.add_original_shard(s).unwrap();
            }
            let res = enc.encode().unwrap();
            // Force the recovery shards to be produced.
            let mut acc = 0u8;
            for r in res.recovery_iter() {
                acc ^= r[0];
            }
            black_box(acc);
        });
        println!("| ({orig},{recov}) | {SHARD_LEN} | {peak:.0} / {med:.0} |");
    }

    println!("\n(reed-solomon-simd reuses its working space via reset() each encode — its");
    println!("true per-encode throughput, not penalized by allocation. Our BcEncoder setup");
    println!("is likewise one-time and excluded from the timed loop for both scalar and NEON,");
    println!("so scalar-vs-NEON stays strictly apples-to-apples.)");
}

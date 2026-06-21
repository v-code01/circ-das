//! circ-das: first measured minimum-distance-ratio (d/n) vs code-rate curve for
//! BLOCK-CIRCULANT LOCAL CODES (Sasidharan/Viterbo/Dau, arXiv:2406.12160, "Block
//! Circulant Codes with Application to Decentralized Systems", 2024) vs 2D
//! Reed-Solomon, for blockchain data availability.
//!
//! Honest-either-way SPIKE: prove a real d/n advantage at high rate, or ship the
//! first measured table showing where there is none.

pub mod gf256;
pub mod grs;
pub mod block_circulant;

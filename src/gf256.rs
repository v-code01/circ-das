//! GF(2^8) arithmetic via scalar log/antilog tables.
//!
//! Field: GF(2) polynomials mod the primitive polynomial
//!   p(x) = x^8 + x^4 + x^3 + x^2 + 1  (0x11D),
//! the standard Reed-Solomon field polynomial (used by e.g. klauspost/reedsolomon,
//! Backblaze). The multiplicative group GF(2^8)* is cyclic of order 255; we use the
//! generator g = 2 (verified at table-build time to have full period 255).
//!
//! WHY log/antilog (not carryless PMULL here): the d/n headline is field-size
//! independent (MDS distances). We only need CORRECT, portable field arithmetic to
//! (a) run the erasure-recovery roundtrip and (b) empirically confirm the distance
//! formulas on small codes. The NEON/TBL throughput bet is a separate, demotable layer.

/// Precomputed log/antilog tables for GF(2^8) with poly 0x11D, generator g=2.
pub struct Gf256 {
    /// antilog[i] = g^i for i in 0..255 (period 255), extended to 0..510 to avoid
    /// modular reduction of exponents in the hot multiply path.
    exp: [u8; 512],
    /// log[x] = discrete log base g of x, for x in 1..=255. log[0] is undefined (set 0).
    log: [u8; 256],
}

/// Reduce a * x (as GF(2)[x] product with x, i.e. left shift) modulo 0x11D.
#[inline]
const fn xtime(a: u16) -> u16 {
    let a = a << 1;
    if a & 0x100 != 0 {
        a ^ 0x11D
    } else {
        a
    }
}

impl Gf256 {
    /// Build the tables. Panics if g=2 does not have full period (it does for 0x11D).
    pub fn new() -> Self {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        // Generate the cyclic group: x = g^i, starting at g^0 = 1.
        let mut x: u16 = 1;
        let mut i = 0usize;
        while i < 255 {
            exp[i] = x as u8;
            log[x as usize] = i as u8;
            x = xtime(x); // multiply by g=2
            i += 1;
        }
        // Full-period invariant: after 255 steps we must return to 1.
        assert!(x == 1, "g=2 is not a generator of GF(2^8) with poly 0x11D");
        // Mirror for exponents 255..510 so exp[i+j] is valid for i,j in 0..=254.
        let mut j = 255usize;
        while j < 510 {
            exp[j] = exp[j - 255];
            j += 1;
        }
        Gf256 { exp, log }
    }

    /// Multiply two field elements. O(1) via logs. mul(0, _) = mul(_, 0) = 0.
    #[inline]
    pub fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        // g^(log a + log b); exponent in 0..=508 covered by the extended table.
        self.exp[self.log[a as usize] as usize + self.log[b as usize] as usize]
    }

    /// Multiplicative inverse. inv(0) is undefined; returns 0 (callers must not divide by 0).
    #[inline]
    pub fn inv(&self, a: u8) -> u8 {
        debug_assert!(a != 0, "GF(2^8) inverse of 0");
        if a == 0 {
            return 0;
        }
        // a^-1 = g^(255 - log a)
        self.exp[255 - self.log[a as usize] as usize]
    }

    /// Divide a / b. div(_, 0) is undefined.
    #[inline]
    pub fn div(&self, a: u8, b: u8) -> u8 {
        if a == 0 {
            return 0;
        }
        debug_assert!(b != 0, "GF(2^8) division by 0");
        // g^(log a - log b + 255) keeps the exponent non-negative and < 510.
        self.exp[self.log[a as usize] as usize + 255 - self.log[b as usize] as usize]
    }

    /// g^i, the i-th power of the generator (i taken mod 255).
    #[inline]
    pub fn exp_g(&self, i: usize) -> u8 {
        self.exp[i % 255]
    }
}

/// Addition/subtraction in GF(2^8) are both XOR.
#[inline]
pub fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_full_period() {
        // Every nonzero element appears exactly once as a power of g.
        let f = Gf256::new();
        let mut seen = [false; 256];
        for i in 0..255 {
            let v = f.exp_g(i);
            assert_ne!(v, 0);
            assert!(!seen[v as usize], "g not a generator: repeat at i={i}");
            seen[v as usize] = true;
        }
        // 1..=255 all seen.
        for (x, s) in seen.iter().enumerate().skip(1) {
            assert!(*s, "element {x} never produced");
        }
    }

    #[test]
    fn mul_zero_and_identity() {
        let f = Gf256::new();
        for a in 0u16..=255 {
            let a = a as u8;
            assert_eq!(f.mul(a, 0), 0);
            assert_eq!(f.mul(0, a), 0);
            assert_eq!(f.mul(a, 1), a);
            assert_eq!(f.mul(1, a), a);
        }
    }

    #[test]
    fn mul_commutative_and_inverse() {
        let f = Gf256::new();
        for a in 1u16..=255 {
            let a = a as u8;
            // inverse
            assert_eq!(f.mul(a, f.inv(a)), 1, "inv failed for {a}");
            for b in 1u16..=255 {
                let b = b as u8;
                assert_eq!(f.mul(a, b), f.mul(b, a));
                // division is the inverse of multiplication
                assert_eq!(f.div(f.mul(a, b), b), a);
            }
        }
    }

    #[test]
    fn distributive() {
        // a*(b+c) = a*b + a*c ; brute over a small but representative sweep.
        let f = Gf256::new();
        for a in 0u16..=255u16 {
            for b in (0u16..=255).step_by(17) {
                for c in (0u16..=255).step_by(19) {
                    let (a, b, c) = (a as u8, b as u8, c as u8);
                    let lhs = f.mul(a, add(b, c));
                    let rhs = add(f.mul(a, b), f.mul(a, c));
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }
}

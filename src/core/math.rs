use std::simd::{StdFloat, prelude::*};

const C1: f32 = 12102203.0;
const C2: f32 = 1064986823.0;
const CLAMP_MIN: f32 = -87.0;
const CLAMP_MAX: f32 = 88.0;

const C1_SIMD: f32x16 = f32x16::splat(C1);
const C2_SIMD: f32x16 = f32x16::splat(C2);
const CLAMP_MIN_SIMD: f32x16 = f32x16::splat(CLAMP_MIN);
const CLAMP_MAX_SIMD: f32x16 = f32x16::splat(CLAMP_MAX);

/// Fast approximate exp(x) for a SIMD lane, via Schraudolph approximation
/// [Schraudolph approximation](https://nic.schraudolph.org/pubs/Schraudolph99.pdf)
#[inline]
pub fn schraudolph_simd(x: f32x16) -> f32x16 {
    let clamped = x.simd_clamp(CLAMP_MIN_SIMD, CLAMP_MAX_SIMD);
    let t = (clamped * C1_SIMD + C2_SIMD).round();
    // SAFETY: t is clamped to a range that safely fits in u32.
    // This allows LLVM to compiles natively to `fcvtzu` or `vcvttps2udq`.
    let bits: u32x16 = unsafe { t.to_int_unchecked::<u32>() };

    f32x16::from_bits(bits.cast::<u32>())
}

/// Fast approximate exp(x) via Schraudolph approximation
/// [Schraudolph approximation](https://nic.schraudolph.org/pubs/Schraudolph99.pdf)
#[inline]
pub fn schraudolph(x: f32) -> f32 {
    let clamped = x.clamp(CLAMP_MIN, CLAMP_MAX);
    // SAFETY: t is clamped to a range that safely fits in u32.
    // This allows LLVM to compile natively to `fcvtzu` or `vcvttps2udq` when auto-vectorizing.
    let bits = unsafe { (clamped * C1 + C2).round().to_int_unchecked::<u32>() };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schraudolph_scalar_accuracy() {
        let test_values = [0.0, -0.5, -1.0, -2.5, -10.0, -50.0, -85.0];
        for &x in &test_values {
            let approx = schraudolph(x);
            let exact = x.exp();
            let diff = (approx - exact).abs();
            assert!(
                diff < 0.05,
                "x={}, approx={}, exact={}, diff={}",
                x,
                approx,
                exact,
                diff
            );
        }
    }

    #[test]
    fn test_schraudolph_simd_accuracy() {
        let test_values = [
            0.0, -0.5, -1.0, -2.5, -10.0, -50.0, -85.0, -87.0, 0.0, -0.5, -1.0, -2.5, -10.0, -50.0,
            -85.0, -87.0,
        ];
        let x_simd = f32x16::from_array(test_values);
        let approx_simd = schraudolph_simd(x_simd);

        for (i, &x) in test_values.iter().enumerate() {
            let approx = approx_simd[i];
            let exact = x.exp();
            let diff = (approx - exact).abs();
            assert!(
                diff < 0.05,
                "lane {}, x={}, approx={}, exact={}",
                i,
                x,
                approx,
                exact
            );
        }
    }
}

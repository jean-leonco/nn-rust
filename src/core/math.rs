const C1: f32 = 12102203.0;
const C2: f32 = 1064986823.0;
const CLAMP_MIN: f32 = -87.0;
const CLAMP_MAX: f32 = 88.0;

/// Applies the Schraudolph approximation to a fixed-size array in place.
///
/// The fixed length lets LLVM auto-vectorize the loop.
#[inline]
pub fn schraudolph_array<const N: usize>(values: &mut [f32; N], offset: f32) {
    for value in values {
        *value = schraudolph(*value - offset);
    }
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
    fn test_schraudolph_array_accuracy() {
        let mut approx = [0.0, -0.5, -1.0, -2.5, -10.0, -50.0, -85.0, -87.0];
        let exact = approx;
        schraudolph_array(&mut approx, 0.0);

        for (approx, x) in approx.into_iter().zip(exact) {
            assert!((approx - x.exp()).abs() < 0.05);
        }
    }
}

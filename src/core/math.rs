use std::simd::{StdFloat, prelude::*};

const C1: f32 = 12102203.0;
const C2: f32 = 1064986823.0;
const CLAMP_MIN: f32 = -87.0;
const CLAMP_MAX: f32 = 88.0;

const C1_SIMD: f32x8 = f32x8::splat(C1);
const C2_SIMD: f32x8 = f32x8::splat(C2);
const CLAMP_MIN_SIMD: f32x8 = f32x8::splat(CLAMP_MIN);
const CLAMP_MAX_SIMD: f32x8 = f32x8::splat(CLAMP_MAX);

/// Fast approximate exp(x) for a SIMD lane, via Schraudolph approximation
/// [Schraudolph approximation](https://nic.schraudolph.org/pubs/Schraudolph99.pdf)
#[inline]
pub fn schraudolph_simd(x: f32x8) -> f32x8 {
    let clamped = x.simd_clamp(CLAMP_MIN_SIMD, CLAMP_MAX_SIMD);
    let t = (clamped * C1_SIMD + C2_SIMD).round();
    let bits = t.cast::<i32>();
    f32x8::from_bits(bits.cast::<u32>())
}

/// Fast approximate exp(x) via Schraudolph approximation
/// [Schraudolph approximation](https://nic.schraudolph.org/pubs/Schraudolph99.pdf)
#[inline]
pub fn schraudolph(x: f32) -> f32 {
    let clamped = x.clamp(CLAMP_MIN, CLAMP_MAX);
    let bits = (clamped * C1 + C2).round() as u32;
    f32::from_bits(bits)
}

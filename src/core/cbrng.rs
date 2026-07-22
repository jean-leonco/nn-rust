use std::simd::prelude::*;

pub const ROUNDS: usize = 10;
const MULTIPLIER_0: u64 = 0xD2511F53;
const MULTIPLIER_1: u64 = 0xCD9E8D57;
const WEYL_0: u32 = 0x9E3779B9;
const WEYL_1: u32 = 0xBB67AE85;
/// The number of output values generated per bernoulli call.
pub const BERNOULLI_BATCH_SIZE: usize = 32;
/// The lane size used for generating random numbers.
pub const LANE_SIZE: usize = 8;
/// The lane iota used for generating random numbers.
pub const LANE_IOTA: u32x8 = u32x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);

pub type KeySchedule = [[u32x8; 2]; ROUNDS];

/// Empty key schedule used when no dropout is present.
pub const EMPTY_KEY_SCHEDULE: KeySchedule = [[u32x8::splat(0); 2]; ROUNDS];

/// Builds the key schedule for the philox algorithm.
///
/// # Arguments
///
/// * `seed` - An array of two 256-bit SIMD vectors.
///
/// # Returns
///
/// An array of 10x2 256-bit SIMD vectors representing the key schedule.
#[inline]
pub fn build_key_schedule(seed: [u32x8; 2]) -> KeySchedule {
    let mut table = [[u32x8::splat(0); 2]; ROUNDS];
    let mut key_0 = seed[0];
    let mut key_1 = seed[1];

    let wey_0 = u32x8::splat(WEYL_0);
    let wey_1 = u32x8::splat(WEYL_1);

    for r in 0..ROUNDS {
        table[r] = [key_0, key_1];
        key_0 += wey_0;
        key_1 += wey_1;
    }

    table
}

/// Calculates eight blocks of philox random number.
///
/// # Arguments
///
/// * `counters` - An array of four 256-bit SIMD vectors.
/// * `seed` - An array of two 256-bit SIMD vectors.
///
/// # Returns
///
/// An array of four 256-bit SIMD vectors containing the generated random number.
#[inline]
pub fn philox(counters: [u32x8; 4], key_schedule: &KeySchedule) -> [u32x8; 4] {
    let mut result = counters;

    let mult_0 = u64x8::splat(MULTIPLIER_0);
    let mult_1 = u64x8::splat(MULTIPLIER_1);

    for r in 0..ROUNDS {
        let prod_0: u64x8 = result[0].cast::<u64>() * mult_0;
        let prod_1: u64x8 = result[2].cast::<u64>() * mult_1;

        let hi0 = (prod_0 >> 32).cast::<u32>();
        let lo0 = prod_0.cast::<u32>();
        let hi1 = (prod_1 >> 32).cast::<u32>();
        let lo1 = prod_1.cast::<u32>();

        let h0 = hi1 ^ result[1] ^ key_schedule[r][0];
        let h1 = hi0 ^ result[3] ^ key_schedule[r][1];

        result = [h0, lo1, h1, lo0];
    }

    result
}

/// Generates a Bernoulli mask using a counter-based RNG.
///
/// # Arguments
///
/// * `counters` - An array of four 256-bit SIMD vectors.
/// * `seed` - An array of two 256-bit SIMD vectors.
/// * `p` - The survival probability.
///
/// # Returns
///
/// An array of four 64-bit SIMD vectors containing the generated mask.
/// 1s for success, 0s for failure.
#[inline]
pub fn bernoulli(counters: [u32x8; 4], key_schedule: &KeySchedule, p: f32) -> [u8x8; 4] {
    let threshold = u32x8::splat((p * u32::MAX as f32) as u32);
    let one = u8x8::splat(1);
    let zero = u8x8::splat(0);

    philox(counters, key_schedule).map(|val| val.simd_lt(threshold).select(one, zero))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::simd::u32x8;

    #[test]
    fn test_philox_deterministic() {
        let counters = [u32x8::splat(0); 4];
        let seed = [u32x8::splat(0); 2];
        let key_schedule = build_key_schedule(seed);

        let result1 = philox(counters, &key_schedule);
        let result2 = philox(counters, &key_schedule);

        assert_eq!(result1, result2);
        assert_ne!(result1[0], counters[0]);
    }

    #[test]
    fn test_philox_avalanche() {
        let seed = [u32x8::splat(0); 2];
        let key_schedule = build_key_schedule(seed);

        let counters1 = [u32x8::splat(0); 4];
        let mut counters2 = [u32x8::splat(0); 4];
        counters2[0][0] = 1;

        let result1 = philox(counters1, &key_schedule);
        let result2 = philox(counters2, &key_schedule);

        assert_ne!(result1[0][0], result2[0][0]);
        assert_ne!(result1[1][0], result2[1][0]);
        assert_ne!(result1[2][0], result2[2][0]);
        assert_ne!(result1[3][0], result2[3][0]);
    }

    #[test]
    fn test_bernoulli_bounds() {
        let counters = [u32x8::splat(0); 4];
        let seed = [u32x8::splat(0); 2];
        let key_schedule = build_key_schedule(seed);

        let ones = bernoulli(counters, &key_schedule, 1.0);
        let zeros = bernoulli(counters, &key_schedule, 0.0);

        assert_eq!(ones[0].to_array(), [1; 8]);
        assert_eq!(zeros[0].to_array(), [0; 8]);
    }
}

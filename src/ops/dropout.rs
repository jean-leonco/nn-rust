use thiserror::Error;

use crate::core::cbrng;
use core::ops::{Range, RangeTo};
use std::simd::prelude::*;

/// Metadata for the Dropout layer.
/// Randomly drops out (sets to zero) a fraction of the neurons.
#[derive(Debug, Clone, PartialEq)]
pub struct DropoutMeta {
    /// The survival probability (1 - dropout probability).
    pub p: f32,
    /// The inverse of the survival probability (1 / p).
    /// This ensures remaining neurons are scaled up to compensate for the dropped out ones.
    pub inv_p: f32,

    pub inv_p_simd: f32x8,
    /// The relative offsets where current layer activations are stored.
    /// Must be multiplied by the batch size to get the absolute offset.
    pub(crate) a_span: Range<usize>,
    /// The relative offsets where current layer mask data is stored.
    /// Must be multiplied by the batch size to get the absolute offset.
    pub(crate) m_span: Range<usize>,
}

/// Error type for Dropout layer.
#[derive(Debug, PartialEq, Error)]
pub enum DropoutError {
    InvalidProbability(f32),
}

impl std::fmt::Display for DropoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DropoutError::InvalidProbability(p) => {
                write!(f, "Probability must be in range 0.0..1.0, got {}", p)
            }
        }
    }
}

impl DropoutMeta {
    pub fn new(p: f32, a_span: Range<usize>, m_span: Range<usize>) -> Result<Self, DropoutError> {
        if !(0.0..1.0).contains(&p) {
            return Err(DropoutError::InvalidProbability(p));
        }

        let survival_prob = 1.0 - p;
        Ok(Self {
            p: survival_prob,
            inv_p: 1.0 / (survival_prob),
            inv_p_simd: f32x8::splat(1.0 / (survival_prob)),
            a_span,
            m_span,
        })
    }

    /// Returns the absolute offsets where current layer activations are stored.
    pub fn activation_offsets(&self, batch_size: usize) -> Range<usize> {
        self.a_span.start * batch_size..self.a_span.end * batch_size
    }

    /// Returns the absolute offsets where current layer mask data is stored.
    pub fn mask_offsets(&self, batch_size: usize) -> Range<usize> {
        self.m_span.start * batch_size..self.m_span.end * batch_size
    }

    /// Returns the absolute offsets where current layer gradients are stored.
    pub fn gradient_offsets(&self, batch_size: usize) -> RangeTo<usize> {
        let dimension = self.a_span.end - self.a_span.start;
        ..dimension * batch_size
    }
}

/// Applies dropout to one 8-wide lane. Handles both full-width (len == 8) and ragged tail (len < 8) slices.
#[inline]
fn apply_dropout_lane(activation_lane: &mut [f32], mask_lane: &mut [u8], mask: u8x8, inv_p: f32x8) {
    assert_eq!(activation_lane.len(), mask_lane.len());

    let survival_mask = mask.cast::<u32>().cast::<f32>();

    let scaled = if activation_lane.len() == cbrng::LANE_SIZE {
        f32x8::from_slice(activation_lane) * survival_mask * inv_p
    } else {
        let mut scratch = [0.0f32; cbrng::LANE_SIZE];
        scratch[..activation_lane.len()].copy_from_slice(activation_lane);
        f32x8::from_slice(&scratch) * survival_mask * inv_p
    };

    activation_lane.copy_from_slice(&scaled.to_array()[..activation_lane.len()]);
    mask_lane.copy_from_slice(&mask.to_array()[..mask_lane.len()]);
}

/// Applies the dropout to given activations and mask in-place.
///
/// # Arguments
///
/// * `meta` - The dropout metadata.
/// * `activations` - The slice of activations to apply dropout to.
/// * `mask` - The slice of mask data to use for dropout.
pub fn forward(
    meta: &DropoutMeta,
    activations: &mut [f32],
    masks: &mut [u8],
    step: usize,
    key_schedule: &cbrng::KeySchedule,
) {
    let mut counters = [
        u32x8::splat(0),
        u32x8::splat(meta.a_span.start as u32),
        u32x8::splat(step as u32),
        u32x8::splat(0 as u32),
    ];

    let mut block_idx = 0u32;
    let mut mask_blocks = masks.chunks_exact_mut(cbrng::BERNOULLI_BATCH_SIZE);
    let mut activation_blocks = activations.chunks_exact_mut(cbrng::BERNOULLI_BATCH_SIZE);

    for (mask_block, activation_block) in (&mut mask_blocks).zip(&mut activation_blocks) {
        counters[0] = u32x8::splat(block_idx * 8) + cbrng::LANE_IOTA;
        let bernoulli_mask = cbrng::bernoulli(counters, key_schedule, meta.p);

        for ((activation_lane, mask_lane), mask) in activation_block
            .chunks_exact_mut(cbrng::LANE_SIZE)
            .zip(mask_block.chunks_exact_mut(cbrng::LANE_SIZE))
            .zip(bernoulli_mask)
        {
            apply_dropout_lane(activation_lane, mask_lane, mask, meta.inv_p_simd);
        }
        block_idx += 1;
    }

    let remaining_masks = mask_blocks.into_remainder();
    let remaining_activations = activation_blocks.into_remainder();

    if !remaining_activations.is_empty() {
        counters[0] = u32x8::splat(block_idx * 8) + cbrng::LANE_IOTA;
        let bernoulli_mask = cbrng::bernoulli(counters, key_schedule, meta.p);

        let mut activation_lanes = remaining_activations.chunks_mut(8);
        let mut mask_lanes = remaining_masks.chunks_mut(8);

        for mask in bernoulli_mask {
            let (Some(activation_lane), Some(mask_lane)) =
                (activation_lanes.next(), mask_lanes.next())
            else {
                break;
            };
            apply_dropout_lane(activation_lane, mask_lane, mask, meta.inv_p_simd);
        }
    }
}

/// Applies the dropout backward pass to the given gradients and mask in-place.
/// # Arguments
///
/// * `meta` - The dropout metadata.
/// * `dz` - The outgoing gradient with respect to the input of this layer.
/// * `da` - The incoming gradient with respect to the output of this layer.
/// * `masks` - The slice containing this layer masks.
///   It must contain the original mask used during the forward pass, as the gradients must be scaled using the exact same random mask.
pub fn backward(meta: &DropoutMeta, dz: &mut [f32], da: &[f32], masks: &[u8]) {
    for ((dz, da), mask) in dz.iter_mut().zip(da.iter()).zip(masks.iter()) {
        *dz = da * *mask as f32 * meta.inv_p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_errors() {
        assert!(DropoutMeta::new(0.2, 1..4, 10..13).is_ok());
        assert!(DropoutMeta::new(1.5, 1..4, 10..13).is_err());
    }

    #[test]
    fn test_offsets() {
        let meta = DropoutMeta::new(0.5, 2..5, 4..8).unwrap();

        assert_eq!(meta.activation_offsets(1), 2..5);
        assert_eq!(meta.mask_offsets(1), 4..8);
        assert_eq!(meta.activation_offsets(3), 6..15);
        assert_eq!(meta.mask_offsets(3), 12..24);
    }

    #[test]
    fn test_forward() {
        let mut meta = DropoutMeta::new(0.5, 0..4, 0..4).unwrap();
        let mut activations = vec![1.0, 2.0, 3.0, 4.0];
        let mut mask = vec![0; 4];
        let step = 0;
        let seed = [u32x8::splat(0); 2];
        let key_schedule = cbrng::build_key_schedule(seed);

        forward(&mut meta, &mut activations, &mut mask, step, &key_schedule);

        assert_eq!(mask, vec![0, 0, 0, 1]);
        assert_eq!(activations, vec![0.0, 0.0, 0.0, 8.0]);
    }

    #[test]
    fn test_backward() {
        let meta = DropoutMeta::new(0.5, 0..4, 0..4).unwrap();
        let mut dz = vec![0.0; 4];
        let da = vec![1.5, 2.5, 3.5, 4.5];
        let mask = vec![0, 0, 0, 1];

        backward(&meta, &mut dz, &da, &mask);

        assert_eq!(dz, vec![0.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn test_forward_and_backward() {
        let mut meta = DropoutMeta::new(0.5, 0..4, 0..4).unwrap();

        let mut activations = vec![1.0, 2.0, 3.0, 4.0];
        let mut mask = vec![0; 4];
        let step = 0;
        let seed = [u32x8::splat(0); 2];
        let key_schedule = cbrng::build_key_schedule(seed);

        let mut dz = vec![0.0; 4];
        let da = vec![1.5, 2.5, 3.5, 4.5];

        forward(&mut meta, &mut activations, &mut mask, step, &key_schedule);
        backward(&meta, &mut dz, &da, &mask);

        assert_eq!(mask, vec![0, 0, 0, 1]);
        assert_eq!(activations, vec![0.0, 0.0, 0.0, 8.0]);
        assert_eq!(dz, vec![0.0, 0.0, 0.0, 9.0]);
    }
}

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
    assert_eq!(activations.len(), masks.len());

    let mut counters = [
        u32x8::splat(0),
        u32x8::splat(meta.a_span.start as u32),
        u32x8::splat(step as u32),
        u32x8::splat(0_u32),
    ];

    let (activation_chunks, remaining_activations) = activations.as_chunks_mut::<32>();
    let (mask_chunks, remaining_masks) = masks.as_chunks_mut::<32>();

    let mut block_idx = 0;
    for (activation_chunk, mask_chunk) in activation_chunks.iter_mut().zip(mask_chunks) {
        counters[0] = u32x8::splat(block_idx * 8) + cbrng::LANE_IOTA;
        let bernoulli_masks = cbrng::bernoulli(counters, key_schedule, meta.p);

        for i in 0..32 {
            let mask = bernoulli_masks[i];
            mask_chunk[i] = mask;
            activation_chunk[i] *= mask as u32 as f32 * meta.inv_p;
        }

        block_idx += 1;
    }

    if !remaining_activations.is_empty() {
        counters[0] = u32x8::splat(block_idx * 8) + cbrng::LANE_IOTA;
        let bernoulli_masks = cbrng::bernoulli(counters, key_schedule, meta.p).to_array();
        let bernoulli_masks_slice = &bernoulli_masks[..remaining_activations.len()];

        for ((activation, mask), remaining_mask) in remaining_activations
            .iter_mut()
            .zip(remaining_masks)
            .zip(bernoulli_masks_slice)
        {
            *mask = *remaining_mask;
            *activation *= *mask as u32 as f32 * meta.inv_p;
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

        assert_eq!(mask, vec![1, 0, 1, 0]);
        assert_eq!(activations, vec![2.0, 0.0, 6.0, 0.0]);
    }

    #[test]
    fn test_backward() {
        let meta = DropoutMeta::new(0.5, 0..4, 0..4).unwrap();
        let mut dz = vec![0.0; 4];
        let da = vec![1.5, 2.5, 3.5, 4.5];
        let mask = vec![1, 0, 1, 0];

        backward(&meta, &mut dz, &da, &mask);

        assert_eq!(dz, vec![3.0, 0.0, 7.0, 0.0]);
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

        assert_eq!(mask, vec![1, 0, 1, 0]);
        assert_eq!(activations, vec![2.0, 0.0, 6.0, 0.0]);
        assert_eq!(dz, vec![3.0, 0.0, 7.0, 0.0]);
    }
}

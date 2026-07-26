use thiserror::Error;

use crate::core::{cbrng, serialization};
use core::ops::{Range, RangeTo};
use std::simd::prelude::*;

/// Dropout layer metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DropoutMeta {
    /// Survival probability (`1 - dropout_rate`).
    pub survival_rate: f32,
    /// Inverse survival probability (`1 / survival_rate`).
    pub inv_survival_rate: f32,
    /// Relative activation range.
    pub(crate) relative_activation_range: Range<usize>,
    /// Relative mask range.
    pub(crate) relative_mask_range: Range<usize>,
}

/// Dropout construction errors.
#[derive(Debug, PartialEq, Error)]
pub enum DropoutError {
    InvalidProbability(f32),
}

impl std::fmt::Display for DropoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DropoutError::InvalidProbability(p) => {
                write!(f, "Probability must be in range 0.0..1.0, got {p}")
            }
        }
    }
}

impl DropoutMeta {
    /// Creates dropout metadata. `dropout_rate` must be in `0.0..1.0`.
    ///
    /// # Errors
    ///
    /// Returns [`DropoutError::InvalidProbability`] if `dropout_rate` is outside `0.0..1.0`.
    pub fn new(
        dropout_rate: f32,
        relative_activation_range: Range<usize>,
        relative_mask_range: Range<usize>,
    ) -> Result<Self, DropoutError> {
        if !(0.0..1.0).contains(&dropout_rate) {
            return Err(DropoutError::InvalidProbability(dropout_rate));
        }

        let survival_rate = 1.0 - dropout_rate;
        Ok(Self {
            survival_rate,
            inv_survival_rate: 1.0 / survival_rate,
            relative_activation_range,
            relative_mask_range,
        })
    }

    /// Activation range.
    pub fn activation_range(&self, batch_size: usize) -> Range<usize> {
        self.relative_activation_range.start * batch_size
            ..self.relative_activation_range.end * batch_size
    }

    /// Mask range.
    pub fn mask_range(&self, batch_size: usize) -> Range<usize> {
        self.relative_mask_range.start * batch_size..self.relative_mask_range.end * batch_size
    }

    /// Gradient range.
    pub fn gradient_range(&self, batch_size: usize) -> RangeTo<usize> {
        let dim = self.relative_activation_range.end - self.relative_activation_range.start;
        ..dim * batch_size
    }
}

/// Errors during dropout layer serialization.
#[derive(Error, Debug)]
pub enum DropoutEncodingError {
    #[error("IO error: {0}")]
    Io(#[from] serialization::SerializationError),
    #[error("Invalid dropout: {0}")]
    InvalidDropout(#[from] DropoutError),
}

impl serialization::Encodable for DropoutMeta {
    type Error = DropoutEncodingError;

    fn encoded_len(&self) -> usize {
        // 1 f32 + 2 ranges
        serialization::F32_WIRE + 2 * serialization::RANGE_WIRE
    }

    fn encode(&self, writer: &mut impl std::io::Write) -> Result<(), Self::Error> {
        serialization::write_f32(writer, self.survival_rate)?;
        serialization::write_range(writer, self.relative_activation_range.clone())?;
        serialization::write_range(writer, self.relative_mask_range.clone())?;
        Ok(())
    }

    fn decode(reader: &mut impl std::io::Read) -> Result<Self, Self::Error> {
        let survival_rate = serialization::read_f32(reader)?;
        let activation_range = serialization::read_range(reader)?;
        let mask_range = serialization::read_range(reader)?;
        let dropout_rate = 1.0 - survival_rate;
        Ok(Self::new(dropout_rate, activation_range, mask_range)?)
    }
}

/// Applies dropout in-place.
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
        u32x8::splat(meta.relative_activation_range.start as u32),
        u32x8::splat(step as u32),
        u32x8::splat(0_u32),
    ];

    let (activation_chunks, remaining_activations) = activations.as_chunks_mut::<32>();
    let (mask_chunks, remaining_masks) = masks.as_chunks_mut::<32>();

    let mut block_idx = 0;
    for (activation_chunk, mask_chunk) in activation_chunks.iter_mut().zip(mask_chunks) {
        counters[0] = u32x8::splat(block_idx * 8) + cbrng::LANE_IOTA;
        let bernoulli_masks = cbrng::bernoulli(counters, key_schedule, meta.survival_rate);

        for i in 0..32 {
            let mask = bernoulli_masks[i];
            mask_chunk[i] = mask;
            activation_chunk[i] *= u32::from(mask) as f32 * meta.inv_survival_rate;
        }

        block_idx += 1;
    }

    if !remaining_activations.is_empty() {
        counters[0] = u32x8::splat(block_idx * 8) + cbrng::LANE_IOTA;
        let bernoulli_masks =
            cbrng::bernoulli(counters, key_schedule, meta.survival_rate).to_array();
        let bernoulli_masks_slice = &bernoulli_masks[..remaining_activations.len()];

        for ((activation, mask), remaining_mask) in remaining_activations
            .iter_mut()
            .zip(remaining_masks)
            .zip(bernoulli_masks_slice)
        {
            *mask = *remaining_mask;
            *activation *= u32::from(*mask) as f32 * meta.inv_survival_rate;
        }
    }
}

/// Applies the dropout derivative in-place.
pub fn backward(meta: &DropoutMeta, dz: &mut [f32], da: &[f32], masks: &[u8]) {
    for ((dz, da), mask) in dz.iter_mut().zip(da.iter()).zip(masks.iter()) {
        *dz = da * f32::from(*mask) * meta.inv_survival_rate;
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

        assert_eq!(meta.activation_range(1), 2..5);
        assert_eq!(meta.mask_range(1), 4..8);
        assert_eq!(meta.activation_range(3), 6..15);
        assert_eq!(meta.mask_range(3), 12..24);
    }

    #[test]
    fn test_forward() {
        let meta = DropoutMeta::new(0.5, 0..4, 0..4).unwrap();
        let mut activations = vec![1.0, 2.0, 3.0, 4.0];
        let mut mask = vec![0; 4];
        let step = 0;
        let seed = [u32x8::splat(0); 2];
        let key_schedule = cbrng::build_key_schedule(seed);

        forward(&meta, &mut activations, &mut mask, step, &key_schedule);

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
        let meta = DropoutMeta::new(0.5, 0..4, 0..4).unwrap();

        let mut activations = vec![1.0, 2.0, 3.0, 4.0];
        let mut mask = vec![0; 4];
        let step = 0;
        let seed = [u32x8::splat(0); 2];
        let key_schedule = cbrng::build_key_schedule(seed);

        let mut dz = vec![0.0; 4];
        let da = vec![1.5, 2.5, 3.5, 4.5];

        forward(&meta, &mut activations, &mut mask, step, &key_schedule);
        backward(&meta, &mut dz, &da, &mask);

        assert_eq!(mask, vec![1, 0, 1, 0]);
        assert_eq!(activations, vec![2.0, 0.0, 6.0, 0.0]);
        assert_eq!(dz, vec![3.0, 0.0, 7.0, 0.0]);
    }
}

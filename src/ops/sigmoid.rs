use core::ops::{Range, RangeTo};
use std::simd::prelude::*;

use crate::core::math;

/// Metadata for the Sigmoid activation function.
#[derive(Debug)]
pub struct SigmoidMeta {
    /// The relative offsets where current layer activations are stored.
    /// Must be multiplied by the batch size to get the absolute offset.
    pub(crate) a_start: usize,
    pub(crate) a_end: usize,
}

impl SigmoidMeta {
    pub fn new(a_start: usize, a_end: usize) -> Self {
        Self { a_start, a_end }
    }

    /// Returns the absolute offsets where current layer activations are stored.
    pub fn activation_offsets(&self, batch_size: usize) -> Range<usize> {
        self.a_start * batch_size..self.a_end * batch_size
    }

    /// Returns the absolute offsets where current layer gradients are stored.
    pub fn gradient_offsets(&self, batch_size: usize) -> RangeTo<usize> {
        let dimension = self.a_end - self.a_start;
        ..dimension * batch_size
    }
}

/// Applies the Sigmoid function to the given activations in-place.
///
/// f(x) = 1 / (1 + exp(-x))
///
/// # Arguments
///
/// * `activations` - The slice to apply the Sigmoid function to.
pub fn forward(activations: &mut [f32]) {
    let one = f32x8::splat(1.0);

    let mut chunks = activations.chunks_exact_mut(8);
    for chunk in &mut chunks {
        let value = f32x8::from_slice(chunk);
        let z = one / (one + math::schraudolph_simd(-value));
        chunk.copy_from_slice(&z.to_array());
    }

    let remaining_activations = chunks.into_remainder();
    if !remaining_activations.is_empty() {
        let mut buf = [0.0; 8];
        buf[..remaining_activations.len()].copy_from_slice(remaining_activations);

        let value = f32x8::from_slice(&buf);
        let z = one / (one + math::schraudolph_simd(-value));
        remaining_activations.copy_from_slice(&z.to_array()[..remaining_activations.len()]);
    }
}

/// Applies the derivative of the Sigmoid function to the given gradients in-place.
///
/// f'(x) = sigmoid(x) * (1 - sigmoid(x))
///
/// # Arguments
///
/// * `dz` - The outgoing gradient with respect to the input of this layer.
/// * `da` - The incoming gradient with respect to the output of this layer.
/// * `activations` - The slice containing this layer activations.
///   It must contain the post-activation values (A) instead of inputs (Z), as derivative is computed directly from the outputs.
pub fn backward(dz: &mut [f32], da: &[f32], activations: &[f32]) {
    assert_eq!(da.len(), dz.len());
    assert_eq!(da.len(), activations.len());

    for ((da, dz), a) in da.iter().zip(dz.iter_mut()).zip(activations.iter()) {
        let derivative = a * (1.0 - a);
        *dz = da * derivative;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offsets() {
        let meta = SigmoidMeta::new(2, 5);

        assert_eq!(meta.activation_offsets(1), 2..5);
        assert_eq!(meta.activation_offsets(3), 6..15);
    }

    #[test]
    fn test_forward() {
        let mut activations = vec![0.0, f32::INFINITY];

        forward(&mut activations);

        assert!((activations[0] - 0.5).abs() < 0.05);
        assert!((activations[1] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_backward() {
        let mut dz = vec![0.0; 2];
        let da = vec![2.0, 4.0];
        let activations = vec![0.5, 1.0];

        backward(&mut dz, &da, &activations);

        assert!((dz[0] - 0.5).abs() < 0.05);
        assert!((dz[1] - 0.0).abs() < 0.05);
    }

    #[test]
    fn test_forward_and_backward() {
        let mut activations = vec![0.0, f32::INFINITY];
        let mut dz = vec![0.0; 2];
        let da = vec![2.0, 4.0];

        forward(&mut activations);
        backward(&mut dz, &da, &activations);

        assert!((activations[0] - 0.5).abs() < 0.05);
        assert!((activations[1] - 1.0).abs() < 0.05);
        assert!((dz[0] - 0.5).abs() < 0.05);
        assert!((dz[1] - 0.0).abs() < 0.05);
    }
}

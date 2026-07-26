use core::ops::Range;
use std::simd::prelude::*;

use crate::core::{math, serialization};

/// Softmax layer metadata.
#[derive(Debug, Clone)]
pub struct SoftmaxMeta {
    /// Relative activation range.
    pub(crate) relative_activation_range: Range<usize>,
    /// Output classes per row.
    pub(crate) output_dim: usize,
}

impl SoftmaxMeta {
    /// Creates metadata for a softmax over `output_dim` classes.
    pub fn new(relative_activation_range: Range<usize>, output_dim: usize) -> Self {
        Self {
            relative_activation_range,
            output_dim,
        }
    }

    /// Activation range for the current layer at `batch_size`.
    pub fn activation_range(&self, batch_size: usize) -> Range<usize> {
        self.relative_activation_range.start * batch_size
            ..self.relative_activation_range.end * batch_size
    }
}

impl serialization::Encodable for SoftmaxMeta {
    type Error = super::serialization::SerializationError;

    fn encoded_len(&self) -> usize {
        // 1 range + 1 u32
        serialization::RANGE_WIRE + serialization::U32_WIRE
    }

    fn encode(&self, writer: &mut impl std::io::Write) -> Result<(), Self::Error> {
        serialization::write_range(writer, self.relative_activation_range.clone())?;
        serialization::write_u32(writer, self.output_dim as u32)
    }

    fn decode(reader: &mut impl std::io::Read) -> Result<Self, Self::Error> {
        let range = serialization::read_range(reader)?;
        let output_dim = serialization::read_u32(reader)? as usize;
        Ok(Self::new(range, output_dim))
    }
}

/// Approximate softmax row-wise in-place.
///
///   `p_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`
///
/// Uses Schraudolph approximate exponential.
///
/// # Panics
///
/// Panics if `output_dim` is zero or `activations.len() % output_dim != 0`.
pub fn forward(meta: &SoftmaxMeta, activations: &mut [f32]) {
    assert!(
        meta.output_dim > 0,
        "softmax output dimension must be non-zero"
    );
    assert_eq!(
        activations.len() % meta.output_dim,
        0,
        "softmax input must contain whole rows"
    );
    for row in &mut activations.chunks_mut(meta.output_dim) {
        let max = row.iter().fold(f32::NEG_INFINITY, |acc, &val| acc.max(val));
        let max_simd = f32x16::splat(max);

        // LLVM refuses to vectorize this loop, since it can't reorder floating-point sum.
        let mut total: f32 = 0.0;
        let mut sum = f32x16::splat(0.0);
        let (chunks, remainder) = row.as_chunks_mut::<16>();
        for chunk in chunks {
            let value = f32x16::from_slice(chunk);
            let e = math::schraudolph_simd(value - max_simd);
            *chunk = e.to_array();
            sum += e;
        }
        for val in remainder {
            *val = math::schraudolph(*val - max);
            total += *val;
        }
        total += sum.reduce_sum();

        for val in row.iter_mut() {
            *val *= 1.0 / total;
        }
    }
}

/// Surrogate gradient for fused softmax + cross-entropy: `dZ = P - Y`.
///
/// Uses the exact softmax derivative despite [`forward`] using approximate
/// exponential. This is a standard surrogate that avoids the full Jacobian.
///
/// # Panics
///
/// Panics if `dz`, `predicted`, and `targets` have unequal lengths.
pub fn backward(dz: &mut [f32], predicted: &[f32], targets: &[f32]) {
    assert_eq!(
        dz.len(),
        predicted.len(),
        "gradient and prediction lengths differ"
    );
    assert_eq!(
        predicted.len(),
        targets.len(),
        "prediction and target lengths differ"
    );
    for ((dz, p), y) in dz.iter_mut().zip(predicted.iter()).zip(targets.iter()) {
        *dz = p - y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offsets() {
        let meta = SoftmaxMeta::new(2..5, 3);

        assert_eq!(meta.activation_range(1), 2..5);
        assert_eq!(meta.activation_range(4), 8..20);
    }

    #[test]
    fn test_forward() {
        let meta = SoftmaxMeta::new(0..6, 3);
        let mut activations = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];

        forward(&meta, &mut activations);

        let sum1 = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp();
        let p0 = 1.0_f32.exp() / sum1;
        let p1 = 2.0_f32.exp() / sum1;
        let p2 = 3.0_f32.exp() / sum1;

        assert!((activations[0] - p0).abs() < 0.05);
        assert!((activations[1] - p1).abs() < 0.05);
        assert!((activations[2] - p2).abs() < 0.05);
        assert!((activations[3] - 1.0 / 3.0).abs() < 0.05);
        assert!((activations[4] - 1.0 / 3.0).abs() < 0.05);
        assert!((activations[5] - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_backward() {
        let predicted = vec![0.1, 0.2, 0.7, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let mut dz = vec![0.0; 6];
        let targets = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        backward(&mut dz, &predicted, &targets);

        assert!((dz[0] - 0.1).abs() < 0.05);
        assert!((dz[1] - 0.2).abs() < 0.05);
        assert!((dz[2] - (0.7 - 1.0)).abs() < 0.05);
        assert!((dz[3] - 1.0 / 3.0).abs() < 0.05);
        assert!((dz[4] - (1.0 / 3.0 - 1.0)).abs() < 0.05);
        assert!((dz[5] - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_forward_and_backward() {
        let meta = SoftmaxMeta::new(0..6, 3);

        let mut activations = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let mut dz = vec![0.0; 6];
        let targets = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        forward(&meta, &mut activations);
        backward(&mut dz, &activations, &targets);

        let sum1 = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp();
        let p0 = 1.0_f32.exp() / sum1;
        let p1 = 2.0_f32.exp() / sum1;
        let p2 = 3.0_f32.exp() / sum1;

        assert!((activations[0] - p0).abs() < 0.05);
        assert!((activations[1] - p1).abs() < 0.05);
        assert!((activations[2] - p2).abs() < 0.05);
        assert!((activations[3] - 1.0 / 3.0).abs() < 0.05);
        assert!((activations[4] - 1.0 / 3.0).abs() < 0.05);
        assert!((activations[5] - 1.0 / 3.0).abs() < 0.05);

        assert!((dz[0] - p0).abs() < 0.05);
        assert!((dz[1] - p1).abs() < 0.05);
        assert!((dz[2] - (p2 - 1.0)).abs() < 0.05);
        assert!((dz[3] - 1.0 / 3.0).abs() < 0.05);
        assert!((dz[4] - (1.0 / 3.0 - 1.0)).abs() < 0.05);
        assert!((dz[5] - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_surrogate_gradient_via_finite_differences() {
        // Verify backward is a reasonable approximation of the numerical gradient
        // of the approximate forward function. The two won't match exactly because
        // forward uses Schraudolph exp and backward uses the exact p - y formula.
        let n_classes = 3;
        let inputs = vec![1.0, 2.0, 3.0];
        let targets = vec![0.0, 0.0, 1.0];
        let eps: f32 = 1e-3;

        let mut numerical_grad = vec![0.0f32; n_classes];
        for i in 0..n_classes {
            let mut inputs_plus = inputs.clone();
            inputs_plus[i] += eps;
            let mut inputs_minus = inputs.clone();
            inputs_minus[i] -= eps;

            let loss_plus = cross_entropy_from_approx(&inputs_plus, &targets);
            let loss_minus = cross_entropy_from_approx(&inputs_minus, &targets);
            numerical_grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        let meta = SoftmaxMeta::new(0..n_classes, n_classes);
        let mut activations = inputs.clone();
        forward(&meta, &mut activations);

        let mut dz = vec![0.0; n_classes];
        backward(&mut dz, &activations, &targets);

        // The surrogate gradient direction should match the numerical gradient.
        // Allow larger tolerance since forward uses approximate exp.
        let dot: f32 = numerical_grad
            .iter()
            .zip(dz.iter())
            .map(|(a, b)| a * b)
            .sum();
        let grad_norm: f32 = numerical_grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        let surr_norm: f32 = dz.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (grad_norm * surr_norm);
        assert!(
            cos_sim > 0.99,
            "surrogate gradient direction diverged: cos_sim={}",
            cos_sim
        );
    }

    /// Softmax + cross-entropy using the approximate Schraudolph exp (same as forward).
    fn cross_entropy_from_approx(inputs: &[f32], targets: &[f32]) -> f32 {
        let max = inputs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = inputs.iter().map(|&x| math::schraudolph(x - max)).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        -targets
            .iter()
            .zip(probs.iter())
            .map(|(t, p)| t * (p + 1e-8).ln())
            .sum::<f32>()
    }
}

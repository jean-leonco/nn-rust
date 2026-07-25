use core::ops::Range;
use std::simd::prelude::*;

use crate::core::math;

/// Metadata for the Softmax Cross-entropy layer.
#[derive(Debug, Clone)]
pub struct SoftmaxMeta {
    /// The relative offsets where current layer activations are stored.
    /// Must be multiplied by the batch size to get the absolute offset.
    pub(crate) a_start: usize,
    pub(crate) a_end: usize,
    /// The number of output classes.
    pub(crate) output_size: usize,
}

impl SoftmaxMeta {
    pub fn new(a_start: usize, a_end: usize, output_size: usize) -> Self {
        Self {
            a_start,
            a_end,
            output_size,
        }
    }

    /// Returns the absolute offsets where current layer activations are stored.
    pub fn activation_offsets(&self, batch_size: usize) -> Range<usize> {
        Range {
            start: self.a_start * batch_size,
            end: self.a_end * batch_size,
        }
    }
}

/// Applies the Softmax loss function to the given activations in-place.
///
/// For each element x_i in a row x of length N, the stable softmax value p_i is computed as:
///
///   p_i = exp(x_i - max(x)) / sum_{j=1..N}(exp(x_j - max(x)))
///
/// Shifting the inputs by the maximum value in the row prevents numerical overflow
/// during the exponential step.
///
/// # Arguments
///
/// * `activations` - The slice to apply the Softmax loss function to.
pub fn forward(meta: &SoftmaxMeta, activations: &mut [f32]) {
    for row in &mut activations.chunks_mut(meta.output_size) {
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

/// Applies the derivative of the Softmax Cross-entropy loss function to the given gradients in-place.
///
/// dz_i = p_i - y_i
///
/// # Arguments
///
/// * `dz` - The outgoing gradient with respect to the input of this layer.
/// * `predicted` - The predicted values.
/// * `y` - The target values.
pub fn backward(dz: &mut [f32], predicted: &[f32], y: &[f32]) {
    for ((dz, p), y) in dz.iter_mut().zip(predicted.iter()).zip(y.iter()) {
        *dz = p - y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offsets() {
        let meta = SoftmaxMeta::new(2, 5, 3);

        assert_eq!(meta.activation_offsets(1), 2..5);
        assert_eq!(meta.activation_offsets(4), 8..20);
    }

    #[test]
    fn test_forward() {
        let meta = SoftmaxMeta::new(0, 6, 3);
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
        let y = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        backward(&mut dz, &predicted, &y);

        assert!((dz[0] - 0.1).abs() < 0.05);
        assert!((dz[1] - 0.2).abs() < 0.05);
        assert!((dz[2] - (0.7 - 1.0)).abs() < 0.05);
        assert!((dz[3] - 1.0 / 3.0).abs() < 0.05);
        assert!((dz[4] - (1.0 / 3.0 - 1.0)).abs() < 0.05);
        assert!((dz[5] - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_forward_and_backward() {
        let meta = SoftmaxMeta::new(0, 6, 3);

        let mut activations = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let mut dz = vec![0.0; 6];
        let y = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        forward(&meta, &mut activations);
        backward(&mut dz, &activations, &y);

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
}

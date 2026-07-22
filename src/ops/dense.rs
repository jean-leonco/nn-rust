use core::ops::{Range, RangeFrom, RangeTo};

use crate::ops::{Initialization, gemm};

/// Metadata for the Dense layer.
#[derive(Debug, Clone)]
pub struct DenseMeta {
    /// The dimension of this layer input.
    /// Same as previous layer number of neurons.
    pub input_dim: usize,
    /// The dimension of this layer output.
    /// Same as the number of neurons for the next layer.
    pub output_dim: usize,
    pub(crate) a_start: usize,
    /// The relative start offset where current layer input activations are stored.
    pub(crate) i_start: usize,
    /// The relative offsets where current layer weights are stored.
    pub weight_offsets: Range<usize>,
    /// The relative offsets where current layer biases are stored.
    pub bias_offsets: Range<usize>,

    /// The initialization method to use for this layer.
    pub initialization: Initialization,
}

impl DenseMeta {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        a_start: usize,
        i_start: usize,
        weight_offsets: Range<usize>,
        bias_offsets: Range<usize>,
        initialization: Initialization,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            a_start,
            i_start,
            weight_offsets,
            bias_offsets,
            initialization,
        }
    }

    /// Returns the absolute offset where activations must be split to get the input and output activations.
    pub fn activations_split_offset(&self, batch_size: usize) -> usize {
        self.a_start * batch_size
    }

    /// Returns the absolute offsets where current layer input activations are stored.
    pub fn input_offsets(&self, batch_size: usize) -> RangeFrom<usize> {
        RangeFrom {
            start: self.i_start * batch_size,
        }
    }

    /// Returns the absolute offsets where current layer output activations are stored.
    pub fn output_offsets(&self, batch_size: usize) -> RangeTo<usize> {
        RangeTo {
            end: self.output_dim * batch_size,
        }
    }

    /// Returns the absolute offsets where current layer dz are stored.
    pub fn dz_offsets(&self, batch_size: usize) -> RangeTo<usize> {
        RangeTo {
            end: self.output_dim * batch_size,
        }
    }

    /// Returns the absolute offsets where current layer da are stored.
    pub fn da_offsets(&self, batch_size: usize) -> RangeTo<usize> {
        RangeTo {
            end: self.input_dim * batch_size,
        }
    }
}

/// Applies the Dense linear transformation to given input, weights, bias and activations in-place.
///
/// f(x) = wx + b
///
/// # Arguments
///
/// * `meta` - The Dense layer metadata.
/// * `batch_size` - The batch size.
/// * `ones` - The ones vector.
/// * `input` - The slice of input activations.
/// * `weights` - The layer weights.
/// * `bias` - The layer bias.
/// * `output` - The slice of output activations, written in-place.
pub fn forward(
    meta: &DenseMeta,
    batch_size: usize,
    ones: &[f32],
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    gemm::sgemm(
        cblas::Transpose::None,
        cblas::Transpose::Ordinary,
        batch_size,
        meta.output_dim,
        meta.input_dim,
        1.0,
        input,
        meta.input_dim,
        weights,
        meta.input_dim,
        0.0,
        output,
        meta.output_dim,
    );

    gemm::sger(
        batch_size,
        meta.output_dim,
        1.0,
        ones,
        bias,
        output,
        meta.output_dim,
    );
}

/// Applies the derivative of the Dense linear transformation function to the given gradients and activations in-place.
///
/// dW = dz * activations
/// dB = dz * ones
///
/// # Arguments
///
/// * `meta` - The Dense layer metadata.
/// * `batch_size` - The batch size.
/// * `ones` - The ones vector.
/// * `dw` - The current layer weights gradient.
/// * `db` - The current layer bias gradient.
/// * `dz` - The incoming gradient with respect to the output of this layer.
/// * `input` - The previous layer output activations.
pub fn backward_parameters(
    meta: &DenseMeta,
    batch_size: usize,
    ones: &[f32],
    dw: &mut [f32],
    db: &mut [f32],
    dz: &[f32],
    input: &[f32],
) {
    gemm::sgemm(
        cblas::Transpose::Ordinary,
        cblas::Transpose::None,
        meta.output_dim,
        meta.input_dim,
        batch_size,
        1.0 / batch_size as f32,
        dz,
        meta.output_dim,
        input,
        meta.input_dim,
        0.0,
        dw,
        meta.input_dim,
    );

    gemm::sgemv(
        cblas::Transpose::Ordinary,
        batch_size,
        meta.output_dim,
        1.0 / batch_size as f32,
        dz,
        meta.output_dim,
        ones,
        0.0,
        db,
    );
}

/// Propagates the gradient through this layer weights, without computing weight or bias gradients.
///
/// dA = dz * W
///
/// # Arguments
///
/// * `meta` - The Dense layer metadata.
/// * `batch_size` - The batch size.
/// * `da` - The gradient with respect to the input of this layer, written in-place.
/// * `dz` - The incoming gradient with respect to the output of this layer.
/// * `weights` - The current layer weights.
pub fn backward_input(
    meta: &DenseMeta,
    batch_size: usize,
    da: &mut [f32],
    dz: &[f32],
    weights: &[f32],
) {
    gemm::sgemm(
        cblas::Transpose::None,
        cblas::Transpose::None,
        batch_size,
        meta.input_dim,
        meta.output_dim,
        1.0,
        dz,
        meta.output_dim,
        weights,
        meta.input_dim,
        0.0,
        da,
        meta.input_dim,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < EPS, "left={:?} right={:?}", a, b);
        }
    }

    #[test]
    fn test_offsets() {
        let meta = DenseMeta::new(3, 2, 5, 2, 1..4, 4..6, Initialization::He);

        assert_eq!(meta.activations_split_offset(2), 10);
        assert_eq!(meta.input_offsets(2), RangeFrom { start: 4 });
        assert_eq!(meta.output_offsets(2), RangeTo { end: 4 });
    }

    #[test]
    fn test_forward() {
        let meta = DenseMeta::new(2, 3, 0, 0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let ones = vec![1.0, 1.0];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];

        forward(
            &meta,
            batch_size,
            &ones,
            &input,
            &weights,
            &bias,
            &mut activations,
        );

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
    }

    #[test]
    fn test_backward_parameters() {
        let meta = DenseMeta::new(2, 3, 0, 0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let ones = vec![1.0, 1.0];
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let mut dw = vec![0.0; 6];
        let mut db = vec![0.0; 3];

        backward_parameters(&meta, batch_size, &ones, &mut dw, &mut db, &dz, &input);

        assert_close(&dw, &[0.5, 2.0, 1.0, 2.5, 1.5, 3.0]);
        assert_close(&db, &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_backward_input() {
        let meta = DenseMeta::new(2, 3, 0, 0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut da = vec![0.0; 4];

        backward_input(&meta, batch_size, &mut da, &dz, &weights);

        assert_close(&da, &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn test_forward_and_backward_parameters() {
        let meta = DenseMeta::new(2, 3, 0, 0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let ones = vec![1.0, 1.0];

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dw = vec![0.0; 6];
        let mut db = vec![0.0; 3];

        forward(
            &meta,
            batch_size,
            &ones,
            &input,
            &weights,
            &bias,
            &mut activations,
        );
        backward_parameters(&meta, batch_size, &ones, &mut dw, &mut db, &dz, &input);

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
        assert_close(&dw, &[6.5, 9.0, 8.5, 12.0, 10.5, 15.0]);
        assert_close(&db, &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_forward_and_backward_input() {
        let meta = DenseMeta::new(2, 3, 0, 0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let ones = vec![1.0, 1.0];

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut da = vec![0.0; 4];

        forward(
            &meta,
            batch_size,
            &ones,
            &input,
            &weights,
            &bias,
            &mut activations,
        );
        backward_input(&meta, batch_size, &mut da, &dz, &weights);

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
        assert_close(&da, &[4.0, 5.0, 10.0, 11.0]);
    }
}

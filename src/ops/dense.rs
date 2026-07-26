use core::ops::{Range, RangeTo};

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
    /// The relative input span for this layer.
    pub relative_input_span: Range<usize>,
    /// The relative output span for this layer.
    pub relative_output_span: Range<usize>,
    /// The relative span where current layer weights are stored.
    pub weight_span: Range<usize>,
    /// The relative span where current layer biases are stored.
    pub bias_span: Range<usize>,
    /// The initialization method to use for this layer.
    pub initialization: Initialization,
}

impl DenseMeta {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        relative_input_span: Range<usize>,
        relative_output_span: Range<usize>,
        weight_span: Range<usize>,
        bias_span: Range<usize>,
        initialization: Initialization,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            relative_input_span,
            relative_output_span,
            weight_span,
            bias_span,
            initialization,
        }
    }

    /// Returns the absolute offset where activations must be split to get the input and output activations.
    pub fn activations_split_offset(&self, batch_size: usize) -> usize {
        self.relative_output_span.start * batch_size
    }

    /// Returns the absolute span where current layer input activations are stored.
    pub fn input_offset(&self, batch_size: usize) -> Range<usize> {
        self.relative_input_span.start * batch_size..self.relative_input_span.end * batch_size
    }

    /// Returns the absolute span where current layer output activations are stored.
    pub fn output_offset(&self, batch_size: usize) -> Range<usize> {
        let len = self.relative_output_span.end - self.relative_output_span.start;
        0..len * batch_size
    }

    /// Returns the absolute span where current layer dz are stored.
    pub fn dz_offset(&self, batch_size: usize) -> RangeTo<usize> {
        ..self.output_dim * batch_size
    }

    /// Returns the absolute span where current layer da are stored.
    pub fn da_offset(&self, batch_size: usize) -> RangeTo<usize> {
        ..self.input_dim * batch_size
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
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    assert_eq!(meta.output_dim, bias.len());

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

    for row in output.chunks_mut(meta.output_dim) {
        for (out, bias) in row.iter_mut().zip(bias.iter()) {
            *out += bias;
        }
    }
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
    dw: &mut [f32],
    db: &mut [f32],
    dz: &[f32],
    input: &[f32],
) {
    assert_eq!(meta.output_dim, db.len());

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

    db.fill(0.0);
    let scale = 1.0 / batch_size as f32;
    for row in dz.chunks(meta.output_dim) {
        for (db_j, dz_j) in db.iter_mut().zip(row) {
            *db_j += scale * dz_j;
        }
    }
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
    fn test_offset() {
        let meta = DenseMeta::new(3, 2, 0..4, 4..8, 1..4, 4..6, Initialization::He);

        assert_eq!(meta.activations_split_offset(2), 8);
        assert_eq!(meta.input_offset(2), 0..8);
        assert_eq!(meta.output_offset(2), 0..8);
    }

    #[test]
    fn test_forward() {
        let meta = DenseMeta::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];

        forward(&meta, batch_size, &input, &weights, &bias, &mut activations);

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
    }

    #[test]
    fn test_backward_parameters() {
        let meta = DenseMeta::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let mut dw = vec![0.0; 6];
        let mut db = vec![0.0; 3];

        backward_parameters(&meta, batch_size, &mut dw, &mut db, &dz, &input);

        assert_close(&dw, &[0.5, 2.0, 1.0, 2.5, 1.5, 3.0]);
        assert_close(&db, &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_backward_input() {
        let meta = DenseMeta::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut da = vec![0.0; 4];

        backward_input(&meta, batch_size, &mut da, &dz, &weights);

        assert_close(&da, &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn test_forward_and_backward_parameters() {
        let meta = DenseMeta::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dw = vec![0.0; 6];
        let mut db = vec![0.0; 3];

        forward(&meta, batch_size, &input, &weights, &bias, &mut activations);
        backward_parameters(&meta, batch_size, &mut dw, &mut db, &dz, &input);

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
        assert_close(&dw, &[6.5, 9.0, 8.5, 12.0, 10.5, 15.0]);
        assert_close(&db, &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_forward_and_backward_input() {
        let meta = DenseMeta::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut da = vec![0.0; 4];

        forward(&meta, batch_size, &input, &weights, &bias, &mut activations);
        backward_input(&meta, batch_size, &mut da, &dz, &weights);

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
        assert_close(&da, &[4.0, 5.0, 10.0, 11.0]);
    }
}

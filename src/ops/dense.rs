use core::ops::{Range, RangeTo};

use crate::core::serialization;
use crate::ops::{Initialization, gemm};
use thiserror::Error;

/// Dense operation.
///
/// Arena ranges are relative. Multiply by `batch_size` for absolute indices.
#[derive(Debug, Clone)]
pub struct Dense {
    /// Input dimension. Equal to the connected layer output dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Relative input activation range.
    pub relative_input_range: Range<usize>,
    /// Relative output activation range.
    pub relative_output_range: Range<usize>,
    /// Weight range.
    pub weight_range: Range<usize>,
    /// Bias range.
    pub bias_range: Range<usize>,
    /// Weight initialization scheme.
    pub initialization: Initialization,
}

impl Dense {
    /// Creates a dense operation.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        relative_input_range: Range<usize>,
        relative_output_range: Range<usize>,
        weight_range: Range<usize>,
        bias_range: Range<usize>,
        initialization: Initialization,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            relative_input_range,
            relative_output_range,
            weight_range,
            bias_range,
            initialization,
        }
    }

    /// Split offset between input and output activations.
    pub fn activation_split_offset(&self, batch_size: usize) -> usize {
        self.relative_output_range.start * batch_size
    }

    /// Input activation range.
    pub fn input_range(&self, batch_size: usize) -> Range<usize> {
        self.relative_input_range.start * batch_size..self.relative_input_range.end * batch_size
    }

    /// Output activation range.
    pub fn output_range(&self, batch_size: usize) -> Range<usize> {
        let len = self.relative_output_range.end - self.relative_output_range.start;
        0..len * batch_size
    }

    /// Output gradient range.
    pub fn gradient_range(&self, batch_size: usize) -> RangeTo<usize> {
        ..self.output_dim * batch_size
    }

    /// Input gradient range.
    pub fn input_gradient_range(&self, batch_size: usize) -> RangeTo<usize> {
        ..self.input_dim * batch_size
    }
}

/// Errors during dense layer serialization.
#[derive(Error, Debug)]
pub enum DenseEncodingError {
    #[error("IO error: {0}")]
    Io(#[from] serialization::SerializationError),
    #[error("Invalid initialization: {0}")]
    InvalidInitialization(String),
}

impl serialization::Encodable for Dense {
    type Error = DenseEncodingError;

    fn encoded_len(&self) -> usize {
        // 2 u32 + 4 ranges + 1 init byte
        2 * serialization::U32_WIRE + 4 * serialization::RANGE_WIRE + 1
    }

    fn encode(&self, writer: &mut impl std::io::Write) -> Result<(), Self::Error> {
        serialization::write_u32(writer, self.input_dim as u32)?;
        serialization::write_u32(writer, self.output_dim as u32)?;
        serialization::write_range(writer, self.relative_input_range.clone())?;
        serialization::write_range(writer, self.relative_output_range.clone())?;
        serialization::write_range(writer, self.weight_range.clone())?;
        serialization::write_range(writer, self.bias_range.clone())?;
        writer
            .write_all(&[self.initialization.as_u8()])
            .map_err(serialization::SerializationError::Io)?;
        Ok(())
    }

    fn decode(reader: &mut impl std::io::Read) -> Result<Self, Self::Error> {
        let input_dim = serialization::read_u32(reader)? as usize;
        let output_dim = serialization::read_u32(reader)? as usize;
        let relative_input_range = serialization::read_range(reader)?;
        let relative_output_range = serialization::read_range(reader)?;
        let weight_range = serialization::read_range(reader)?;
        let bias_range = serialization::read_range(reader)?;
        let mut init_buf = [0u8; 1];
        reader
            .read_exact(&mut init_buf)
            .map_err(serialization::SerializationError::Io)?;
        let initialization = Initialization::try_from(init_buf[0])
            .map_err(|e| DenseEncodingError::InvalidInitialization(e.to_string()))?;

        Ok(Self::new(
            input_dim,
            output_dim,
            relative_input_range,
            relative_output_range,
            weight_range,
            bias_range,
            initialization,
        ))
    }
}

/// Forward pass: `Y = X Wᵀ + b`.
///
/// Shapes: `X [B, input_dim]`, `W [output_dim, input_dim]`, `b [output_dim]`,
/// `Y [B, output_dim]`. All row-major.
///
/// # Panics
///
/// Panics if `bias.len() != output_dim`.
pub fn forward(
    operation: &Dense,
    batch_size: usize,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    assert_eq!(operation.output_dim, bias.len());

    gemm::sgemm(
        cblas::Transpose::None,
        cblas::Transpose::Ordinary,
        batch_size,
        operation.output_dim,
        operation.input_dim,
        1.0,
        input,
        operation.input_dim,
        weights,
        operation.input_dim,
        0.0,
        output,
        operation.output_dim,
    );

    for row in output.chunks_mut(operation.output_dim) {
        for (out, bias) in row.iter_mut().zip(bias.iter()) {
            *out += bias;
        }
    }
}

/// Parameter gradients averaged over the batch: `dW = dZᵀ X / B`, `dB = Σ_rows(dZ) / B`.
///
/// Shapes: `dZ [B, output_dim]`, `X [B, input_dim]`, `dW [output_dim, input_dim]`,
/// `dB [output_dim]`. All row-major.
///
/// # Panics
///
/// Panics if `db.len() != output_dim`.
pub fn backward_parameters(
    operation: &Dense,
    batch_size: usize,
    dw: &mut [f32],
    db: &mut [f32],
    dz: &[f32],
    input: &[f32],
) {
    assert_eq!(operation.output_dim, db.len());

    gemm::sgemm(
        cblas::Transpose::Ordinary,
        cblas::Transpose::None,
        operation.output_dim,
        operation.input_dim,
        batch_size,
        1.0 / batch_size as f32,
        dz,
        operation.output_dim,
        input,
        operation.input_dim,
        0.0,
        dw,
        operation.input_dim,
    );

    db.fill(0.0);
    let scale = 1.0 / batch_size as f32;
    for row in dz.chunks(operation.output_dim) {
        for (db_j, dz_j) in db.iter_mut().zip(row) {
            *db_j += scale * dz_j;
        }
    }
}

/// Input gradient: `dA = dZ W`.
///
/// Shapes: `dZ [B, output_dim]`, `W [output_dim, input_dim]`,
/// `dA [B, input_dim]`. All row-major.
pub fn backward_input(
    operation: &Dense,
    batch_size: usize,
    da: &mut [f32],
    dz: &[f32],
    weights: &[f32],
) {
    gemm::sgemm(
        cblas::Transpose::None,
        cblas::Transpose::None,
        batch_size,
        operation.input_dim,
        operation.output_dim,
        1.0,
        dz,
        operation.output_dim,
        weights,
        operation.input_dim,
        0.0,
        da,
        operation.input_dim,
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
        let operation = Dense::new(3, 2, 0..4, 4..8, 1..4, 4..6, Initialization::He);

        assert_eq!(operation.activation_split_offset(2), 8);
        assert_eq!(operation.input_range(2), 0..8);
        assert_eq!(operation.output_range(2), 0..8);
    }

    #[test]
    fn test_forward() {
        let operation = Dense::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];

        forward(
            &operation,
            batch_size,
            &input,
            &weights,
            &bias,
            &mut activations,
        );

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
    }

    #[test]
    fn test_backward_parameters() {
        let operation = Dense::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let mut dw = vec![0.0; 6];
        let mut db = vec![0.0; 3];

        backward_parameters(&operation, batch_size, &mut dw, &mut db, &dz, &input);

        assert_close(&dw, &[0.5, 2.0, 1.0, 2.5, 1.5, 3.0]);
        assert_close(&db, &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_backward_input() {
        let operation = Dense::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut da = vec![0.0; 4];

        backward_input(&operation, batch_size, &mut da, &dz, &weights);

        assert_close(&da, &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn test_forward_and_backward_parameters() {
        let operation = Dense::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dw = vec![0.0; 6];
        let mut db = vec![0.0; 3];

        forward(
            &operation,
            batch_size,
            &input,
            &weights,
            &bias,
            &mut activations,
        );
        backward_parameters(&operation, batch_size, &mut dw, &mut db, &dz, &input);

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
        assert_close(&dw, &[6.5, 9.0, 8.5, 12.0, 10.5, 15.0]);
        assert_close(&db, &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_forward_and_backward_input() {
        let operation = Dense::new(2, 3, 0..0, 0..0, 0..0, 0..0, Initialization::He);
        let batch_size = 2;

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![0.5, -0.5, 1.0];
        let mut activations = vec![0.0; 6];
        let dz = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut da = vec![0.0; 4];

        forward(
            &operation,
            batch_size,
            &input,
            &weights,
            &bias,
            &mut activations,
        );
        backward_input(&operation, batch_size, &mut da, &dz, &weights);

        assert_close(&activations, &[1.5, 1.5, 4.0, 3.5, 3.5, 8.0]);
        assert_close(&da, &[4.0, 5.0, 10.0, 11.0]);
    }
}

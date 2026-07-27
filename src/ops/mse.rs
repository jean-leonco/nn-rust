use crate::core::serialization;
use core::ops::Range;

/// Mean squared error operation.
#[derive(Debug, Clone)]
pub struct MeanSquaredError {
    /// Relative activation range.
    pub(crate) relative_activation_range: Range<usize>,
}

impl MeanSquaredError {
    /// Creates a mean squared error operation.
    pub fn new(relative_activation_range: Range<usize>) -> Self {
        Self {
            relative_activation_range,
        }
    }

    /// Activation range for the current layer at `batch_size`.
    pub fn activation_range(&self, batch_size: usize) -> Range<usize> {
        self.relative_activation_range.start * batch_size
            ..self.relative_activation_range.end * batch_size
    }
}

impl serialization::Encodable for MeanSquaredError {
    type Error = super::serialization::SerializationError;

    fn encoded_len(&self) -> usize {
        // 1 range
        serialization::RANGE_WIRE
    }

    fn encode(&self, writer: &mut impl std::io::Write) -> Result<(), Self::Error> {
        serialization::write_range(writer, self.relative_activation_range.clone())
    }

    fn decode(reader: &mut impl std::io::Read) -> Result<Self, Self::Error> {
        let range = serialization::read_range(reader)?;
        Ok(Self::new(range))
    }
}

/// Computes the mean squared error gradient: `dZ = 2(P - Y) / D`.
///
/// # Panics
///
/// Panics if `dz`, `predicted`, and `targets` have unequal lengths.
pub fn backward(operation: &MeanSquaredError, dz: &mut [f32], predicted: &[f32], targets: &[f32]) {
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

    let output_dim = operation.relative_activation_range.len();
    let scale = 2.0 / output_dim as f32;

    for ((dz, p), y) in dz.iter_mut().zip(predicted.iter()).zip(targets.iter()) {
        *dz = scale * (p - y);
    }
}

use core::ops::{Range, RangeTo};

use crate::core::serialization;

/// `ReLU` layer metadata.
#[derive(Debug, Clone)]
pub struct ReluMeta {
    /// Relative activation range.
    pub(crate) relative_activation_range: Range<usize>,
}

impl ReluMeta {
    /// Creates metadata for an in-place `ReLU`.
    pub fn new(relative_activation_range: Range<usize>) -> Self {
        Self {
            relative_activation_range,
        }
    }

    /// Activation range.
    pub fn activation_range(&self, batch_size: usize) -> Range<usize> {
        self.relative_activation_range.start * batch_size
            ..self.relative_activation_range.end * batch_size
    }

    /// Gradient range.
    pub fn gradient_range(&self, batch_size: usize) -> RangeTo<usize> {
        let dim = self.relative_activation_range.end - self.relative_activation_range.start;
        ..dim * batch_size
    }
}

impl serialization::Encodable for ReluMeta {
    type Error = super::serialization::SerializationError;

    fn encoded_len(&self) -> usize {
        // 1 range
        serialization::RANGE_WIRE
    }

    fn encode(&self, writer: &mut impl std::io::Write) -> Result<(), Self::Error> {
        serialization::write_range(writer, self.relative_activation_range.clone())
    }

    fn decode(reader: &mut impl std::io::Read) -> Result<Self, Self::Error> {
        Ok(Self::new(serialization::read_range(reader)?))
    }
}

/// Applies `ReLU` in-place: `f(x) = max(0, x)`.
pub fn forward(activations: &mut [f32]) {
    for val in activations {
        *val = val.max(0.0);
    }
}

/// Applies the `ReLU` derivative in-place: `f'(x) = 1 if x > 0, else 0`.
pub fn backward(dz: &mut [f32], da: &[f32], activations: &[f32]) {
    for ((da, dz), a) in da.iter().zip(dz.iter_mut()).zip(activations.iter()) {
        let derivative = if *a > 0.0 { 1.0 } else { 0.0 };
        *dz = da * derivative;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offsets() {
        let meta = ReluMeta::new(2..5);

        assert_eq!(meta.activation_range(1), 2..5);
        assert_eq!(meta.activation_range(3), 6..15);
    }

    #[test]
    fn test_forward() {
        let mut activations = vec![-2.0, -0.0, 1.0, 3.5];

        forward(&mut activations);
        assert_eq!(activations, vec![0.0, 0.0, 1.0, 3.5]);
    }

    #[test]
    fn test_backward() {
        let mut dz = vec![0.0; 4];
        let da = vec![1.5, -2.0, 3.0, 0.5];
        let activations = vec![0.0, 0.0, 1.0, 3.5];

        backward(&mut dz, &da, &activations);
        assert_eq!(dz, vec![0.0, 0.0, 3.0, 0.5]);
    }

    #[test]
    fn test_forward_and_backward() {
        let mut activations = vec![-2.0, -0.0, 1.0, 3.5];
        let mut dz = vec![0.0; 4];
        let da = vec![1.5, -2.0, 3.0, 0.5];

        forward(&mut activations);
        backward(&mut dz, &da, &activations);

        assert_eq!(activations, vec![0.0, 0.0, 1.0, 3.5]);
        assert_eq!(dz, vec![0.0, 0.0, 3.0, 0.5]);
    }
}

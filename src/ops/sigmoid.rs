use core::ops::{Range, RangeTo};

use crate::core::{math, serialization};

/// In-place sigmoid operation.
#[derive(Debug, Clone)]
pub struct Sigmoid {
    /// Relative activation range.
    pub(crate) relative_activation_range: Range<usize>,
}

impl Sigmoid {
    /// Creates an in-place sigmoid operation.
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

impl serialization::Encodable for Sigmoid {
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

impl Sigmoid {
    /// Applies sigmoid in-place: `f(x) = 1 / (1 + exp(-x))`.
    pub fn forward(&self, activations: &mut [f32]) {
        for val in activations {
            let z = 1.0 / (1.0 + math::schraudolph(-*val));
            *val = z;
        }
    }

    /// Applies the sigmoid derivative in-place: `f'(x) = x * (1 - x)`.
    pub fn backward(&self, dz: &mut [f32], da: &[f32], activations: &[f32]) {
        assert_eq!(da.len(), dz.len());
        assert_eq!(da.len(), activations.len());

        for ((da, dz), a) in da.iter().zip(dz.iter_mut()).zip(activations.iter()) {
            let derivative = a * (1.0 - a);
            *dz = da * derivative;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offsets() {
        let operation = Sigmoid::new(2..5);

        assert_eq!(operation.activation_range(1), 2..5);
        assert_eq!(operation.activation_range(3), 6..15);
    }

    #[test]
    fn test_forward() {
        let mut activations = vec![0.0, f32::INFINITY];

        let operation = Sigmoid::new(0..2);
        operation.forward(&mut activations);

        assert!((activations[0] - 0.5).abs() < 0.05);
        assert!((activations[1] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_backward() {
        let mut dz = vec![0.0; 2];
        let da = vec![2.0, 4.0];
        let activations = vec![0.5, 1.0];

        let operation = Sigmoid::new(0..2);
        operation.backward(&mut dz, &da, &activations);

        assert!((dz[0] - 0.5).abs() < 0.05);
        assert!((dz[1] - 0.0).abs() < 0.05);
    }

    #[test]
    fn test_forward_and_backward() {
        let mut activations = vec![0.0, f32::INFINITY];
        let mut dz = vec![0.0; 2];
        let da = vec![2.0, 4.0];

        let operation = Sigmoid::new(0..2);
        operation.forward(&mut activations);
        operation.backward(&mut dz, &da, &activations);

        assert!((activations[0] - 0.5).abs() < 0.05);
        assert!((activations[1] - 1.0).abs() < 0.05);
        assert!((dz[0] - 0.5).abs() < 0.05);
        assert!((dz[1] - 0.0).abs() < 0.05);
    }
}

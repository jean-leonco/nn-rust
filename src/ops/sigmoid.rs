use core::ops::{Range, RangeTo};

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
        Range {
            start: self.a_start * batch_size,
            end: self.a_end * batch_size,
        }
    }

    /// Returns the absolute offsets where current layer gradients are stored.
    pub fn gradient_offsets(&self, batch_size: usize) -> RangeTo<usize> {
        let dimension = self.a_end - self.a_start;
        RangeTo {
            end: dimension * batch_size,
        }
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
    for val in activations {
        *val = 1.0 / (1.0 + (-*val).exp());
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

        assert!((activations[0] - 0.5).abs() < f32::EPSILON);
        assert!((activations[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_backward() {
        let mut dz = vec![0.0; 2];
        let da = vec![2.0, 4.0];
        let activations = vec![0.5, 1.0];

        backward(&mut dz, &da, &activations);

        assert!((dz[0] - 0.5).abs() < f32::EPSILON);
        assert!((dz[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_forward_and_backward() {
        let mut activations = vec![0.0, f32::INFINITY];
        let mut dz = vec![0.0; 2];
        let da = vec![2.0, 4.0];

        forward(&mut activations);
        backward(&mut dz, &da, &activations);

        assert!((activations[0] - 0.5).abs() < f32::EPSILON);
        assert!((activations[1] - 1.0).abs() < f32::EPSILON);
        assert!((dz[0] - 0.5).abs() < f32::EPSILON);
        assert!((dz[1] - 0.0).abs() < f32::EPSILON);
    }
}

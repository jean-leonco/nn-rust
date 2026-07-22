use core::ops::{Range, RangeTo};

/// Metadata for the ReLU activation function.
#[derive(Debug, Clone)]
pub struct ReluMeta {
    /// The relative offsets where current layer activations are stored.
    /// Must be multiplied by the batch size to get the absolute offset.
    pub(crate) a_start: usize,
    pub(crate) a_end: usize,
}

impl ReluMeta {
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

/// Applies the ReLU function to the given activations in-place.
///
/// f(x) = max(0, x)
///
/// # Arguments
///
/// * `activations` - The slice to apply the ReLU function to.
pub fn forward(activations: &mut [f32]) {
    for val in activations {
        *val = val.max(0.0);
    }
}

/// Applies the derivative of the ReLU function to the given gradients in-place.
///
/// f'(x) = 1 if x > 0, 0 otherwise
///
/// # Arguments
///
/// * `dz` - The outgoing gradient with respect to the input of this layer.
/// * `da` - The incoming gradient with respect to the output of this layer.
/// * `activations` - The slice containing this layer activations.
///   It must contain the post-activation values (A) instead of inputs (Z), as derivative is computed directly from the outputs.
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
        let meta = ReluMeta::new(2, 5);

        assert_eq!(meta.activation_offsets(1), 2..5);
        assert_eq!(meta.activation_offsets(3), 6..15);
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

use core::ops::{Range, RangeTo};

use rand::{Rng, distr};
use rand_distr::Distribution;

/// Metadata for the Dropout layer.
/// Randomly drops out (sets to zero) a fraction of the neurons.
#[derive(Debug, PartialEq)]
pub struct DropoutMeta {
    /// Dropout probability.
    pub p: f32,
    /// Scaling factor for the inverse of dropout probability.
    pub inv_p: f32,
    distribution: distr::Bernoulli,
    /// The relative offsets where current layer activations are stored.
    /// Must be multiplied by the batch size to get the absolute offset.
    pub(crate) a_start: usize,
    pub(crate) a_end: usize,
    /// The relative offsets where current layer mask data is stored.
    /// Must be multiplied by the batch size to get the absolute offset.
    pub(crate) m_start: usize,
    pub(crate) m_end: usize,
}

impl DropoutMeta {
    pub fn new(
        p: f32,
        d_start: usize,
        d_end: usize,
        m_start: usize,
        m_end: usize,
    ) -> Result<Self, distr::BernoulliError> {
        Ok(Self {
            p,
            inv_p: 1.0 / (1.0 - p),
            distribution: distr::Bernoulli::new(1.0 - (p as f64))?,
            a_start: d_start,
            a_end: d_end,
            m_start,
            m_end,
        })
    }

    /// Returns the absolute offsets where current layer activations are stored.
    pub fn activation_offsets(&self, batch_size: usize) -> Range<usize> {
        Range {
            start: self.a_start * batch_size,
            end: self.a_end * batch_size,
        }
    }

    /// Returns the absolute offsets where current layer mask data is stored.
    pub fn mask_offsets(&self, batch_size: usize) -> Range<usize> {
        Range {
            start: self.m_start * batch_size,
            end: self.m_end * batch_size,
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

/// Applies the dropout to given activations and mask in-place.
///
/// # Arguments
///
/// * `meta` - The dropout metadata.
/// * `activations` - The slice of activations to apply dropout to.
/// * `mask` - The slice of mask data to use for dropout.
/// * `rng` - The random number generator to use for dropout.
pub fn forward<R: Rng + ?Sized>(
    meta: &DropoutMeta,
    activations: &mut [f32],
    masks: &mut [u8],
    rng: &mut R,
) {
    for (val, mask) in activations.iter_mut().zip(masks.iter_mut()) {
        *mask = meta.distribution.sample(rng) as u8;
        *val *= *mask as f32 * meta.inv_p;
    }
}

/// Applies the dropout backward pass to the given gradients and mask in-place.
/// # Arguments
///
/// * `meta` - The dropout metadata.
/// * `dz` - The outgoing gradient with respect to the input of this layer.
/// * `da` - The incoming gradient with respect to the output of this layer.
/// * `masks` - The slice containing this layer masks.
///   It must contain the original mask used during the forward pass, as the gradients must be scaled using the exact same random mask.
pub fn backward(meta: &DropoutMeta, dz: &mut [f32], da: &[f32], masks: &[u8]) {
    for ((dz, da), mask) in dz.iter_mut().zip(da.iter()).zip(masks.iter()) {
        *dz = da * *mask as f32 * meta.inv_p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn test_new_and_errors() {
        assert!(DropoutMeta::new(0.2, 1, 4, 10, 13).is_ok());
        assert!(DropoutMeta::new(1.5, 1, 4, 10, 13).is_err());
    }

    #[test]
    fn test_offsets() {
        let meta = DropoutMeta::new(0.5, 2, 5, 4, 8).unwrap();

        assert_eq!(meta.activation_offsets(1), 2..5);
        assert_eq!(meta.mask_offsets(1), 4..8);
        assert_eq!(meta.activation_offsets(3), 6..15);
        assert_eq!(meta.mask_offsets(3), 12..24);
    }

    #[test]
    fn test_forward() {
        let mut meta = DropoutMeta::new(0.5, 0, 4, 0, 4).unwrap();
        let mut activations = vec![1.0, 2.0, 3.0, 4.0];
        let mut mask = vec![0; 4];
        let mut rng = SmallRng::seed_from_u64(42);

        forward(&mut meta, &mut activations, &mut mask, &mut rng);

        assert_eq!(mask, vec![0, 1, 0, 0]);
        assert_eq!(activations, vec![0.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn test_backward() {
        let meta = DropoutMeta::new(0.5, 0, 4, 0, 4).unwrap();
        let mut dz = vec![0.0; 4];
        let da = vec![1.5, 2.5, 3.5, 4.5];
        let mask = vec![0, 1, 0, 0];

        backward(&meta, &mut dz, &da, &mask);

        assert_eq!(dz, vec![0.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn test_forward_and_backward() {
        let mut meta = DropoutMeta::new(0.5, 0, 4, 0, 4).unwrap();

        let mut activations = vec![1.0, 2.0, 3.0, 4.0];
        let mut mask = vec![0; 4];
        let mut dz = vec![0.0; 4];
        let da = vec![1.5, 2.5, 3.5, 4.5];

        let mut rng = SmallRng::seed_from_u64(42);

        forward(&mut meta, &mut activations, &mut mask, &mut rng);
        backward(&meta, &mut dz, &da, &mask);

        assert_eq!(mask, vec![0, 1, 0, 0]);
        assert_eq!(activations, vec![0.0, 4.0, 0.0, 0.0]);
        assert_eq!(dz, vec![0.0, 5.0, 0.0, 0.0]);
    }
}

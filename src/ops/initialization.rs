use rand::Rng;
use rand_distr::{Distribution, Normal};
use thiserror::Error;

/// Initialization for weight matrices.
#[derive(Debug, PartialEq)]
pub enum Initialization {
    /// He (or Kaiming) initialization. Best suited for ReLU activation.
    He,
    /// Xavier (or Glorot) initialization. Best suited for sigmoid/tanh activation.
    Xavier,
}

#[derive(Error, Debug)]
pub enum InitializationError {
    #[error("Normal distribution error: {0}")]
    NormalDistr(#[from] rand_distr::NormalError),
}

impl Initialization {
    /// Returns the standard deviation for the given initialization scheme.
    fn std_dev(&self, input: usize, output: usize) -> f32 {
        match self {
            Self::He => 2.0 / (input as f32),
            Self::Xavier => 2.0 / ((input + output) as f32),
        }
        .sqrt()
    }

    pub fn to_u8(&self) -> u8 {
        match self {
            Self::He => 0,
            Self::Xavier => 1,
        }
    }

    /// Initializes the params for the given input and output sizes.
    pub fn init<R: Rng + ?Sized>(
        &self,
        input: usize,
        output: usize,
        params: &mut Vec<f32>,
        rng: &mut R,
    ) -> Result<(), InitializationError> {
        let std_dev = self.std_dev(input, output);
        let normal = Normal::new(0.0, std_dev)?;

        params.reserve_exact((input * output) + output);

        for _ in 0..(input * output) {
            params.push(normal.sample(rng));
        }
        params.extend(std::iter::repeat_n(0.0, output));

        Ok(())
    }
}

impl TryFrom<u8> for Initialization {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::He),
            1 => Ok(Self::Xavier),
            _ => Err("Invalid initialization code"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn test_std_dev() {
        let he = Initialization::He;
        let xavier = Initialization::Xavier;

        let expected_he = (2.0 / 10.0_f32).sqrt();
        assert!((he.std_dev(10, 20) - expected_he).abs() < f32::EPSILON);

        let expected_xavier = (2.0 / 30.0_f32).sqrt();
        assert!((xavier.std_dev(10, 20) - expected_xavier).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serialization() {
        let he = Initialization::He;
        let xavier = Initialization::Xavier;

        assert_eq!(he.to_u8(), 0);
        assert_eq!(xavier.to_u8(), 1);

        assert_eq!(Initialization::try_from(0), Ok(Initialization::He));
        assert_eq!(Initialization::try_from(1), Ok(Initialization::Xavier));
        assert!(Initialization::try_from(2).is_err());
    }

    #[test]
    fn test_init_he() {
        let mut rng = SmallRng::seed_from_u64(42);
        let input = 1000;
        let output = 1000;
        let mut params = Vec::new();

        let he = Initialization::He;
        he.init(input, output, &mut params, &mut rng).unwrap();

        let expected_std_dev = he.std_dev(input, output);
        let weights = &params[..1_000_000];

        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance: f32 =
            weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();

        assert!(mean.abs() < 0.01);
        assert!((std_dev - expected_std_dev).abs() < 0.01);
    }

    #[test]
    fn test_init_xavier() {
        let mut rng = SmallRng::seed_from_u64(42);
        let input = 1000;
        let output = 1000;
        let mut params = Vec::new();

        let xavier = Initialization::Xavier;
        xavier.init(input, output, &mut params, &mut rng).unwrap();

        let expected_std_dev = xavier.std_dev(input, output);
        let weights = &params[..1_000_000];

        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance: f32 =
            weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();

        assert!(mean.abs() < 0.01);
        assert!((std_dev - expected_std_dev).abs() < 0.01);
    }
}

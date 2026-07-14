use rand::Rng;
use rand_distr::{Distribution, Normal};
use thiserror::Error;

use crate::sequential;

#[derive(Error, Debug)]
pub enum WeightsError {
    #[error("Normal distribution error: {0}")]
    NormalDistr(#[from] rand_distr::NormalError),
}
pub type Result<T> = std::result::Result<T, WeightsError>;

#[derive(Debug)]
pub struct Weights {
    pub values: Vec<f32>,
}

impl Weights {
    pub fn init<R: Rng + ?Sized>(blueprint: &sequential::Sequential, rng: &mut R) -> Result<Self> {
        let mut values = Vec::with_capacity(blueprint.weights_size);

        for node in &blueprint.nodes {
            if let sequential::SequentialExecutionNode::Dense {
                input_dim,
                output_dim,
                initializer,
                ..
            } = node
            {
                let std_dev = initializer.std_dev(*input_dim, *output_dim);
                let normal = Normal::new(0.0, std_dev)?;
                for _ in 0..(*input_dim * *output_dim) {
                    values.push(normal.sample(rng));
                }
                values.extend(std::iter::repeat_n(0.0, *output_dim));
            }
        }

        Ok(Self { values })
    }
}

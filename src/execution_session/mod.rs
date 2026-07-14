use cblas::Transpose;
use rand::rngs::SmallRng;
use thiserror::Error;

use crate::{
    sequential::{Sequential, SequentialExecutionNode},
    weights,
};

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Bernoulli distribution error: {0}")]
    BernoulliDistr(#[from] rand::distr::BernoulliError),
}
pub type Result<T> = std::result::Result<T, ExecutionError>;

pub(crate) mod gemm;

pub struct ExecutionSession<'a> {
    blueprint: &'a Sequential,
    rng: &'a mut SmallRng,
    batch_size: usize,

    /**
     * Stores the activations for each layer, used for backpropagation.
     * Same as sum(nodes.output if node is Dense)
     */
    activations: Vec<f32>,
    /**
     * Stores the gradients for optimizer step.
     * Same size as weights, since it stores the dW and dB gradients.
     */
    gradients: Vec<f32>,

    /**
     * Ping-pong buffer for gradients, required for backpropagation.
     * Stores dA & dZ.
     */
    grad_buf: (Vec<f32>, Vec<f32>),

    /**
     * Stores the ones vector for use in gemm operations.
     */
    ones: Vec<f32>,
}

impl<'a> ExecutionSession<'a> {
    pub fn new(blueprint: &'a Sequential, rng: &'a mut SmallRng, batch_size: usize) -> Self {
        let activations = vec![0.0; batch_size * blueprint.a_size];
        let gradients = vec![0.0; blueprint.weights_size];

        Self {
            blueprint,
            rng,
            batch_size,
            activations,
            gradients,
            grad_buf: (
                vec![0.0; batch_size * blueprint.max_dim],
                vec![0.0; batch_size * blueprint.max_dim],
            ),
            ones: vec![1.0f32; batch_size],
        }
    }

    pub fn forward(&mut self, weights: &weights::Weights, x: &[f32]) -> Result<&[f32]> {
        let mut is_first = true;
        let mut output_start = 0;
        let mut output_end = 0;

        for node in &self.blueprint.nodes {
            match node {
                SequentialExecutionNode::Dense {
                    input_dim,
                    output_dim,
                    w_start,
                    w_end,
                    b_start,
                    b_end,
                    a_start,
                    input_a_start,
                    ..
                } => {
                    let split_point = a_start * self.batch_size;
                    let (left, right) = self.activations.split_at_mut(split_point);

                    let input_slice = if is_first {
                        x
                    } else {
                        let input_start = *input_a_start * self.batch_size;
                        &left[input_start..]
                    };

                    let output_slice = &mut right[..(output_dim * self.batch_size)];
                    let weights_slice = &weights.values[*w_start..*w_end];
                    let bias_slice = &weights.values[*b_start..*b_end];

                    // output = bias, broadcast across batch_size rows
                    for row in output_slice.chunks_mut(*output_dim) {
                        row.copy_from_slice(bias_slice);
                    }

                    gemm::gemm_f32(
                        Transpose::None,
                        Transpose::Ordinary,
                        self.batch_size,
                        *output_dim,
                        *input_dim,
                        1.0,
                        input_slice,
                        *input_dim,
                        weights_slice,
                        *input_dim,
                        1.0,
                        output_slice,
                        *output_dim,
                    );
                }
                SequentialExecutionNode::Dropout {
                    p,
                    data_start,
                    data_end,
                    mask_start,
                    mask_end,
                    inv_p,
                } => {
                    let d_start = data_start * self.batch_size;
                    let d_end = data_end * self.batch_size;

                    let (left, right) = self.activations.split_at_mut(mask_start * self.batch_size);

                    let data_slice = &mut left[d_start..d_end];
                    let mask_slice = &mut right[..((mask_end - mask_start) * self.batch_size)];

                    let distribution = rand::distr::Bernoulli::new(1.0 - (*p as f64))?;
                    for (val, mask) in data_slice.iter_mut().zip(mask_slice.iter_mut()) {
                        *mask = rand::distr::Distribution::sample(&distribution, &mut self.rng)
                            as u8 as f32;
                        *val *= *mask * inv_p;
                    }
                }
                SequentialExecutionNode::Relu { a_start, a_end } => {
                    let start = a_start * self.batch_size;
                    let end = a_end * self.batch_size;

                    for val in &mut self.activations[start..end] {
                        *val = val.max(0.0);
                    }
                }
                SequentialExecutionNode::Sigmoid { a_start, a_end } => {
                    let start = a_start * self.batch_size;
                    let end = a_end * self.batch_size;

                    for val in &mut self.activations[start..end] {
                        *val = 1.0 / (1.0 + (-*val).exp());
                    }
                }
                SequentialExecutionNode::SoftmaxCrossEntropy {
                    a_start,
                    a_end,
                    output_dim,
                } => {
                    output_start = a_start * self.batch_size;
                    output_end = a_end * self.batch_size;

                    for row in
                        &mut self.activations[output_start..output_end].chunks_mut(*output_dim)
                    {
                        let max = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
                        for val in row.iter_mut() {
                            *val = (*val - max).exp();
                        }

                        let sum: f32 = row.iter().sum();
                        for val in row.iter_mut() {
                            *val /= sum;
                        }
                    }
                }
            }

            is_first = false;
        }

        Ok(&self.activations[output_start..output_end])
    }

    pub fn backward(&mut self, weights: &weights::Weights, x: &[f32], y: &[f32]) -> &[f32] {
        for node in self.blueprint.nodes.iter().rev() {
            let (read_buf, write_buf) = &mut self.grad_buf;

            match node {
                SequentialExecutionNode::Dense {
                    input_dim,
                    output_dim,
                    w_start,
                    w_end,
                    b_start,
                    b_end,
                    a_start,
                    input_a_start,
                    ..
                } => {
                    let dz_slice = &read_buf[..(output_dim * self.batch_size)];

                    let forward_a_prev = if *a_start == 0 {
                        x
                    } else {
                        let input_start = *input_a_start * self.batch_size;
                        &self.activations[input_start..(*a_start * self.batch_size)]
                    };

                    let w_slice = &weights.values[*w_start..*w_end];

                    let (left, right) = self.gradients.split_at_mut(*w_end);
                    let dw_slice = &mut left[*w_start..*w_end];
                    let db_slice = &mut right[..(*b_end - *b_start)];

                    gemm::gemm_f32(
                        Transpose::Ordinary,
                        Transpose::None,
                        *output_dim,
                        *input_dim,
                        self.batch_size,
                        1.0 / self.batch_size as f32,
                        dz_slice,
                        *output_dim,
                        forward_a_prev,
                        *input_dim,
                        0.0,
                        dw_slice,
                        *input_dim,
                    );

                    gemm::sgemv_f32(
                        Transpose::Ordinary,
                        self.batch_size,
                        *output_dim,
                        1.0 / self.batch_size as f32,
                        dz_slice,
                        *output_dim,
                        &self.ones,
                        0.0,
                        db_slice,
                    );

                    if *a_start > 0 {
                        let da_prev_slice = &mut write_buf[..(*input_dim * self.batch_size)];
                        gemm::gemm_f32(
                            Transpose::None,
                            Transpose::None,
                            self.batch_size,
                            *input_dim,
                            *output_dim,
                            1.0,
                            dz_slice,
                            *output_dim,
                            w_slice,
                            *input_dim,
                            0.0,
                            da_prev_slice,
                            *input_dim,
                        );
                    }
                }
                SequentialExecutionNode::Dropout {
                    inv_p,
                    data_start,
                    data_end,
                    mask_start,
                    mask_end,
                    ..
                } => {
                    let dim = data_end - data_start;
                    let da_slice = &read_buf[..(dim * self.batch_size)];
                    let dz_slice = &mut write_buf[..(dim * self.batch_size)];

                    let m_start = mask_start * self.batch_size;
                    let m_end = mask_end * self.batch_size;

                    for ((dz, da), mask) in dz_slice
                        .iter_mut()
                        .zip(da_slice.iter())
                        .zip(self.activations[m_start..m_end].iter())
                    {
                        *dz = da * mask * inv_p;
                    }
                }
                SequentialExecutionNode::Relu { a_start, a_end } => {
                    let dim = a_end - a_start;
                    let start = a_start * self.batch_size;
                    let end = a_end * self.batch_size;

                    let da_slice = &read_buf[..(dim * self.batch_size)];
                    let dz_slice = &mut write_buf[..(dim * self.batch_size)];

                    for ((da, dz), a) in da_slice
                        .iter()
                        .zip(dz_slice.iter_mut())
                        .zip(self.activations[start..end].iter())
                    {
                        let derivative = if *a > 0.0 { 1.0 } else { 0.0 };
                        *dz = da * derivative;
                    }
                }
                SequentialExecutionNode::Sigmoid { a_start, a_end } => {
                    let dim = a_end - a_start;
                    let start = a_start * self.batch_size;
                    let end = a_end * self.batch_size;

                    let da_slice = &read_buf[..(dim * self.batch_size)];
                    let dz_slice = &mut write_buf[..(dim * self.batch_size)];

                    for ((da, dz), a) in da_slice
                        .iter()
                        .zip(dz_slice.iter_mut())
                        .zip(self.activations[start..end].iter())
                    {
                        let derivative = a * (1.0 - a);
                        *dz = da * derivative;
                    }
                }
                SequentialExecutionNode::SoftmaxCrossEntropy { a_start, a_end, .. } => {
                    let start = a_start * self.batch_size;
                    let end = a_end * self.batch_size;
                    let predictions = &self.activations[start..end];

                    let dz_slice = &mut write_buf[..predictions.len()];
                    for ((dz, p), target) in
                        dz_slice.iter_mut().zip(predictions.iter()).zip(y.iter())
                    {
                        *dz = p - target;
                    }
                }
            };

            std::mem::swap(read_buf, write_buf);
        }

        &self.gradients
    }
}

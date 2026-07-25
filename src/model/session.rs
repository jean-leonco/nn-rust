use std::simd::prelude::*;

use crate::{
    core::cbrng,
    model::DefinitionGraph,
    ops::{Op, dense, dropout, relu, sigmoid, softmax},
};

/// A session for executing a sequential model.
#[derive(Debug)]
pub struct Session {
    /// The batch size using between runs. It can't be changed between calls.
    batch_size: usize,

    /// Activations for each layer, used for backpropagation.
    /// Same as sum(nodes.output if node is Dense)
    activations: Vec<f32>,

    /// Masks for dropout layers.
    masks: Vec<u8>,

    /// Gradients for optimizer step.
    /// Same size as weights, since it stores the dW and dB gradients.
    gradients: Vec<f32>,

    /// Ping-pong buffer for gradients, required for backpropagation.
    /// Stores dA & dZ.
    gradient_buffer: (Vec<f32>, Vec<f32>),

    /// Current step. Advanced each time `forward` is called.
    step: usize,

    /// Philox key schedule.
    key_schedule: cbrng::KeySchedule,
}

impl Session {
    pub fn new(graph: &DefinitionGraph, batch_size: usize, seed: Option<[u32; 2]>) -> Self {
        let activations = vec![0.0; batch_size * graph.activation_size];
        let gradients = vec![0.0; graph.params_size];
        let masks = vec![0u8; batch_size * graph.mask_size];

        let session_seed = match seed {
            Some(seed) => [u32x8::splat(seed[0]), u32x8::splat(seed[1])],
            None => [u32x8::splat(0), u32x8::splat(0)],
        };

        Self {
            batch_size,
            activations,
            masks,
            step: 0,
            gradients,
            gradient_buffer: (
                vec![0.0; batch_size * graph.max_dimension],
                vec![0.0; batch_size * graph.max_dimension],
            ),
            key_schedule: if graph
                .train_ops
                .iter()
                .any(|op| matches!(op, Op::Dropout(_)))
            {
                cbrng::build_key_schedule(session_seed)
            } else {
                cbrng::EMPTY_KEY_SCHEDULE
            },
        }
    }

    /// Runs the forward pass of the model and updates the activations.
    pub fn forward(&mut self, ops: &Vec<Op>, params: &mut [f32], x: &[f32]) -> &[f32] {
        let mut output_start = 0;
        let mut output_end = 0;
        self.step += 1;

        for op in ops {
            match op {
                Op::Input(meta) => {
                    let split_offset = meta.activations_split_offset(self.batch_size);
                    let output = &mut self.activations[split_offset..];

                    let output_slice = &mut output[meta.output_offsets(self.batch_size)];

                    let layer_weights = &params[meta.weight_offsets.clone()];
                    let layer_bias = &params[meta.bias_offsets.clone()];

                    dense::forward(
                        meta,
                        self.batch_size,
                        x,
                        layer_weights,
                        layer_bias,
                        output_slice,
                    );
                }
                Op::Dense(meta) => {
                    let (input, output) = self
                        .activations
                        .split_at_mut(meta.activations_split_offset(self.batch_size));

                    let input_slice = &input[meta.input_offsets(self.batch_size)];
                    let output_slice = &mut output[meta.output_offsets(self.batch_size)];

                    let layer_weights = &params[meta.weight_offsets.clone()];
                    let layer_bias = &params[meta.bias_offsets.clone()];

                    dense::forward(
                        meta,
                        self.batch_size,
                        input_slice,
                        layer_weights,
                        layer_bias,
                        output_slice,
                    );
                }
                Op::Dropout(meta) => {
                    let activations =
                        &mut self.activations[meta.activation_offsets(self.batch_size)];
                    let masks = &mut self.masks[meta.mask_offsets(self.batch_size)];

                    dropout::forward(meta, activations, masks, self.step, &self.key_schedule);
                }
                Op::Relu(meta) => {
                    let activations =
                        &mut self.activations[meta.activation_offsets(self.batch_size)];

                    relu::forward(activations);
                }
                Op::Sigmoid(meta) => {
                    let activations =
                        &mut self.activations[meta.activation_offsets(self.batch_size)];

                    sigmoid::forward(activations);
                }
                Op::Softmax(meta) => {
                    let activation_offsets = meta.activation_offsets(self.batch_size);
                    output_start = activation_offsets.start;
                    output_end = activation_offsets.end;

                    let activations = &mut self.activations[activation_offsets];
                    softmax::forward(meta, activations);
                }
            }
        }

        &self.activations[output_start..output_end]
    }

    /// Runs the backward pass of the model and updates the gradients.
    pub fn backward(&mut self, ops: &[Op], params: &[f32], x: &[f32], y: &[f32]) -> &[f32] {
        for op in ops.iter().rev() {
            let (read_buf, write_buf) = &mut self.gradient_buffer;

            match op {
                Op::Input(meta) => {
                    let layer_gradients =
                        &mut self.gradients[meta.weight_offsets.start..meta.bias_offsets.end];
                    let w_len = meta.weight_offsets.end - meta.weight_offsets.start;

                    let (dw, db) = layer_gradients.split_at_mut(w_len);
                    let dz = &read_buf[meta.dz_offsets(self.batch_size)];

                    dense::backward_parameters(meta, self.batch_size, dw, db, dz, x);
                }
                Op::Dense(meta) => {
                    let layer_gradients =
                        &mut self.gradients[meta.weight_offsets.start..meta.bias_offsets.end];
                    let w_len = meta.weight_offsets.end - meta.weight_offsets.start;

                    let (dw, db) = layer_gradients.split_at_mut(w_len);
                    let dz = &read_buf[meta.dz_offsets(self.batch_size)];

                    let activations =
                        &self.activations[..meta.activations_split_offset(self.batch_size)];
                    let input_slice = &activations[meta.input_offsets(self.batch_size)];

                    dense::backward_parameters(meta, self.batch_size, dw, db, dz, input_slice);

                    let da = &mut write_buf[meta.da_offsets(self.batch_size)];
                    let layer_weights = &params[meta.weight_offsets.clone()];
                    dense::backward_input(meta, self.batch_size, da, dz, layer_weights);
                }
                Op::Dropout(meta) => {
                    let dz = &mut write_buf[meta.gradient_offsets(self.batch_size)];
                    let da = &read_buf[meta.gradient_offsets(self.batch_size)];
                    let masks = &self.masks[meta.mask_offsets(self.batch_size)];

                    dropout::backward(meta, dz, da, masks);
                }
                Op::Relu(meta) => {
                    let dz = &mut write_buf[meta.gradient_offsets(self.batch_size)];
                    let da = &read_buf[meta.gradient_offsets(self.batch_size)];
                    let activations = &self.activations[meta.activation_offsets(self.batch_size)];

                    relu::backward(dz, da, activations);
                }
                Op::Sigmoid(meta) => {
                    let dz = &mut write_buf[meta.gradient_offsets(self.batch_size)];
                    let da = &read_buf[meta.gradient_offsets(self.batch_size)];
                    let activations = &self.activations[meta.activation_offsets(self.batch_size)];

                    sigmoid::backward(dz, da, activations);
                }
                Op::Softmax(meta) => {
                    let predictions = &self.activations[meta.activation_offsets(self.batch_size)];
                    let dz = &mut write_buf[..predictions.len()];

                    softmax::backward(dz, predictions, y);
                }
            }

            std::mem::swap(read_buf, write_buf);
        }

        &self.gradients
    }
}

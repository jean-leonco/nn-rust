use std::simd::prelude::*;

use crate::{
    core::cbrng,
    model::SequentialModel,
    ops::{Operation, dense, dropout, relu, sigmoid, softmax},
};

/// Reusable execution buffers for a sequential model at a fixed batch size.
#[derive(Debug)]
pub struct Session {
    batch_size: usize,
    activations: Vec<f32>,
    masks: Vec<u8>,
    gradients: Vec<f32>,
    gradient_buffer: (Vec<f32>, Vec<f32>),
    step: usize,
    key_schedule: cbrng::KeySchedule,
}

impl Session {
    /// Allocates buffers for `model` at `batch_size`.
    pub fn new(model: &SequentialModel, batch_size: usize, seed: Option<[u32; 2]>) -> Self {
        let activations = vec![0.0; batch_size * model.layout.activations_len];
        let gradients = vec![0.0; model.layout.params_len];
        let masks = vec![0; batch_size * model.layout.masks_len];

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
                vec![0.0; batch_size * model.layout.max_neurons],
                vec![0.0; batch_size * model.layout.max_neurons],
            ),
            key_schedule: if model
                .train_ops
                .iter()
                .any(|op| matches!(op, Operation::Dropout(_)))
            {
                cbrng::build_key_schedule(session_seed)
            } else {
                cbrng::EMPTY_KEY_SCHEDULE
            },
        }
    }

    /// Forward pass. Returns the output slice of the last layer.
    pub fn forward(&mut self, ops: &[Operation], params: &mut [f32], x: &[f32]) -> &[f32] {
        let mut output_start = 0;
        let mut output_end = 0;
        self.step += 1;

        for op in ops {
            match op {
                Operation::Input(meta) => {
                    let split_offset = meta.activation_split_offset(self.batch_size);
                    let output = &mut self.activations[split_offset..];

                    let output_slice = &mut output[meta.output_range(self.batch_size)];

                    let layer_weights = &params[meta.weight_range.clone()];
                    let layer_bias = &params[meta.bias_range.clone()];

                    dense::forward(
                        meta,
                        self.batch_size,
                        x,
                        layer_weights,
                        layer_bias,
                        output_slice,
                    );
                }
                Operation::Dense(meta) => {
                    let (input, output) = self
                        .activations
                        .split_at_mut(meta.activation_split_offset(self.batch_size));

                    let input_slice = &input[meta.input_range(self.batch_size)];
                    let output_slice = &mut output[meta.output_range(self.batch_size)];

                    let layer_weights = &params[meta.weight_range.clone()];
                    let layer_bias = &params[meta.bias_range.clone()];

                    dense::forward(
                        meta,
                        self.batch_size,
                        input_slice,
                        layer_weights,
                        layer_bias,
                        output_slice,
                    );
                }
                Operation::Dropout(meta) => {
                    let activations = &mut self.activations[meta.activation_range(self.batch_size)];
                    let masks = &mut self.masks[meta.mask_range(self.batch_size)];

                    dropout::forward(meta, activations, masks, self.step, &self.key_schedule);
                }
                Operation::Relu(meta) => {
                    let activations = &mut self.activations[meta.activation_range(self.batch_size)];

                    relu::forward(activations);
                }
                Operation::Sigmoid(meta) => {
                    let activations = &mut self.activations[meta.activation_range(self.batch_size)];

                    sigmoid::forward(activations);
                }
                Operation::Softmax(meta) => {
                    let activation_range = meta.activation_range(self.batch_size);
                    output_start = activation_range.start;
                    output_end = activation_range.end;

                    let activations = &mut self.activations[activation_range];
                    softmax::forward(meta, activations);
                }
            }
        }

        &self.activations[output_start..output_end]
    }

    /// Backward pass. Returns parameter gradients.
    pub fn backward(&mut self, ops: &[Operation], params: &[f32], x: &[f32], y: &[f32]) -> &[f32] {
        for op in ops.iter().rev() {
            let (read_buf, write_buf) = &mut self.gradient_buffer;

            match op {
                Operation::Input(meta) => {
                    let layer_gradients =
                        &mut self.gradients[meta.weight_range.start..meta.bias_range.end];
                    let weights_len = meta.weight_range.end - meta.weight_range.start;

                    let (dw, db) = layer_gradients.split_at_mut(weights_len);
                    let dz = &read_buf[meta.gradient_range(self.batch_size)];

                    dense::backward_parameters(meta, self.batch_size, dw, db, dz, x);
                }
                Operation::Dense(meta) => {
                    let layer_gradients =
                        &mut self.gradients[meta.weight_range.start..meta.bias_range.end];
                    let weights_len = meta.weight_range.end - meta.weight_range.start;

                    let (dw, db) = layer_gradients.split_at_mut(weights_len);
                    let dz = &read_buf[meta.gradient_range(self.batch_size)];

                    let activations =
                        &self.activations[..meta.activation_split_offset(self.batch_size)];
                    let input_slice = &activations[meta.input_range(self.batch_size)];

                    dense::backward_parameters(meta, self.batch_size, dw, db, dz, input_slice);

                    let da = &mut write_buf[meta.input_gradient_range(self.batch_size)];
                    let layer_weights = &params[meta.weight_range.clone()];
                    dense::backward_input(meta, self.batch_size, da, dz, layer_weights);
                }
                Operation::Dropout(meta) => {
                    let dz = &mut write_buf[meta.gradient_range(self.batch_size)];
                    let da = &read_buf[meta.gradient_range(self.batch_size)];
                    let masks = &self.masks[meta.mask_range(self.batch_size)];

                    dropout::backward(meta, dz, da, masks);
                }
                Operation::Relu(meta) => {
                    let dz = &mut write_buf[meta.gradient_range(self.batch_size)];
                    let da = &read_buf[meta.gradient_range(self.batch_size)];
                    let activations = &self.activations[meta.activation_range(self.batch_size)];

                    relu::backward(dz, da, activations);
                }
                Operation::Sigmoid(meta) => {
                    let dz = &mut write_buf[meta.gradient_range(self.batch_size)];
                    let da = &read_buf[meta.gradient_range(self.batch_size)];
                    let activations = &self.activations[meta.activation_range(self.batch_size)];

                    sigmoid::backward(dz, da, activations);
                }
                Operation::Softmax(meta) => {
                    let predictions = &self.activations[meta.activation_range(self.batch_size)];
                    let dz = &mut write_buf[..predictions.len()];

                    softmax::backward(dz, predictions, y);
                }
            }

            std::mem::swap(read_buf, write_buf);
        }

        &self.gradients
    }
}

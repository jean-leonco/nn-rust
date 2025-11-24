use std::{f32, fs::File};

use crate::{activation::ActivationFn, dataloader::Dataloader, metrics};
use bincode;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NeuralNetworkEncodeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Encode error: {0}")]
    Encode(#[from] bincode::error::EncodeError),
}

#[derive(Debug, Error)]
pub enum NeuralNetworkDecodeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Decode error: {0}")]
    Decode(#[from] bincode::error::DecodeError),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork<A: ActivationFn> {
    weights: Vec<Array2<f32>>,
    num_of_layers: usize,
    bias: Vec<Array1<f32>>,
    activation_fn: A,
}

impl<A: ActivationFn + Serialize + for<'a> Deserialize<'a>> NeuralNetwork<A> {
    pub fn new(topology: &[usize], activation_fn: A) -> Self {
        let num_of_connections = topology.len() - 1;

        let mut bias = Vec::with_capacity(num_of_connections);
        let mut weights = Vec::with_capacity(num_of_connections);

        for i in 0..num_of_connections {
            let input_size = topology[i];
            let output_size = topology[i + 1];

            weights.push(A::init_weights(input_size, output_size));

            bias.push(Array1::zeros(output_size));
        }

        Self {
            num_of_layers: topology.len(),
            weights,
            bias,
            activation_fn,
        }
    }

    pub fn save(&self, path: &str) -> Result<(), NeuralNetworkEncodeError> {
        let mut f = File::create(path)?;
        bincode::serde::encode_into_std_write(self, &mut f, bincode::config::standard())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, NeuralNetworkDecodeError> {
        let mut f = File::open(path)?;
        let decoded = bincode::serde::decode_from_std_read(&mut f, bincode::config::standard())?;
        Ok(decoded)
    }

    fn forward_pass(
        weights: &[Array2<f32>],
        bias: &[Array1<f32>],
        num_of_layers: usize,
        input: &ArrayView2<f32>,
        activation_fn: &mut A,
    ) {
        activation_fn.init_layers(input);

        for i in 0..num_of_layers - 1 {
            let bias = &bias[i];
            let weight = &weights[i];
            let x = activation_fn.get_activation(i);

            let z = x.dot(weight) + bias;

            if i == num_of_layers - 2 {
                activation_fn.set_output(Self::softmax(&z).view());
            } else {
                activation_fn.activate(i, &z);
            };
        }
    }

    fn softmax(z: &Array2<f32>) -> Array2<f32> {
        let max = z.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let shifted = z - max.insert_axis(Axis(1));
        let exp = shifted.mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp / sum
    }

    pub fn predict(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let mut activation_fn = A::new(self.num_of_layers);

        Self::forward_pass(
            &self.weights,
            &self.bias,
            self.num_of_layers,
            input,
            &mut activation_fn,
        );
        activation_fn.get_output().to_owned()
    }

    fn forward_propagation(&mut self, input: &ArrayView2<f32>) {
        Self::forward_pass(
            &self.weights,
            &self.bias,
            self.num_of_layers,
            input,
            &mut self.activation_fn,
        );
    }

    fn backward_propagation(&mut self, y: &ArrayView2<f32>, learning_rate: f32) {
        let output_activation = self.activation_fn.get_output();

        let batch_size = y.nrows() as f32;
        let mut dz = &output_activation - y;

        for i in (0..self.num_of_layers - 1).rev() {
            let a = self.activation_fn.get_activation(i);
            let weight_gradient = a.t().dot(&dz) / batch_size;
            let bias_gradient = dz.sum_axis(Axis(0)) / batch_size;

            if i > 0 {
                let foo = &dz.dot(&self.weights[i].t());
                dz = foo * self.activation_fn.derivative(i);
            }

            Zip::from(&mut self.weights[i])
                .and(&weight_gradient)
                .for_each(|w, &g| *w -= g * learning_rate);

            Zip::from(&mut self.bias[i])
                .and(&bias_gradient)
                .for_each(|w, &g| *w -= g * learning_rate);
        }
    }

    pub fn train(
        &mut self,
        loader: &mut impl for<'a> Dataloader<'a>,
        epoch: usize,
        learning_rate: f32,
    ) {
        for e in 0..epoch {
            println!("Starting epoch {}/{}", e + 1, epoch);

            let mut total_loss = 0.0;
            let mut total_correct = 0.0;
            let mut samples = 0;

            let num_of_batches = loader.num_of_batches();
            for (i, (x, y)) in loader.train_batches().enumerate() {
                let curr_batch_size = x.nrows() as f32;

                if i % 50 == 0 {
                    println!("Batch {i}/{num_of_batches}");
                }

                samples += x.nrows();

                self.forward_propagation(&x);
                self.backward_propagation(&y, learning_rate);
                let output = self.activation_fn.get_output();

                total_loss += metrics::cross_entropy_loss(&output, &y) * curr_batch_size;
                total_correct += metrics::accuracy(&output, &y) * curr_batch_size;
            }

            let loss = total_loss / samples as f32;
            let accuracy = total_correct / samples as f32;
            let (eval_loss, eval_accuracy) = self.eval(loader);

            println!(
                "Epoch {}/{} — Loss: {:.4}, Accuracy: {:.4}, Eval Loss: {:.4}, Eval Accuracy: {:.4}",
                e + 1,
                epoch,
                loss,
                accuracy,
                eval_loss,
                eval_accuracy
            );
        }

        let (training_loss, training_accuracy) = self.eval(loader);
        println!(
            "Training finished — Validation Loss: {training_loss:.4}, Validation Accuracy: {training_accuracy:.4}"
        );
    }

    fn eval(&self, loader: &mut impl for<'a> Dataloader<'a>) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_correct = 0.0;
        let mut samples = 0;

        for (x, y) in loader.validation_batches() {
            let curr_batch_size = x.nrows() as f32;

            samples += x.nrows();

            let prediction = self.predict(&x);
            total_loss += metrics::cross_entropy_loss(&prediction.view(), &y) * curr_batch_size;
            total_correct += metrics::accuracy(&prediction.view(), &y) * curr_batch_size;
        }

        let loss = total_loss / samples as f32;
        let accuracy = total_correct / samples as f32;

        (loss, accuracy)
    }
}

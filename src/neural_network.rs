use std::f32;

use crate::dataset::TrainingSet;
use ndarray::{Array1, Array2, Axis, Zip};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct NeuralNetwork {
    weights: Vec<Array2<f32>>,
    num_of_layers: usize,
    bias: Vec<Array1<f32>>,
    activation_cache: Vec<Array2<f32>>,
}

impl NeuralNetwork {
    pub fn new(topology: &[usize]) -> Self {
        let num_connections = topology.len() - 1;

        let mut bias = Vec::with_capacity(num_connections);
        let mut weights = Vec::with_capacity(num_connections);
        let activation_cache = Vec::with_capacity(topology.len());

        for i in 0..num_connections {
            let input_size = topology[i];
            let output_size = topology[i + 1];

            weights.push(Self::xavier(input_size, output_size));
            bias.push(Array1::zeros(output_size));
        }

        Self {
            num_of_layers: weights.len(),
            weights,
            bias,
            activation_cache,
        }
    }

    fn xavier(input_size: usize, output_size: usize) -> Array2<f32> {
        let std_dev = 2.0 / ((input_size + output_size) as f32);
        let std_dev = std_dev.sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        Array2::random((input_size, output_size), normal)
    }

    fn forward_pass(
        weights: &[Array2<f32>],
        bias: &[Array1<f32>],
        num_of_layers: usize,
        input: Array2<f32>,
        cache: &mut Vec<Array2<f32>>,
    ) {
        cache.push(input);

        for i in 0..num_of_layers {
            let x = &cache[i];
            let bias = &bias[i];
            let weight = &weights[i];

            let z = x.dot(weight) + bias;

            let a = if i == num_of_layers - 1 {
                Self::softmax(&z)
            } else {
                Self::sigmoid(&z)
            };

            cache.push(a);
        }
    }

    pub fn predict(&self, input: Array2<f32>) -> Array2<f32> {
        let mut cache = Vec::with_capacity(self.num_of_layers);
        Self::forward_pass(
            &self.weights,
            &self.bias,
            self.num_of_layers,
            input,
            &mut cache,
        );
        cache.last().unwrap().to_owned()
    }

    fn forward_propagation(&mut self, input: Array2<f32>) {
        self.activation_cache.clear();
        Self::forward_pass(
            &self.weights,
            &self.bias,
            self.num_of_layers,
            input,
            &mut self.activation_cache,
        );
    }

    fn sigmoid(z: &Array2<f32>) -> Array2<f32> {
        z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
        x * (1.0 - x)
    }

    fn softmax(z: &Array2<f32>) -> Array2<f32> {
        let max = z.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let shifted = z - max.insert_axis(Axis(1));
        let exp = shifted.mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp / sum
    }

    fn backward_propagation(&mut self, y: Array2<f32>, learning_rate: f32) {
        let output_activation = self.activation_cache.last().unwrap();

        let batch_size = y.nrows() as f32;
        let mut dz = output_activation - y;

        for i in (0..self.num_of_layers).rev() {
            let a = &self.activation_cache[i];

            let weight_gradient = a.t().dot(&dz) / batch_size;
            let bias_gradient = dz.sum_axis(Axis(0)) / batch_size;

            if i > 0 {
                dz = &dz.dot(&self.weights[i].t()) * Self::sigmoid_derivative(a);
            }

            self.weights[i] -= &(weight_gradient * learning_rate);
            self.bias[i] -= &(bias_gradient * learning_rate);
        }
    }

    pub fn train(
        &mut self,
        dataset: &TrainingSet,
        epoch: usize,
        batch_size: usize,
        learning_rate: f32,
    ) {
        for e in 0..epoch {
            println!("Starting epoch {}/{}", e + 1, epoch);

            let mut total_loss = 0.0;
            let mut total_correct = 0.0;
            let mut samples = 0.0;

            let mut indices: Vec<usize> = (0..dataset.train.data.nrows()).collect();
            indices.shuffle(&mut rand::rng());

            for (i, batch) in indices.chunks(batch_size).enumerate() {
                if i % 50 == 0 {
                    println!("Batch {}/{}", i, indices.len() / batch_size);
                }

                let curr_batch_size = batch.len() as f32;

                let x = dataset.train.data.select(Axis(0), batch);
                let y = dataset.train.labels.select(Axis(0), batch);

                samples += x.nrows() as f32;

                self.forward_propagation(x);
                self.backward_propagation(y.clone(), learning_rate);

                let output = self.activation_cache.last().unwrap();

                total_loss += Self::cross_entropy_loss(output, &y) * curr_batch_size;
                total_correct += Self::accuracy(output, &y) * curr_batch_size;
            }

            let loss = total_loss / samples;
            let accuracy = total_correct / samples;

            println!(
                "Epoch {}/{} — Loss: {:.4}, Accuracy: {:.4}",
                e + 1,
                epoch,
                loss,
                accuracy
            );
        }
    }

    fn cross_entropy_loss(output: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let probabilities = (output + 1e-8).mapv(f32::ln);
        let loss = -(y * &probabilities).sum();
        loss / output.nrows() as f32
    }

    fn accuracy(output: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let matches = Zip::from(&Self::argmax(output))
            .and(&Self::argmax(y))
            .map_collect(|&a, &b| a == b)
            .map(|&b| b as usize)
            .sum();

        matches as f32 / output.nrows() as f32
    }

    fn argmax(x: &Array2<f32>) -> Array1<usize> {
        x.axis_iter(Axis(0))
            .map(|row| {
                let (max_idx, _) = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, x), (_, y)| x.total_cmp(y))
                    .unwrap();
                max_idx
            })
            .collect()
    }
}

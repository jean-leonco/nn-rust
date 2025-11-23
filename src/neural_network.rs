use std::f32;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal};

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

    pub fn forward_propagation(&mut self, input: Array2<f32>) {
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

    pub fn backward_propagation(&mut self, target: Array2<f32>, learning_rate: f32) {
        let output_activation = self.activation_cache.last().unwrap();

        let batch_size = target.nrows() as f32;
        let mut dz = output_activation - target;

        for i in (0..self.num_of_layers).rev() {
            let a_prev = &self.activation_cache[i];

            let weight_gradient = a_prev.t().dot(&dz) / batch_size;
            let bias_gradient = dz.sum_axis(Axis(0)) / batch_size;

            if i > 0 {
                dz = &dz.dot(&self.weights[i].t()) * Self::sigmoid_derivative(a_prev);
            }

            self.weights[i] = &self.weights[i] - (learning_rate * weight_gradient);
            self.bias[i] = &self.bias[i] - (learning_rate * bias_gradient);
        }
    }
}

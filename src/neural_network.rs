use std::f32;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal};

#[derive(Debug)]
pub struct NeuralNetwork {
    weights: Vec<Array2<f32>>,
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

    pub fn forward_propagation(&mut self, input: Array2<f32>) {
        self.activation_cache.clear();
        self.activation_cache.push(input);

        let num_of_layers = self.weights.len();
        for i in 0..num_of_layers {
            let x = &self.activation_cache[i];
            let bias = &self.bias[i];
            let weight = &self.weights[i];

            let z = x.dot(weight) + bias;

            let a = if i == num_of_layers - 1 {
                Self::softmax(&z)
            } else {
                Self::sigmoid(&z)
            };

            self.activation_cache.push(a);
        }
    }

    fn sigmoid(z: &Array2<f32>) -> Array2<f32> {
        z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn softmax(z: &Array2<f32>) -> Array2<f32> {
        let max = z.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let shifted = z - max.insert_axis(Axis(1));
        let exp = shifted.mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp / sum
    }
}

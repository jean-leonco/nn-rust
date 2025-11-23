use std::f32;

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};

use crate::dataset::NUM_OF_CLASSES;

#[derive(Debug)]
pub struct NeuralNetwork {
    weights: Vec<Array2<f32>>,
    bias: Vec<Array1<f32>>,
    activation_cache: Vec<Array2<f32>>,
}

impl NeuralNetwork {
    pub fn new(hidden_layers: usize, input: usize, neurons: usize) -> Self {
        let mut rng = rand::rng();

        let num_connections = hidden_layers + 1;
        let output_layer_idx = num_connections - 1;

        let mut bias = Vec::with_capacity(num_connections);
        let mut weights = Vec::with_capacity(num_connections);
        let activation_cache = Vec::with_capacity(hidden_layers + 2);

        for i in 0..num_connections {
            let weights_shape = if i == 0 {
                (input, neurons)
            } else if i == output_layer_idx {
                (neurons, NUM_OF_CLASSES)
            } else {
                (neurons, neurons)
            };

            let std_dev = 2.0 / ((weights_shape.0 + weights_shape.1) as f32);
            let std_dev = std_dev.sqrt();
            let normal = Normal::new(0.0, std_dev).unwrap();

            let initialized_weights =
                Array2::from_shape_fn(weights_shape, |_| normal.sample(&mut rng));
            weights.push(initialized_weights);

            let bias_shape = if i == output_layer_idx {
                NUM_OF_CLASSES
            } else {
                neurons
            };

            bias.push(Array1::zeros(bias_shape));
        }

        Self {
            weights,
            bias,
            activation_cache,
        }
    }

    pub fn forward_propagation(&mut self, input: Array2<f32>) {
        self.activation_cache.push(input);

        let num_of_layers = self.weights.len();
        for i in 0..num_of_layers {
            let x = &self.activation_cache[i];
            let bias = &self.bias[i];
            let weight = &self.weights[i];

            let z = x.dot(weight) + bias;

            let a = if i < num_of_layers - 1 {
                Self::sigmoid(&z)
            } else {
                Self::softmax(&z)
            };

            self.activation_cache.push(a);
        }
    }

    fn sigmoid(z: &Array2<f32>) -> Array2<f32> {
        z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn softmax(z: &Array2<f32>) -> Array2<f32> {
        let max = z.map_axis(Axis(1), |col| col.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let shifted = z + max.insert_axis(Axis(1));
        let exp = shifted.mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp / sum
    }
}

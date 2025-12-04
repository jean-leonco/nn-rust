use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis, linalg::general_mat_mul};
use ndarray_rand::{RandomExt, rand_distr::Normal};

use crate::layer::{Layer, LayerParams, LayerType};

#[derive(Debug)]
pub struct Dense {
    pub(crate) weights: Array2<f32>,
    pub(crate) bias: Array1<f32>,
    pub(crate) x: Array2<f32>,
}

#[derive(Debug, Clone, Copy)]
pub enum Initialization {
    Xavier,
    He,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize, initialization: Initialization) -> Self {
        Self {
            weights: match initialization {
                Initialization::Xavier => {
                    let std_dev = 2.0 / ((input_size + output_size) as f32);
                    let std_dev = std_dev.sqrt();
                    let normal = Normal::new(0.0, std_dev).unwrap();
                    Array2::random((input_size, output_size), normal)
                }
                Initialization::He => {
                    let std_dev = 2.0 / (input_size as f32);
                    let std_dev = std_dev.sqrt();
                    let normal = Normal::new(0.0, std_dev).unwrap();
                    Array2::random((input_size, output_size), normal)
                }
            },
            bias: Array1::zeros(output_size),
            x: Array2::zeros((0, 0)),
        }
    }

    pub fn from_params(weights: Array2<f32>, bias: Array1<f32>) -> Self {
        Self {
            weights,
            bias,
            x: Array2::zeros((0, 0)),
        }
    }
}

impl Layer for Dense {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Dense
    }

    fn get_params(&self) -> Option<LayerParams<'_>> {
        Some(LayerParams {
            weights: self.weights.view(),
            bias: self.bias.view(),
        })
    }

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        output.assign(&self.bias);
        // z = W * x + b
        general_mat_mul(1.0, input, &self.weights, 1.0, output);
    }

    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        if self.x.shape() != input.shape() {
            self.x = Array2::zeros((input.nrows(), input.ncols()));
        }

        self.x.assign(input);

        output.assign(&self.bias);
        // z = W * x + b
        general_mat_mul(1.0, input, &self.weights, 1.0, output);
    }

    fn backward(
        &mut self,
        grad_input: &mut ArrayViewMut2<f32>,
        grad_output: &ArrayView2<f32>,
        learning_rate: f32,
    ) {
        let batch_size = grad_output.nrows() as f32;
        let learning_rate = learning_rate / batch_size;

        // dX = dY · W^T
        general_mat_mul(1.0, grad_output, &self.weights.t(), 0.0, grad_input);

        // dW = X^T · dY
        general_mat_mul(
            -learning_rate,
            &self.x.t(),
            grad_output,
            1.0,
            &mut self.weights.view_mut(),
        );

        // db = sum(dY)
        let bias_gradient = grad_output.sum_axis(Axis(0));
        self.bias.scaled_add(-learning_rate, &bias_gradient);
    }
}

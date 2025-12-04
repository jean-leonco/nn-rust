use ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};

use crate::layer::{Layer, LayerParams, LayerType};

#[derive(Debug)]
pub struct Sigmoid {
    pub(crate) a: Array2<f32>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {
            a: Array2::zeros((0, 0)),
        }
    }
}

impl Layer for Sigmoid {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Sigmoid
    }

    fn get_params(&self) -> Option<LayerParams<'_>> {
        None
    }

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        output.assign(&input);

        // f(x) = (1 / (1 + e ^ -x))
        output.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    }

    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        if self.a.shape() != input.shape() {
            self.a = Array2::zeros((input.nrows(), input.ncols()));
        }

        self.a.assign(&input);

        // f(x) = (1 / (1 + e ^ -x))
        self.a.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        output.assign(&self.a);
    }

    fn backward(
        &mut self,
        grad_input: &mut ArrayViewMut2<f32>,
        grad_output: &ArrayView2<f32>,
        _learning_rate: f32,
    ) {
        // f'(x) = f(x) * (1.0 - f(x)) = a * (1.0 - a)
        Zip::from(grad_input)
            .and(grad_output)
            .and(&self.a)
            .for_each(|d_in, d_out, a| {
                *d_in = d_out * (a * (1.0 - a));
            });
    }
}

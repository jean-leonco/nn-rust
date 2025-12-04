use ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};

use crate::layer::{Layer, LayerParams, LayerType};

#[derive(Debug)]
pub struct Relu {
    pub(crate) z: Array2<f32>,
}

impl Default for Relu {
    fn default() -> Self {
        Self::new()
    }
}

impl Relu {
    pub fn new() -> Self {
        Self {
            z: Array2::zeros((0, 0)),
        }
    }
}

impl Layer for Relu {
    fn get_layer_type(&self) -> LayerType {
        LayerType::Relu
    }

    fn get_params(&self) -> Option<LayerParams<'_>> {
        None
    }

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        output.assign(input);

        // f(x) = max(0,x)
        output.mapv_inplace(|x| x.max(0.0));
    }

    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        if self.z.shape() != input.shape() {
            self.z = Array2::zeros((input.nrows(), input.ncols()));
        }

        output.assign(input);

        // f(x) = max(0,x)
        output.mapv_inplace(|x| x.max(0.0));

        self.z.assign(output);
    }

    fn backward(
        &mut self,
        grad_input: &mut ArrayViewMut2<f32>,
        grad_output: &ArrayView2<f32>,
        _learning_rate: f32,
    ) {
        // f'(x) = 0 if x < 0
        //         1 if x > 0
        Zip::from(grad_input)
            .and(grad_output)
            .and(&self.z)
            .for_each(|d_in, &d_out, z| {
                *d_in = if *z > 0.0 { d_out } else { 0.0 };
            });
    }
}

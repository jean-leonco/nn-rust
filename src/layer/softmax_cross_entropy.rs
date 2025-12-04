use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

use crate::layer::{Layer, LayerParams, LayerType};

#[derive(Debug)]
pub struct SoftmaxCrossEntropy {
    pub(crate) predicted: Array2<f32>,
}

impl Default for SoftmaxCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftmaxCrossEntropy {
    pub fn new() -> Self {
        Self {
            predicted: Array2::zeros((0, 0)),
        }
    }
}

impl Layer for SoftmaxCrossEntropy {
    fn get_layer_type(&self) -> LayerType {
        LayerType::SoftmaxCrossEntropy
    }

    fn get_params(&self) -> Option<LayerParams<'_>> {
        None
    }

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        let max = input.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let shifted = input - max.insert_axis(Axis(1));
        let exp = shifted.mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        output.assign(&(exp / sum));
    }

    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        if self.predicted.shape() != input.shape() {
            self.predicted = Array2::zeros((input.nrows(), input.ncols()));
        }

        let max = input.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let shifted = input - max.insert_axis(Axis(1));
        let exp = shifted.mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));

        self.predicted.assign(&(exp / sum));
        output.assign(&self.predicted);
    }

    fn backward(
        &mut self,
        grad_input: &mut ArrayViewMut2<f32>,
        grad_output: &ArrayView2<f32>,
        _learning_rate: f32,
    ) {
        grad_input.assign(&(&self.predicted - grad_output));
    }
}

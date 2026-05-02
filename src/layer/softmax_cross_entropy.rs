use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

use crate::{layer::Layer, model::encoder};

#[derive(Debug)]
pub struct SoftmaxCrossEntropy {
    predicted: Array2<f32>,
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
    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), encoder::SerializationError> {
        writer.write_all(&[super::LayerType::SoftmaxCrossEntropy as u8])?;

        Ok(())
    }

    fn read(_reader: &mut impl std::io::Read) -> Result<Self, encoder::SerializationError> {
        Ok(Self::new())
    }

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        output.assign(input);
        for mut row in output.axis_iter_mut(Axis(0)) {
            let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max).exp());
            let sum = row.sum();
            row.mapv_inplace(|x| x / sum);
        }
    }

    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        if self.predicted.shape() != input.shape() {
            self.predicted = Array2::zeros((input.nrows(), input.ncols()));
        }

        self.predicted.assign(input);
        for mut row in self.predicted.axis_iter_mut(Axis(0)) {
            let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max).exp());
            let sum = row.sum();
            row.mapv_inplace(|x| x / sum);
        }
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

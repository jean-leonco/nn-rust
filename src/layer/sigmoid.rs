use ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};

use crate::{layer::Layer, model::encoder};

#[derive(Debug)]
pub struct Sigmoid {
    a: Array2<f32>,
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {
            a: Array2::zeros((0, 0)),
        }
    }
}

impl Layer for Sigmoid {
    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), encoder::SerializationError> {
        writer.write_all(&[super::LayerType::Sigmoid as u8])?;
        Ok(())
    }

    fn read(_reader: &mut impl std::io::Read) -> Result<Self, encoder::SerializationError> {
        Ok(Self::new())
    }

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        Zip::from(output).and(input).for_each(|out, &in_| {
            // f(x) = (1 / (1 + e ^ -x))
            *out = 1.0 / (1.0 + (-in_).exp());
        });
    }

    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        if self.a.shape() != input.shape() {
            self.a = Array2::zeros((input.nrows(), input.ncols()));
        }

        Zip::from(&mut self.a)
            .and(output)
            .and(input)
            .for_each(|a, out, &in_| {
                // f(x) = (1 / (1 + e ^ -x))
                let val = 1.0 / (1.0 + (-in_).exp());
                *a = val;
                *out = val;
            });
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

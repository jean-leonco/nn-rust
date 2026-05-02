use ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};

use crate::{layer::Layer, model::encoder};

#[derive(Debug)]
pub struct Relu {
    z: Array2<f32>,
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
    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), encoder::SerializationError> {
        writer.write_all(&[super::LayerType::Relu as u8])?;
        Ok(())
    }

    fn read(_reader: &mut impl std::io::Read) -> Result<Self, encoder::SerializationError> {
        Ok(Self::new())
    }

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        Zip::from(output).and(input).for_each(|out, &in_| {
            // f(x) = max(0,x)
            *out = in_.max(0.0);
        });
    }

    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        if self.z.shape() != input.shape() {
            self.z = Array2::zeros((input.nrows(), input.ncols()));
        }

        Zip::from(&mut self.z)
            .and(output)
            .and(input)
            .for_each(|z, out, &in_| {
                // f(x) = max(0,x)
                *out = in_.max(0.0);
                *z = *out;
            });
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

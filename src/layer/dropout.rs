use ndarray::{Array2, Zip};
use ndarray_rand::{
    RandomExt,
    rand::{SeedableRng, rngs::SmallRng},
    rand_distr::{Bernoulli, Distribution},
};

use crate::{layer::Layer, model::encoder};

#[derive(Debug)]
pub struct Dropout {
    mask: Array2<u8>,
    distribution: Bernoulli,
    drop_rate: f32,
    inv_p: f32,
    rng: SmallRng,
}

impl Dropout {
    pub fn new(drop_rate: f32) -> Self {
        let p = 1.0 - drop_rate;
        Self {
            mask: Array2::zeros((0, 0)),
            distribution: Bernoulli::new(p as f64).unwrap(),
            drop_rate,
            inv_p: 1.0 / p,
            rng: SmallRng::from_os_rng(),
        }
    }
}

impl Layer for Dropout {
    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), encoder::SerializationError> {
        writer.write_all(&[super::LayerType::Dropout as u8])?;
        encoder::write_f32(writer, self.drop_rate)?;

        Ok(())
    }

    fn read(reader: &mut impl std::io::Read) -> Result<Self, encoder::SerializationError> {
        let drop_rate = encoder::read_f32(reader)?;
        Ok(Self::new(drop_rate))
    }

    fn forward(&self, input: &ndarray::ArrayView2<f32>, output: &mut ndarray::ArrayViewMut2<f32>) {
        output.assign(input);
    }

    fn forward_train(
        &mut self,
        input: &ndarray::ArrayView2<f32>,
        output: &mut ndarray::ArrayViewMut2<f32>,
    ) {
        if self.mask.shape() != input.shape() {
            self.mask =
                Array2::random((input.nrows(), input.ncols()), self.distribution).map(|m| *m as u8);
        } else {
            self.mask.map_inplace(|m| {
                *m = self.distribution.sample(&mut self.rng) as u8;
            });
        }

        // out = activation * mask / (1 - p)
        Zip::from(&mut *output)
            .and(input)
            .and(&self.mask)
            .for_each(|out, &in_, &m| {
                *out = in_ * m as f32 * self.inv_p;
            });
    }

    fn backward(
        &mut self,
        grad_input: &mut ndarray::ArrayViewMut2<f32>,
        grad_output: &ndarray::ArrayView2<f32>,
        _learning_rate: f32,
    ) {
        Zip::from(&mut *grad_input)
            .and(grad_output)
            .and(&self.mask)
            .for_each(|out, &in_, &m| {
                *out = in_ * m as f32 * self.inv_p;
            });
    }
}

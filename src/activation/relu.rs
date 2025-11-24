use ndarray::{Array2, ArrayView2};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use serde::{Deserialize, Serialize};

use crate::activation::{ActivationFn, cache::Cache};

#[derive(Debug, Serialize, Deserialize)]
pub struct ReLuActivationFn {
    num_of_layers: usize,
    activation_cache: Cache,
    pre_activation_cache: Cache,
}

impl ActivationFn for ReLuActivationFn {
    fn new(num_of_layers: usize) -> Self {
        Self {
            num_of_layers,
            activation_cache: Cache::new(num_of_layers),
            pre_activation_cache: Cache::new(num_of_layers),
        }
    }

    fn init_weights(input_size: usize, output_size: usize) -> Array2<f32> {
        let std_dev = 2.0 / (input_size as f32);
        let std_dev = std_dev.sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        Array2::random((input_size, output_size), normal)
    }

    fn init_layers(&mut self, input: &ArrayView2<f32>) {
        self.activation_cache.init_layers(input);
        self.pre_activation_cache.init_layers(input);
    }

    fn get_activation(&self, i: usize) -> ArrayView2<f32> {
        self.activation_cache.get(i)
    }

    fn activate(&mut self, i: usize, z: &Array2<f32>) {
        self.pre_activation_cache.set_layer_state(i + 1, &z.view());

        let a = z.mapv(|x| x.max(0.0));
        self.activation_cache.set_layer_state(i + 1, &a.view());
    }

    fn derivative(&self, i: usize) -> Array2<f32> {
        let z = self.pre_activation_cache.get(i);
        z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn set_output(&mut self, a: ArrayView2<f32>) {
        self.activation_cache
            .set_layer_state(self.num_of_layers - 1, &a.view());
    }

    fn get_output(&self) -> ArrayView2<f32> {
        self.activation_cache
            .get_last()
            .expect("Failed to get output")
    }
}

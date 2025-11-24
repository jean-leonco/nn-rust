use ndarray::{Array2, ArrayView2};

pub mod cache;
pub mod relu;
pub mod sigmoid;

pub trait ActivationFn {
    fn new(num_of_layers: usize) -> Self;
    fn init_weights(input_size: usize, output_size: usize) -> Array2<f32>;
    fn init_layers(&mut self, input: &ArrayView2<f32>);
    fn get_activation(&self, i: usize) -> ArrayView2<f32>;
    fn activate(&mut self, i: usize, z: &Array2<f32>);
    fn derivative(&self, i: usize) -> Array2<f32>;
    fn set_output(&mut self, a: ArrayView2<f32>);
    fn get_output(&self) -> ArrayView2<f32>;
}

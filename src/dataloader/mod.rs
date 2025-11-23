use ndarray::Array2;

pub mod mnist_loader;

pub trait Dataloader {
    fn num_of_batches(&self) -> usize;
    fn train_batches(&self) -> impl Iterator<Item = (Array2<f32>, Array2<f32>)>;
    fn validation_batches(&self) -> impl Iterator<Item = (Array2<f32>, Array2<f32>)>;
}

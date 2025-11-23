use ndarray::ArrayView2;

pub mod mnist_loader;

pub trait Dataloader<'a> {
    fn num_of_batches(&self) -> usize;
    fn train_batches(
        &'a mut self,
    ) -> impl Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>;
    fn validation_batches(
        &'a self,
    ) -> impl Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>;
}

use crate::{
    dataloader::Dataloader,
    layer::Layer,
    metrics::{accuracy, cross_entropy_loss},
    model::{
        builder::{ModelBuilder, NoInput},
        encoder::{SerializationError, decode_model, encode_model},
    },
};
use ndarray::{Array2, ArrayView2, s};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    max_layer_dim: usize,
    pub layer_dims: Vec<usize>,
    cached_batch_size: Option<usize>,
    input_buffer: Array2<f32>,
    output_buffer: Array2<f32>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>, layer_dims: Vec<usize>) -> Self {
        Self {
            layers,
            max_layer_dim: *layer_dims
                .iter()
                .max_by(std::cmp::Ord::cmp)
                .expect("Topology must contain at least one layer dimension"),
            layer_dims,
            cached_batch_size: None,
            input_buffer: Array2::zeros((0, 0)),
            output_buffer: Array2::zeros((0, 0)),
        }
    }

    pub fn builder() -> ModelBuilder<NoInput> {
        ModelBuilder::new()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SerializationError> {
        let f = File::create(path)?;
        let mut writer = BufWriter::new(f);

        encode_model(self, &mut writer)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, SerializationError> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        decode_model(&mut reader)
    }

    fn ensure_cache_size(&mut self, batch_size: usize) {
        match self.cached_batch_size {
            Some(value) if value == batch_size => {}
            _ => {
                self.cached_batch_size = Some(batch_size);
                self.input_buffer = Array2::zeros((batch_size, self.max_layer_dim));
                self.output_buffer = Array2::zeros((batch_size, self.max_layer_dim));
            }
        }
    }

    pub fn predict(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let batch_size = x.nrows();

        let mut input_buffer = Array2::zeros((batch_size, self.max_layer_dim));
        let mut output_buffer = Array2::zeros((batch_size, self.max_layer_dim));

        input_buffer.slice_mut(s![.., ..x.ncols()]).assign(x);

        for i in 0..self.layers.len() {
            let input_dim = self.layer_dims[i];
            let output_dim = self.layer_dims[i + 1];

            let input_slice = input_buffer.slice(s![.., ..input_dim]);
            let mut output_slice = output_buffer.slice_mut(s![.., ..output_dim]);

            self.layers[i].forward(&input_slice, &mut output_slice);

            std::mem::swap(&mut input_buffer, &mut output_buffer);
        }

        let output_size = self.layer_dims.last().expect("Unable to get output size");
        input_buffer.slice(s![.., ..*output_size]).to_owned()
    }

    fn forward_train(&mut self, x: &ArrayView2<f32>) -> Array2<f32> {
        self.ensure_cache_size(x.nrows());

        self.input_buffer.slice_mut(s![.., ..x.ncols()]).assign(x);

        for i in 0..self.layers.len() {
            let input_dim = self.layer_dims[i];
            let output_dim = self.layer_dims[i + 1];

            let input_slice = self.input_buffer.slice(s![.., ..input_dim]);
            let mut output_slice = self.output_buffer.slice_mut(s![.., ..output_dim]);

            self.layers[i].forward_train(&input_slice, &mut output_slice);

            std::mem::swap(&mut self.input_buffer, &mut self.output_buffer);
        }

        let output_size = self.layer_dims.last().expect("Unable to get output size");
        self.input_buffer.slice(s![.., ..*output_size]).to_owned()
    }

    fn backward(&mut self, y: &ArrayView2<f32>, learning_rate: f32) {
        self.ensure_cache_size(y.nrows());

        self.output_buffer.slice_mut(s![.., ..y.ncols()]).assign(y);

        for i in (0..self.layers.len()).rev() {
            let input_dim = self.layer_dims[i];
            let output_dim = self.layer_dims[i + 1];

            let mut input_slice = self.input_buffer.slice_mut(s![.., ..input_dim]);
            let output_slice = self.output_buffer.slice(s![.., ..output_dim]);

            self.layers[i].backward(&mut input_slice, &output_slice, learning_rate);

            std::mem::swap(&mut self.input_buffer, &mut self.output_buffer);
        }
    }

    pub fn train(
        &mut self,
        loader: &mut impl for<'a> Dataloader<'a>,
        epoch: usize,
        learning_rate: f32,
    ) {
        for e in 0..epoch {
            println!("Starting epoch {}/{}", e + 1, epoch);

            let mut total_loss = 0.0;
            let mut total_correct = 0.0;
            let mut total_samples = 0;

            let num_of_batches = loader.num_of_batches();
            for (i, (x, y)) in loader.train_batches().enumerate() {
                if i % 50 == 0 {
                    println!("Batch {i}/{num_of_batches}");
                }

                let output = self.forward_train(&x);

                self.backward(&y, learning_rate);

                total_samples += x.nrows();
                let batch_size = x.nrows() as f32;
                total_loss += cross_entropy_loss(&output.view(), &y) * batch_size;
                total_correct += accuracy(&output.view(), &y) * batch_size;
            }

            let loss = total_loss / total_samples as f32;
            let accuracy = total_correct / total_samples as f32;

            println!(
                "Epoch {}/{} — Loss: {:.4}, Accuracy: {:.4}",
                e + 1,
                epoch,
                loss,
                accuracy,
            );
        }
    }

    pub fn eval(&self, loader: &mut impl for<'a> Dataloader<'a>) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_correct = 0.0;
        let mut total_samples = 0;

        for (x, y) in loader.validation_batches() {
            let prediction = self.predict(&x);

            total_samples += x.nrows();
            let batch_size = x.nrows() as f32;
            total_loss += cross_entropy_loss(&prediction.view(), &y) * batch_size;
            total_correct += accuracy(&prediction.view(), &y) * batch_size;
        }

        let loss = total_loss / total_samples as f32;
        let accuracy = total_correct / total_samples as f32;

        (loss, accuracy)
    }
}

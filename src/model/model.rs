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
    max_dim: usize,
    pub topology: Vec<usize>,
    last_batch_size: Option<usize>,

    forward_input: Array2<f32>,
    forward_output: Array2<f32>,

    backward_input: Array2<f32>,
    backward_output: Array2<f32>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>, topology: Vec<usize>) -> Self {
        Self {
            layers,
            max_dim: *topology
                .iter()
                .max_by(std::cmp::Ord::cmp)
                .expect("Topology must contain at least one layer dimension"),
            topology,
            last_batch_size: None,
            forward_input: Array2::zeros((0, 0)),
            forward_output: Array2::zeros((0, 0)),
            backward_input: Array2::zeros((0, 0)),
            backward_output: Array2::zeros((0, 0)),
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

    pub fn predict(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let batch_size = x.nrows();

        let mut input = Array2::zeros((batch_size, self.max_dim));
        let mut output = Array2::zeros((batch_size, self.max_dim));

        input.slice_mut(s![.., ..x.ncols()]).assign(x);

        for i in 0..self.layers.len() {
            let input_bound = self.topology[i];
            let output_bound = self.topology[i + 1];

            let input_slice = input.slice(s![.., ..input_bound]);
            let mut output_slice = output.slice_mut(s![.., ..output_bound]);

            self.layers[i].forward(&input_slice, &mut output_slice);

            std::mem::swap(&mut input, &mut output);
        }

        let output_size = self.topology.last().expect("Unable to get output size");
        input.slice_mut(s![.., ..*output_size]).to_owned()
    }

    fn forward_propagation(&mut self, x: &ArrayView2<f32>) -> Array2<f32> {
        let batch_size = x.nrows();

        match self.last_batch_size {
            Some(value) if value == batch_size => {}
            _ => {
                self.last_batch_size = Some(batch_size);
                self.forward_input = Array2::zeros((batch_size, self.max_dim));
                self.forward_output = Array2::zeros((batch_size, self.max_dim));
            }
        }

        self.forward_input.slice_mut(s![.., ..x.ncols()]).assign(x);

        for i in 0..self.layers.len() {
            let input_bound = self.topology[i];
            let output_bound = self.topology[i + 1];

            let input_slice = self.forward_input.slice(s![.., ..input_bound]);
            let mut output_slice = self.forward_output.slice_mut(s![.., ..output_bound]);

            self.layers[i].forward_train(&input_slice, &mut output_slice);

            std::mem::swap(&mut self.forward_input, &mut self.forward_output);
        }

        self.last_batch_size = None;

        let output_size = self.topology.last().expect("Unable to get output size");
        self.forward_input
            .slice_mut(s![.., ..*output_size])
            .to_owned()
    }

    fn backward_propagation(&mut self, y: &ArrayView2<f32>, learning_rate: f32) {
        let batch_size = y.nrows();

        match self.last_batch_size {
            Some(value) if value == batch_size => {}
            _ => {
                self.last_batch_size = Some(batch_size);
                self.backward_input = Array2::zeros((batch_size, self.max_dim));
                self.backward_output = Array2::zeros((batch_size, self.max_dim));
            }
        }

        self.backward_output
            .slice_mut(s![.., ..y.ncols()])
            .assign(y);

        for i in (0..self.layers.len()).rev() {
            let input_bound = self.topology[i];
            let output_bound = self.topology[i + 1];

            let mut input_slice = self.backward_input.slice_mut(s![.., ..input_bound]);
            let output_slice = self.backward_output.slice(s![.., ..output_bound]);

            self.layers[i].backward(&mut input_slice, &output_slice, learning_rate);

            std::mem::swap(&mut self.backward_input, &mut self.backward_output);
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
            let mut samples = 0;

            let num_of_batches = loader.num_of_batches();
            for (i, (x, y)) in loader.train_batches().enumerate() {
                if i % 50 == 0 {
                    println!("Batch {i}/{num_of_batches}");
                }

                let output = self.forward_propagation(&x);

                self.backward_propagation(&y, learning_rate);

                samples += x.nrows();
                let batch_size = x.nrows() as f32;
                total_loss += cross_entropy_loss(&output.view(), &y) * batch_size;
                total_correct += accuracy(&output.view(), &y) * batch_size;
            }

            let loss = total_loss / samples as f32;
            let accuracy = total_correct / samples as f32;

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
        let mut samples = 0;

        for (x, y) in loader.validation_batches() {
            let prediction = self.predict(&x);

            samples += x.nrows();
            let batch_size = x.nrows() as f32;
            total_loss += cross_entropy_loss(&prediction.view(), &y) * batch_size;
            total_correct += accuracy(&prediction.view(), &y) * batch_size;
        }

        let loss = total_loss / samples as f32;
        let accuracy = total_correct / samples as f32;

        (loss, accuracy)
    }
}

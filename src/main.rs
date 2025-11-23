use ndarray::s;

use crate::dataset::NUM_OF_CLASSES;

mod dataset;
mod neural_network;

fn main() {
    match dataset::TrainingSet::load("train", "t10k") {
        Ok(dataset) => {
            let cols = dataset.train.data.shape()[1];
            let topology = [cols, 100, 100, 100, NUM_OF_CLASSES];
            let mut network = neural_network::NeuralNetwork::new(&topology);

            let input = dataset.validation.data.slice(s!(..1, ..)).to_owned();
            network.train(&dataset, 10, 128, 0.5);
            network.predict(input);
        }
        Err(err) => println!("{err:?}"),
    }
}

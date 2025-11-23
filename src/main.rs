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

            network.train(&dataset, 10, 128, 0.5);

            let label = dataset.validation.labels.slice(s!(..1, ..)).to_owned();
            let input = dataset.validation.data.slice(s!(..1, ..)).to_owned();

            let prediction = network.predict(input);

            let predicted = neural_network::NeuralNetwork::argmax(&prediction)[0];
            let actual = neural_network::NeuralNetwork::argmax(&label)[0];
            println!("Predicted: {}, Actual: {}", predicted, actual);
        }
        Err(err) => println!("{err:?}"),
    }
}

use ndarray::s;

use crate::dataset::NUM_OF_CLASSES;

mod dataset;
mod neural_network;

fn main() {
    match dataset::Dataset::load("train-labels.idx1-ubyte", "train-images.idx3-ubyte") {
        Ok(dataset) => {
            println!(
                "total labels of {} and images of {} with shape {:?}",
                dataset.labels.len(),
                dataset.images.len(),
                dataset.images.shape(),
            );

            let cols = dataset.images.shape()[1];

            let topology = [cols, 100, 100, 100, NUM_OF_CLASSES];
            let mut network = neural_network::NeuralNetwork::new(&topology);

            let input = dataset.images.slice(s![..32, ..]);
            let target = dataset.labels.slice(s![..32, ..]);
            network.forward_propagation(input.to_owned());
            network.backward_propagation(target.to_owned(), 10.0);
        }
        Err(err) => println!("{err:?}"),
    }
}

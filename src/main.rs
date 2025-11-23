use ndarray::s;

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
            let mut network = neural_network::NeuralNetwork::new(3, cols, 100);
            let input = dataset.images.slice(s![..32, ..]);
            network.forward_propagation(input.to_owned());
        }
        Err(err) => println!("{err:?}"),
    }
}

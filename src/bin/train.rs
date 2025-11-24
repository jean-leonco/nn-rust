use nn_rust::{
    activation::{ActivationFn, relu},
    dataloader::mnist_loader::MNistLoader,
    neural_network::NeuralNetwork,
};

fn main() {
    let topology = [784, 100, 100, 100, 100, 10];
    let mut network = NeuralNetwork::new(&topology, relu::ReLuActivationFn::new(topology.len()));

    let mut loader = MNistLoader::load(128).unwrap();

    network.train(&mut loader, 10, 0.01);
    network.save("model").unwrap();
}

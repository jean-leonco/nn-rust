use nn_rust::{
    dataloader::mnist_loader::MNistLoader, layer::dense::Initialization, model::model::Model,
};

fn main() {
    let mut loader = MNistLoader::load(128).expect("Unable to load MNIST dataset");

    let mut relu_model = Model::builder()
        .input(784)
        .dense(100, Initialization::He)
        .relu()
        .dense(100, Initialization::He)
        .relu()
        .dense(100, Initialization::He)
        .relu()
        .dense(100, Initialization::He)
        .relu()
        .dense(100, Initialization::He)
        .relu()
        .dense(10, Initialization::He)
        .softmax_cross_entropy()
        .build();

    relu_model.train(&mut loader, 20, 0.02);
    let (training_loss, training_accuracy) = relu_model.eval(&mut loader);
    println!(
        "ReLU - Validation Loss: {training_loss:.4}, Validation Accuracy: {training_accuracy:.4}"
    );
    relu_model.save("relu_model").expect("Unable to save model");

    let mut sigmoid_model = Model::builder()
        .input(784)
        .dense(100, Initialization::Xavier)
        .sigmoid()
        .dense(100, Initialization::Xavier)
        .sigmoid()
        .dense(100, Initialization::Xavier)
        .sigmoid()
        .dense(100, Initialization::Xavier)
        .sigmoid()
        .dense(100, Initialization::Xavier)
        .sigmoid()
        .dense(10, Initialization::Xavier)
        .softmax_cross_entropy()
        .build();

    sigmoid_model.train(&mut loader, 20, 0.2);
    let (training_loss, training_accuracy) = sigmoid_model.eval(&mut loader);
    println!(
        "Sigmoid - Validation Loss: {training_loss:.4}, Validation Accuracy: {training_accuracy:.4}"
    );
    sigmoid_model
        .save("sigmoid_model")
        .expect("Unable to save model");
}

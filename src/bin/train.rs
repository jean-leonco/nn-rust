use nn_rust::{
    dataloader::mnist_loader::MNistLoader, layer::dense::Initialization, model::model::Model,
};

fn train_model(
    model_name: &str,
    display_name: &str,
    mut model: Model,
    loader: &mut MNistLoader,
    epochs: usize,
    learning_rate: f32,
) {
    println!("\n=== {display_name} Model ===");
    model.train(loader, epochs, learning_rate);

    let (validation_loss, validation_accuracy) = model.eval(loader);
    println!(
        "Validation Loss: {validation_loss:.4}, Validation Accuracy: {validation_accuracy:.4}"
    );

    model
        .save(model_name)
        .expect("Failed to save model: {model_name}");
}

fn main() {
    let mut loader = MNistLoader::load(128).expect("Failed to load MNIST dataset");

    let relu_model = Model::builder()
        .input(784)
        .dense(256, Initialization::He)
        .relu()
        .dense(64, Initialization::He)
        .relu()
        .dense(10, Initialization::He)
        .softmax_cross_entropy()
        .build();

    train_model("relu_model", "ReLU", relu_model, &mut loader, 15, 0.05);

    let sigmoid_model = Model::builder()
        .input(784)
        .dense(256, Initialization::Xavier)
        .sigmoid()
        .dense(64, Initialization::Xavier)
        .sigmoid()
        .dense(10, Initialization::Xavier)
        .softmax_cross_entropy()
        .build();

    train_model(
        "sigmoid_model",
        "Sigmoid",
        sigmoid_model,
        &mut loader,
        15,
        0.2,
    );
}

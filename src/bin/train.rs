use nn_rust::{
    dataset::mnist_dataset::MnistDataset,
    execution_session::ExecutionSession,
    metrics::{accuracy, cross_entropy_loss},
    optimizer::SgdOptimizer,
    sequential::{Initializer, Sequential, SequentialModel},
};
use rand::{SeedableRng, rngs::SmallRng};

fn train_model(
    model_name: &str,
    display_name: &str,
    model: &mut SequentialModel,
    rng: &mut SmallRng,
    dataset: &mut MnistDataset,
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
) {
    println!("\n=== {display_name} Model ===");

    let mut train_rng = rng.clone();

    let mut session = ExecutionSession::new(&model.blueprint, rng, batch_size);

    let sgd = SgdOptimizer::new(learning_rate);

    let mut x = vec![0.0f32; batch_size * 784];

    for epoch in 0..epochs {
        let mut e_loss = 0.0;
        let mut e_correct = 0.0;
        let mut e_samples = 0;

        for (x_raw, y) in dataset.train_batches(&mut train_rng) {
            MnistDataset::convert_to_px(x_raw, &mut x);
            let predictions = session.forward(&model.weights, &x).unwrap();
            e_samples += batch_size;
            e_loss += cross_entropy_loss(predictions, y, batch_size) * batch_size as f32;
            e_correct += accuracy(predictions, y, batch_size).unwrap() * batch_size as f32;

            let gradients = session.backward(&model.weights, &x, y);
            sgd.step(&mut model.weights, gradients);
        }

        println!(
            "Epoch {}/{epochs} - Loss: {:.4}, Accuracy: {:.4}",
            epoch + 1,
            e_loss / e_samples as f32,
            e_correct / e_samples as f32
        );
    }

    let mut val_loss = 0.0;
    let mut val_correct = 0.0;
    let mut val_samples = 0;
    for (x_raw, y) in dataset.validation_batches() {
        MnistDataset::convert_to_px(x_raw, &mut x);
        let predictions = session.forward(&model.weights, &x).unwrap();
        val_samples += batch_size;
        val_loss += cross_entropy_loss(predictions, y, batch_size) * batch_size as f32;
        val_correct += accuracy(predictions, y, batch_size).unwrap() * batch_size as f32;
    }
    println!(
        "Val Loss: {:.4}, Val Acc: {:.4}",
        val_loss / val_samples as f32,
        val_correct / val_samples as f32
    );
    model.save(model_name).unwrap();
}

fn main() {
    let batch_size = 128;
    let mut dataset = MnistDataset::load(batch_size).expect("Failed to load MNIST dataset");
    let mut rng = SmallRng::seed_from_u64(42);

    let mut relu_model = Sequential::builder()
        .input(784)
        .dense(128, Initializer::He)
        .relu()
        .dropout(0.2)
        .dense(64, Initializer::He)
        .relu()
        .dropout(0.2)
        .dense(10, Initializer::He)
        .softmax_cross_entropy()
        .build(&mut rng)
        .unwrap();

    train_model(
        "relu_model",
        "ReLU",
        &mut relu_model,
        &mut rng,
        &mut dataset,
        15,
        batch_size,
        0.05,
    );

    let mut sigmoid_model = Sequential::builder()
        .input(784)
        .dense(256, Initializer::Xavier)
        .sigmoid()
        .dense(64, Initializer::Xavier)
        .sigmoid()
        .dense(10, Initializer::Xavier)
        .softmax_cross_entropy()
        .build(&mut rng)
        .unwrap();

    train_model(
        "sigmoid_model",
        "Sigmoid",
        &mut sigmoid_model,
        &mut rng,
        &mut dataset,
        20,
        batch_size,
        0.2,
    );
}

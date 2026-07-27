use nn_rust::{
    core::TrainMetrics,
    dataset::mnist::MnistDataset,
    model::{SequentialModel, Session},
    ops::Initialization,
    optim::sgd::SgdOptimizer,
};
use rand::{SeedableRng, rngs::SmallRng};

const BATCH_SIZE: usize = 128;
const INPUT_SIZE: usize = 784;
const SEED: [u32; 2] = [42; 2];

fn train_model(
    model_name: &str,
    display_name: &str,
    model: &mut SequentialModel,
    rng: &mut SmallRng,
    dataset: &mut MnistDataset,
    epochs: usize,
    learning_rate: f32,
) {
    println!("\n=== {display_name} Model ===");

    let mut session = Session::new(model, BATCH_SIZE, Some(SEED));
    let optimizer = SgdOptimizer::new(learning_rate);

    let mut x = vec![0.0f32; BATCH_SIZE * INPUT_SIZE];

    for epoch in 0..epochs {
        let mut train_metrics = TrainMetrics::new(BATCH_SIZE);
        for (x_batch, y) in dataset.train_batches(rng) {
            MnistDataset::convert_to_px(x_batch, &mut x);

            let prediction = session.forward(&model.train_ops, &mut model.params, &x);
            train_metrics
                .update(prediction, y)
                .expect("metrics shape mismatch");

            let gradients = session.backward(&model.train_ops, &model.params, &x, y);
            optimizer.step(&mut model.params, gradients);
        }

        println!("Epoch {epoch}/{epochs}: {train_metrics}");
    }

    let mut validation_metrics = TrainMetrics::new(BATCH_SIZE);

    for (x_validation, y) in dataset.validation_batches() {
        MnistDataset::convert_to_px(x_validation, &mut x);
        let prediction = session.forward(&model.inference_ops, &mut model.params, &x);
        validation_metrics
            .update(prediction, y)
            .expect("metrics shape mismatch");
    }

    println!("Validation: {validation_metrics}");

    model.save(model_name).expect("Failed to save model");
}

fn main() {
    let mut dataset = MnistDataset::load(BATCH_SIZE).expect("Failed to load MNIST dataset");
    let mut rng = SmallRng::seed_from_u64(42);

    let mut relu_model = SequentialModel::builder()
        .input(784)
        .dense(128, Initialization::He)
        .relu()
        .dropout(0.2)
        .unwrap()
        .dense(64, Initialization::He)
        .relu()
        .dropout(0.2)
        .unwrap()
        .dense(10, Initialization::He)
        .softmax()
        .build();
    relu_model
        .initialize_params(&mut rng)
        .expect("Failed to initialize parameters");

    train_model(
        "relu_model",
        "ReLU",
        &mut relu_model,
        &mut rng,
        &mut dataset,
        50,
        0.05,
    );

    let mut sigmoid_model = SequentialModel::builder()
        .input(784)
        .dense(256, Initialization::Xavier)
        .sigmoid()
        .dropout(0.2)
        .unwrap()
        .dense(64, Initialization::Xavier)
        .sigmoid()
        .dropout(0.2)
        .unwrap()
        .dense(10, Initialization::Xavier)
        .softmax()
        .build();
    sigmoid_model
        .initialize_params(&mut rng)
        .expect("Failed to initialize parameters");

    train_model(
        "sigmoid_model",
        "Sigmoid",
        &mut sigmoid_model,
        &mut rng,
        &mut dataset,
        50,
        0.2,
    );
}

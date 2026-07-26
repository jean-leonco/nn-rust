use nn_rust::{
    core::TrainMetrics,
    dataset::mnist::MnistDataset,
    model::{SequentialModel, Session},
    ops::Initialization,
    optim::sgd::SgdOptimizer,
};
use rand::{SeedableRng, rngs::SmallRng};

const BATCH_SIZE: usize = 256;
const INPUT_SIZE: usize = 784;
const SEED: [u32; 2] = [42; 2];
const LEARNING_RATE: f32 = 0.005;
const EPOCHS: usize = 20;

fn main() {
    let mut dataset = MnistDataset::load(BATCH_SIZE).expect("Failed to load MNIST dataset");
    let mut rng = SmallRng::seed_from_u64(42);

    let mut model = SequentialModel::builder()
        .input(784)
        .dense(2048, Initialization::He)
        .relu()
        .dropout(0.2)
        .unwrap()
        .dense(1024, Initialization::He)
        .relu()
        .dropout(0.1)
        .unwrap()
        .dense(512, Initialization::He)
        .relu()
        .dense(10, Initialization::He)
        .softmax()
        .build();
    model
        .initialize_params(&mut rng)
        .expect("Failed to initialize parameters");

    let mut session = Session::new(&model, BATCH_SIZE, Some(SEED));
    let optimizer = SgdOptimizer::new(LEARNING_RATE);

    let mut x = vec![0.0f32; BATCH_SIZE * INPUT_SIZE];

    for epoch in 0..EPOCHS {
        let mut train_metrics = TrainMetrics::new(BATCH_SIZE);
        for (x_batch, y) in dataset.train_batches(&mut rng) {
            MnistDataset::convert_to_px(x_batch, &mut x);

            let prediction = session.forward(&model.train_ops, &mut model.params, &x);
            train_metrics.update(prediction, y);

            let gradients = session.backward(&model.train_ops, &model.params, &x, y);
            optimizer.step(&mut model.params, gradients);
        }

        println!("Epoch {epoch}/{EPOCHS}: {train_metrics}");
    }

    let mut validation_metrics = TrainMetrics::new(BATCH_SIZE);

    for (x_validation, y) in dataset.validation_batches() {
        MnistDataset::convert_to_px(x_validation, &mut x);
        let prediction = session.forward(&model.inference_ops, &mut model.params, &x);
        validation_metrics.update(prediction, y);
    }

    println!("Validation: {validation_metrics}");
}

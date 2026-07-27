#![feature(portable_simd)]

use std::simd::prelude::*;

use nn_rust::{
    core::{LANE_IOTA, bernoulli, build_key_schedule},
    dataset::mnist::MnistDataset,
    model::{SequentialModel, Session},
    ops::Initialization,
    optim::sgd::SgdOptimizer,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

const BATCH_SIZE: usize = 128;
const INPUT_DIM: usize = 784;
const EPOCHS: usize = 100;
const LEARNING_RATE: f32 = 0.08;
const SURVIVAL_RATE: f32 = 0.8;
const SEED: [u32; 2] = [42; 2];
const DISPLAY_SEED: u64 = 0xA11CE;
const DISPLAY_STEP: usize = 0xA11CE;

fn main() {
    let mut dataset = MnistDataset::load(BATCH_SIZE).expect("Failed to load MNIST dataset");
    let mut rng = SmallRng::seed_from_u64(42);
    let key_schedule = build_key_schedule([u32x8::splat(SEED[0]), u32x8::splat(SEED[1])]);

    let mut model = SequentialModel::builder()
        .input(INPUT_DIM)
        .dense(256, Initialization::He)
        .relu()
        .dense(128, Initialization::He)
        .relu()
        .dense(256, Initialization::He)
        .relu()
        .dense(INPUT_DIM, Initialization::Xavier)
        .sigmoid()
        .mse()
        .build();
    model
        .initialize_params(&mut rng)
        .expect("Failed to initialize parameters");

    let mut session = Session::new(&model, BATCH_SIZE, Some(SEED));
    let optimizer = SgdOptimizer::new(LEARNING_RATE);
    let mut clean = vec![0.0; BATCH_SIZE * INPUT_DIM];
    let mut corrupted = vec![0.0; BATCH_SIZE * INPUT_DIM];
    let mut step = 0;

    for epoch in 0..EPOCHS {
        let mut metrics = model.loss_kind().metrics(BATCH_SIZE);

        for (images, _) in dataset.train_batches(&mut rng) {
            MnistDataset::convert_to_px(images, &mut clean);
            corrupt(&clean, &mut corrupted, step, &key_schedule);
            step += 1;

            let prediction = session.forward(&model.train_ops, &mut model.params, &corrupted);
            metrics
                .update(prediction, &clean)
                .expect("metrics shape mismatch");

            let gradients = session.backward(&model.train_ops, &model.params, &corrupted, &clean);
            optimizer.step(&mut model.params, gradients);
        }

        println!("Epoch {}/{EPOCHS}: {metrics}", epoch + 1);
    }

    let mut validation_metrics = model.loss_kind().metrics(BATCH_SIZE);
    for (images, _) in dataset.validation_batches() {
        MnistDataset::convert_to_px(images, &mut clean);
        corrupt(&clean, &mut corrupted, step, &key_schedule);
        step += 1;

        let prediction = session.forward(&model.inference_ops, &mut model.params, &corrupted);
        validation_metrics
            .update(prediction, &clean)
            .expect("metrics shape mismatch");
    }
    println!("Validation: {validation_metrics}");

    let mut display_rng = SmallRng::seed_from_u64(DISPLAY_SEED);
    let n_batches = dataset.validation_batches().count();
    let batch_index = display_rng.random_range(0..n_batches);
    let (images, _) = dataset
        .validation_batches()
        .nth(batch_index)
        .expect("validation batch exists");
    let sample_index = display_rng.random_range(0..BATCH_SIZE);
    let start = sample_index * INPUT_DIM;
    let end = start + INPUT_DIM;

    let mut sample = vec![0.0; INPUT_DIM];
    MnistDataset::convert_to_px(&images[start..end], &mut sample);
    let mut noisy_sample = vec![0.0; INPUT_DIM];
    corrupt(&sample, &mut noisy_sample, DISPLAY_STEP, &key_schedule);
    let reconstruction = model
        .predict(&noisy_sample)
        .expect("Failed to reconstruct image");

    println!("\nOriginal");
    print_ascii(&sample);
    println!("\nCorrupted");
    print_ascii(&noisy_sample);
    println!("\nReconstruction");
    print_ascii(&reconstruction);
}

fn corrupt(
    clean: &[f32],
    corrupted: &mut [f32],
    step: usize,
    key_schedule: &nn_rust::core::KeySchedule,
) {
    assert_eq!(clean.len(), corrupted.len());

    for (block_index, (input, output)) in clean.chunks(32).zip(corrupted.chunks_mut(32)).enumerate()
    {
        let counters = [
            u32x8::splat((block_index * 8) as u32) + LANE_IOTA,
            u32x8::splat(0),
            u32x8::splat(step as u32),
            u32x8::splat(0),
        ];
        let mask = bernoulli(counters, key_schedule, SURVIVAL_RATE).to_array();

        for ((source, destination), keep) in input.iter().zip(output).zip(mask) {
            *destination = source * f32::from(keep);
        }
    }
}

fn print_ascii(pixels: &[f32]) {
    const SHADES: [char; 8] = [' ', '.', '-', '=', '+', 'O', '#', '@'];

    for row in pixels.as_chunks::<28>().0 {
        for &pixel in row {
            let index = (pixel.clamp(0.0, 1.0) * (SHADES.len() - 1) as f32) as usize;
            print!("{}", SHADES[index]);
        }
        println!();
    }
}

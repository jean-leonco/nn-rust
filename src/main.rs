use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array2;

mod dataloader;
mod neural_network;
extern crate blas_src;

fn load_test_image() -> (Array2<f32>, Array2<f32>) {
    let mut label = Array2::zeros((1, 10));
    label[[0, 3]] = 1.0;

    let img = image::open("test_image.png")
        .expect("Failed to open image")
        .to_luma8();
    let img = DynamicImage::ImageLuma8(img).resize_exact(28, 28, FilterType::Nearest);

    let mut data = Array2::zeros((1, 784));
    for y in 0..28 {
        for x in 0..28 {
            let px = img.get_pixel(x, y)[0] as f32;
            let inv = (255.0 - px) / 255.0;
            data[[0, (y * 28 + x) as usize]] = inv;
        }
    }

    (label, data)
}

fn main() {
    let topology = [784, 100, 100, 100, 10];
    let mut network = neural_network::NeuralNetwork::new(&topology);

    let mut loader = dataloader::mnist_loader::MNistLoader::load(128).unwrap();

    network.train(&mut loader, 10, 0.5);

    let (test_label, test_data) = load_test_image();
    let prediction = network.predict(test_data.view());

    let predicted = neural_network::NeuralNetwork::argmax(&prediction.view())[0];
    let actual = neural_network::NeuralNetwork::argmax(&test_label.view())[0];

    println!("Predicted: {}, Actual: {}", predicted, actual);
}

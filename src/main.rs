use image::{GenericImageView, Pixel, imageops::FilterType};
use ndarray::Array2;

mod dataloader;
mod neural_network;
extern crate blas_src;

fn load_test_image() -> (Array2<f32>, Array2<f32>) {
    let mut label = Array2::zeros((1, 10));
    label[[0, 3]] = 1.0;

    let img = image::open("test_image.jpeg")
        .expect("Failed to open image")
        .resize(28, 28, FilterType::Nearest);
    let (width, height) = img.dimensions();

    let mut data = Array2::zeros((1, 784));
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let ch = pixel.to_rgb().0;
            let r = ch[0] as f32;
            let g = ch[1] as f32;
            let b = ch[2] as f32;

            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let inverted = (255.0 - gray) / 255.0;
            let i = y as usize * 28 + x as usize;
            data[[0, i]] = inverted;
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

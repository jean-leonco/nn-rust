use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array2;
use nn_rust::neural_network::NeuralNetwork;

fn main() {
    let network = NeuralNetwork::load("model").unwrap();

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

    let prediction = network.predict(data.view());

    let predicted = NeuralNetwork::argmax(&prediction.view())[0];
    let actual = NeuralNetwork::argmax(&label.view())[0];

    println!("Predicted: {}, Actual: {}", predicted, actual);
}

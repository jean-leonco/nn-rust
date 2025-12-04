use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array2;
use nn_rust::{metrics, model::model::Model};

fn run_model(model_name: &str, display_name: &str, data: &Array2<f32>, true_label: usize) {
    println!("\n=== {display_name} Model ===");

    let model = Model::load(model_name).expect(&format!("Failed to load model: {model_name}"));

    let prediction = model.predict(&data.view());
    let predicted = metrics::argmax(&prediction.view())[0];

    println!("Predicted: {predicted} | Actual: {true_label}");
    println!("Class Probabilities:");

    for (class, &prob) in prediction.row(0).iter().enumerate() {
        if class == predicted {
            println!("  {class}: {prob:.4}  <-- predicted");
        } else {
            println!("  {class}: {prob:.4}");
        }
    }
}

fn main() {
    let mut label = Array2::zeros((1, 10));
    label[[0, 3]] = 1.0;

    let img = image::open("test_image.png")
        .expect("Failed to open image")
        .to_luma8();
    let img = DynamicImage::ImageLuma8(img).resize_exact(28, 28, FilterType::Nearest);

    let mut data = Array2::zeros((1, 784));
    for y in 0..28 {
        for x in 0..28 {
            let px = f32::from(img.get_pixel(x, y)[0]);
            let inv = (255.0 - px) / 255.0;
            data[[0, (y * 28 + x) as usize]] = inv;
        }
    }

    let true_label = metrics::argmax(&label.view())[0];

    run_model("relu_model", "ReLU", &data, true_label);
    run_model("sigmoid_model", "Sigmoid", &data, true_label);
}

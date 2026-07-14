use image::{DynamicImage, GenericImageView, imageops::FilterType};
use nn_rust::{metrics, sequential::SequentialModel};

fn run_model(model_name: &str, display_name: &str, data: &[f32], true_label: usize) {
    println!("\n=== {display_name} Model ===");

    let model = SequentialModel::load(model_name).unwrap();

    let prediction = model.predict(data).unwrap();
    let predicted = metrics::argmax(&prediction, 1).unwrap()[0];

    println!("Predicted: {predicted} | Actual: {true_label}");
    println!("Class Probabilities:");

    for (class, &prob) in prediction.iter().enumerate() {
        if class == predicted {
            println!("  {class}: {prob:.4}  <-- predicted");
        } else {
            println!("  {class}: {prob:.4}");
        }
    }
}

fn main() {
    let mut label = vec![0.0; 10];
    label[3] = 1.0;

    let img = image::open("test_image.png")
        .expect("Failed to open image")
        .to_luma8();
    let img = DynamicImage::ImageLuma8(img).resize_exact(28, 28, FilterType::Nearest);

    let mut data = vec![0.0; 784];
    for y in 0..28 {
        for x in 0..28 {
            let px = f32::from(img.get_pixel(x, y)[0]);
            let inv = (255.0 - px) / 255.0;
            data[(y * 28 + x) as usize] = inv;
        }
    }

    let true_label = metrics::argmax(&label, 1).unwrap()[0];

    run_model("relu_model", "ReLU", &data, true_label);
    run_model("sigmoid_model", "Sigmoid", &data, true_label);
}

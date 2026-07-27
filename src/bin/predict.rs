use image::{DynamicImage, GenericImageView, imageops::FilterType};
use nn_rust::{core::metrics, model::SequentialModel};

fn run_model(model_name: &str, display_name: &str, data: &[f32], true_label: usize) {
    println!("\n=== {display_name} Model ===");

    let mut model = SequentialModel::load(model_name, None).expect("Failed to load model");

    let prediction = model.predict(data).expect("Invalid prediction input");
    let predicted = metrics::argmax(&prediction, 1).expect("Failed to get argmax")[0];

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

            let v = data[(y * 28 + x) as usize];
            let ch = match v {
                v if v > 0.75 => '#',
                v if v > 0.5 => '+',
                v if v > 0.25 => '.',
                _ => ' ',
            };
            print!("{ch}");
        }
        println!();
    }

    let true_label = metrics::argmax(&label, 1).expect("Failed to get argmax")[0];

    run_model("relu_model", "ReLU", &data, true_label);
    run_model("sigmoid_model", "Sigmoid", &data, true_label);
}

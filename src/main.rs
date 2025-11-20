mod dataset;

fn main() {
    match dataset::Dataset::load("train-labels.idx1-ubyte", "train-images.idx3-ubyte") {
        Ok(dataset) => println!(
            "total labels of {} and images of {}",
            dataset.labels.len(),
            dataset.images.len()
        ),
        Err(err) => println!("{err:?}"),
    }
}

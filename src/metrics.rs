use ndarray::{Array1, ArrayView2, Axis, Zip};

pub fn cross_entropy_loss(output: &ArrayView2<f32>, y: &ArrayView2<f32>) -> f32 {
    let mut probabilities = output + 1e-8;
    probabilities.mapv_inplace(f32::ln);

    let loss = -(y * &probabilities).sum();
    loss / output.nrows() as f32
}

pub fn accuracy(output: &ArrayView2<f32>, y: &ArrayView2<f32>) -> f32 {
    let matches = Zip::from(&argmax(output))
        .and(&argmax(y))
        .map_collect(|&a, &b| f32::from(a == b))
        .sum();
    matches / output.nrows() as f32
}

pub fn argmax(x: &ArrayView2<f32>) -> Array1<usize> {
    x.axis_iter(Axis(0))
        .map(|row| {
            let (max_idx, _) = row
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.total_cmp(y))
                .unwrap();
            max_idx
        })
        .collect()
}

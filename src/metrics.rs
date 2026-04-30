use ndarray::{Array1, ArrayView2, Axis, Zip};

pub fn cross_entropy_loss(output: &ArrayView2<f32>, y: &ArrayView2<f32>) -> f32 {
    let mut total_loss = 0.0;
    Zip::from(output).and(y).for_each(|&o, &y_val| {
        total_loss -= y_val * (o + 1e-8).ln();
    });
    total_loss / output.nrows() as f32
}

pub fn accuracy(output: &ArrayView2<f32>, y: &ArrayView2<f32>) -> f32 {
    let mut matches = 0.0;
    for (out_row, y_row) in output.axis_iter(Axis(0)).zip(y.axis_iter(Axis(0))) {
        if argmax_row(&out_row) == argmax_row(&y_row) {
            matches += 1.0;
        }
    }
    matches / output.nrows() as f32
}

fn argmax_row(row: &ndarray::ArrayView1<f32>) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(idx, _)| idx)
        .unwrap()
}

pub fn argmax(x: &ArrayView2<f32>) -> Array1<usize> {
    x.axis_iter(Axis(0)).map(|row| argmax_row(&row)).collect()
}

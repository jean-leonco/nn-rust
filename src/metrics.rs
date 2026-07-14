use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Metrics error: empty row")]
    EmptyRow,
}
pub type Result<T> = std::result::Result<T, MetricsError>;

pub fn cross_entropy_loss(predictions: &[f32], targets: &[f32], n_rows: usize) -> f32 {
    let mut loss = 0.0;
    for (p, t) in predictions.iter().zip(targets.iter()) {
        loss -= t * (p + 1e-8).ln()
    }
    loss / n_rows as f32
}

pub fn accuracy(predictions: &[f32], targets: &[f32], n_rows: usize) -> Result<f32> {
    let mut matches = 0.0;
    let cols = predictions.len() / n_rows;

    let predictions_slice = predictions.chunks_exact(cols);
    let targets_slice = targets.chunks_exact(cols);

    for (p, t) in predictions_slice.zip(targets_slice) {
        if argmax_row(&p)? == argmax_row(&t)? {
            matches += 1.0;
        }
    }
    Ok(matches / n_rows as f32)
}

fn argmax_row(row: &[f32]) -> Result<usize> {
    row.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(idx, _)| idx)
        .ok_or(MetricsError::EmptyRow)
}

pub fn argmax(predictions: &[f32], n_rows: usize) -> Result<Vec<usize>> {
    let cols = predictions.len() / n_rows;

    predictions
        .chunks_exact(cols)
        .map(|row| argmax_row(&row))
        .collect()
}

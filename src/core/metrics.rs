use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Metrics error: empty row")]
    EmptyRow,
}

/// Computes the cross-entropy loss between predictions and target values.
pub fn cross_entropy_loss(predictions: &[f32], targets: &[f32], n_rows: usize) -> f32 {
    let mut loss = 0.0;
    for (p, t) in predictions.iter().zip(targets.iter()) {
        loss -= t * (p + 1e-8).ln()
    }
    loss / n_rows as f32
}

/// Computes the accuracy of predictions compared to target values.
pub fn accuracy(predictions: &[f32], targets: &[f32], n_rows: usize) -> Result<f32, MetricsError> {
    let mut matches = 0.0;
    let cols = predictions.len() / n_rows;

    let predictions_slice = predictions.chunks_exact(cols);
    let targets_slice = targets.chunks_exact(cols);

    for (p, t) in predictions_slice.zip(targets_slice) {
        if argmax_row(p)? == argmax_row(t)? {
            matches += 1.0;
        }
    }
    Ok(matches / n_rows as f32)
}

/// Finds the index of the maximum value in a row.
fn argmax_row(row: &[f32]) -> Result<usize, MetricsError> {
    row.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(idx, _)| idx)
        .ok_or(MetricsError::EmptyRow)
}

/// Finds the index of the maximum value in each row of a matrix.
pub fn argmax(predictions: &[f32], n_rows: usize) -> Result<Vec<usize>, MetricsError> {
    let cols = predictions.len() / n_rows;

    predictions.chunks_exact(cols).map(argmax_row).collect()
}

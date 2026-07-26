use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Metrics error: empty row")]
    EmptyRow,
}

/// Mean cross-entropy loss across `n_rows`.
pub fn cross_entropy_loss(predictions: &[f32], targets: &[f32], n_rows: usize) -> f32 {
    let mut loss = 0.0;
    for (p, t) in predictions.iter().zip(targets.iter()) {
        loss -= t * (p + 1e-8).ln()
    }
    loss / n_rows as f32
}

/// Fraction of rows where the predicted class matches the true class.
pub fn accuracy(predictions: &[f32], targets: &[f32], n_rows: usize) -> Result<f32, MetricsError> {
    if n_rows == 0 || predictions.is_empty() {
        return Err(MetricsError::EmptyRow);
    }

    let mut matches = 0.0;
    let cols = predictions.len() / n_rows;

    for (p, t) in predictions.chunks_exact(cols).zip(targets.chunks_exact(cols)) {
        if argmax_row(p)? == argmax_row(t)? {
            matches += 1.0;
        }
    }
    Ok(matches / n_rows as f32)
}

fn argmax_row(row: &[f32]) -> Result<usize, MetricsError> {
    row.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(idx, _)| idx)
        .ok_or(MetricsError::EmptyRow)
}

/// Index of the maximum value per row in a flat row-major matrix.
pub fn argmax(predictions: &[f32], n_rows: usize) -> Result<Vec<usize>, MetricsError> {
    if n_rows == 0 || predictions.is_empty() {
        return Err(MetricsError::EmptyRow);
    }

    let cols = predictions.len() / n_rows;
    predictions.chunks_exact(cols).map(argmax_row).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn cross_entropy_single_row() {
        // loss = -sum(t * ln(p + 1e-8)) / n_rows
        // predictions=[0.9, 0.1], targets=[1.0, 0.0], n_rows=1
        // loss = -(1.0 * ln(0.9)) = ~0.10536
        let loss = cross_entropy_loss(&[0.9, 0.1], &[1.0, 0.0], 1);
        assert!((loss - 0.10536).abs() < EPS);
    }

    #[test]
    fn cross_entropy_two_rows() {
        // Row 1: -(1.0 * ln(0.9)) ≈ 0.10536
        // Row 2: -(1.0 * ln(0.8)) ≈ 0.22314
        // Mean: (0.10536 + 0.22314) / 2 ≈ 0.16425
        let loss = cross_entropy_loss(
            &[0.9, 0.1, 0.2, 0.8],
            &[1.0, 0.0, 0.0, 1.0],
            2,
        );
        assert!((loss - 0.16425).abs() < EPS);
    }

    #[test]
    fn cross_entropy_perfect_prediction() {
        // Predicting 1.0 for the correct class: -ln(1.0 + 1e-8) ≈ 0
        let loss = cross_entropy_loss(&[0.0, 1.0], &[0.0, 1.0], 1);
        assert!(loss < EPS);
    }

    #[test]
    fn cross_entropy_epsilon_prevents_log_zero() {
        // Predicting 0.0 for the correct class should not panic or return -inf
        let loss = cross_entropy_loss(&[1.0, 0.0], &[0.0, 1.0], 1);
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn accuracy_perfect() {
        let predictions = &[0.1, 0.8, 0.1, 0.7, 0.2, 0.1];
        let targets = &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        assert!((accuracy(predictions, targets, 2).unwrap() - 1.0).abs() < EPS);
    }

    #[test]
    fn accuracy_zero() {
        let predictions = &[0.8, 0.1, 0.1, 0.1, 0.8, 0.1];
        let targets = &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!((accuracy(predictions, targets, 2).unwrap()).abs() < EPS);
    }

    #[test]
    fn accuracy_partial() {
        let predictions = &[0.1, 0.7, 0.2, 0.8, 0.1, 0.1, 0.3, 0.3, 0.4];
        let targets = &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert!((accuracy(predictions, targets, 3).unwrap() - 1.0).abs() < EPS);
    }

    #[test]
    fn accuracy_mixed() {
        let predictions = &[0.1, 0.7, 0.2, 0.8, 0.1, 0.1, 0.3, 0.3, 0.4];
        let targets = &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        assert!((accuracy(predictions, targets, 3).unwrap() - 2.0 / 3.0).abs() < EPS);
    }

    #[test]
    fn accuracy_empty_returns_error() {
        assert!(matches!(accuracy(&[], &[], 1), Err(MetricsError::EmptyRow)));
    }

    #[test]
    fn accuracy_zero_rows_returns_error() {
        assert!(matches!(accuracy(&[0.1, 0.2], &[0.3, 0.4], 0), Err(MetricsError::EmptyRow)));
    }

    #[test]
    fn argmax_single_row() {
        let idx = argmax(&[0.1, 0.5, 0.4], 1).unwrap();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn argmax_multiple_rows() {
        let idx = argmax(&[0.1, 0.5, 0.4, 0.9, 0.05, 0.05], 2).unwrap();
        assert_eq!(idx, vec![1, 0]);
    }

    #[test]
    fn argmax_tie_returns_last() {
        // max_by on nightly resolves ties by keeping the last element
        let idx = argmax(&[1.0, 1.0, 0.5], 1).unwrap();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn argmax_empty_returns_error() {
        assert!(matches!(argmax(&[], 1), Err(MetricsError::EmptyRow)));
    }

    #[test]
    fn argmax_zero_rows_returns_error() {
        assert!(matches!(argmax(&[0.1, 0.2], 0), Err(MetricsError::EmptyRow)));
    }
}

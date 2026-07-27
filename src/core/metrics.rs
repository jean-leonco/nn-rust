use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("empty matrix: no rows to process")]
    EmptyRow,
    #[error("predictions and targets have different lengths: {predictions_len} vs {targets_len}")]
    LengthMismatch {
        predictions_len: usize,
        targets_len: usize,
    },
    #[error("{values_len} values cannot be divided into {rows} equal rows")]
    InvalidMatrixShape { values_len: usize, rows: usize },
}

/// Mean categorical cross-entropy over a row-major probability matrix.
///
/// Each row is one sample, each column is one class. `predictions` holds
/// post-softmax probabilities; `targets` holds the corresponding class
/// distributions.
///
/// # Errors
///
/// Returns [`MetricsError::LengthMismatch`] if the slices differ in length,
/// or [`MetricsError::InvalidMatrixShape`] if `rows` is zero or does not
/// evenly divide the slice length.
pub fn cross_entropy_loss(
    predictions: &[f32],
    targets: &[f32],
    rows: usize,
) -> Result<f32, MetricsError> {
    if predictions.len() != targets.len() {
        return Err(MetricsError::LengthMismatch {
            predictions_len: predictions.len(),
            targets_len: targets.len(),
        });
    }
    validate_matrix(predictions, rows)?;

    let mut loss = 0.0;
    for (p, t) in predictions.iter().zip(targets.iter()) {
        if *t != 0.0 {
            loss -= t * (p + 1e-8).ln();
        }
    }
    Ok(loss / rows as f32)
}

/// Fraction of rows whose argmax class matches the target argmax class.
///
/// Both matrices must be equally shaped and row-major.
///
/// # Errors
///
/// Returns [`MetricsError::LengthMismatch`] if the slices differ in length,
/// or [`MetricsError::InvalidMatrixShape`] if `rows` is zero or does not
/// evenly divide the slice length.
pub fn accuracy(predictions: &[f32], targets: &[f32], rows: usize) -> Result<f32, MetricsError> {
    if predictions.len() != targets.len() {
        return Err(MetricsError::LengthMismatch {
            predictions_len: predictions.len(),
            targets_len: targets.len(),
        });
    }
    let columns = validate_matrix(predictions, rows)?;

    let mut matches = 0.0;
    for (p, t) in predictions.chunks(columns).zip(targets.chunks(columns)) {
        if argmax_row(p)? == argmax_row(t)? {
            matches += 1.0;
        }
    }
    Ok(matches / rows as f32)
}

fn argmax_row(row: &[f32]) -> Result<usize, MetricsError> {
    row.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(idx, _)| idx)
        .ok_or(MetricsError::EmptyRow)
}

/// Index of the maximum value in each row of a flat row-major matrix.
///
/// # Errors
///
/// Returns [`MetricsError::InvalidMatrixShape`] if `rows` is zero or does
/// not evenly divide the slice length.
pub fn argmax(predictions: &[f32], rows: usize) -> Result<Vec<usize>, MetricsError> {
    let columns = validate_matrix(predictions, rows)?;
    predictions.chunks(columns).map(argmax_row).collect()
}

fn validate_matrix(values: &[f32], rows: usize) -> Result<usize, MetricsError> {
    if values.is_empty() {
        return Err(MetricsError::EmptyRow);
    }
    if rows == 0 || !values.len().is_multiple_of(rows) {
        return Err(MetricsError::InvalidMatrixShape {
            values_len: values.len(),
            rows,
        });
    }

    let columns = values.len() / rows;
    if columns == 0 {
        return Err(MetricsError::InvalidMatrixShape {
            values_len: values.len(),
            rows,
        });
    }
    Ok(columns)
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
        let loss = cross_entropy_loss(&[0.9, 0.1], &[1.0, 0.0], 1).unwrap();
        assert!((loss - 0.10536).abs() < EPS);
    }

    #[test]
    fn cross_entropy_two_rows() {
        // Row 1: -(1.0 * ln(0.9)) ≈ 0.10536
        // Row 2: -(1.0 * ln(0.8)) ≈ 0.22314
        // Mean: (0.10536 + 0.22314) / 2 ≈ 0.16425
        let loss = cross_entropy_loss(&[0.9, 0.1, 0.2, 0.8], &[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        assert!((loss - 0.16425).abs() < EPS);
    }

    #[test]
    fn cross_entropy_perfect_prediction() {
        let loss = cross_entropy_loss(&[0.0, 1.0], &[0.0, 1.0], 1).unwrap();
        assert!(loss < EPS);
    }

    #[test]
    fn cross_entropy_epsilon_prevents_log_zero() {
        let loss = cross_entropy_loss(&[1.0, 0.0], &[0.0, 1.0], 1).unwrap();
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
        assert!(matches!(
            accuracy(&[0.1, 0.2], &[0.3, 0.4], 0),
            Err(MetricsError::InvalidMatrixShape { .. })
        ));
    }

    #[test]
    fn matrix_shape_errors_are_reported() {
        assert!(matches!(
            cross_entropy_loss(&[0.9, 0.1], &[1.0], 1),
            Err(MetricsError::LengthMismatch { .. })
        ));
        assert!(matches!(
            accuracy(&[0.1, 0.2], &[0.3, 0.4], 3),
            Err(MetricsError::InvalidMatrixShape { .. })
        ));
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
        let idx = argmax(&[1.0, 1.0, 0.5], 1).unwrap();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn argmax_empty_returns_error() {
        assert!(matches!(argmax(&[], 1), Err(MetricsError::EmptyRow)));
    }

    #[test]
    fn argmax_zero_rows_returns_error() {
        assert!(matches!(
            argmax(&[0.1, 0.2], 0),
            Err(MetricsError::InvalidMatrixShape { .. })
        ));
    }
}

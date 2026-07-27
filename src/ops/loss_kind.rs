use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossKind {
    SoftmaxCrossEntropy,
    MeanSquaredError,
}

impl LossKind {
    pub fn metrics(self, batch_size: usize) -> LossMetrics {
        assert!(batch_size > 0, "metrics batch size must be non-zero");
        match self {
            Self::SoftmaxCrossEntropy => {
                LossMetrics::SoftmaxCrossEntropy(CrossEntropyMetrics::new(batch_size))
            }
            Self::MeanSquaredError => {
                LossMetrics::MeanSquaredError(MeanSquaredErrorMetrics::new(batch_size))
            }
        }
    }
}

#[derive(Debug)]
pub enum LossMetrics {
    SoftmaxCrossEntropy(CrossEntropyMetrics),
    MeanSquaredError(MeanSquaredErrorMetrics),
}

impl LossMetrics {
    pub fn update(&mut self, predictions: &[f32], targets: &[f32]) -> Result<(), LossMetricsError> {
        match self {
            Self::SoftmaxCrossEntropy(metrics) => metrics.update(predictions, targets),
            Self::MeanSquaredError(metrics) => metrics.update(predictions, targets),
        }
    }
}

impl std::fmt::Display for LossMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SoftmaxCrossEntropy(metrics) => metrics.fmt(f),
            Self::MeanSquaredError(metrics) => metrics.fmt(f),
        }
    }
}

#[derive(Debug)]
pub struct CrossEntropyMetrics {
    batch_size: usize,
    samples: usize,
    loss: f32,
    correct: usize,
}

impl CrossEntropyMetrics {
    fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            samples: 0,
            loss: 0.0,
            correct: 0,
        }
    }

    pub fn update(&mut self, predictions: &[f32], targets: &[f32]) -> Result<(), LossMetricsError> {
        if predictions.len() != targets.len() {
            return Err(LossMetricsError::LengthMismatch {
                predictions_len: predictions.len(),
                targets_len: targets.len(),
            });
        }

        let columns = validate_matrix(predictions, self.batch_size)?;
        for (prediction, target) in predictions.chunks(columns).zip(targets.chunks(columns)) {
            for (&probability, &expected) in prediction.iter().zip(target) {
                if expected != 0.0 {
                    self.loss -= expected * (probability + 1e-8).ln();
                }
            }
            if argmax_row(prediction)? == argmax_row(target)? {
                self.correct += 1;
            }
        }
        self.samples += self.batch_size;
        Ok(())
    }
}

impl std::fmt::Display for CrossEntropyMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let loss = self.loss / self.samples as f32;
        let accuracy = self.correct as f32 / self.samples as f32;
        write!(f, "Loss: {loss:.4}; Accuracy: {accuracy:.4}")
    }
}

#[derive(Debug)]
pub struct MeanSquaredErrorMetrics {
    batch_size: usize,
    elements: usize,
    squared_error: f32,
    absolute_error: f32,
}

impl MeanSquaredErrorMetrics {
    fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            elements: 0,
            squared_error: 0.0,
            absolute_error: 0.0,
        }
    }

    pub fn update(&mut self, predictions: &[f32], targets: &[f32]) -> Result<(), LossMetricsError> {
        if predictions.len() != targets.len() {
            return Err(LossMetricsError::LengthMismatch {
                predictions_len: predictions.len(),
                targets_len: targets.len(),
            });
        }

        validate_matrix(predictions, self.batch_size)?;
        for (&prediction, &target) in predictions.iter().zip(targets) {
            let error = prediction - target;
            self.squared_error += error * error;
            self.absolute_error += error.abs();
        }
        self.elements += predictions.len();
        Ok(())
    }
}

impl std::fmt::Display for MeanSquaredErrorMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mse = self.squared_error / self.elements as f32;
        let mae = self.absolute_error / self.elements as f32;
        let rmse = mse.sqrt();
        write!(f, "MSE: {mse:.4}; MAE: {mae:.4}; RMSE: {rmse:.4}")
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum LossMetricsError {
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

pub fn argmax(predictions: &[f32], rows: usize) -> Result<Vec<usize>, LossMetricsError> {
    let columns = validate_matrix(predictions, rows)?;
    predictions.chunks(columns).map(argmax_row).collect()
}

fn argmax_row(row: &[f32]) -> Result<usize, LossMetricsError> {
    row.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(index, _)| index)
        .ok_or(LossMetricsError::EmptyRow)
}

fn validate_matrix(values: &[f32], rows: usize) -> Result<usize, LossMetricsError> {
    if values.is_empty() {
        return Err(LossMetricsError::EmptyRow);
    }
    if rows == 0 || !values.len().is_multiple_of(rows) {
        return Err(LossMetricsError::InvalidMatrixShape {
            values_len: values.len(),
            rows,
        });
    }
    Ok(values.len() / rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_metrics_report_loss_and_accuracy() {
        let mut metrics = LossKind::SoftmaxCrossEntropy.metrics(2);
        metrics
            .update(&[0.9, 0.1, 0.2, 0.8], &[1.0, 0.0, 0.0, 1.0])
            .unwrap();

        assert_eq!(metrics.to_string(), "Loss: 0.1643; Accuracy: 1.0000");
    }

    #[test]
    fn mse_metrics_report_relevant_errors() {
        let mut metrics = LossKind::MeanSquaredError.metrics(1);
        metrics.update(&[2.5, 4.0, 2.2], &[3.0, 5.0, 2.0]).unwrap();

        assert_eq!(
            metrics.to_string(),
            "MSE: 0.4300; MAE: 0.5667; RMSE: 0.6557"
        );
    }

    #[test]
    fn argmax_handles_multiple_rows() {
        assert_eq!(
            argmax(&[0.1, 0.5, 0.4, 0.9, 0.05, 0.05], 2).unwrap(),
            vec![1, 0]
        );
    }
}

use crate::core::{MetricsError, metrics};

/// Stores the training metrics for a model.
#[derive(Debug)]
pub struct TrainMetrics {
    /// The batch size used for training or evaluation.
    batch_size: usize,
    /// The number of samples processed.
    samples: usize,
    /// The total loss.
    loss: f32,
    /// The number of correct predictions.
    correct: f32,
}

impl std::fmt::Display for TrainMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let loss = self.loss / self.samples as f32;
        let accuracy = self.correct / self.samples as f32;

        write!(f, "Loss: {loss:.4}; Accuracy: {accuracy:.4}")
    }
}

impl TrainMetrics {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            samples: 0,
            loss: 0.0,
            correct: 0.0,
        }
    }

    /// Updates the metrics with epoch results.
    pub fn update(&mut self, prediction: &[f32], target: &[f32]) -> Result<(), MetricsError> {
        self.samples += self.batch_size;
        self.loss += metrics::cross_entropy_loss(prediction, target, self.batch_size)?
            * self.batch_size as f32;
        self.correct +=
            metrics::accuracy(prediction, target, self.batch_size)? * self.batch_size as f32;
        Ok(())
    }
}

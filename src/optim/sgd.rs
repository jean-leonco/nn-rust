/// Stochastic Gradient Descent (SGD) optimizer.
#[derive(Debug)]
pub struct SgdOptimizer {
    /// Learning rate for the optimizer.
    learning_rate: f32,
}

impl SgdOptimizer {
    /// Creates SGD with the supplied learning rate.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Performs a single step of the SGD optimizer and updates the parameters.
    pub fn step(&self, params: &mut [f32], gradients: &[f32]) {
        for (w, g) in params.iter_mut().zip(gradients.iter()) {
            // w = w - (learning_rate * slope)
            *w -= self.learning_rate * g;
        }
    }
}

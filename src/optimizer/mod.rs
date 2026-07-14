use crate::weights;

#[derive(Debug)]
pub struct SgdOptimizer {
    learning_rate: f32,
}

impl SgdOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    pub fn step(&self, weights: &mut weights::Weights, gradients: &[f32]) {
        for (w, g) in weights.values.iter_mut().zip(gradients.iter()) {
            // w = w - (learning_rate * slope)
            *w -= self.learning_rate * g;
        }
    }
}

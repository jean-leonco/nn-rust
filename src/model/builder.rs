use std::marker::PhantomData;

use crate::{
    core::ArenaLayout,
    model::{SessionCache, sequential::SequentialModel},
    ops::{
        Dense, Dropout, Initialization, MeanSquaredError, Operation, Relu, Sigmoid,
        SoftmaxCrossEntropy, dropout::DropoutError,
    },
};

/// No input dimension set.
pub struct NoInput;
/// Input dimension set, no layers added.
pub struct HasInputSize;
/// At least one dense layer added.
pub struct HasInputLayer;
/// Softmax added.
pub struct HasSoftmax;
/// Terminal Loss added.
pub struct HasLoss;

#[derive(Debug)]
/// Sequential model builder.
pub struct ModelBuilder<State> {
    layout: ArenaLayout,
    current_dim: usize,
    ops: Vec<Operation>,
    session_cache: Option<SessionCache>,
    _state: PhantomData<State>,
}

impl<State> ModelBuilder<State> {
    fn add_dense(&mut self, output_dim: usize, initialization: Initialization) -> Dense {
        assert!(output_dim > 0, "dense output dimension must be non-zero");
        let input_dim = self.current_dim;
        self.current_dim = output_dim;

        let weight_range = self.layout.reserve_params(input_dim * output_dim);
        let bias_range = self.layout.reserve_params(output_dim);
        let input_range = self.layout.last_activation_range.clone();
        let output_range = self.layout.reserve_activations(output_dim);

        Dense::new(
            input_dim,
            output_dim,
            input_range,
            output_range,
            weight_range,
            bias_range,
            initialization,
        )
    }

    fn add_operation<NewState>(mut self, operation: Operation) -> ModelBuilder<NewState> {
        self.ops.push(operation);

        ModelBuilder {
            layout: self.layout,
            current_dim: self.current_dim,
            ops: self.ops,
            session_cache: self.session_cache,
            _state: PhantomData,
        }
    }
}

impl Default for ModelBuilder<NoInput> {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder<NoInput> {
    /// Creates an empty model builder.
    pub fn new() -> Self {
        Self {
            layout: ArenaLayout::default(),
            current_dim: 0,
            ops: Vec::new(),
            session_cache: None,
            _state: PhantomData,
        }
    }

    /// Sets the input dimension.
    pub fn input(mut self, dim: usize) -> ModelBuilder<HasInputSize> {
        assert!(dim > 0, "input dimension must be non-zero");
        self.layout.reserve_activations(dim);
        ModelBuilder {
            layout: self.layout,
            current_dim: dim,
            ops: self.ops,
            session_cache: self.session_cache,
            _state: PhantomData,
        }
    }
}

impl ModelBuilder<HasInputSize> {
    /// Adds a Dense layer. First layer reads raw input.
    pub fn dense(
        mut self,
        output_dim: usize,
        initialization: Initialization,
    ) -> ModelBuilder<HasInputLayer> {
        let operation = self.add_dense(output_dim, initialization);
        self.add_operation(Operation::Input(operation))
    }
}

impl ModelBuilder<HasInputLayer> {
    /// Adds a Dense layer.
    pub fn dense(
        mut self,
        output_dim: usize,
        initialization: Initialization,
    ) -> ModelBuilder<HasInputLayer> {
        let operation = self.add_dense(output_dim, initialization);
        self.add_operation(Operation::Dense(operation))
    }

    /// Adds a Sigmoid activation.
    pub fn sigmoid(self) -> ModelBuilder<HasInputLayer> {
        let activation_range = self.layout.last_activation_range.clone();
        self.add_operation(Operation::Sigmoid(Sigmoid::new(activation_range)))
    }

    /// Adds a `ReLU` activation.
    pub fn relu(self) -> ModelBuilder<HasInputLayer> {
        let activation_range = self.layout.last_activation_range.clone();
        self.add_operation(Operation::Relu(Relu::new(activation_range)))
    }

    /// Adds a Dropout layer. `dropout_rate` must be in `0.0..1.0`.
    pub fn dropout(
        mut self,
        dropout_rate: f32,
    ) -> Result<ModelBuilder<HasInputLayer>, DropoutError> {
        let activation_range = self.layout.last_activation_range.clone();
        let mask_range = self.layout.reserve_masks(self.current_dim);

        Ok(self.add_operation(Operation::Dropout(Dropout::new(
            dropout_rate,
            activation_range,
            mask_range,
        )?)))
    }

    /// Adds a Softmax layer.
    pub fn softmax(self) -> ModelBuilder<HasSoftmax> {
        ModelBuilder {
            layout: self.layout,
            current_dim: self.current_dim,
            ops: self.ops,
            session_cache: self.session_cache,
            _state: PhantomData,
        }
    }

    /// Adds the final Mean Squared Error loss.
    pub fn mse(self) -> ModelBuilder<HasLoss> {
        let activation_range = self.layout.last_activation_range.clone();

        self.add_operation(Operation::MeanSquaredError(MeanSquaredError::new(
            activation_range,
        )))
    }
}

impl ModelBuilder<HasSoftmax> {
    /// Adds the Softmax Cross-entropy loss.
    pub fn cross_entropy(self) -> ModelBuilder<HasLoss> {
        let activation_range = self.layout.last_activation_range.clone();
        let output_dim = self.current_dim;

        self.add_operation(Operation::SoftmaxCrossEntropy(SoftmaxCrossEntropy::new(
            activation_range,
            output_dim,
        )))
    }
}

impl ModelBuilder<HasLoss> {
    /// Sets the session cache for inference.
    pub fn session_cache(mut self, session_cache: SessionCache) -> Self {
        self.session_cache = Some(session_cache);
        self
    }

    /// Builds the model.
    pub fn build(self) -> SequentialModel {
        SequentialModel::new(self.ops, self.layout, self.session_cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Initialization;

    #[test]
    fn test_builder_allocations() {
        let model = ModelBuilder::new()
            .input(10)
            .dense(5, Initialization::He)
            .relu()
            .softmax()
            .cross_entropy()
            .build();

        assert_eq!(model.layout.params_len, 55);
        assert_eq!(model.layout.activations_len, 15);
        assert_eq!(model.layout.max_neurons, 10);
    }

    #[test]
    #[should_panic(expected = "input dimension must be non-zero")]
    fn rejects_zero_input_dimension() {
        let _ = ModelBuilder::new().input(0);
    }

    #[test]
    #[should_panic(expected = "dense output dimension must be non-zero")]
    fn rejects_zero_dense_dimension() {
        let _ = ModelBuilder::new().input(10).dense(0, Initialization::He);
    }
}

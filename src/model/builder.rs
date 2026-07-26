use std::marker::PhantomData;

use crate::{
    core::ArenaLayout,
    model::{SessionCache, sequential::SequentialModel},
    ops::{
        DenseMeta, DropoutMeta, Initialization, Op, ReluMeta, SigmoidMeta, SoftmaxMeta,
        dropout::DropoutError,
    },
};

pub struct NoInput;
pub struct HasInputSize;
pub struct HasInputLayer;
pub struct HasLoss;

#[derive(Debug)]
pub struct ModelBuilder<State> {
    /// The arena layout for the model buffers.
    layout: ArenaLayout,
    /// The current dimension of the input/output.
    current_dim: usize,
    /// The list of operations.
    ops: Vec<Op>,
    /// The session cache.
    session_cache: Option<SessionCache>,
    _state: PhantomData<State>,
}

impl<State> ModelBuilder<State> {
    fn add_dense(&mut self, output_dim: usize, initialization: Initialization) -> DenseMeta {
        let input_dim = self.current_dim;
        self.current_dim = output_dim;

        let weight_span = self.layout.reserve_params(input_dim * output_dim);
        let bias_span = self.layout.reserve_params(output_dim);
        let input_span = self.layout.last_activation_span.clone();
        let output_span = self.layout.reserve_activations(output_dim);

        DenseMeta::new(
            input_dim,
            output_dim,
            input_span,
            output_span,
            weight_span,
            bias_span,
            initialization,
        )
    }

    fn add_node<NewState>(mut self, node: Op) -> ModelBuilder<NewState> {
        self.ops.push(node);

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
    pub fn new() -> Self {
        Self {
            layout: ArenaLayout::default(),
            current_dim: 0,
            ops: Vec::new(),
            session_cache: None,
            _state: PhantomData,
        }
    }

    /// Defines the input size of the model.
    pub fn input(mut self, dim: usize) -> ModelBuilder<HasInputSize> {
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
    /// Defines a dense layer.
    pub fn dense(
        mut self,
        output_dim: usize,
        initialization: Initialization,
    ) -> ModelBuilder<HasInputLayer> {
        let dense_meta = self.add_dense(output_dim, initialization);
        self.add_node(Op::Input(dense_meta))
    }
}

impl ModelBuilder<HasInputLayer> {
    /// Defines a dense layer.
    pub fn dense(
        mut self,
        output_dim: usize,
        initialization: Initialization,
    ) -> ModelBuilder<HasInputLayer> {
        let dense_meta = self.add_dense(output_dim, initialization);
        self.add_node(Op::Dense(dense_meta))
    }

    // Defines a sigmoid activation layer.
    pub fn sigmoid(self) -> ModelBuilder<HasInputLayer> {
        let a_span = self.layout.last_activation_span.clone();
        self.add_node(Op::Sigmoid(SigmoidMeta::new(a_span.start, a_span.end)))
    }

    // Defines a ReLU activation layer.
    pub fn relu(self) -> ModelBuilder<HasInputLayer> {
        let a_span = self.layout.last_activation_span.clone();
        self.add_node(Op::Relu(ReluMeta::new(a_span.start, a_span.end)))
    }

    // Defines a dropout layer.
    pub fn dropout(mut self, p: f32) -> Result<ModelBuilder<HasInputLayer>, DropoutError> {
        let a_span = self.layout.last_activation_span.clone();
        let m_span = self.layout.reserve_masks(self.current_dim);

        Ok(self.add_node(Op::Dropout(DropoutMeta::new(p, a_span, m_span)?)))
    }

    // Defines a softmax activation layer.
    pub fn softmax(self) -> ModelBuilder<HasLoss> {
        let a_span = self.layout.last_activation_span.clone();
        let output_size = self.current_dim;

        self.add_node(Op::Softmax(SoftmaxMeta::new(
            a_span.start,
            a_span.end,
            output_size,
        )))
    }
}

impl ModelBuilder<HasLoss> {
    pub fn session_cache(mut self, session_cache: SessionCache) -> Self {
        self.session_cache = Some(session_cache);
        self
    }

    /// Compiles the model. No further changes can be made to the model after this is called.
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
            .build();

        assert_eq!(model.layout.params_len, 55);
        assert_eq!(model.layout.activations_len, 15);
        assert_eq!(model.layout.max_neurons, 10);
    }
}

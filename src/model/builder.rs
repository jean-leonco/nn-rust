use core::ops::Range;
use std::marker::PhantomData;

use crate::{
    model::sequential::{DefinitionGraph, SequentialModel, SessionCache},
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
enum NodeType {
    Dense(usize),
    Dropout,
    Relu,
    Sigmoid,
    SoftMax,
}

#[derive(Debug)]
pub struct ModelBuilder<State> {
    /// The current offset into the parameter buffer.
    params_offset: usize,
    /// The current offset into the mask buffer.
    masks_offset: usize,
    /// The current offset into the activation buffer.
    activations_offset: usize,
    /// The current dimension of the input/output.
    current_dim: usize,
    /// The maximum dimension of the input/output.
    max_dim: usize,
    /// The last data start offset in the activation buffer.
    last_data_start: usize,
    /// The list of operations.
    ops: Vec<Op>,
    /// The session cache.
    session_cache: Option<SessionCache>,
    _state: PhantomData<State>,
}

impl<State> ModelBuilder<State> {
    fn increment_offset(&mut self, node: NodeType) -> Range<usize> {
        match node {
            NodeType::Dense(output_dim) => {
                let a_start = self.activations_offset;
                self.activations_offset += output_dim;
                let a_end = self.activations_offset;
                a_start..a_end
            }
            _ => {
                let a_start = self.activations_offset - self.current_dim;
                let a_end = self.activations_offset;
                a_start..a_end
            }
        }
    }

    fn add_dense(&mut self, output_dim: usize, initialization: Initialization) -> DenseMeta {
        let i_start = self.last_data_start;

        let input_dim = self.current_dim;
        self.current_dim = output_dim;
        if output_dim > self.max_dim {
            self.max_dim = output_dim;
        }

        let w_start = self.params_offset;
        self.params_offset += input_dim * output_dim;
        let w_end = self.params_offset;

        let b_start = w_end;
        let b_end = b_start + output_dim;
        self.params_offset += output_dim;

        let a_span = self.increment_offset(NodeType::Dense(output_dim));
        self.last_data_start = a_span.start;

        DenseMeta::new(
            input_dim,
            output_dim,
            a_span.start,
            i_start,
            w_start..w_end,
            b_start..b_end,
            initialization,
        )
    }

    fn add_node<NewState>(mut self, node: Op) -> ModelBuilder<NewState> {
        self.ops.push(node);

        ModelBuilder {
            params_offset: self.params_offset,
            activations_offset: self.activations_offset,
            masks_offset: self.masks_offset,
            current_dim: self.current_dim,
            max_dim: self.max_dim,
            last_data_start: self.last_data_start,
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
            params_offset: 0,
            activations_offset: 0,
            masks_offset: 0,
            current_dim: 0,
            max_dim: 0,
            last_data_start: 0,
            ops: Vec::new(),
            session_cache: None,
            _state: PhantomData,
        }
    }

    /// Defines the input size of the model.
    pub fn input(self, dim: usize) -> ModelBuilder<HasInputSize> {
        ModelBuilder {
            params_offset: 0,
            activations_offset: dim,
            masks_offset: 0,
            current_dim: dim,
            max_dim: dim,
            last_data_start: 0,
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
    pub fn sigmoid(mut self) -> ModelBuilder<HasInputLayer> {
        let a_span = self.increment_offset(NodeType::Sigmoid);
        self.add_node(Op::Sigmoid(SigmoidMeta::new(a_span.start, a_span.end)))
    }

    // Defines a ReLU activation layer.
    pub fn relu(mut self) -> ModelBuilder<HasInputLayer> {
        let a_span = self.increment_offset(NodeType::Relu);
        self.add_node(Op::Relu(ReluMeta::new(a_span.start, a_span.end)))
    }

    // Defines a dropout layer.
    pub fn dropout(mut self, p: f32) -> Result<ModelBuilder<HasInputLayer>, DropoutError> {
        let a_span = self.increment_offset(NodeType::Dropout);

        let m_start = self.masks_offset;
        self.masks_offset += self.current_dim;
        let m_end = self.masks_offset;

        Ok(self.add_node(Op::Dropout(DropoutMeta::new(p, a_span, m_start..m_end)?)))
    }

    // Defines a softmax activation layer.
    pub fn softmax(mut self) -> ModelBuilder<HasLoss> {
        let a_span = self.increment_offset(NodeType::SoftMax);
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
        let graph = DefinitionGraph::new(
            self.ops,
            self.params_offset,
            self.masks_offset,
            self.activations_offset,
            self.max_dim,
        );
        SequentialModel::new(graph, self.session_cache)
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

        assert_eq!(model.graph.params_size, 55);
        assert_eq!(model.graph.activation_size, 15);
        assert_eq!(model.graph.max_dimension, 10);
    }
}

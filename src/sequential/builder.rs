use std::marker::PhantomData;

use rand::Rng;

use crate::sequential::{Initializer, Sequential, SequentialExecutionNode, SequentialModel};

pub struct NoInput;
pub struct HasInput;
pub struct HasOutput;

#[derive(Debug)]
enum NodeType {
    Dense(usize),
    Dropout,
    Relu,
    Sigmoid,
    SoftmaxCrossEntropy,
}

pub struct SequentialBuilder<State> {
    weights_offset: usize,
    a_offset: usize,
    current_dim: usize,
    max_dim: usize,
    last_data_start: usize,
    nodes: Vec<SequentialExecutionNode>,
    _state: PhantomData<State>,
}

pub type NewSequentialBuilder = SequentialBuilder<NoInput>;

impl SequentialBuilder<NoInput> {
    pub fn new() -> Self {
        Self {
            weights_offset: 0,
            a_offset: 0,
            current_dim: 0,
            max_dim: 0,
            last_data_start: 0,
            nodes: Vec::new(),
            _state: PhantomData,
        }
    }

    pub fn input(self, dim: usize) -> SequentialBuilder<HasInput> {
        SequentialBuilder {
            weights_offset: 0,
            a_offset: 0,
            current_dim: dim,
            max_dim: dim,
            last_data_start: 0,
            nodes: self.nodes,
            _state: PhantomData,
        }
    }
}

impl<State> SequentialBuilder<State> {
    fn increment_offset(&mut self, node: NodeType) -> (usize, usize) {
        match node {
            NodeType::Dense(output_dim) => {
                let a_start = self.a_offset;
                self.a_offset += output_dim;
                let a_end = self.a_offset;
                (a_start, a_end)
            }
            _ => {
                let a_start = self.a_offset - self.current_dim;
                let a_end = self.a_offset;
                (a_start, a_end)
            }
        }
    }

    fn add_node<NewState>(mut self, node: SequentialExecutionNode) -> SequentialBuilder<NewState> {
        self.nodes.push(node);

        SequentialBuilder {
            weights_offset: self.weights_offset,
            a_offset: self.a_offset,
            current_dim: self.current_dim,
            max_dim: self.max_dim,
            last_data_start: self.last_data_start,
            nodes: self.nodes,
            _state: PhantomData,
        }
    }
}

impl SequentialBuilder<HasInput> {
    pub fn dense(
        mut self,
        output_dim: usize,
        initializer: Initializer,
    ) -> SequentialBuilder<HasInput> {
        let input_a_start = self.last_data_start;

        let input_dim = self.current_dim;
        self.current_dim = output_dim;
        if output_dim > self.max_dim {
            self.max_dim = output_dim;
        }

        let w_start = self.weights_offset;
        self.weights_offset += input_dim * output_dim;
        let w_end = self.weights_offset;

        let b_start = w_end;
        let b_end = b_start + output_dim;
        self.weights_offset += output_dim;

        let (a_start, a_end) = self.increment_offset(NodeType::Dense(output_dim));
        self.last_data_start = a_start;

        self.add_node(SequentialExecutionNode::Dense {
            input_dim,
            output_dim,
            initializer,
            w_start,
            b_start,
            w_end,
            b_end,
            a_start,
            a_end,
            input_a_start,
        })
    }

    pub fn sigmoid(mut self) -> SequentialBuilder<HasInput> {
        let (a_start, a_end) = self.increment_offset(NodeType::Sigmoid);

        self.add_node(SequentialExecutionNode::Sigmoid { a_start, a_end })
    }

    pub fn relu(mut self) -> SequentialBuilder<HasInput> {
        let (a_start, a_end) = self.increment_offset(NodeType::Relu);

        self.add_node(SequentialExecutionNode::Relu { a_start, a_end })
    }

    pub fn dropout(mut self, p: f32) -> SequentialBuilder<HasInput> {
        let (a_start, a_end) = self.increment_offset(NodeType::Dropout);

        let mask_start = self.a_offset;
        self.a_offset += self.current_dim;
        let mask_end = self.a_offset;

        let inv_p = 1.0 / (1.0 - p);

        self.add_node(SequentialExecutionNode::Dropout {
            p,
            inv_p,
            data_start: a_start,
            data_end: a_end,
            mask_start,
            mask_end,
        })
    }

    pub fn softmax_cross_entropy(mut self) -> SequentialBuilder<HasOutput> {
        let (a_start, a_end) = self.increment_offset(NodeType::SoftmaxCrossEntropy);
        let output_dim = self.current_dim;

        self.add_node(SequentialExecutionNode::SoftmaxCrossEntropy {
            a_start,
            a_end,
            output_dim,
        })
    }
}

impl SequentialBuilder<HasOutput> {
    pub fn build<R: Rng + ?Sized>(self, rng: &mut R) -> crate::sequential::sequential_model::Result<SequentialModel> {
        let blueprint = Sequential::new(
            self.nodes,
            self.weights_offset,
            self.a_offset,
            self.max_dim,
            self.last_data_start,
        );
        SequentialModel::new(blueprint, rng)
    }
}

use std::marker::PhantomData;

use crate::{
    layer::{
        Layer,
        dense::{Dense, Initialization},
        relu::Relu,
        sigmoid::Sigmoid,
        softmax_cross_entropy::SoftmaxCrossEntropy,
    },
    model::model::Model,
};

pub struct NoInput;
pub struct HasInput;
pub struct HasLoss;

pub struct ModelBuilder<State> {
    layers: Vec<Box<dyn Layer>>,
    topology: Vec<usize>,
    current_dim: usize,
    _state: PhantomData<State>,
}

impl ModelBuilder<NoInput> {
    pub(crate) fn new() -> Self {
        Self {
            layers: Vec::new(),
            topology: Vec::new(),
            current_dim: 0,
            _state: PhantomData,
        }
    }

    pub fn input(self, size: usize) -> ModelBuilder<HasInput> {
        ModelBuilder {
            layers: self.layers,
            topology: self.topology,
            current_dim: size,
            _state: PhantomData,
        }
    }
}

impl<State> ModelBuilder<State> {
    fn add_layer<NewState>(
        mut self,
        layer: impl Layer + 'static,
        new_dim: Option<usize>,
    ) -> ModelBuilder<NewState> {
        self.topology.push(self.current_dim);
        self.layers.push(Box::new(layer));

        ModelBuilder {
            layers: self.layers,
            topology: self.topology,
            current_dim: match new_dim {
                Some(value) => value,
                None => self.current_dim,
            },
            _state: PhantomData,
        }
    }
}

impl ModelBuilder<HasInput> {
    pub fn dense(
        self,
        output_size: usize,
        initialization: Initialization,
    ) -> ModelBuilder<HasInput> {
        let input_size = self.current_dim;

        self.add_layer(
            Dense::new(input_size, output_size, initialization),
            Some(output_size),
        )
    }

    pub fn sigmoid(self) -> ModelBuilder<HasInput> {
        self.add_layer(Sigmoid::new(), None)
    }

    pub fn relu(self) -> ModelBuilder<HasInput> {
        self.add_layer(Relu::new(), None)
    }

    pub fn softmax_cross_entropy(self) -> ModelBuilder<HasLoss> {
        self.add_layer(SoftmaxCrossEntropy::new(), None)
    }
}

impl ModelBuilder<HasLoss> {
    pub fn build(mut self) -> Model {
        self.topology.push(self.current_dim);

        Model::new(self.layers, self.topology)
    }
}

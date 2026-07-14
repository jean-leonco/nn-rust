pub use builder::SequentialBuilder;
pub use execution_node::{Initializer, SequentialExecutionNode};
pub use sequential_model::SequentialModel;

pub mod builder;
pub mod execution_node;
pub mod sequential_model;

#[derive(Debug)]
pub struct Sequential {
    pub nodes: Vec<SequentialExecutionNode>,
    pub weights_size: usize,
    pub a_size: usize,
    pub max_dim: usize,
    pub last_data_start: usize,
}

impl Sequential {
    pub fn new(
        nodes: Vec<execution_node::SequentialExecutionNode>,
        weights_size: usize,
        a_size: usize,
        max_dim: usize,
        last_data_start: usize,
    ) -> Self {
        Self {
            nodes,
            weights_size,
            a_size,
            max_dim,
            last_data_start,
        }
    }

    pub fn builder() -> builder::NewSequentialBuilder {
        SequentialBuilder::new()
    }
}

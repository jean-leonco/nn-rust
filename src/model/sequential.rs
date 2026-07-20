use std::{
    io::{Read, Write},
    path::Path,
};

use rand::Rng;
use thiserror::Error;

use crate::{
    core::serialization,
    model::{Session, builder},
    ops::{Op, OpSerializationError, initialization},
};

/// Model graph and definition.
#[derive(Debug)]
pub struct DefinitionGraph {
    /// The sequence of operations to be performed by the model.
    pub ops: Vec<Op>,
    /// The total number of parameters in the model, both weights and biases.
    pub params_size: usize,
    /// The total number of mask parameters in the model. Only present when using dropout layers.
    pub mask_size: usize,
    /// The total number of activation parameters in the model.
    pub activation_size: usize,
    /// The maximum number of neurons in the model.
    pub max_dimension: usize,
}

impl DefinitionGraph {
    pub fn new(
        ops: Vec<Op>,
        params_size: usize,
        mask_size: usize,
        activation_size: usize,
        max_dimension: usize,
    ) -> Self {
        Self {
            ops,
            params_size,
            mask_size,
            activation_size,
            max_dimension,
        }
    }
}

#[derive(Error, Debug)]
pub enum SequentialModelSerializationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("File magic number mismatch")]
    MagicNumberMismatch,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serialization::SerializationError),
    #[error("Op serialization error: {0}")]
    OpSerialization(#[from] OpSerializationError),
}

const MAGIC_NUMBER: [u8; 4] = *b"NNRS";
const VERSION: u32 = 1;

/// Represents a model consisting of its parameters and definition.
#[derive(Debug)]
pub struct SequentialModel {
    /// The definition used during the model lifecycle. It dictates the structure and operations used.
    pub graph: DefinitionGraph,
    /// Stores the weights and biases of the model.
    pub params: Vec<f32>,
}

impl SequentialModel {
    pub fn new(graph: DefinitionGraph) -> Self {
        Self {
            params: Vec::with_capacity(graph.params_size),
            graph,
        }
    }

    /// Initializes the model parameters according to the dense layers initialization method.
    pub fn initialize_params<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<(), initialization::InitializationError> {
        for op in &self.graph.ops {
            match op {
                Op::Input(meta) | Op::Dense(meta) => {
                    meta.initialization.init(
                        meta.input_dim,
                        meta.output_dim,
                        &mut self.params,
                        rng,
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn builder() -> builder::ModelBuilder<builder::NoInput> {
        builder::ModelBuilder::new()
    }

    /// Runs the model on the given input data and returns the prediction.
    pub fn predict(&mut self, x: &[f32]) -> Vec<f32> {
        let input_dim = match &self.graph.ops[0] {
            Op::Input(meta) => meta.input_dim,
            _ => panic!("First layer must be Input"),
        };
        let batch_size = x.len() / input_dim;
        let mut session = Session::new(&self.graph, batch_size);
        session.infer(&mut self.params, x).to_vec()
    }

    /// Saves the model to the given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SequentialModelSerializationError> {
        let mut file = std::fs::File::create(path)?;

        file.write_all(&MAGIC_NUMBER)?;
        serialization::write_u32(&mut file, VERSION)?;

        serialization::write_u32(&mut file, self.graph.params_size as u32)?;
        serialization::write_u32(&mut file, self.graph.mask_size as u32)?;
        serialization::write_u32(&mut file, self.graph.activation_size as u32)?;
        serialization::write_u32(&mut file, self.graph.max_dimension as u32)?;

        let n_ops = self.graph.ops.len() as u32;
        serialization::write_u32(&mut file, n_ops)?;
        for op in &self.graph.ops {
            file.write_all(&op.to_bytes())?;
        }

        let params_bytes: &[u8] = bytemuck::cast_slice(&self.params);
        serialization::write_u32(&mut file, self.params.len() as u32)?;
        file.write_all(params_bytes)?;
        Ok(())
    }

    /// Loads a model from the given path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, SequentialModelSerializationError> {
        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != MAGIC_NUMBER {
            return Err(SequentialModelSerializationError::MagicNumberMismatch);
        }

        let version = serialization::read_u32(&mut file)?;
        if version != VERSION {
            return Err(SequentialModelSerializationError::UnsupportedVersion(
                version,
            ));
        }

        let params_size = serialization::read_u32(&mut file)? as usize;
        let mask_size = serialization::read_u32(&mut file)? as usize;
        let activation_size = serialization::read_u32(&mut file)? as usize;
        let max_dimension = serialization::read_u32(&mut file)? as usize;

        let n_ops = serialization::read_u32(&mut file)?;
        let mut ops = Vec::new();
        for _ in 0..n_ops {
            ops.push(Op::from_reader(&mut file)?);
        }

        let graph =
            DefinitionGraph::new(ops, params_size, mask_size, activation_size, max_dimension);

        let n_params = serialization::read_u32(&mut file)? as usize;
        let mut params_bytes = vec![0u8; n_params * 4];
        file.read_exact(&mut params_bytes)?;

        let params: Vec<f32> = bytemuck::cast_slice(&params_bytes).to_vec();

        Ok(Self { graph, params })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Initialization;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn test_model_initialization_length() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut model = SequentialModel::builder()
            .input(10)
            .dense(20, Initialization::He)
            .dense(5, Initialization::Xavier)
            .softmax()
            .build();

        model.initialize_params(&mut rng).unwrap();

        assert_eq!(model.params.len(), model.graph.params_size);
    }
}

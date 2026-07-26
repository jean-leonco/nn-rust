use std::{
    io::{Read, Write},
    path::Path,
};

use rand::Rng;
use thiserror::Error;

use crate::{
    core::{ArenaLayout, Encodable, serialization},
    model::{Session, SessionCache, builder},
    ops::{OpSerializationError, Operation, initialization},
};

#[derive(Error, Debug)]
/// Errors during model serialization.
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

/// Sequential model with parameters and operation plan.
#[derive(Debug)]
pub struct SequentialModel {
    /// Operations executed during training.
    pub train_ops: Vec<Operation>,
    /// Operations executed during inference.
    pub inference_ops: Vec<Operation>,
    /// Arena layout for session buffer sizing.
    pub layout: ArenaLayout,
    /// Model parameters.
    pub params: Vec<f32>,
    session_cache: SessionCache,
}

impl SequentialModel {
    /// Creates a model from operations and layout.
    pub fn new(
        ops: Vec<Operation>,
        layout: ArenaLayout,
        session_cache: Option<SessionCache>,
    ) -> Self {
        Self {
            params: vec![0.0; layout.params_len],
            session_cache: session_cache.unwrap_or_default(),
            inference_ops: Operation::inference_ops(&ops),
            train_ops: ops,
            layout,
        }
    }

    /// Initializes parameters using each layer initialization scheme.
    pub fn initialize_params<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<(), initialization::InitializationError> {
        for op in &self.train_ops {
            match op {
                Operation::Input(meta) | Operation::Dense(meta) => {
                    let weights = &mut self.params[meta.weight_range.clone()];

                    meta.initialization
                        .init(meta.input_dim, meta.output_dim, weights, rng)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Creates a model builder.
    pub fn builder() -> builder::ModelBuilder<builder::NoInput> {
        builder::ModelBuilder::new()
    }

    /// Runs inference.
    pub fn predict(&mut self, x: &[f32]) -> Vec<f32> {
        let input_dim = match &self.train_ops[0] {
            Operation::Input(meta) => meta.input_dim,
            _ => panic!("First layer must be Input"),
        };
        let batch_size = x.len() / input_dim;

        let session = if let Some(session) = self.session_cache.get(batch_size) {
            session
        } else {
            self.session_cache
                .put(batch_size, Session::new(self, batch_size, None));
            self.session_cache.get(batch_size).unwrap()
        };

        session
            .forward(&self.inference_ops, &mut self.params, x)
            .to_vec()
    }

    /// Saves the model to `path`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SequentialModelSerializationError> {
        let mut file = std::fs::File::create(path)?;

        file.write_all(&MAGIC_NUMBER)?;
        serialization::write_u32(&mut file, VERSION)?;
        self.layout.encode(&mut file)?;

        let n_ops = self.train_ops.len() as u32;
        serialization::write_u32(&mut file, n_ops)?;
        for op in &self.train_ops {
            file.write_all(&op.to_bytes())?;
        }

        let params_bytes: &[u8] = bytemuck::cast_slice(&self.params);
        serialization::write_u32(&mut file, self.params.len() as u32)?;
        file.write_all(params_bytes)?;
        Ok(())
    }

    /// Loads a model from `path`.
    pub fn load<P: AsRef<Path>>(
        path: P,
        session_cache: Option<SessionCache>,
    ) -> Result<Self, SequentialModelSerializationError> {
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

        let layout = ArenaLayout::decode(&mut file)?;

        let n_ops = serialization::read_u32(&mut file)?;
        let mut ops = Vec::new();
        for _ in 0..n_ops {
            ops.push(Operation::decode(&mut file)?);
        }

        let n_params = serialization::read_u32(&mut file)? as usize;
        let mut params_bytes = vec![0u8; n_params * 4];
        file.read_exact(&mut params_bytes)?;
        let params: Vec<f32> = bytemuck::cast_slice(&params_bytes).to_vec();

        Ok(Self {
            inference_ops: Operation::inference_ops(&ops),
            train_ops: ops,
            layout,
            params,
            session_cache: session_cache.unwrap_or_default(),
        })
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

        assert_eq!(model.params.len(), model.layout.params_len);
    }
}

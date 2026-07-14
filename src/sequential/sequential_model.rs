use std::{
    io::{Read, Write},
    path::Path,
};
use thiserror::Error;

use rand::{
    Rng, SeedableRng,
    rngs::{SmallRng, SysRng},
};

use crate::{execution_session::ExecutionSession, sequential, weights, io::{read_u32, write_u32}};

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Weights error: {0}")]
    Weights(#[from] crate::weights::WeightsError),
    #[error("Random generation error: {0}")]
    Rand(String),
    #[error("Execution error: {0}")]
    Execution(#[from] crate::execution_session::ExecutionError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Crate IO error: {0}")]
    CrateIo(#[from] crate::io::IoError),
    #[error("Node error: {0}")]
    Node(#[from] crate::sequential::execution_node::NodeError),
    #[error("File magic number mismatch")]
    MagicNumberMismatch,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
}
pub type Result<T> = std::result::Result<T, ModelError>;

const MAGIC_NUMBER: [u8; 4] = *b"NNRS";
const VERSION: u32 = 1;

#[derive(Debug)]
pub struct SequentialModel {
    pub blueprint: sequential::Sequential,
    pub weights: weights::Weights,
}

impl SequentialModel {
    pub fn new<R: Rng + ?Sized>(blueprint: sequential::Sequential, rng: &mut R) -> Result<Self> {
        Ok(Self {
            weights: weights::Weights::init(&blueprint, rng)?,
            blueprint,
        })
    }

    pub fn predict(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut rng = SmallRng::try_from_rng(&mut SysRng).map_err(|e| ModelError::Rand(e.to_string()))?;
        let mut session = ExecutionSession::new(&self.blueprint, &mut rng, 1);

        let output = session.forward(&self.weights, x)?;
        Ok(output.to_vec())
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = std::fs::File::create(path)?;

        // 0. Write Header
        file.write_all(&MAGIC_NUMBER)?;
        write_u32(&mut file, VERSION)?;

        // 1. Write blueprint metadata
        write_u32(&mut file, self.blueprint.weights_size as u32)?;
        write_u32(&mut file, self.blueprint.a_size as u32)?;
        write_u32(&mut file, self.blueprint.max_dim as u32)?;
        write_u32(&mut file, self.blueprint.last_data_start as u32)?;

        let n_nodes = self.blueprint.nodes.len() as u32;
        write_u32(&mut file, n_nodes)?;
        for node in &self.blueprint.nodes {
            file.write_all(&node.to_bytes())?;
        }

        let weights_bytes: &[u8] = bytemuck::cast_slice(&self.weights.values);
        write_u32(&mut file, self.weights.values.len() as u32)?;
        file.write_all(weights_bytes)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        
        // 0. Verify Header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != MAGIC_NUMBER {
            return Err(ModelError::MagicNumberMismatch);
        }
        
        let version = read_u32(&mut file)?;
        if version != VERSION {
            return Err(ModelError::UnsupportedVersion(version));
        }

        let weights_size = read_u32(&mut file)? as usize;
        let a_size = read_u32(&mut file)? as usize;
        let max_dim = read_u32(&mut file)? as usize;
        let last_data_start = read_u32(&mut file)? as usize;

        let n_nodes = read_u32(&mut file)?;
        let mut nodes = Vec::new();
        for _ in 0..n_nodes {
            nodes.push(sequential::SequentialExecutionNode::from_reader(&mut file)?);
        }

        let blueprint =
            sequential::Sequential::new(nodes, weights_size, a_size, max_dim, last_data_start);

        let n_weights = read_u32(&mut file)? as usize;
        let mut weights_bytes = vec![0u8; n_weights * 4];
        file.read_exact(&mut weights_bytes)?;

        let values: Vec<f32> = bytemuck::cast_slice(&weights_bytes).to_vec();

        let weights = weights::Weights { values };

        Ok(Self { blueprint, weights })
    }
}

pub mod dense;
pub mod dropout;
pub(crate) mod gemm;
pub mod initialization;
pub mod relu;
pub mod sigmoid;
pub mod softmax;

use std::io::Read;

pub use dense::{DenseEncodingError, DenseMeta};
pub use dropout::{DropoutEncodingError, DropoutMeta};
pub use initialization::Initialization;
pub use relu::ReluMeta;
pub use sigmoid::SigmoidMeta;
pub use softmax::SoftmaxMeta;
use thiserror::Error;

use crate::core::{Encodable, serialization};

#[derive(Debug, Clone)]
/// A single operation in a sequential model execution plan.
pub enum Operation {
    /// First dense layer. Reads raw input instead of the activation arena.
    Input(DenseMeta),
    /// Dense layer reading from the activation arena.
    Dense(DenseMeta),
    /// Dropout regularization. Training only.
    Dropout(DropoutMeta),
    /// `ReLU` activation.
    Relu(ReluMeta),
    /// Sigmoid activation.
    Sigmoid(SigmoidMeta),
    /// Softmax output.
    Softmax(SoftmaxMeta),
}

#[derive(Error, Debug)]
/// Errors during `Operation` serialization.
pub enum OpSerializationError {
    #[error("Dense serialization error: {0}")]
    Dense(#[from] DenseEncodingError),
    #[error("Dropout serialization error: {0}")]
    Dropout(#[from] DropoutEncodingError),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serialization::SerializationError),
    #[error("Unknown operation variant: {0}")]
    UnknownOperationVariant(u8),
}

impl Operation {
    pub const INPUT_ID: u8 = 0;
    pub const DENSE_ID: u8 = 1;
    pub const DROPOUT_ID: u8 = 2;
    pub const RELU_ID: u8 = 3;
    pub const SIGMOID_ID: u8 = 4;
    pub const SOFTMAX_ID: u8 = 5;

    fn as_u8(&self) -> u8 {
        match self {
            Self::Input(..) => Self::INPUT_ID,
            Self::Dense(..) => Self::DENSE_ID,
            Self::Dropout(..) => Self::DROPOUT_ID,
            Self::Relu(..) => Self::RELU_ID,
            Self::Sigmoid(..) => Self::SIGMOID_ID,
            Self::Softmax(..) => Self::SOFTMAX_ID,
        }
    }

    /// Encodes the operation.
    pub fn encode(&self, writer: &mut impl std::io::Write) -> Result<(), OpSerializationError> {
        let mut buf = Vec::new();

        let buf = match self {
            Self::Input(meta) | Self::Dense(meta) => {
                buf.reserve(1 + meta.encoded_len());
                buf.push(self.as_u8());
                meta.encode(&mut buf)?;
                buf
            }
            Self::Dropout(meta) => {
                buf.reserve(1 + meta.encoded_len());
                buf.push(self.as_u8());
                meta.encode(&mut buf)?;
                buf
            }
            Self::Relu(meta) => {
                buf.reserve(1 + meta.encoded_len());
                buf.push(self.as_u8());
                meta.encode(&mut buf)?;
                buf
            }
            Self::Sigmoid(meta) => {
                buf.reserve(1 + meta.encoded_len());
                buf.push(self.as_u8());
                meta.encode(&mut buf)?;
                buf
            }
            Self::Softmax(meta) => {
                buf.reserve(1 + meta.encoded_len());
                buf.push(self.as_u8());
                meta.encode(&mut buf)?;
                buf
            }
        };
        writer
            .write_all(&buf)
            .map_err(serialization::SerializationError::Io)?;

        Ok(())
    }

    /// Decodes an operation.
    pub fn decode(reader: &mut impl Read) -> Result<Self, OpSerializationError> {
        let mut tag = [0u8; 1];
        reader
            .read_exact(&mut tag)
            .map_err(serialization::SerializationError::Io)?;

        match tag[0] {
            Self::INPUT_ID => Ok(Self::Input(DenseMeta::decode(reader)?)),
            Self::DENSE_ID => Ok(Self::Dense(DenseMeta::decode(reader)?)),
            Self::DROPOUT_ID => Ok(Self::Dropout(DropoutMeta::decode(reader)?)),
            Self::RELU_ID => Ok(Self::Relu(ReluMeta::decode(reader)?)),
            Self::SIGMOID_ID => Ok(Self::Sigmoid(SigmoidMeta::decode(reader)?)),
            Self::SOFTMAX_ID => Ok(Self::Softmax(SoftmaxMeta::decode(reader)?)),
            _ => Err(OpSerializationError::UnknownOperationVariant(tag[0])),
        }
    }

    /// Filters out training-only operations (e.g. dropout).
    pub fn inference_ops(base_ops: &[Operation]) -> Vec<Operation> {
        base_ops
            .iter()
            .filter_map(|op| match op {
                Self::Dropout(_) => None,
                _ => Some(op.clone()),
            })
            .collect::<Vec<_>>()
    }
}

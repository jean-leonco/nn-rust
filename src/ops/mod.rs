use std::io::Read;
use thiserror::Error;

pub mod dense;
pub mod dropout;
pub(crate) mod gemm;
pub mod initialization;
pub mod loss_kind;
pub mod mse;
pub mod relu;
pub mod sigmoid;
pub mod softmax_cross_entropy;

pub use dense::{Dense, DenseEncodingError};
pub use dropout::{Dropout, DropoutEncodingError};
pub use initialization::Initialization;
pub use loss_kind::{LossKind, LossMetrics, LossMetricsError, argmax};
pub use mse::MeanSquaredError;
pub use relu::Relu;
pub use sigmoid::Sigmoid;
pub use softmax_cross_entropy::SoftmaxCrossEntropy;

use crate::core::{Encodable, serialization};

macro_rules! encode_ops {
    ( $( $variant:ident($operation:ident) => $tag:expr ),+ $(,)? ) => {
        pub(crate) fn encode(&self, writer: &mut impl std::io::Write) -> Result<(), OpSerializationError> {
            let mut buf = Vec::new();
            let buf = match self {
                $(
                    Self::$variant($operation) => {
                        buf.reserve(1 + $operation.encoded_len());
                        buf.push($tag);
                        $operation.encode(&mut buf)?;
                        buf
                    }
                )+
            };
            writer
                .write_all(&buf)
                .map_err(serialization::SerializationError::Io)?;
            Ok(())
        }
    };
}

#[derive(Debug, Clone)]
/// A single operation in a sequential model execution plan.
pub enum Operation {
    /// First dense layer. Reads raw input instead of the activation arena.
    Input(Dense),
    /// Dense layer reading from the activation arena.
    Dense(Dense),
    /// Dropout regularization. Training only.
    Dropout(Dropout),
    /// `ReLU` activation.
    Relu(Relu),
    /// Sigmoid activation.
    Sigmoid(Sigmoid),
    /// Softmax Cross-Entropy Loss.
    SoftmaxCrossEntropy(SoftmaxCrossEntropy),
    /// Mean Squared Error Loss.
    MeanSquaredError(MeanSquaredError),
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
    pub const SOFTMAX_CROSS_ENTROPY_ID: u8 = 5;
    pub const MSE_ID: u8 = 6;

    encode_ops! {
        Input(operation) => Self::INPUT_ID,
        Dense(operation) => Self::DENSE_ID,
        Dropout(operation) => Self::DROPOUT_ID,
        Relu(operation) => Self::RELU_ID,
        Sigmoid(operation) => Self::SIGMOID_ID,
        SoftmaxCrossEntropy(operation) => Self::SOFTMAX_CROSS_ENTROPY_ID,
        MeanSquaredError(operation) => Self::MSE_ID,
    }

    /// Decodes an operation.
    pub fn decode(reader: &mut impl Read) -> Result<Self, OpSerializationError> {
        let mut tag = [0u8; 1];
        reader
            .read_exact(&mut tag)
            .map_err(serialization::SerializationError::Io)?;

        match tag[0] {
            Self::INPUT_ID => Ok(Self::Input(Dense::decode(reader)?)),
            Self::DENSE_ID => Ok(Self::Dense(Dense::decode(reader)?)),
            Self::DROPOUT_ID => Ok(Self::Dropout(Dropout::decode(reader)?)),
            Self::RELU_ID => Ok(Self::Relu(Relu::decode(reader)?)),
            Self::SIGMOID_ID => Ok(Self::Sigmoid(Sigmoid::decode(reader)?)),
            Self::SOFTMAX_CROSS_ENTROPY_ID => Ok(Self::SoftmaxCrossEntropy(
                SoftmaxCrossEntropy::decode(reader)?,
            )),
            Self::MSE_ID => Ok(Self::MeanSquaredError(MeanSquaredError::decode(reader)?)),
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

    pub fn loss_kind(&self) -> Option<LossKind> {
        match self {
            Self::SoftmaxCrossEntropy(_) => Some(LossKind::SoftmaxCrossEntropy),
            Self::MeanSquaredError(_) => Some(LossKind::MeanSquaredError),
            _ => None,
        }
    }
}

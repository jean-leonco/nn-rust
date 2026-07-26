pub mod dense;
pub mod dropout;
pub(crate) mod gemm;
pub mod initialization;
pub mod relu;
pub mod sigmoid;
pub mod softmax;

use std::io::Read;

pub use dense::DenseMeta;
pub use dropout::DropoutMeta;
pub use initialization::Initialization;
pub use relu::ReluMeta;
pub use sigmoid::SigmoidMeta;
pub use softmax::SoftmaxMeta;
use thiserror::Error;

use crate::{
    core::{SerializationError, serialization},
    ops::initialization::InitializationError,
};

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
    #[error("Serialization error: {0}")]
    SerializationError(#[from] SerializationError),
    #[error("Standard IO error: {0}")]
    StdIo(#[from] std::io::Error),
    #[error("Initialization error: {0}")]
    InitializationError(#[from] InitializationError),
    #[error("Unknown operation variant: {0}")]
    UnknownOperationVariant(u8),
    #[error("Dropout error: {0}")]
    DropoutError(#[from] dropout::DropoutError),
}

impl Operation {
    /// Converts operation to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = match self {
            Self::Input { .. } | Self::Dense { .. } => 1 + 8 * 4 + 1,
            Self::Dropout { .. } => 1 + 4 + 4 * 4,
            Self::Relu { .. } | Self::Sigmoid { .. } => 1 + 2 * 4,
            Self::Softmax { .. } => 1 + 3 * 4,
        };
        let mut buf = Vec::with_capacity(size);

        match self {
            Self::Input(meta) | Self::Dense(meta) => {
                buf.push(u8::from(!matches!(self, Self::Input(_))));
                serialization::write_u32(&mut buf, meta.input_dim as u32).unwrap();
                serialization::write_u32(&mut buf, meta.output_dim as u32).unwrap();

                serialization::write_range(&mut buf, meta.relative_input_range.clone()).unwrap();
                serialization::write_range(&mut buf, meta.relative_output_range.clone()).unwrap();
                serialization::write_range(&mut buf, meta.weight_range.clone()).unwrap();
                serialization::write_range(&mut buf, meta.bias_range.clone()).unwrap();
                buf.push(meta.initialization.as_u8());
            }
            Self::Dropout(meta) => {
                buf.push(2);
                serialization::write_f32(&mut buf, meta.survival_rate).unwrap();
                serialization::write_range(&mut buf, meta.relative_activation_range.clone())
                    .unwrap();
                serialization::write_range(&mut buf, meta.relative_mask_range.clone()).unwrap();
            }
            Self::Relu(meta) => {
                buf.push(3);
                serialization::write_range(&mut buf, meta.relative_activation_range.clone())
                    .unwrap();
            }
            Self::Sigmoid(meta) => {
                buf.push(4);
                serialization::write_range(&mut buf, meta.relative_activation_range.clone())
                    .unwrap();
            }
            Self::Softmax(meta) => {
                buf.push(5);
                serialization::write_range(&mut buf, meta.relative_activation_range.clone())
                    .unwrap();
                serialization::write_u32(&mut buf, meta.output_dim as u32).unwrap();
            }
        }

        buf
    }

    /// Deserializes an operation from a reader.
    pub fn decode(reader: &mut impl Read) -> Result<Operation, OpSerializationError> {
        let mut variant = [0u8; 1];
        reader.read_exact(&mut variant)?;

        match variant[0] {
            0 | 1 => {
                let input_dim = serialization::read_u32(reader)? as usize;
                let output_dim = serialization::read_u32(reader)? as usize;
                let input_range = serialization::read_range(reader)?;
                let output_range = serialization::read_range(reader)?;
                let weight_range = serialization::read_range(reader)?;
                let bias_range = serialization::read_range(reader)?;

                let mut init_buf = [0u8; 1];
                reader.read_exact(&mut init_buf)?;
                let initialization = Initialization::try_from(init_buf[0])
                    .map_err(|_| OpSerializationError::UnknownOperationVariant(init_buf[0]))?;

                let meta = DenseMeta::new(
                    input_dim,
                    output_dim,
                    input_range,
                    output_range,
                    weight_range,
                    bias_range,
                    initialization,
                );

                if variant[0] == 0 {
                    Ok(Self::Input(meta))
                } else {
                    Ok(Self::Dense(meta))
                }
            }
            2 => {
                let p = serialization::read_f32(reader)?;
                let activation_range = serialization::read_range(reader)?;
                let mask_range = serialization::read_range(reader)?;
                Ok(Self::Dropout(DropoutMeta::new(
                    p,
                    activation_range,
                    mask_range,
                )?))
            }
            3 => {
                let activation_range = serialization::read_range(reader)?;
                Ok(Self::Relu(ReluMeta::new(activation_range)))
            }
            4 => {
                let activation_range = serialization::read_range(reader)?;
                Ok(Self::Sigmoid(SigmoidMeta::new(activation_range)))
            }
            5 => {
                let activation_range = serialization::read_range(reader)?;
                let output_dim = serialization::read_u32(reader)? as usize;
                Ok(Self::Softmax(SoftmaxMeta::new(
                    activation_range,
                    output_dim,
                )))
            }
            _ => Err(OpSerializationError::UnknownOperationVariant(variant[0])),
        }
    }

    /// Filters out training-only operations (e.g. dropout).
    pub fn inference_ops(base_ops: &[Operation]) -> Vec<Operation> {
        base_ops
            .iter()
            .filter_map(|op| match op {
                Operation::Dropout(_) => None,
                _ => Some(op.clone()),
            })
            .collect::<Vec<_>>()
    }
}

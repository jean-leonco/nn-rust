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

#[derive(Debug)]
pub enum Op {
    Input(DenseMeta),
    Dense(DenseMeta),
    Dropout(DropoutMeta),
    Relu(ReluMeta),
    Sigmoid(SigmoidMeta),
    Softmax(SoftmaxMeta),
}

#[derive(Error, Debug)]
pub enum OpSerializationError {
    #[error("Serialization error: {0}")]
    SerializationError(#[from] SerializationError),
    #[error("Standard IO error: {0}")]
    StdIo(#[from] std::io::Error),
    #[error("Initialization error: {0}")]
    InitializationError(#[from] InitializationError),
    #[error("Unknown execution node variant: {0}")]
    UnknownNodeVariant(u8),
    #[error("Bernoulli error: {0}")]
    BernoulliDistr(#[from] rand::distr::BernoulliError),
}

impl Op {
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
                buf.push(if matches!(self, Self::Input(_)) { 0 } else { 1 });
                serialization::write_u32(&mut buf, meta.input_dim as u32).unwrap();
                serialization::write_u32(&mut buf, meta.output_dim as u32).unwrap();
                serialization::write_u32(&mut buf, meta.a_start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.i_start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.weight_offsets.start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.weight_offsets.end as u32).unwrap();
                serialization::write_u32(&mut buf, meta.bias_offsets.start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.bias_offsets.end as u32).unwrap();
                buf.push(meta.initialization.to_u8());
            }
            Self::Dropout(meta) => {
                buf.push(2);
                serialization::write_f32(&mut buf, meta.p).unwrap();
                serialization::write_u32(&mut buf, meta.a_start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.a_end as u32).unwrap();
                serialization::write_u32(&mut buf, meta.m_start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.m_end as u32).unwrap();
            }
            Self::Relu(meta) => {
                buf.push(3);
                serialization::write_u32(&mut buf, meta.a_start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.a_end as u32).unwrap();
            }
            Self::Sigmoid(meta) => {
                buf.push(4);
                serialization::write_u32(&mut buf, meta.a_start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.a_end as u32).unwrap();
            }
            Self::Softmax(meta) => {
                buf.push(5);
                serialization::write_u32(&mut buf, meta.a_start as u32).unwrap();
                serialization::write_u32(&mut buf, meta.a_end as u32).unwrap();
                serialization::write_u32(&mut buf, meta.output_size as u32).unwrap();
            }
        };

        buf
    }

    /// Deserializes an operation from a reader.
    pub fn from_reader(reader: &mut impl Read) -> Result<Op, OpSerializationError> {
        let mut variant = [0u8; 1];
        reader.read_exact(&mut variant)?;

        match variant[0] {
            0 | 1 => {
                let input_dim = serialization::read_u32(reader)? as usize;
                let output_dim = serialization::read_u32(reader)? as usize;
                let a_start = serialization::read_u32(reader)? as usize;
                let i_start = serialization::read_u32(reader)? as usize;
                let w_start = serialization::read_u32(reader)? as usize;
                let w_end = serialization::read_u32(reader)? as usize;
                let b_start = serialization::read_u32(reader)? as usize;
                let b_end = serialization::read_u32(reader)? as usize;
                let mut init_buf = [0u8; 1];
                reader.read_exact(&mut init_buf)?;
                let initialization = Initialization::try_from(init_buf[0])
                    .map_err(|_| OpSerializationError::UnknownNodeVariant(init_buf[0]))?;

                let meta = DenseMeta::new(
                    input_dim,
                    output_dim,
                    a_start,
                    i_start,
                    w_start..w_end,
                    b_start..b_end,
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
                let d_start = serialization::read_u32(reader)? as usize;
                let d_end = serialization::read_u32(reader)? as usize;
                let m_start = serialization::read_u32(reader)? as usize;
                let m_end = serialization::read_u32(reader)? as usize;
                Ok(Self::Dropout(DropoutMeta::new(
                    p, d_start, d_end, m_start, m_end,
                )?))
            }
            3 => {
                let a_start = serialization::read_u32(reader)? as usize;
                let a_end = serialization::read_u32(reader)? as usize;
                Ok(Self::Relu(ReluMeta::new(a_start, a_end)))
            }
            4 => {
                let a_start = serialization::read_u32(reader)? as usize;
                let a_end = serialization::read_u32(reader)? as usize;
                Ok(Self::Sigmoid(SigmoidMeta::new(a_start, a_end)))
            }
            5 => {
                let a_start = serialization::read_u32(reader)? as usize;
                let a_end = serialization::read_u32(reader)? as usize;
                let output_dim = serialization::read_u32(reader)? as usize;
                Ok(Self::Softmax(SoftmaxMeta::new(a_start, a_end, output_dim)))
            }
            _ => Err(OpSerializationError::UnknownNodeVariant(variant[0])),
        }
    }
}

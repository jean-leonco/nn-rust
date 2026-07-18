use crate::io::{read_f32, read_u32};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NodeError {
    #[error("IO error: {0}")]
    Io(#[from] crate::io::IoError),
    #[error("Standard IO error: {0}")]
    StdIo(#[from] std::io::Error),
    #[error("Unknown initializer variant: {0}")]
    UnknownInitializerVariant(u8),
    #[error("Unknown execution node variant: {0}")]
    UnknownNodeVariant(u8),
}
pub type Result<T> = std::result::Result<T, NodeError>;

#[derive(Debug)]
pub enum Initializer {
    He,
    Xavier,
}

impl Initializer {
    pub fn std_dev(&self, input: usize, output: usize) -> f32 {
        match self {
            Self::He => 2.0 / ((input + output) as f32),
            Self::Xavier => 1.0 / ((input + output) as f32),
        }
        .sqrt()
    }

    pub fn to_u8(&self) -> u8 {
        match self {
            Self::He => 0,
            Self::Xavier => 1,
        }
    }
}

#[derive(Debug)]
pub enum SequentialExecutionNode {
    Dense {
        input_dim: usize,
        output_dim: usize,
        initializer: Initializer,
        w_start: usize,
        b_start: usize,
        w_end: usize,
        b_end: usize,
        a_start: usize,
        a_end: usize,
        input_a_start: usize,
    },
    Dropout {
        p: f32,
        inv_p: f32,
        data_start: usize,
        data_end: usize,
        mask_start: usize,
        mask_end: usize,
    },
    Relu {
        a_start: usize,
        a_end: usize,
    },
    Sigmoid {
        a_start: usize,
        a_end: usize,
    },
    SoftmaxCrossEntropy {
        a_start: usize,
        a_end: usize,
        output_dim: usize,
    },
}

impl SequentialExecutionNode {
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = match self {
            Self::Dense { .. } => 1 + 9 * 4 + 1,
            Self::Dropout { .. } => 1 + 2 * 4 + 4 * 4,
            Self::Relu { .. } | Self::Sigmoid { .. } => 1 + 2 * 4,
            Self::SoftmaxCrossEntropy { .. } => 1 + 3 * 4,
        };
        let mut buf = Vec::with_capacity(size);

        match self {
            SequentialExecutionNode::Dense {
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
            } => {
                buf.push(0);
                buf.extend_from_slice(&(*input_dim as u32).to_le_bytes());
                buf.extend_from_slice(&(*output_dim as u32).to_le_bytes());
                buf.push(initializer.to_u8());
                buf.extend_from_slice(&(*w_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*b_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*w_end as u32).to_le_bytes());
                buf.extend_from_slice(&(*b_end as u32).to_le_bytes());
                buf.extend_from_slice(&(*a_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*a_end as u32).to_le_bytes());
                buf.extend_from_slice(&(*input_a_start as u32).to_le_bytes());
            }
            SequentialExecutionNode::Dropout {
                p,
                inv_p,
                data_start,
                data_end,
                mask_start,
                mask_end,
            } => {
                buf.push(1);
                buf.extend_from_slice(&p.to_le_bytes());
                buf.extend_from_slice(&inv_p.to_le_bytes());
                buf.extend_from_slice(&(*data_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*data_end as u32).to_le_bytes());
                buf.extend_from_slice(&(*mask_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*mask_end as u32).to_le_bytes());
            }
            SequentialExecutionNode::Relu { a_start, a_end } => {
                buf.push(2);
                buf.extend_from_slice(&(*a_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*a_end as u32).to_le_bytes());
            }
            SequentialExecutionNode::Sigmoid { a_start, a_end } => {
                buf.push(3);
                buf.extend_from_slice(&(*a_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*a_end as u32).to_le_bytes());
            }
            SequentialExecutionNode::SoftmaxCrossEntropy {
                a_start,
                a_end,
                output_dim,
            } => {
                buf.push(4);
                buf.extend_from_slice(&(*a_start as u32).to_le_bytes());
                buf.extend_from_slice(&(*a_end as u32).to_le_bytes());
                buf.extend_from_slice(&(*output_dim as u32).to_le_bytes());
            }
        };

        buf
    }

    pub fn from_reader(reader: &mut impl std::io::Read) -> Result<Self> {
        let mut variant = [0u8; 1];
        reader.read_exact(&mut variant)?;

        match variant[0] {
            0 => {
                let input_dim = read_u32(reader)? as usize;
                let output_dim = read_u32(reader)? as usize;
                let mut init_u8 = [0u8; 1];
                reader.read_exact(&mut init_u8)?;
                let initializer = match init_u8[0] {
                    0 => Initializer::He,
                    1 => Initializer::Xavier,
                    _ => return Err(NodeError::UnknownInitializerVariant(init_u8[0])),
                };
                let w_start = read_u32(reader)? as usize;
                let b_start = read_u32(reader)? as usize;
                let w_end = read_u32(reader)? as usize;
                let b_end = read_u32(reader)? as usize;
                let a_start = read_u32(reader)? as usize;
                let a_end = read_u32(reader)? as usize;
                let input_a_start = read_u32(reader)? as usize;

                Ok(SequentialExecutionNode::Dense {
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
            1 => {
                let p = read_f32(reader)?;
                let inv_p = read_f32(reader)?;
                let data_start = read_u32(reader)? as usize;
                let data_end = read_u32(reader)? as usize;
                let mask_start = read_u32(reader)? as usize;
                let mask_end = read_u32(reader)? as usize;

                Ok(SequentialExecutionNode::Dropout {
                    p,
                    inv_p,
                    data_start,
                    data_end,
                    mask_start,
                    mask_end,
                })
            }
            2 => {
                let a_start = read_u32(reader)? as usize;
                let a_end = read_u32(reader)? as usize;
                Ok(SequentialExecutionNode::Relu { a_start, a_end })
            }
            3 => {
                let a_start = read_u32(reader)? as usize;
                let a_end = read_u32(reader)? as usize;
                Ok(SequentialExecutionNode::Sigmoid { a_start, a_end })
            }
            4 => {
                let a_start = read_u32(reader)? as usize;
                let a_end = read_u32(reader)? as usize;
                let output_dim = read_u32(reader)? as usize;
                Ok(SequentialExecutionNode::SoftmaxCrossEntropy {
                    a_start,
                    a_end,
                    output_dim,
                })
            }
            _ => Err(NodeError::UnknownNodeVariant(variant[0])),
        }
    }
}

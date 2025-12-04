use crate::{
    layer::{
        Layer, LayerType, dense::Dense, relu::Relu, sigmoid::Sigmoid,
        softmax_cross_entropy::SoftmaxCrossEntropy,
    },
    model::model::Model,
};
use ndarray::{Array1, Array2};
use std::io::{Read, Write};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid layer count: {0}")]
    InvalidLayerCount(#[from] std::num::TryFromIntError),

    #[error("Invalid layer type: {0}")]
    InvalidLayerType(u8),

    #[error("Layer missing parameters")]
    MissingParams,
}

pub fn encode_model(model: &Model, writer: &mut impl Write) -> Result<(), SerializationError> {
    let n_topology = u32::try_from(model.topology.len())?;
    write_u32(writer, n_topology)?;

    for value in &model.topology {
        write_u32(writer, u32::try_from(*value)?)?;
    }

    let n_layers = u32::try_from(model.layers.len())?;
    write_u32(writer, n_layers)?;

    for layer in &model.layers {
        let layer_type = layer.get_layer_type();
        writer.write_all(&[layer_type as u8])?;

        match layer.get_layer_type() {
            LayerType::Dense => {
                let params = layer
                    .get_params()
                    .ok_or(SerializationError::MissingParams)?;
                let (w_rows, w_cols) = params.weights.dim();
                let b_dim = params.bias.len();

                write_u32(writer, u32::try_from(w_rows)?)?;
                write_u32(writer, u32::try_from(w_cols)?)?;
                write_u32(writer, u32::try_from(b_dim)?)?;

                for &value in params.weights.iter() {
                    write_f32(writer, value)?;
                }

                for &value in params.bias.iter() {
                    write_f32(writer, value)?;
                }
            }
            _ => {}
        }
    }

    writer.flush()?;

    Ok(())
}

pub fn decode_model(reader: &mut impl Read) -> Result<Model, SerializationError> {
    let n_topology = read_u32(reader)? as usize;
    let mut topology = Vec::with_capacity(n_topology);

    for _ in 0..n_topology {
        topology.push(read_u32(reader)? as usize);
    }

    let n_layers = read_u32(reader)? as usize;
    let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(n_layers);

    for _ in 0..n_layers {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;

        let layer_type = LayerType::try_from(buf[0])
            .map_err(|_| SerializationError::InvalidLayerType(buf[0]))?;

        match layer_type {
            LayerType::Dense => {
                let w_rows = read_u32(reader)? as usize;
                let w_cols = read_u32(reader)? as usize;
                let b_dim = read_u32(reader)? as usize;

                let mut weights = Array2::zeros((w_rows, w_cols));
                for value in weights.iter_mut() {
                    *value = read_f32(reader)?;
                }

                let mut bias = Array1::zeros(b_dim);
                for value in bias.iter_mut() {
                    *value = read_f32(reader)?;
                }

                layers.push(Box::new(Dense::from_params(weights, bias)));
            }
            LayerType::Sigmoid => {
                layers.push(Box::new(Sigmoid::new()));
            }
            LayerType::Relu => {
                layers.push(Box::new(Relu::new()));
            }
            LayerType::SoftmaxCrossEntropy => {
                layers.push(Box::new(SoftmaxCrossEntropy::new()));
            }
        }
    }

    Ok(Model::new(layers, topology))
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<(), SerializationError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32, SerializationError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_f32(writer: &mut impl Write, value: f32) -> Result<(), SerializationError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_f32(reader: &mut impl Read) -> Result<f32, SerializationError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

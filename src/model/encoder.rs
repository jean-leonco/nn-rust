use crate::{
    layer::{
        Layer, LayerType, dense::Dense, dropout::Dropout, relu::Relu, sigmoid::Sigmoid,
        softmax_cross_entropy::SoftmaxCrossEntropy,
    },
    model::model::Model,
};
use bytemuck::cast_slice;
use std::io::Read;
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

pub fn encode_model(
    model: &Model,
    writer: &mut dyn std::io::Write,
) -> Result<(), SerializationError> {
    let n_layer_dims = u32::try_from(model.layer_dims.len())?;
    write_u32(writer, n_layer_dims)?;

    let layer_dims = model
        .layer_dims
        .iter()
        .map(|value| u32::try_from(*value))
        .collect::<Result<Vec<u32>, _>>()?;
    writer.write_all(cast_slice(layer_dims.as_slice()))?;

    let n_layers = u32::try_from(model.layers.len())?;
    write_u32(writer, n_layers)?;

    for layer in &model.layers {
        layer.write(writer)?;
    }

    writer.flush()?;

    Ok(())
}

pub fn decode_model(reader: &mut impl Read) -> Result<Model, SerializationError> {
    let n_layer_dims = read_u32(reader)? as usize;

    let byte_len = n_layer_dims * 4;
    let mut dims_bytes = vec![0u8; byte_len];
    reader.read_exact(&mut dims_bytes)?;

    let layer_dims: Vec<usize> = dims_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as usize)
        .collect();

    let n_layers = read_u32(reader)? as usize;
    let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(n_layers);

    for _ in 0..n_layers {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;

        let layer_type = LayerType::try_from(buf[0])
            .map_err(|_| SerializationError::InvalidLayerType(buf[0]))?;

        match layer_type {
            LayerType::Dense => {
                layers.push(Box::new(Dense::read(reader)?));
            }
            LayerType::Sigmoid => {
                layers.push(Box::new(Sigmoid::read(reader)?));
            }
            LayerType::Relu => {
                layers.push(Box::new(Relu::read(reader)?));
            }
            LayerType::SoftmaxCrossEntropy => {
                layers.push(Box::new(SoftmaxCrossEntropy::read(reader)?));
            }
            LayerType::Dropout => {
                layers.push(Box::new(Dropout::read(reader)?));
            }
        }
    }

    Ok(Model::new(layers, layer_dims))
}

pub fn write_u32(writer: &mut dyn std::io::Write, value: u32) -> Result<(), SerializationError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub fn write_f32(writer: &mut dyn std::io::Write, value: f32) -> Result<(), SerializationError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub fn read_u32(reader: &mut impl Read) -> Result<u32, SerializationError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_f32(reader: &mut impl Read) -> Result<f32, SerializationError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

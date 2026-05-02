use std::io::{Read, Write};

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2};

use crate::model::encoder::SerializationError;

pub mod dense;
pub mod dropout;
pub mod relu;
pub mod sigmoid;
pub mod softmax_cross_entropy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Dense = 0,
    Sigmoid,
    Relu,
    SoftmaxCrossEntropy,
    Dropout,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense => write!(f, "Dense"),
            Self::Sigmoid => write!(f, "Sigmoid"),
            Self::Relu => write!(f, "ReLU"),
            Self::SoftmaxCrossEntropy => write!(f, "Softmax and Cross-Entropy"),
            Self::Dropout => write!(f, "Dropout"),
        }
    }
}

impl TryFrom<u8> for LayerType {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(LayerType::Dense),
            1 => Ok(LayerType::Sigmoid),
            2 => Ok(LayerType::Relu),
            3 => Ok(LayerType::SoftmaxCrossEntropy),
            4 => Ok(LayerType::Dropout),
            _ => Err("Invalid u8 {value} value for LayerType"),
        }
    }
}

#[derive(Debug)]
pub struct LayerParams<'a> {
    pub weights: ArrayView2<'a, f32>,
    pub bias: ArrayView1<'a, f32>,
}

pub trait Layer {
    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>);
    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>);
    fn backward(
        &mut self,
        grad_input: &mut ArrayViewMut2<f32>,
        grad_output: &ArrayView2<f32>,
        learning_rate: f32,
    );

    fn write(&self, writer: &mut dyn Write) -> Result<(), SerializationError>;

    fn read(reader: &mut impl Read) -> Result<Self, SerializationError>
    where
        Self: Sized;
}

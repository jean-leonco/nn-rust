use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2};

pub mod dense;
pub mod relu;
pub mod sigmoid;
pub mod softmax_cross_entropy;

#[derive(Debug)]
pub enum LayerType {
    Dense = 0,
    Sigmoid,
    Relu,
    SoftmaxCrossEntropy,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense => write!(f, "Dense"),
            Self::Sigmoid => write!(f, "Sigmoid"),
            Self::Relu => write!(f, "ReLU"),
            Self::SoftmaxCrossEntropy => write!(f, "Softmax and Cross-Entropy"),
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
    fn get_layer_type(&self) -> LayerType;
    fn get_params(&self) -> Option<LayerParams<'_>>;

    fn forward(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>);
    fn forward_train(&mut self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>);
    fn backward(
        &mut self,
        grad_input: &mut ArrayViewMut2<f32>,
        grad_output: &ArrayView2<f32>,
        learning_rate: f32,
    );
}

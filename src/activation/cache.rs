use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Cache {
    state: Vec<Array2<f32>>,
    num_of_layers: usize,
}

impl Cache {
    pub fn new(num_of_layers: usize) -> Self {
        Self {
            state: Vec::with_capacity(num_of_layers),
            num_of_layers,
        }
    }

    pub fn init_layers(&mut self, input: &ArrayView2<f32>) {
        if self.state.len() < self.num_of_layers {
            self.state.clear();
            self.state.push(input.to_owned());

            for _ in 0..self.num_of_layers - 1 {
                self.state.push(Array2::zeros((0, 0)));
            }
        }

        self.set_layer_state(0, input);
    }

    pub fn get(&self, i: usize) -> ArrayView2<f32> {
        self.state[i].view()
    }

    pub fn get_last(&self) -> Option<ArrayView2<f32>> {
        self.state.last().map(|v| v.view())
    }

    pub fn set_layer_state(&mut self, i: usize, value: &ArrayView2<'_, f32>) {
        if self.state[i].dim() == value.dim() {
            self.state[i].assign(&value);
        } else {
            self.state[i] = value.to_owned();
        }
    }
}

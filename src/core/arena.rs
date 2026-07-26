use core::ops::Range;

use crate::core::serialization;

/// Represents the layout of the buffers using during session.
#[derive(Debug, Default)]
pub struct ArenaLayout {
    /// The length of the parameter buffer.
    pub params_len: usize,
    /// The relative length of the mask buffer. Must be used along with batch size to get the absolute length.
    pub masks_len: usize,
    /// The relative length of the activation buffer. Must be used along with batch size to get the absolute length.
    pub activations_len: usize,
    /// The last relative span of the activation buffer. Must be used along with batch size to get the absolute length.
    pub last_activation_span: Range<usize>,
    /// The maximum number of neurons in a single layer.
    pub max_neurons: usize,
}

impl ArenaLayout {
    /// Reserve a span for parameters.
    pub fn reserve_params(&mut self, size: usize) -> Range<usize> {
        let start = self.params_len;
        self.params_len += size;
        start..self.params_len
    }

    /// Reserve a span for masks.
    pub fn reserve_masks(&mut self, size: usize) -> Range<usize> {
        let start = self.masks_len;
        self.masks_len += size;
        start..self.masks_len
    }

    /// Reserve a span for activations.
    pub fn reserve_activations(&mut self, dimension: usize) -> Range<usize> {
        if dimension > self.max_neurons {
            self.max_neurons = dimension;
        }

        let start = self.activations_len;
        let end = self.activations_len + dimension;
        self.activations_len += dimension;
        self.last_activation_span = start..end;

        start..end
    }
}

impl serialization::Encodable for ArenaLayout {
    type Error = super::SerializationError;

    fn write(
        &self,
        writer: &mut impl std::io::prelude::Write,
    ) -> Result<(), super::SerializationError> {
        serialization::write_u32(writer, self.params_len as u32)?;
        serialization::write_u32(writer, self.masks_len as u32)?;
        serialization::write_u32(writer, self.activations_len as u32)?;
        serialization::write_span(writer, self.last_activation_span.clone())?;
        serialization::write_u32(writer, self.max_neurons as u32)?;
        Ok(())
    }

    fn from_reader(
        reader: &mut impl std::io::prelude::Read,
    ) -> Result<Self, super::SerializationError>
    where
        Self: Sized,
    {
        let params_len = serialization::read_u32(reader)? as usize;
        let masks_len = serialization::read_u32(reader)? as usize;
        let activations_len = serialization::read_u32(reader)? as usize;
        let last_activation_span = serialization::read_span(reader)?;
        let max_neurons = serialization::read_u32(reader)? as usize;

        Ok(Self {
            params_len,
            masks_len,
            activations_len,
            last_activation_span,
            max_neurons,
        })
    }
}

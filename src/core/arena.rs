use core::ops::Range;

use crate::core::serialization;

/// Arena layout for session execution buffers.
#[derive(Debug, Default)]
pub struct ArenaLayout {
    /// Total parameter count (weights + biases).
    pub params_len: usize,
    /// Relative mask count.
    pub masks_len: usize,
    /// Relative activation count.
    pub activations_len: usize,
    /// Relative range of the last reserved activation slice.
    pub last_activation_range: Range<usize>,
    /// Largest single-layer width.
    pub max_neurons: usize,
}

impl ArenaLayout {
    /// Reserves `size` parameter slots.
    pub fn reserve_params(&mut self, size: usize) -> Range<usize> {
        let start = self.params_len;
        self.params_len += size;
        start..self.params_len
    }

    /// Reserves `size` mask slots.
    pub fn reserve_masks(&mut self, size: usize) -> Range<usize> {
        let start = self.masks_len;
        self.masks_len += size;
        start..self.masks_len
    }

    /// Reserves `dim` activation slots.
    pub fn reserve_activations(&mut self, dim: usize) -> Range<usize> {
        if dim > self.max_neurons {
            self.max_neurons = dim;
        }

        let start = self.activations_len;
        let end = self.activations_len + dim;
        self.activations_len += dim;
        self.last_activation_range = start..end;

        start..end
    }
}

impl serialization::Encodable for ArenaLayout {
    type Error = super::serialization::SerializationError;

    fn encoded_len(&self) -> usize {
        // 5 u32
        5 * serialization::U32_WIRE
    }

    fn encode(
        &self,
        writer: &mut impl std::io::prelude::Write,
    ) -> Result<(), super::SerializationError> {
        serialization::write_u32(writer, self.params_len as u32)?;
        serialization::write_u32(writer, self.masks_len as u32)?;
        serialization::write_u32(writer, self.activations_len as u32)?;
        serialization::write_range(writer, self.last_activation_range.clone())?;
        serialization::write_u32(writer, self.max_neurons as u32)?;
        Ok(())
    }

    fn decode(reader: &mut impl std::io::prelude::Read) -> Result<Self, super::SerializationError>
    where
        Self: Sized,
    {
        let params_len = serialization::read_u32(reader)? as usize;
        let masks_len = serialization::read_u32(reader)? as usize;
        let activations_len = serialization::read_u32(reader)? as usize;
        let last_activation_range = serialization::read_range(reader)?;
        let max_neurons = serialization::read_u32(reader)? as usize;

        Ok(Self {
            params_len,
            masks_len,
            activations_len,
            last_activation_range,
            max_neurons,
        })
    }
}

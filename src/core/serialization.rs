use core::ops::Range;
use std::io::{Read, Write};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Writes a 32-bit unsigned integer to the writer in little-endian format.
pub fn write_u32(writer: &mut impl Write, value: u32) -> Result<(), SerializationError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Writes a 32-bit floating-point number to the writer in little-endian format.
pub fn write_f32(writer: &mut impl Write, value: f32) -> Result<(), SerializationError> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Reads a 32-bit unsigned integer from the reader in little-endian format.
pub fn read_u32(reader: &mut impl Read) -> Result<u32, SerializationError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_usize_range(reader: &mut impl Read) -> Result<Range<usize>, SerializationError> {
    let start = read_u32(reader)? as usize;
    let end = read_u32(reader)? as usize;
    Ok(start..end)
}

/// Reads a 32-bit floating-point number from the reader in little-endian format.
pub fn read_f32(reader: &mut impl Read) -> Result<f32, SerializationError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Reads a 32-bit unsigned integer from the reader in big-endian format.
pub fn read_u32_be(reader: &mut impl Read) -> Result<u32, SerializationError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

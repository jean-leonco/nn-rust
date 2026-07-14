use std::io::{Read, Write};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, IoError>;

pub fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub fn write_f32(writer: &mut impl Write, value: f32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

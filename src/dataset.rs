use ndarray::{Array2, Axis};
use std::fs::File;
use std::io::{BufReader, Read};
use thiserror::Error;

const LABELS_MAGIC_NUMBER: u32 = 2049;
const IMAGES_MAGIC_NUMBER: u32 = 2051;
pub const NUM_OF_CLASSES: usize = 10;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("Magic number mismatch, expected {0}, got {1}")]
    MagicNumber(u32, u32),

    #[error("File {0} corrupted: expected EOF but found {1} extra bytes")]
    TrailingBytes(String, usize),

    #[error(
        "Invalid dimensions resulting in overflow or zero-size: count {0}, rows {1} and columns {2}"
    )]
    InvalidDimensions(usize, usize, usize),
}

fn read_u32(reader: &mut impl Read) -> Result<u32, DatasetError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_usize(reader: &mut impl Read) -> Result<usize, DatasetError> {
    Ok(read_u32(reader)? as usize)
}

fn load_labels(path: &str) -> Result<Array2<f32>, DatasetError> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);

    let magic = read_u32(&mut reader)?;
    if magic != LABELS_MAGIC_NUMBER {
        return Err(DatasetError::MagicNumber(LABELS_MAGIC_NUMBER, magic));
    }

    let count = read_usize(&mut reader)?;
    let mut data = vec![0u8; count];
    reader.read_exact(&mut data)?;

    let mut labels = Array2::zeros((count, NUM_OF_CLASSES));
    for (i, mut row) in labels.axis_iter_mut(Axis(0)).enumerate() {
        let label = data[i] as usize;
        row[label] = 1.0;
    }

    let mut remaining_bytes = Vec::with_capacity(1);
    if reader.read_to_end(&mut remaining_bytes)? > 0 {
        return Err(DatasetError::TrailingBytes(
            path.to_string(),
            remaining_bytes.len(),
        ));
    }

    Ok(labels)
}

fn load_images(path: &str) -> Result<Array2<f32>, DatasetError> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);

    let magic = read_u32(&mut reader)?;
    if magic != IMAGES_MAGIC_NUMBER {
        return Err(DatasetError::MagicNumber(IMAGES_MAGIC_NUMBER, magic));
    }

    let count = read_usize(&mut reader)?;
    let rows = read_usize(&mut reader)?;
    let cols = read_usize(&mut reader)?;

    let size = count
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(cols))
        .ok_or(DatasetError::InvalidDimensions(count, rows, cols))?;

    let mut data = vec![0u8; size];
    reader.read_exact(&mut data)?;
    let images: Vec<f32> = data.iter().map(|&byte| f32::from(byte) / 255.0).collect();

    if images.len() != size {
        return Err(
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "File size mismatch").into(),
        );
    }

    Ok(Array2::from_shape_vec((count, rows * cols), images)?)
}

#[derive(Debug)]
pub struct Dataset {
    pub labels: Array2<f32>,
    pub images: Array2<f32>,
}

impl Dataset {
    pub fn load(labels_path: &str, images_path: &str) -> Result<Self, DatasetError> {
        Ok(Self {
            labels: load_labels(labels_path)?,
            images: load_images(images_path)?,
        })
    }
}

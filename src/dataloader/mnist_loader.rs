use ndarray::{Array2, ArrayView2, Axis, Zip, s};
use ndarray_rand::rand::{self, Rng};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use thiserror::Error;

use crate::dataloader::Dataloader;

const LABELS_MAGIC_NUMBER: u32 = 2049;
const IMAGES_MAGIC_NUMBER: u32 = 2051;
const NUM_OF_CLASSES: usize = 10;

#[derive(Debug, Error)]
pub enum MNistLoaderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("Magic number mismatch, expected {0}, got {1}")]
    MagicNumber(u32, u32),

    #[error("Expected EOF but found {0} extra bytes")]
    TrailingBytes(usize),

    #[error(
        "Invalid dimensions resulting in overflow or zero-size: count {0}, rows {1} and columns {2}"
    )]
    InvalidDimensions(usize, usize, usize),
}

#[derive(Debug)]
pub struct MNistLoader {
    num_of_batches: usize,
    batch_size: usize,
    train_x: Array2<f32>,
    train_y: Array2<f32>,
    validation_x: Array2<f32>,
    validation_y: Array2<f32>,
}

impl MNistLoader {
    pub fn load(batch_size: usize) -> Result<Self, MNistLoaderError> {
        let train_x = Self::load_images("mnist/train-images.idx3-ubyte")?;
        let train_y = Self::load_labels("mnist/train-labels.idx1-ubyte")?;

        let validation_x = Self::load_images("mnist/t10k-images.idx3-ubyte")?;
        let validation_y = Self::load_labels("mnist/t10k-labels.idx1-ubyte")?;

        let num_of_batches = train_x.nrows().div_ceil(batch_size);

        Ok(Self {
            num_of_batches,
            batch_size,
            train_x,
            train_y,
            validation_x,
            validation_y,
        })
    }

    fn load_labels(path: impl AsRef<Path>) -> Result<Array2<f32>, MNistLoaderError> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        let magic = Self::read_u32(&mut reader)?;
        if magic != LABELS_MAGIC_NUMBER {
            return Err(MNistLoaderError::MagicNumber(LABELS_MAGIC_NUMBER, magic));
        }

        let count = Self::read_usize(&mut reader)?;
        let mut data = vec![0u8; count];
        reader.read_exact(&mut data)?;

        let mut labels = Array2::zeros((count, NUM_OF_CLASSES));
        for (i, mut row) in labels.axis_iter_mut(Axis(0)).enumerate() {
            let label = data[i] as usize;
            row[label] = 1.0;
        }

        let mut remaining_bytes = Vec::with_capacity(1);
        if reader.read_to_end(&mut remaining_bytes)? > 0 {
            return Err(MNistLoaderError::TrailingBytes(remaining_bytes.len()));
        }

        Ok(labels)
    }

    fn load_images(path: &str) -> Result<Array2<f32>, MNistLoaderError> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        let magic = Self::read_u32(&mut reader)?;
        if magic != IMAGES_MAGIC_NUMBER {
            return Err(MNistLoaderError::MagicNumber(IMAGES_MAGIC_NUMBER, magic));
        }

        let count = Self::read_usize(&mut reader)?;
        let rows = Self::read_usize(&mut reader)?;
        let cols = Self::read_usize(&mut reader)?;

        let size = count
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(cols))
            .ok_or(MNistLoaderError::InvalidDimensions(count, rows, cols))?;

        let mut data = vec![0u8; size];
        reader.read_exact(&mut data)?;
        let images: Vec<f32> = data.iter().map(|&byte| f32::from(byte) / 255.0).collect();

        if images.len() != size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "File size mismatch",
            )
            .into());
        }

        Ok(Array2::from_shape_vec((count, rows * cols), images)?)
    }

    fn read_u32(reader: &mut impl Read) -> Result<u32, MNistLoaderError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    fn read_usize(reader: &mut impl Read) -> Result<usize, MNistLoaderError> {
        Ok(Self::read_u32(reader)? as usize)
    }

    fn swap_rows(value: &mut Array2<f32>, i: usize, j: usize) {
        // equivalent of vec.swap(i, j)
        let (mut row_i, mut row_j) = value.multi_slice_mut((s![i, ..], s![j, ..]));
        Zip::from(&mut row_i)
            .and(&mut row_j)
            .for_each(std::mem::swap);
    }
}

impl Dataloader<'_> for MNistLoader {
    fn num_of_batches(&self) -> usize {
        self.num_of_batches
    }

    fn train_batches(
        &mut self,
    ) -> impl Iterator<Item = (ArrayView2<'_, f32>, ArrayView2<'_, f32>)> {
        let n = self.train_x.nrows();

        let mut rng = rand::rng();

        // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        for i in 0..(n - 1) {
            let j = Rng::random_range(&mut rng, i..n);

            // if row was selected to be swapped
            if i != j {
                Self::swap_rows(&mut self.train_x, i, j);
                Self::swap_rows(&mut self.train_y, i, j);
            }
        }

        let num_batches = n / self.batch_size;
        let mut batches = Vec::with_capacity(num_batches);

        for i in 0..num_batches {
            let start = i * self.batch_size;
            let end = start + self.batch_size;

            let xb = self.train_x.slice(s![start..end, ..]);
            let yb = self.train_y.slice(s![start..end, ..]);
            batches.push((xb, yb));
        }

        batches.into_iter()
    }

    fn validation_batches(
        &self,
    ) -> impl Iterator<Item = (ArrayView2<'_, f32>, ArrayView2<'_, f32>)> {
        std::iter::once((self.validation_x.view(), self.validation_y.view()))
    }
}

use ndarray::{Array2, ArrayView2, Axis, s};
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
    train_x: Array2<u8>,
    train_y: Array2<f32>,
    validation_x: Array2<u8>,
    validation_y: Array2<f32>,
}

impl MNistLoader {
    pub fn load(batch_size: usize) -> Result<Self, MNistLoaderError> {
        let train_x = Self::load_images("mnist/train-images.idx3-ubyte")?;
        let train_y = Self::load_labels("mnist/train-labels.idx1-ubyte")?;

        let validation_x = Self::load_images("mnist/t10k-images.idx3-ubyte")?;
        let validation_y = Self::load_labels("mnist/t10k-labels.idx1-ubyte")?;

        let num_of_batches = train_x.nrows() / batch_size;

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

    fn load_images(path: &str) -> Result<Array2<u8>, MNistLoaderError> {
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

        let mut images = vec![0u8; size];
        reader.read_exact(&mut images)?;

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
}

impl Dataloader<'_> for MNistLoader {
    fn num_of_batches(&self) -> usize {
        self.num_of_batches
    }

    fn train_batches(&mut self) -> impl Iterator<Item = (ArrayView2<'_, u8>, ArrayView2<'_, f32>)> {
        let n = self.train_x.nrows();
        let mut rng = rand::rng();

        // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        let tx_ptr = self.train_x.as_mut_ptr();
        let ty_ptr = self.train_y.as_mut_ptr();
        let x_cols = self.train_x.ncols();
        let y_cols = self.train_y.ncols();

        for i in 0..(n - 1) {
            let j = Rng::random_range(&mut rng, i..n);

            // if row was selected to be swapped
            if i != j {
                unsafe {
                    std::ptr::swap_nonoverlapping(
                        tx_ptr.add(i * x_cols),
                        tx_ptr.add(j * x_cols),
                        x_cols,
                    );
                    std::ptr::swap_nonoverlapping(
                        ty_ptr.add(i * y_cols),
                        ty_ptr.add(j * y_cols),
                        y_cols,
                    );
                }
            }
        }

        let batch_size = self.batch_size;
        let train_x = &self.train_x;
        let train_y = &self.train_y;

        (0..self.num_of_batches).map(move |i| {
            let start = i * batch_size;
            let end = start + batch_size;

            let xb = train_x.slice(s![start..end, ..]);
            let yb = train_y.slice(s![start..end, ..]);
            (xb, yb)
        })
    }

    fn validation_batches(
        &self,
    ) -> impl Iterator<Item = (ArrayView2<'_, u8>, ArrayView2<'_, f32>)> {
        std::iter::once((self.validation_x.view(), self.validation_y.view()))
    }
}

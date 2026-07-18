use rand::{Rng, RngExt};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use thiserror::Error;

const LABELS_MAGIC_NUMBER: u32 = 2049;
const IMAGES_MAGIC_NUMBER: u32 = 2051;
const NUM_OF_CLASSES: usize = 10;

#[derive(Debug, Error)]
pub enum MnistDatasetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

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
pub struct MnistDataset {
    batch_size: usize,
    n_train_imgs: usize,
    n_validation_imgs: usize,
    train_x: Vec<u8>,
    train_y: Vec<f32>,
    validation_x: Vec<u8>,
    validation_y: Vec<f32>,
    train_x_cols: usize,
    train_y_cols: usize,
}

impl MnistDataset {
    pub fn load(batch_size: usize) -> Result<Self, MnistDatasetError> {
        let (train_x, n_train_imgs) = Self::load_images("Mnist/train-images.idx3-ubyte")?;
        let train_y = Self::load_labels("Mnist/train-labels.idx1-ubyte")?;

        let (validation_x, n_validation_imgs) = Self::load_images("Mnist/t10k-images.idx3-ubyte")?;
        let validation_y = Self::load_labels("Mnist/t10k-labels.idx1-ubyte")?;

        Ok(Self {
            batch_size,
            n_train_imgs,
            n_validation_imgs,
            train_x_cols: train_x.len() / n_train_imgs,
            train_y_cols: train_y.len() / n_train_imgs,
            train_x,
            train_y,
            validation_x,
            validation_y,
        })
    }

    fn load_labels(path: impl AsRef<Path>) -> Result<Vec<f32>, MnistDatasetError> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        let magic = Self::read_u32(&mut reader)?;
        if magic != LABELS_MAGIC_NUMBER {
            return Err(MnistDatasetError::MagicNumber(LABELS_MAGIC_NUMBER, magic));
        }

        let count = Self::read_usize(&mut reader)?;
        let mut data = vec![0u8; count];
        reader.read_exact(&mut data)?;

        let mut labels = vec![0.0; count * NUM_OF_CLASSES];
        for (i, label) in data.iter().enumerate() {
            let label = *label as usize;
            labels[i * NUM_OF_CLASSES + label] = 1.0;
        }

        let mut remaining_bytes = Vec::with_capacity(1);
        if reader.read_to_end(&mut remaining_bytes)? > 0 {
            return Err(MnistDatasetError::TrailingBytes(remaining_bytes.len()));
        }

        Ok(labels)
    }

    fn load_images(path: &str) -> Result<(Vec<u8>, usize), MnistDatasetError> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        let magic = Self::read_u32(&mut reader)?;
        if magic != IMAGES_MAGIC_NUMBER {
            return Err(MnistDatasetError::MagicNumber(IMAGES_MAGIC_NUMBER, magic));
        }

        let count = Self::read_usize(&mut reader)?;
        let rows = Self::read_usize(&mut reader)?;
        let cols = Self::read_usize(&mut reader)?;

        let size = count
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(cols))
            .ok_or(MnistDatasetError::InvalidDimensions(count, rows, cols))?;

        let mut images = vec![0u8; size];
        reader.read_exact(&mut images)?;

        if images.len() != size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "File size mismatch",
            )
            .into());
        }

        Ok((images, count))
    }

    fn read_u32(reader: &mut impl Read) -> Result<u32, MnistDatasetError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    fn read_usize(reader: &mut impl Read) -> Result<usize, MnistDatasetError> {
        Ok(Self::read_u32(reader)? as usize)
    }

    pub fn train_batches<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> impl Iterator<Item = (&'_ [u8], &'_ [f32])> {
        self.shuffle(rng);
        let x_chunk = self.batch_size * self.train_x_cols;
        let y_chunk = self.batch_size * self.train_y_cols;

        self.train_x
            .chunks_exact(x_chunk)
            .zip(self.train_y.chunks_exact(y_chunk))
    }

    pub fn validation_batches(&self) -> impl Iterator<Item = (&'_ [u8], &'_ [f32])> {
        let x_cols = self.validation_x.len() / self.n_validation_imgs;
        let y_cols = self.validation_y.len() / self.n_validation_imgs;
        let x_chunk = self.batch_size * x_cols;
        let y_chunk = self.batch_size * y_cols;

        self.validation_x
            .chunks_exact(x_chunk)
            .zip(self.validation_y.chunks_exact(y_chunk))
    }

    pub fn convert_to_px(raw_x: &[u8], x: &mut [f32]) {
        const INV_255: f32 = 1.0 / 255.0;
        for (out, &val) in x.iter_mut().zip(raw_x.iter()) {
            *out = (val as f32) * INV_255;
        }
    }

    pub fn shuffle<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        let tx_ptr = self.train_x.as_mut_ptr();
        let ty_ptr = self.train_y.as_mut_ptr();

        for i in 0..(self.n_train_imgs - 1) {
            let j = rng.random_range(i..self.n_train_imgs);

            // if row was selected to be swapped
            if i != j {
                unsafe {
                    std::ptr::swap_nonoverlapping(
                        tx_ptr.add(i * self.train_x_cols),
                        tx_ptr.add(j * self.train_x_cols),
                        self.train_x_cols,
                    );
                    std::ptr::swap_nonoverlapping(
                        ty_ptr.add(i * self.train_y_cols),
                        ty_ptr.add(j * self.train_y_cols),
                        self.train_y_cols,
                    );
                }
            }
        }
    }
}

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::similar_names)]
#![feature(portable_simd)]

extern crate blas_src;

pub mod core;
pub mod dataset;
pub mod model;
pub mod ops;
pub mod optim;

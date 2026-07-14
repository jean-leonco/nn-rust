#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::return_self_not_must_use)]

extern crate blas_src;

pub mod dataset;
pub mod execution_session;
pub mod metrics;
pub mod optimizer;
pub mod sequential;
pub mod weights;
pub mod io;

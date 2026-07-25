#![allow(clippy::too_many_arguments)]

/// Performs a Single-precision General Matrix Multiply.
/// [LAPACK Reference](https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html)
pub(crate) fn sgemm(
    trans_a: cblas::Transpose,
    trans_b: cblas::Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}

/// Performs a Single-precision General Matrix-Vector multiplication.
/// [LAPACK Reference](https://www.netlib.org/lapack/explore-html/d7/dda/group__gemv_ga0d35d880b663ad18204bb23bd186e380.html)
pub(crate) fn sgemv(
    trans: cblas::Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    beta: f32,
    y: &mut [f32],
) {
    unsafe {
        cblas::sgemv(
            cblas::Layout::RowMajor,
            trans,
            m as i32,
            n as i32,
            alpha,
            a,
            lda as i32,
            x,
            1,
            beta,
            y,
            1,
        );
    }
}

/// Performs a Single-precision rank-1 update of a matrix.
/// [LAPACK Reference](https://www.netlib.org/lapack/explore-html/d8/d75/group__ger_ga95baec6bb0a84393d7bc67212b566ab0.html#ga95baec6bb0a84393d7bc67212b566ab0)
pub(crate) fn sger(
    m: usize,
    n: usize,
    alpha: f32,
    x: &[f32],
    y: &[f32],
    a: &mut [f32],
    lda: usize,
) {
    unsafe {
        cblas::sger(
            cblas::Layout::RowMajor,
            m as i32,
            n as i32,
            alpha,
            x,
            1,
            y,
            1,
            a,
            lda as i32,
        );
    }
}

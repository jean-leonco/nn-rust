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

use cblas::{Layout, sgemm, sgemv, sger};

pub(crate) fn gemm_f32(
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
        sgemm(
            Layout::RowMajor,
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

pub(crate) fn sgemv_f32(
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
        sgemv(
            Layout::RowMajor,
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

pub(crate) fn sger_f32(
    m: usize,
    n: usize,
    alpha: f32,
    x: &[f32],
    y: &[f32],
    a: &mut [f32],
    lda: usize,
) {
    unsafe {
        sger(
            Layout::RowMajor,
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

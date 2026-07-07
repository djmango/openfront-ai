//! GIL-free hot paths for openfront-ai.
//!
//! Targets the two stable, allocation-heavy loops that profiled hot in
//! Python and whose formats are frozen (cache-bc CACHE_FORMAT=1, obs v4):
//!
//!   decode_frame   zstd frame -> (owner slots, packed fallout), one copy
//!   collate_grids  pad+stack C-contiguous (C,h,w) arrays to (B,C,gh,gw)
//!   collate_masks  pad+stack (h,w) arrays to (B,gh,gw)
//!
//! All three release the GIL for the heavy part; collate additionally
//! parallelizes the batch copy with rayon.

use half::f16;
use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;

/// zstd-decompress one cache-bc frame blob and split it into the owner-slot
/// grid (hr, wr) and the packbits fallout plane (hr, wr/8).
#[pyfunction]
fn decode_frame<'py>(
    py: Python<'py>,
    blob: &Bound<'py, PyBytes>,
    hr: usize,
    wr: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u8>>)> {
    let src = blob.as_bytes();
    let hw = hr * wr;
    let packed_w = wr / 8; // wr is a multiple of REGION=8
    let total = hw + hr * packed_w;
    let raw = py
        .allow_threads(|| zstd::bulk::decompress(src, total))
        .map_err(|e| PyValueError::new_err(format!("zstd: {e}")))?;
    if raw.len() != total {
        return Err(PyValueError::new_err(format!(
            "frame size {} != {total}",
            raw.len()
        )));
    }
    let owners = PyArray1::from_slice(py, &raw[..hw])
        .reshape([hr, wr])
        .map_err(|e| PyValueError::new_err(format!("owners: {e}")))?;
    let fallout = PyArray1::from_slice(py, &raw[hw..])
        .reshape([hr, packed_w])
        .map_err(|e| PyValueError::new_err(format!("fallout: {e}")))?;
    Ok((owners, fallout))
}

struct Src {
    ptr: usize, // *const T as usize so it's Send
    c: usize,
    h: usize,
    w: usize,
}

fn collate_generic<'py, T: Element + Copy + Send + Sync + Default>(
    py: Python<'py>,
    grids: &Bound<'py, PyList>,
    gh: usize,
    gw: usize,
) -> PyResult<Bound<'py, PyArray4<T>>> {
    let mut srcs: Vec<Src> = Vec::with_capacity(grids.len());
    // Keep the readonly guards alive for the whole copy.
    let mut guards = Vec::with_capacity(grids.len());
    let mut c_all = 0usize;
    for item in grids.iter() {
        let arr: Bound<'py, PyArray3<T>> = item.extract()?;
        let ro = arr.readonly();
        if !ro.is_c_contiguous() {
            return Err(PyValueError::new_err("grid must be C-contiguous"));
        }
        let sh = ro.shape().to_vec();
        let (c, h, w) = (sh[0], sh[1], sh[2]);
        if c_all == 0 {
            c_all = c;
        } else if c != c_all {
            return Err(PyValueError::new_err("channel mismatch"));
        }
        if h > gh || w > gw {
            return Err(PyValueError::new_err("grid larger than target"));
        }
        srcs.push(Src {
            ptr: ro.as_slice()?.as_ptr() as usize,
            c,
            h,
            w,
        });
        guards.push(ro);
    }
    let b = srcs.len();
    let out = unsafe { PyArray4::<T>::new(py, [b, c_all, gh, gw], false) };
    let out_ptr = unsafe { out.as_slice_mut()?.as_mut_ptr() as usize };
    let plane = gh * gw;
    py.allow_threads(|| {
        (0..b).into_par_iter().for_each(|i| {
            let s = &srcs[i];
            let src = unsafe { std::slice::from_raw_parts(s.ptr as *const T, s.c * s.h * s.w) };
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_ptr as *mut T).add(i * c_all * plane),
                    c_all * plane,
                )
            };
            dst.fill(T::default());
            for c in 0..s.c {
                for y in 0..s.h {
                    let so = (c * s.h + y) * s.w;
                    let d_o = (c * gh + y) * gw;
                    dst[d_o..d_o + s.w].copy_from_slice(&src[so..so + s.w]);
                }
            }
        });
    });
    Ok(out)
}

/// Pad+stack a list of (C, h, w) float32 grids into (B, C, gh, gw).
#[pyfunction]
fn collate_grids_f32<'py>(
    py: Python<'py>,
    grids: &Bound<'py, PyList>,
    gh: usize,
    gw: usize,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    collate_generic::<f32>(py, grids, gh, gw)
}

/// Pad+stack a list of (C, h, w) float16 grids into (B, C, gh, gw).
#[pyfunction]
fn collate_grids_f16<'py>(
    py: Python<'py>,
    grids: &Bound<'py, PyList>,
    gh: usize,
    gw: usize,
) -> PyResult<Bound<'py, PyArray4<f16>>> {
    collate_generic::<f16>(py, grids, gh, gw)
}

/// Pad+stack a list of (h, w) float32 masks into (B, gh, gw).
#[pyfunction]
fn collate_masks<'py>(
    py: Python<'py>,
    masks: &Bound<'py, PyList>,
    gh: usize,
    gw: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut srcs: Vec<Src> = Vec::with_capacity(masks.len());
    let mut guards = Vec::with_capacity(masks.len());
    for item in masks.iter() {
        let arr: PyReadonlyArray2<f32> = item.extract()?;
        if !arr.is_c_contiguous() {
            return Err(PyValueError::new_err("mask must be C-contiguous"));
        }
        let sh = arr.shape().to_vec();
        if sh[0] > gh || sh[1] > gw {
            return Err(PyValueError::new_err("mask larger than target"));
        }
        srcs.push(Src {
            ptr: arr.as_slice()?.as_ptr() as usize,
            c: 1,
            h: sh[0],
            w: sh[1],
        });
        guards.push(arr);
    }
    let b = srcs.len();
    let out = unsafe { PyArray3::<f32>::new(py, [b, gh, gw], false) };
    let out_ptr = unsafe { out.as_slice_mut()?.as_mut_ptr() as usize };
    let plane = gh * gw;
    py.allow_threads(|| {
        (0..b).into_par_iter().for_each(|i| {
            let s = &srcs[i];
            let src = unsafe { std::slice::from_raw_parts(s.ptr as *const f32, s.h * s.w) };
            let dst = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(i * plane), plane)
            };
            dst.fill(0.0);
            for y in 0..s.h {
                dst[y * gw..y * gw + s.w].copy_from_slice(&src[y * s.w..(y + 1) * s.w]);
            }
        });
    });
    Ok(out)
}

// gil_used = false: safe under free-threaded (nogil) CPython; all shared
// state is function-local and the copies release the GIL anyway.
#[pymodule(gil_used = false)]
fn ofrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_frame, m)?)?;
    m.add_function(wrap_pyfunction!(collate_grids_f32, m)?)?;
    m.add_function(wrap_pyfunction!(collate_grids_f16, m)?)?;
    m.add_function(wrap_pyfunction!(collate_masks, m)?)?;
    Ok(())
}

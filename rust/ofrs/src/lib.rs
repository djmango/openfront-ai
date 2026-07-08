//! GIL-free hot paths for openfront-ai.
//!
//! Targets the two stable, allocation-heavy loops that profiled hot in
//! Python and whose formats are frozen (cache-bc CACHE_FORMAT=1, obs v4):
//!
//!   decode_frame   zstd frame -> (owner slots, packed fallout), one copy
//!   collate_grids  pad+stack C-contiguous (C,h,w) arrays to (B,C,gh,gw)
//!   collate_masks  pad+stack (h,w) arrays to (B,gh,gw)
//!
//! All three release the GIL for the heavy part; collate parallelizes with
//! rayon only when the batch is large enough to amortize thread overhead.

use half::f16;
use numpy::{
    Element, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;
use std::cell::RefCell;
use std::mem::size_of;
use std::ptr;

thread_local! {
    static ZSTD_SCRATCH: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

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

    let owners = PyArray2::<u8>::zeros(py, [hr, wr], false);
    let fallout = PyArray2::<u8>::zeros(py, [hr, packed_w], false);
    let (o_ptr, f_ptr) = unsafe {
        (
            owners.as_slice_mut()?.as_mut_ptr() as usize,
            fallout.as_slice_mut()?.as_mut_ptr() as usize,
        )
    };

    py.allow_threads(|| -> PyResult<()> {
        ZSTD_SCRATCH.with(|cell| -> PyResult<()> {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            scratch.extend_from_slice(
                &zstd::bulk::decompress(src, total)
                    .map_err(|e| PyValueError::new_err(format!("zstd: {e}")))?,
            );
            if scratch.len() != total {
                return Err(PyValueError::new_err(format!(
                    "frame size {} != {total}",
                    scratch.len()
                )));
            }
            unsafe {
                ptr::copy_nonoverlapping(scratch.as_ptr(), o_ptr as *mut u8, hw);
                ptr::copy_nonoverlapping(
                    scratch.as_ptr().add(hw),
                    f_ptr as *mut u8,
                    hr * packed_w,
                );
            }
            Ok(())
        })
    })?;
    Ok((owners, fallout))
}

struct Src {
    ptr: usize, // *const T as usize so it's Send
    c: usize,
    h: usize,
    w: usize,
}

fn copy_grid<T: Copy>(out_ptr: usize, plane: usize, c_all: usize, gh: usize, gw: usize, s: &Src) {
    let src_len = s.c * s.h * s.w;
    let src = unsafe { std::slice::from_raw_parts(s.ptr as *const T, src_len) };
    let dst = unsafe {
        std::slice::from_raw_parts_mut(
            out_ptr as *mut T,
            c_all * plane,
        )
    };
    unsafe {
        ptr::write_bytes(
            dst.as_mut_ptr() as *mut u8,
            0,
            c_all * plane * size_of::<T>(),
        );
    }
    if s.h == gh && s.w == gw {
        dst[..src_len].copy_from_slice(src);
        return;
    }
    for c in 0..s.c {
        for y in 0..s.h {
            let so = (c * s.h + y) * s.w;
            let d_o = (c * gh + y) * gw;
            dst[d_o..d_o + s.w].copy_from_slice(&src[so..so + s.w]);
        }
    }
}

fn collate_generic<'py, T: Element + Copy + Send + Sync>(
    py: Python<'py>,
    grids: &Bound<'py, PyList>,
    gh: usize,
    gw: usize,
) -> PyResult<Bound<'py, PyArray4<T>>> {
    let mut srcs: Vec<Src> = Vec::with_capacity(grids.len());
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
    let bytes: usize = srcs.iter().map(|s| s.c * s.h * s.w).sum::<usize>() * size_of::<T>();
    let par = b >= 4 && bytes >= 256 * 1024;

    py.allow_threads(|| {
        if par {
            srcs.par_iter().enumerate().for_each(|(i, s)| {
                let dst = unsafe { (out_ptr as *mut T).add(i * c_all * plane) as usize };
                copy_grid::<T>(dst, plane, c_all, gh, gw, s);
            });
        } else {
            for (i, s) in srcs.iter().enumerate() {
                let dst = unsafe { (out_ptr as *mut T).add(i * c_all * plane) as usize };
                copy_grid::<T>(dst, plane, c_all, gh, gw, s);
            }
        }
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

fn copy_mask(out_ptr: usize, plane: usize, gh: usize, gw: usize, s: &Src) {
    let src = unsafe { std::slice::from_raw_parts(s.ptr as *const f32, s.h * s.w) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, plane) };
    unsafe {
        ptr::write_bytes(
            dst.as_mut_ptr() as *mut u8,
            0,
            plane * size_of::<f32>(),
        );
    }
    if s.h == gh && s.w == gw {
        dst[..s.h * s.w].copy_from_slice(src);
        return;
    }
    for y in 0..s.h {
        dst[y * gw..y * gw + s.w].copy_from_slice(&src[y * s.w..(y + 1) * s.w]);
    }
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
    let par = b >= 4 && b * plane * size_of::<f32>() >= 256 * 1024;

    py.allow_threads(|| {
        if par {
            srcs.par_iter().enumerate().for_each(|(i, s)| {
                let dst = unsafe { (out_ptr as *mut f32).add(i * plane) as usize };
                copy_mask(dst, plane, gh, gw, s);
            });
        } else {
            for (i, s) in srcs.iter().enumerate() {
                let dst = unsafe { (out_ptr as *mut f32).add(i * plane) as usize };
                copy_mask(dst, plane, gh, gw, s);
            }
        }
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

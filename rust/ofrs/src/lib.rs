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

mod feat;
mod sampler;

use half::f16;
use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
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

fn stack_generic<'py, T: Element + Copy + Send + Sync>(
    py: Python<'py>,
    arrays: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let mut srcs: Vec<(usize, usize)> = Vec::with_capacity(arrays.len()); // (ptr, len)
    let mut guards = Vec::with_capacity(arrays.len());
    let mut numel = 0usize;
    for item in arrays.iter() {
        let arr: Bound<'py, numpy::PyArrayDyn<T>> = item.extract()?;
        let ro = arr.readonly();
        let sl = ro.as_slice().map_err(|_| PyValueError::new_err("stack: need C-contiguous"))?;
        if numel == 0 {
            numel = sl.len();
        } else if sl.len() != numel {
            return Err(PyValueError::new_err("stack: shape mismatch"));
        }
        srcs.push((sl.as_ptr() as usize, sl.len()));
        guards.push(ro);
    }
    let b = srcs.len();
    let out = unsafe { PyArray2::<T>::new(py, [b, numel], false) };
    let out_ptr = unsafe { out.as_slice_mut()?.as_mut_ptr() as usize };
    py.allow_threads(|| {
        (0..b).into_par_iter().for_each(|i| {
            let src = unsafe { std::slice::from_raw_parts(srcs[i].0 as *const T, numel) };
            let dst = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut T).add(i * numel), numel)
            };
            dst.copy_from_slice(src);
        });
    });
    Ok(out)
}

/// Stack equal-shape float32 arrays into (B, numel); reshape Python-side.
#[pyfunction]
fn stack_f32<'py>(py: Python<'py>, arrays: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    stack_generic::<f32>(py, arrays)
}

/// Stack equal-shape float16 arrays into (B, numel); reshape Python-side.
#[pyfunction]
fn stack_f16<'py>(py: Python<'py>, arrays: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyArray2<f16>>> {
    stack_generic::<f16>(py, arrays)
}

/// Parse one env-worker message (rl/vec.py binary protocol, replaces pickle):
///   u32 LE header length ++ header json ++ concatenated array buffers.
/// Header: {"arrays": [[key, dtype_str, [shape...]], ...], ...anything else}.
/// Returns (rest_of_header_json, {key: ndarray}).
#[pyfunction]
fn unpack_arrays<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyBytes>,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let buf = payload.as_bytes();
    if buf.len() < 4 {
        return Err(PyValueError::new_err("payload too short"));
    }
    let hlen = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let header: serde_json::Value = serde_json::from_slice(&buf[4..4 + hlen])
        .map_err(|e| PyValueError::new_err(format!("header: {e}")))?;
    let mut data = &buf[4 + hlen..];
    let out = PyDict::new(py);
    let arrays = header["arrays"]
        .as_array()
        .ok_or_else(|| PyValueError::new_err("header missing arrays"))?;
    for spec in arrays {
        let key = spec[0].as_str().ok_or_else(|| PyValueError::new_err("array key"))?;
        let dt = spec[1].as_str().ok_or_else(|| PyValueError::new_err("array dtype"))?;
        let shape: Vec<usize> = spec[2]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
            .unwrap_or_default();
        let numel: usize = shape.iter().product();
        macro_rules! take {
            ($t:ty) => {{
                let nb = numel * std::mem::size_of::<$t>();
                if data.len() < nb {
                    return Err(PyValueError::new_err("payload truncated"));
                }
                let (head, rest) = data.split_at(nb);
                data = rest;
                // Byte-copy handles unaligned buffers (headers are odd-length).
                let mut v: Vec<$t> = vec![Default::default(); numel];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        head.as_ptr(),
                        v.as_mut_ptr() as *mut u8,
                        nb,
                    );
                }
                let arr = PyArray1::from_vec(py, v);
                out.set_item(
                    key,
                    arr.reshape(shape.clone())
                        .map_err(|e| PyValueError::new_err(format!("reshape: {e}")))?,
                )?;
            }};
        }
        match dt {
            "|u1" => take!(u8),
            "<f4" => take!(f32),
            "<f2" => take!(f16),
            "<i8" => take!(i64),
            other => {
                return Err(PyValueError::new_err(format!("unsupported dtype {other}")))
            }
        }
    }
    let mut rest = header;
    rest.as_object_mut().unwrap().remove("arrays");
    Ok((rest.to_string(), out))
}

// gil_used = false: safe under free-threaded (nogil) CPython; all shared
// state is function-local and the copies release the GIL anyway.
#[pymodule(gil_used = false)]
fn ofrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_frame, m)?)?;
    m.add_function(wrap_pyfunction!(collate_grids_f32, m)?)?;
    m.add_function(wrap_pyfunction!(collate_grids_f16, m)?)?;
    m.add_function(wrap_pyfunction!(collate_masks, m)?)?;
    m.add_function(wrap_pyfunction!(stack_f32, m)?)?;
    m.add_function(wrap_pyfunction!(stack_f16, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_arrays, m)?)?;
    m.add_class::<sampler::Sampler>()?;
    Ok(())
}

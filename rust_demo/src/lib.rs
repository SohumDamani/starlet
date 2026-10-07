use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float64Array, StructArray};
use arrow_schema::{DataType, Field};
use pyo3::prelude::*;
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::PyArray;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Earth radius used by EPSG:3857 (spherical Web Mercator), in metres.
const R: f64 = 6_378_137.0;

/// (N, 2) lon/lat degrees -> (N, 2) Web Mercator metres. Borrows the input, allocates the output.
#[pyfunction]
fn to_mercator<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let input = coords.as_array();
    let n = input.nrows();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let lon = input[[i, 0]];
        let lat = input[[i, 1]];
        out[[i, 0]] = R * lon.to_radians();
        out[[i, 1]] = R * (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan().ln();
    }
    PyArray2::from_owned_array(py, out)
}

/// Returns the address of the first element Rust sees; equal to numpy's `arr.ctypes.data` if borrowed, not copied.
#[pyfunction]
fn data_address(coords: PyReadonlyArray2<'_, f64>) -> usize {
    coords.as_array().as_ptr() as usize
}

/// Takes the array and returns it unchanged; times the hand-off alone.
#[pyfunction]
fn passthrough<'py>(coords: Bound<'py, PyArray2<f64>>) -> Bound<'py, PyArray2<f64>> {
    coords
}

/// GeoArrow-style points: Arrow struct<x: double, y: double>. Borrows the x/y buffers, returns a new struct array.
#[pyfunction]
fn to_mercator_arrow(py: Python<'_>, arr: PyArray) -> PyArrowResult<Py<PyAny>> {
    let (array, _field) = arr.into_inner();
    let s = array.as_struct();
    let lon = s.column(0).as_primitive::<Float64Type>();
    let lat = s.column(1).as_primitive::<Float64Type>();
    let x: Float64Array = lon.values().iter().map(|v| R * v.to_radians()).collect();
    let y: Float64Array = lat
        .values()
        .iter()
        .map(|v| R * (std::f64::consts::FRAC_PI_4 + v.to_radians() / 2.0).tan().ln())
        .collect();
    let fields = vec![
        Arc::new(Field::new("x", DataType::Float64, false)),
        Arc::new(Field::new("y", DataType::Float64, false)),
    ];
    let out = StructArray::new(fields.into(), vec![Arc::new(x) as ArrayRef, Arc::new(y)], None);
    Ok(PyArray::from_array_ref(Arc::new(out)).to_pyarrow(py)?.unbind())
}

/// Address of the x child's data as Rust sees it (compare with pyarrow's buffer address).
#[pyfunction]
fn arrow_x_address(arr: PyArray) -> usize {
    let (array, _field) = arr.into_inner();
    array.as_struct().column(0).as_primitive::<Float64Type>().values().as_ptr() as usize
}

// ---------- Rust parses the WKB column itself ----------

/// 2D little-endian WKB Point only (1 byte order + 4 type + 16 coords = 21 bytes). Demo parser, not general.
fn parse_point(b: &[u8]) -> Option<(f64, f64)> {
    if b.len() != 21 || b[0] != 1 || u32::from_le_bytes(b[1..5].try_into().ok()?) != 1 {
        return None;
    }
    Some((
        f64::from_le_bytes(b[5..13].try_into().ok()?),
        f64::from_le_bytes(b[13..21].try_into().ok()?),
    ))
}

fn merc(lon: f64, lat: f64) -> (f64, f64) {
    (R * lon.to_radians(), R * (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan().ln())
}

fn pool(threads: usize) -> Arc<ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let mut m = POOLS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    m.entry(threads)
        .or_insert_with(|| Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()))
        .clone()
}

/// Parse n WKB points via `get(i)`, reproject, return (x, y); Err if any non-null row is not a 2D LE point.
fn run<'a, F>(n: usize, threads: usize, get: F) -> Result<(Vec<f64>, Vec<f64>), ()>
where
    F: Fn(usize) -> Option<&'a [u8]> + Sync + Send,
{
    let mut x = vec![0f64; n];
    let mut y = vec![0f64; n];
    let bad = AtomicBool::new(false);
    let work = |i: usize, xo: &mut f64, yo: &mut f64| match get(i) {
        None => {
            *xo = f64::NAN;
            *yo = f64::NAN;
        }
        Some(b) => match parse_point(b) {
            Some((lon, lat)) => (*xo, *yo) = merc(lon, lat),
            None => bad.store(true, Ordering::Relaxed),
        },
    };
    if threads == 0 {
        for (i, (xo, yo)) in x.iter_mut().zip(y.iter_mut()).enumerate() {
            work(i, xo, yo);
        }
    } else {
        pool(threads).install(|| {
            x.par_iter_mut()
                .zip(y.par_iter_mut())
                .enumerate()
                .for_each(|(i, (xo, yo))| work(i, xo, yo));
        });
    }
    if bad.load(Ordering::Relaxed) { Err(()) } else { Ok((x, y)) }
}

/// WKB binary column -> Arrow struct<x,y> in Web Mercator. threads=0: single thread; else a rayon pool of that size. GIL released.
#[pyfunction]
fn wkb_to_mercator_arrow(py: Python<'_>, arr: PyArray, threads: usize) -> PyArrowResult<Py<PyAny>> {
    let (array, _field) = arr.into_inner();
    let result = py.detach(|| match array.data_type() {
        DataType::Binary => {
            let a = array.as_binary::<i32>();
            run(a.len(), threads, |i| if a.is_null(i) { None } else { Some(a.value(i)) })
        }
        DataType::LargeBinary => {
            let a = array.as_binary::<i64>();
            run(a.len(), threads, |i| if a.is_null(i) { None } else { Some(a.value(i)) })
        }
        _ => Err(()),
    });
    let (x, y) = result.map_err(|_| pyo3::exceptions::PyValueError::new_err("column is not 2D little-endian WKB points"))?;
    let fields = vec![
        Arc::new(Field::new("x", DataType::Float64, true)),
        Arc::new(Field::new("y", DataType::Float64, true)),
    ];
    let out = StructArray::new(
        fields.into(),
        vec![Arc::new(Float64Array::from(x)) as ArrayRef, Arc::new(Float64Array::from(y))],
        None,
    );
    Ok(PyArray::from_array_ref(Arc::new(out)).to_pyarrow(py)?.unbind())
}

/// Address of the WKB values buffer as Rust sees it (compare with pyarrow's buffers()[2].address).
#[pyfunction]
fn wkb_values_address(arr: PyArray) -> usize {
    let (array, _field) = arr.into_inner();
    match array.data_type() {
        DataType::Binary => array.as_binary::<i32>().values().as_ptr() as usize,
        _ => array.as_binary::<i64>().values().as_ptr() as usize,
    }
}

// ---------- General WKB (any geometry type): walk the structure, reproject every coordinate, write WKB back ----------

fn rd_u32(b: &[u8], p: usize, le: bool) -> Option<u32> {
    let a: [u8; 4] = b.get(p..p + 4)?.try_into().ok()?;
    Some(if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
}

fn rd_f64(b: &[u8], p: usize, le: bool) -> Option<f64> {
    let a: [u8; 8] = b.get(p..p + 8)?.try_into().ok()?;
    Some(if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) })
}

fn wr_f64(b: &mut [u8], p: usize, le: bool, v: f64) -> Option<()> {
    b.get_mut(p..p + 8)?.copy_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
    Some(())
}

/// Reproject n coordinates of `dims` doubles each starting at *pos (x,y only; Z/M are left as they are).
fn coords(src: &[u8], dst: &mut [u8], pos: &mut usize, n: usize, dims: usize, le: bool) -> Option<()> {
    for _ in 0..n {
        let (x, y) = merc(rd_f64(src, *pos, le)?, rd_f64(src, *pos + 8, le)?);
        wr_f64(dst, *pos, le, x)?;
        wr_f64(dst, *pos + 8, le, y)?;
        *pos += 8 * dims;
    }
    Some(())
}

/// Walk one WKB geometry (Point..GeometryCollection, ISO or EWKB Z/M/SRID, either byte order).
fn walk(src: &[u8], dst: &mut [u8], pos: &mut usize) -> Option<()> {
    let le = match *src.get(*pos)? {
        1 => true,
        0 => false,
        _ => return None,
    };
    *pos += 1;
    let t = rd_u32(src, *pos, le)?;
    *pos += 4;
    let code = t & 0x0FFF_FFFF;
    let (iso_dim, base) = (code / 1000, code % 1000);
    let z = (t & 0x8000_0000 != 0) || iso_dim == 1 || iso_dim == 3;
    let m = (t & 0x4000_0000 != 0) || iso_dim == 2 || iso_dim == 3;
    let dims = 2 + z as usize + m as usize;
    if t & 0x2000_0000 != 0 {
        *pos += 4; // SRID, left as is
    }
    match base {
        1 => coords(src, dst, pos, 1, dims, le),
        2 => {
            let n = rd_u32(src, *pos, le)? as usize;
            *pos += 4;
            coords(src, dst, pos, n, dims, le)
        }
        3 => {
            let rings = rd_u32(src, *pos, le)? as usize;
            *pos += 4;
            for _ in 0..rings {
                let n = rd_u32(src, *pos, le)? as usize;
                *pos += 4;
                coords(src, dst, pos, n, dims, le)?;
            }
            Some(())
        }
        4..=7 => {
            let parts = rd_u32(src, *pos, le)? as usize;
            *pos += 4;
            for _ in 0..parts {
                walk(src, dst, pos)?;
            }
            Some(())
        }
        _ => None,
    }
}

/// WKB binary column (any geometry types) -> WKB binary column in Web Mercator. threads=0: single thread.
/// Reprojection never changes a geometry's byte length, so the output reuses the input's offsets.
#[pyfunction]
fn wkb_reproject_column(py: Python<'_>, arr: PyArray, threads: usize) -> PyArrowResult<Py<PyAny>> {
    let (array, _field) = arr.into_inner();
    if array.data_type() != &DataType::Binary {
        return Err(pyo3::exceptions::PyValueError::new_err("expected a binary (int32 offsets) column").into());
    }
    let a = array.as_binary::<i32>();
    let n = a.len();
    let offs = a.value_offsets();
    let start = offs[0] as usize;
    let rel: Vec<i32> = offs.iter().map(|o| o - offs[0]).collect();
    let src_vals = &a.values()[start..offs[n] as usize];
    let mut vals = src_vals.to_vec();

    let ok = py.detach(|| {
        let bad = AtomicBool::new(false);
        let mut rows: Vec<&mut [u8]> = Vec::with_capacity(n);
        let mut rest: &mut [u8] = &mut vals[..];
        for i in 0..n {
            let (head, tail) = rest.split_at_mut((rel[i + 1] - rel[i]) as usize);
            rows.push(head);
            rest = tail;
        }
        let work = |i: usize, dst: &mut &mut [u8]| {
            if a.is_null(i) || dst.is_empty() {
                return;
            }
            let src = &src_vals[rel[i] as usize..rel[i + 1] as usize];
            if walk(src, dst, &mut 0).is_none() {
                bad.store(true, Ordering::Relaxed);
            }
        };
        if threads == 0 {
            for (i, d) in rows.iter_mut().enumerate() {
                work(i, d);
            }
        } else {
            pool(threads).install(|| rows.par_iter_mut().enumerate().for_each(|(i, d)| work(i, d)));
        }
        !bad.load(Ordering::Relaxed)
    });
    if !ok {
        return Err(pyo3::exceptions::PyValueError::new_err("malformed or unsupported WKB").into());
    }
    let out = arrow_array::BinaryArray::new(
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(rel)),
        arrow_buffer::Buffer::from_vec(vals),
        a.nulls().cloned(),
    );
    Ok(PyArray::from_array_ref(Arc::new(out)).to_pyarrow(py)?.unbind())
}

#[pymodule]
fn merc_demo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_mercator, m)?)?;
    m.add_function(wrap_pyfunction!(data_address, m)?)?;
    m.add_function(wrap_pyfunction!(passthrough, m)?)?;
    m.add_function(wrap_pyfunction!(to_mercator_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_x_address, m)?)?;
    m.add_function(wrap_pyfunction!(wkb_to_mercator_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(wkb_values_address, m)?)?;
    m.add_function(wrap_pyfunction!(wkb_reproject_column, m)?)?;
    Ok(())
}

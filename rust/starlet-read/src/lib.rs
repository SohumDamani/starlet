//! Read Parquet row groups for starlet's Read step, one thread per row group, and hand them to Python as Arrow.
//! The footer (and page index, if the file has one) is parsed once and kept in memory. The file itself is opened per
//! read and closed right after, so nothing holds it open between tiles.
//!
//! `read_row_groups` reads whole row groups. `read_kept_rows` reads the bbox columns first, filters, then reads the other
//! columns only where needed: whole groups that have a match, or with `use_pages`, only the pages that have a match.

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{BooleanArray, RecordBatch, RecordBatchReader};
use arrow_schema::Schema;
use arrow_select::concat::concat_batches;
use arrow_select::filter::filter_record_batch;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection, RowSelectionPolicy,
};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use pyo3::exceptions::{PyIOError, PyKeyError};
use pyo3::prelude::*;
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::PyTable;
use rayon::prelude::*;
use rayon::ThreadPool;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const BBOX_COLS: [&str; 4] = ["_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax"];
type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// One thread pool per thread count, built once. Never more threads than this machine has cores.
fn pool(threads: usize) -> Arc<ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let threads = threads.clamp(1, cores);
    POOLS.get_or_init(Default::default).lock().unwrap().entry(threads)
        .or_insert_with(|| Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()))
        .clone()
}

/// Consecutive runs of true in `keep` as a row selection.
fn rows_to_selection(keep: &[bool]) -> RowSelection {
    let mut ranges = Vec::new();
    let mut start = None;
    for (i, &k) in keep.iter().enumerate() {
        match (k, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                ranges.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        ranges.push(s..keep.len());
    }
    RowSelection::from_consecutive_ranges(ranges.into_iter(), keep.len())
}

/// What `read_late` returns for one row group.
struct GroupRead {
    kept: Option<RecordBatch>, // the kept rows: requested columns, then the 4 bbox columns
    bbox_rows: usize,          // rows whose bbox was read
    geometry_rows: usize,      // rows in the geometry pages read
    kept_rows: usize,
}

/// A Parquet file with its footer cached.
#[pyclass]
struct ParquetFile {
    path: String,
    meta: ArrowReaderMetadata,
}

impl ParquetFile {
    /// All rows of one row group: only the requested columns, decoded to Arrow.
    fn read_one(&self, row_group: usize, mask: &ProjectionMask) -> parquet::errors::Result<Vec<RecordBatch>> {
        let file = File::open(&self.path)?;
        let rows = self.meta.metadata().row_group(row_group).num_rows() as usize;
        ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.meta.clone())
            .with_projection(mask.clone())
            .with_row_groups(vec![row_group])
            .with_batch_size(rows.max(1))
            .build()?
            .map(|batch| batch.map_err(Into::into))
            .collect()
    }

    /// Requested columns of one row group, optionally only some rows, as one batch.
    fn read_selected(&self, rg: usize, mask: &ProjectionMask, selection: Option<RowSelection>) -> Res<RecordBatch> {
        let rows = self.meta.metadata().row_group(rg).num_rows() as usize;
        let mut builder = ParquetRecordBatchReaderBuilder::new_with_metadata(File::open(&self.path)?, self.meta.clone())
            .with_projection(mask.clone())
            .with_row_groups(vec![rg])
            .with_batch_size(rows.max(1));
        if let Some(selection) = selection {
            builder = builder.with_row_selection(selection).with_row_selection_policy(RowSelectionPolicy::Selectors);
        }
        let reader = builder.build()?;
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        Ok(concat_batches(&schema, &batches)?)
    }

    fn leaf(&self, name: &str) -> Res<usize> {
        self.meta.parquet_schema().columns().iter().position(|c| c.name() == name)
            .ok_or_else(|| format!("no column {name}").into())
    }

    /// For each row of a row group: is it in pages whose bbox min/max can overlap the query?
    fn candidate_rows(&self, rg: usize, q: &[f64; 4]) -> Res<Vec<bool>> {
        let md = self.meta.metadata();
        let n = md.row_group(rg).num_rows() as usize;
        let (columns, offsets) = md.column_index().zip(md.offset_index())
            .ok_or_else(|| format!("{} has no page index", self.path))?;
        let mut keep = vec![true; n];
        for (i, name) in BBOX_COLS.iter().enumerate() {
            let leaf = self.leaf(name)?;
            let ColumnIndexMetaData::DOUBLE(index) = &columns[rg][leaf] else { continue };
            let pages = offsets[rg][leaf].page_locations();
            for (p, page) in pages.iter().enumerate() {
                let start = page.first_row_index as usize;
                let end = pages.get(p + 1).map_or(n, |next| next.first_row_index as usize);
                let overlaps = match i {
                    0 => index.min_value(p).is_none_or(|v| *v <= q[2]), // xmin <= query xmax
                    1 => index.min_value(p).is_none_or(|v| *v <= q[3]), // ymin <= query ymax
                    2 => index.max_value(p).is_none_or(|v| *v >= q[0]), // xmax >= query xmin
                    _ => index.max_value(p).is_none_or(|v| *v >= q[1]), // ymax >= query ymin
                };
                if !overlaps {
                    keep[start..end].fill(false);
                }
            }
        }
        Ok(keep)
    }

    /// How many rows are in the geometry pages that hold a kept row.
    fn rows_in_hit_pages(&self, rg: usize, geometry: &str, candidates: &[bool], hit: &BooleanArray) -> Res<usize> {
        let md = self.meta.metadata();
        let n = candidates.len();
        let pages = md.offset_index().ok_or("no page index")?[rg][self.leaf(geometry)?].page_locations();
        let mut hits = hit.values().iter();
        let kept: Vec<bool> = candidates.iter().map(|&c| c && hits.next().unwrap_or(false)).collect();
        Ok(pages.iter().enumerate().map(|(p, page)| {
            let start = page.first_row_index as usize;
            let end = pages.get(p + 1).map_or(n, |next| next.first_row_index as usize);
            if kept[start..end].iter().any(|&k| k) { end - start } else { 0 }
        }).sum())
    }

    /// One row group: bbox columns first, then the other columns only for what matches.
    fn read_late(&self, rg: usize, rest: &ProjectionMask, bbox: &ProjectionMask, geometry: &str, q: &[f64; 4], use_pages: bool) -> Res<GroupRead> {
        let n = self.meta.metadata().row_group(rg).num_rows() as usize;

        // 1. candidate rows: with a page index, only pages whose min/max can overlap
        let candidates = if use_pages { Some(self.candidate_rows(rg, q)?) } else { None };
        let selection = candidates.as_deref().map(rows_to_selection);
        if selection.as_ref().is_some_and(|s| !s.selects_any()) {
            return Ok(GroupRead { kept: None, bbox_rows: 0, geometry_rows: 0, kept_rows: 0 });
        }

        // 2. read the bbox columns of those rows and filter exactly
        let bbox_batch = self.read_selected(rg, bbox, selection.clone())?;
        let column = |name: &str| bbox_batch.column_by_name(name).map(|c| c.as_primitive::<Float64Type>().clone()).ok_or(format!("no column {name}"));
        let (a, b, c, d) = (column(BBOX_COLS[0])?, column(BBOX_COLS[1])?, column(BBOX_COLS[2])?, column(BBOX_COLS[3])?);
        let hit: BooleanArray = (0..bbox_batch.num_rows())
            .map(|i| Some(c.value(i) >= q[0] && a.value(i) <= q[2] && d.value(i) >= q[1] && b.value(i) <= q[3]))
            .collect();
        let kept_rows = hit.true_count();
        let bbox_rows = bbox_batch.num_rows();
        if kept_rows == 0 {
            return Ok(GroupRead { kept: None, bbox_rows, geometry_rows: 0, kept_rows });
        }

        // 3. the other columns: only the pages with a match, or the whole group
        let (rest_batch, geometry_rows) = match (&selection, &candidates) {
            (Some(selection), Some(candidates)) => {
                let wanted = selection.and_then(&RowSelection::from_filters(std::slice::from_ref(&hit)));
                (self.read_selected(rg, rest, Some(wanted))?, self.rows_in_hit_pages(rg, geometry, candidates, &hit)?)
            }
            _ => (filter_record_batch(&self.read_selected(rg, rest, None)?, &hit)?, n),
        };

        // 4. kept rows: requested columns + their bbox columns
        let kept_bbox = filter_record_batch(&bbox_batch, &hit)?;
        let mut fields: Vec<_> = rest_batch.schema().fields().iter().cloned().collect();
        fields.extend(kept_bbox.schema().fields().iter().cloned());
        let mut arrays = rest_batch.columns().to_vec();
        arrays.extend(kept_bbox.columns().iter().cloned());
        let kept = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        Ok(GroupRead { kept: Some(kept), bbox_rows, geometry_rows, kept_rows })
    }
}

#[pymethods]
impl ParquetFile {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let file = File::open(&path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
        let meta = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional))
            .map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
        Ok(Self { path, meta })
    }

    #[getter]
    fn num_row_groups(&self) -> usize {
        self.meta.metadata().num_row_groups()
    }

    /// The given row groups as one Arrow table. `collect` keeps them in the order asked for, whichever thread
    /// finishes first, so ties in Pick still go to the earliest row. Columns come back in file order.
    #[pyo3(signature = (row_groups, columns, threads=8))]
    fn read_row_groups(&self, py: Python<'_>, row_groups: Vec<usize>, columns: Vec<String>, threads: usize) -> PyArrowResult<Py<PyAny>> {
        let schema = self.meta.schema();
        let mut roots = Vec::with_capacity(columns.len());
        for name in &columns {
            roots.push(schema.index_of(name).map_err(|_| PyKeyError::new_err(format!("no column {name}")))?);
        }
        roots.sort_unstable();
        let projected = Arc::new(schema.project(&roots)?);
        let mask = ProjectionMask::roots(self.meta.parquet_schema(), roots);

        let read = py.detach(|| {
            pool(threads).install(|| {
                row_groups.par_iter().map(|&rg| self.read_one(rg, &mask)).collect::<parquet::errors::Result<Vec<_>>>()
            })
        });
        let batches: Vec<RecordBatch> = read.map_err(|e| PyIOError::new_err(e.to_string()))?.into_iter().flatten().collect();
        Ok(PyTable::try_new(batches, projected)?.into_pyarrow(py)?.unbind())
    }

    /// Late materialization. `columns` must not include the bbox columns, and the first one is the geometry. `query` is
    /// [xmin, ymin, xmax, ymax]. Returns (kept rows as a table, rows whose bbox was read, rows in geometry pages read, kept rows).
    #[pyo3(signature = (row_groups, columns, query, threads=8, use_pages=false))]
    fn read_kept_rows(&self, py: Python<'_>, row_groups: Vec<usize>, columns: Vec<String>, query: [f64; 4], threads: usize, use_pages: bool)
        -> PyArrowResult<(Py<PyAny>, usize, usize, usize)> {
        let schema = self.meta.schema();
        let root = |name: &str| schema.index_of(name).map_err(|_| PyKeyError::new_err(format!("no column {name}")));
        let mut rest_roots = columns.iter().map(|n| root(n)).collect::<PyResult<Vec<_>>>()?;
        let mut bbox_roots = BBOX_COLS.iter().map(|n| root(n)).collect::<PyResult<Vec<_>>>()?;
        rest_roots.sort_unstable();
        bbox_roots.sort_unstable();
        let fields: Vec<_> = rest_roots.iter().chain(&bbox_roots).map(|&i| schema.field(i).clone()).collect();
        let out_schema = Arc::new(Schema::new(fields));
        let rest = ProjectionMask::roots(self.meta.parquet_schema(), rest_roots);
        let bbox = ProjectionMask::roots(self.meta.parquet_schema(), bbox_roots);
        let geometry = &columns[0];

        let read = py.detach(|| {
            pool(threads).install(|| {
                row_groups.par_iter().map(|&rg| self.read_late(rg, &rest, &bbox, geometry, &query, use_pages)).collect::<Res<Vec<_>>>()
            })
        });
        let groups = read.map_err(|e| PyIOError::new_err(e.to_string()))?;
        let (mut bbox_rows, mut geometry_rows, mut kept_rows) = (0, 0, 0);
        let mut batches = Vec::new();
        for group in groups {
            bbox_rows += group.bbox_rows;
            geometry_rows += group.geometry_rows;
            kept_rows += group.kept_rows;
            batches.extend(group.kept);
        }
        let table = PyTable::try_new(batches, out_schema)?.into_pyarrow(py)?.unbind();
        Ok((table, bbox_rows, geometry_rows, kept_rows))
    }
}

/// Takes a table and gives it straight back: measures the cost of crossing between Python and Rust.
#[pyfunction]
fn pass_through(py: Python<'_>, table: PyTable) -> PyArrowResult<Py<PyAny>> {
    Ok(table.into_pyarrow(py)?.unbind())
}

#[pymodule]
fn starlet_read(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParquetFile>()?;
    m.add_function(wrap_pyfunction!(pass_through, m)?)?;
    Ok(())
}

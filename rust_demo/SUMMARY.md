# Rust + Python integration: Web Mercator reprojection demo

**Question:** can Rust be called from Starlet's Python, what does the data hand-off cost, and is GeoArrow needed?
**Answer so far:** integration works with PyO3 + maturin. Rust can read the stored WKB column directly, so GeoArrow is not required for the reprojection step. The cost that remains is rebuilding shapely objects, not Rust.

## What was built (Starlet source unchanged)
`rust_demo/` holds a small Rust crate (`merc_demo`, PyO3 0.29, built with maturin into `starlet/.venv`). It reprojects EPSG:4326 to EPSG:3857 and was compared with Starlet's own `reproject_geometries` / `reproject_table` (`starlet/_internal/tiling/crs.py`).

## Findings

**1. Borrowing works and is free.** Rust reads NumPy and Arrow buffers in place. Checked by buffer-address match, so it is borrowed and not copied.

**2. Where the time goes (500k points, WKB column to shapely output, today's Python = 689 ms):**

| Stage | Time |
|---|---|
| `to_numpy` of the WKB column | 24 ms |
| `from_wkb` decode (shapely) | 239 ms |
| `reproject_geometries` | 389 ms |
| Rust parses WKB + reprojects, 1 thread | 13 ms |
| Rust parses WKB + reprojects, 8 threads | 3 ms |
| Rebuilding shapely points afterwards | 178 ms |

With the shapely rebuild included, the Rust path is about **3.6-3.8x** faster end to end. The rebuild is the biggest cost left.

**3. GeoArrow vs Rust parsing WKB (points).**

| | Rust parses WKB | GeoArrow column |
|---|---|---|
| Per run, 1 thread | 13 ms | 11 ms |
| One-time cost | none | 276 ms |
| Python / stored-data changes | none | many (WKB used in 87 places across 15 files; sampling priority is a CRC of the WKB bytes) |

GeoArrow saved about 2 ms per run. The earlier 6x gain over extracting from shapely objects each run is real, but that comparison is against the extraction step, not against Rust parsing WKB itself.

**4. Lines and polygons also work from WKB** (general WKB walker, same output as Python):

| Data | Python today | Rust 1 thread | Rust 8 threads |
|---|---|---|---|
| Rails lines, 145k rows / 3.05M coords | 1,475 ms | 77 ms (19x) | 32 ms (46x) |
| NE polygons, 96k rows / 27.2M coords | 9,414 ms | 769 ms (12x) | 257 ms (37x) |

Lines match to 7.5e-9 m. For polygons, 722 of 1.29M coordinates differ, all at latitude -90 (a Mercator singularity at the pole, not a parser bug).

**5. Triage and sparsify are different.** They work on property values and numeric scores, not geometry, so they need plain Arrow columns, not GeoArrow. Their properties live in Python dicts, which Rust cannot borrow, so the cost to measure there is building columns from the dicts.

## Caveats (not verified)
- The machine was not idle (browsers and chat apps open); CPU load not sampled. Treat ratios as more reliable than milliseconds.
- Threads stop helping around 8 (24 threads was no faster).
- Polygon test data was one file repeated 21x to get a measurable size.
- Geometry-changing steps (clip, simplify, `make_valid`, MVT encode) are untested and would need a real Rust geometry library.
- Starlet's own test suite was not run, since nothing in Starlet changed.

## Proposed next steps (one stage at a time)
1. Map the pipeline: input type, output type, data owner and code location per stage.
2. Profile each stage for its share of total time on fresh runs.
3. For each candidate stage, decide what data Rust would need.
4. Measure the hand-off cost per stage before porting any logic.
5. Ask the GeoArrow question only for stages that need geometry.

## Reproduce
From `starlet/`: `.venv/Scripts/python.exe ../rust_demo/bench_wkb.py` (also `bench_geoarrow.py`, `bench_full.py`, `bench_shapes.py`).

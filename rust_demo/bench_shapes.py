"""Lines and polygons: Starlet's real reproject_table (Python) vs Rust WKB walker. WKB in -> WKB out."""
import glob, json, statistics, time
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, shapely
from shapely import from_wkb
import merc_demo
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, geoparquet_crs, reproject_table

ROOT = Path(__file__).parent.parent / "starlet"
RUNS = 5


def stats(fn):
    fn(); ts = []
    for _ in range(RUNS):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), min(ts), max(ts)


f = lambda s: f"{s[0]:8.1f} [{s[1]:.0f}-{s[2]:.0f}] ms"


def report(name, col):
    types = {shapely.get_geometry_type if False else None}
    geoms = from_wkb(col.to_numpy(zero_copy_only=False))
    tnames = sorted({str(shapely.get_type_id(g)) for g in geoms[:200000] if g is not None})
    ncoords = len(shapely.get_coordinates(geoms))
    tbl = pa.table({"geometry": col})
    ref_tbl, _ = reproject_table(tbl, "geometry", WGS84_CRS, WEB_MERCATOR_CRS)
    got = merc_demo.wkb_reproject_column(col, 8)
    rc = shapely.get_coordinates(from_wkb(ref_tbl["geometry"].to_numpy(zero_copy_only=False)))
    gc = shapely.get_coordinates(from_wkb(got.to_numpy(zero_copy_only=False)))
    diff = float(np.abs(rc - gc).max())
    same_struct = bool((shapely.get_num_coordinates(from_wkb(got.to_numpy(zero_copy_only=False)))
                        == shapely.get_num_coordinates(geoms)).all())
    py = stats(lambda: reproject_table(tbl, "geometry", WGS84_CRS, WEB_MERCATOR_CRS))
    r = {t: stats(lambda t=t: merc_demo.wkb_reproject_column(col, t)) for t in (0, 4, 8, 24)}
    print(f"\n=== {name}: {len(col):,} rows, {ncoords:,} coordinates, shapely type ids {tnames}, "
          f"WKB size {col.nbytes/1e6:.0f} MB")
    print(f"parity vs reproject_table: max coord diff {diff:.2e} m | same vertex counts: {same_struct}")
    print(f"  {'Python reproject_table (today)':34s}{f(py)}")
    for t, s in r.items():
        print(f"  {('Rust, 1 thread' if t == 0 else f'Rust, rayon {t} threads'):34s}{f(s)}   {py[0]/s[0]:6.1f}x")


# ---- lines: Starlet's own rails output ----
files = sorted(glob.glob(str(ROOT / "rails_indexed" / "**" / "*.parquet"), recursive=True), key=lambda p: -Path(p).stat().st_size)
t = pq.read_table(files[0])
crs = geoparquet_crs(t.schema, "geometry")
print("rails file:", Path(files[0]).name[:40], "| CRS:", str(crs)[:50].replace("\n", " "))
col = t["geometry"].combine_chunks().cast(pa.binary())
report("LINES (rails tile)", col)

# ---- polygons: Natural Earth states/provinces, repeated to get a measurable size ----
fc = json.load(open(ROOT / "benchmark_data" / "NE_states_provinces.geojson", encoding="utf-8"))
geoms = np.array([shapely.geometry.shape(ft["geometry"]) for ft in fc["features"] if ft["geometry"]], dtype=object)
reps = max(1, 100_000 // len(geoms))
wkb = shapely.to_wkb(np.tile(geoms, reps), hex=False)
col = pa.array(list(wkb), type=pa.binary())
print(f"\nNE polygons: {len(geoms):,} features x {reps} repeats (repeated only to get a measurable size)")
report("POLYGONS (NE states/provinces)", col)

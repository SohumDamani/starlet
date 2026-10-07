"""Rust on shapely geometries (extract each run) vs Rust on a GeoArrow-style Arrow column (built once)."""
import statistics, time
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, shapely
from shapely import from_wkb
import merc_demo
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, reproject_geometries

DATA = Path(__file__).parent.parent / "starlet" / "benchmark_data" / "osm21_pois_sample.parquet"
RUNS = 7


def stats(fn):
    fn(); ts = []
    for _ in range(RUNS):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), min(ts), max(ts)


f = lambda s: f"{s[0]:8.1f} [{s[1]:.0f}-{s[2]:.0f}] ms"

geoms = from_wkb(pq.read_table(DATA, columns=["geometry"])["geometry"].to_numpy(zero_copy_only=False))


def build_geoarrow():
    c = shapely.get_coordinates(geoms)
    return pa.StructArray.from_arrays([pa.array(c[:, 0]), pa.array(c[:, 1])], names=["x", "y"])


one_time = stats(build_geoarrow)          # paid once, when the column is created/stored
col = build_geoarrow()                    # the stored, Python-owned Arrow column

# ---- parity ----
ref = shapely.get_coordinates(reproject_geometries(geoms, WGS84_CRS, WEB_MERCATOR_CRS)[0])
out = merc_demo.to_mercator_arrow(col)
got = np.column_stack([out.field("x").to_numpy(), out.field("y").to_numpy()])
print("output type:", out.type, "| length:", len(out))
print(f"parity vs reproject_geometries: max diff {np.abs(ref - got).max():.2e} m")
print("zero-copy (x child borrowed):", merc_demo.arrow_x_address(col) == col.field("x").buffers()[1].address)

# ---- per-run paths ----
A = stats(lambda: merc_demo.to_mercator(shapely.get_coordinates(geoms)))       # shapely -> extract -> Rust (numpy)
B = stats(lambda: merc_demo.to_mercator_arrow(col))                            # Arrow column -> Rust -> Arrow
py_whole = stats(lambda: reproject_geometries(geoms, WGS84_CRS, WEB_MERCATOR_CRS))
out_x, out_y = out.field("x").to_numpy(), out.field("y").to_numpy()
rebuild = stats(lambda: shapely.points(out_x, out_y))

print(f"\npoints: {len(geoms):,}   runs: {RUNS} (median [min-max])\n")
print(f"{'one-time: build GeoArrow column':44s}{f(one_time)}")
print(f"{'per-run  A: extract + Rust (no GeoArrow)':44s}{f(A)}")
print(f"{'per-run  B: Rust on Arrow column':44s}{f(B)}")
print(f"{'per-run  rebuild shapely (if still needed)':44s}{f(rebuild)}")
print(f"{'Python reproject_geometries (today)':44s}{f(py_whole)}")
print(f"\nB vs A (per-run, no rebuild):             {A[0]/B[0]:.1f}x")
print(f"B + rebuild vs today's Python:            {py_whole[0]/(B[0]+rebuild[0]):.2f}x")
print(f"A + rebuild vs today's Python:            {py_whole[0]/(A[0]+rebuild[0]):.2f}x")
print(f"B (no rebuild) vs today's Python:         {py_whole[0]/B[0]:.1f}x")

"""Python vs Rust: WGS84 -> Web Mercator on the 500k-point OSM sample."""
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shapely
from shapely import from_wkb

import merc_demo
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, reproject_geometries

DATA = Path(__file__).parent.parent / "starlet" / "benchmark_data" / "osm21_pois_sample.parquet"
RUNS = 5
R = 6_378_137.0


def median_ms(fn, runs=RUNS):
    fn()  # warm-up (pyproj transformer build, page faults)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def numpy_mercator(c):
    out = np.empty_like(c)
    out[:, 0] = R * np.radians(c[:, 0])
    out[:, 1] = R * np.log(np.tan(np.pi / 4 + np.radians(c[:, 1]) / 2))
    return out


table = pq.read_table(DATA, columns=["geometry"])
geoms = from_wkb(table["geometry"].to_numpy(zero_copy_only=False))
print(f"points: {len(geoms):,}  nulls: {int(shapely.is_missing(geoms).sum())}  empty: {int(shapely.is_empty(geoms).sum())}")

coords = shapely.get_coordinates(geoms)  # (N, 2) float64
print("coords:", coords.shape, coords.dtype, "C-contiguous:", coords.flags["C_CONTIGUOUS"])

# --- parity ---
py_result = shapely.get_coordinates(reproject_geometries(geoms, WGS84_CRS, WEB_MERCATOR_CRS)[0])
rs_result = merc_demo.to_mercator(coords)
diff = np.abs(py_result - rs_result)
print(f"parity vs pyproj path: max abs diff = {diff.max():.3e} m  (max |value| = {np.abs(py_result).max():.3e})")
print(f"parity vs numpy:       max abs diff = {np.abs(numpy_mercator(coords) - rs_result).max():.3e} m")

# --- zero-copy check ---
print("borrowed, not copied:", merc_demo.data_address(coords) == coords.ctypes.data)

# --- timings ---
rows = [
    ("extract coords (get_coordinates)", median_ms(lambda: shapely.get_coordinates(geoms))),
    ("hand-off only (Rust passthrough)", median_ms(lambda: merc_demo.passthrough(coords))),
    ("Python: reproject_geometries", median_ms(lambda: reproject_geometries(geoms, WGS84_CRS, WEB_MERCATOR_CRS))),
    ("Python: numpy closed-form", median_ms(lambda: numpy_mercator(coords))),
    ("Rust: to_mercator", median_ms(lambda: merc_demo.to_mercator(coords))),
]
print()
for name, ms in rows:
    print(f"{name:36s} {ms:9.2f} ms")
t = dict(rows)
print()
print(f"Rust vs reproject_geometries : {t['Python: reproject_geometries'] / t['Rust: to_mercator']:.1f}x")
print(f"Rust vs numpy closed-form    : {t['Python: numpy closed-form'] / t['Rust: to_mercator']:.1f}x")
print(f"Rust incl. extraction vs reproject_geometries: "
      f"{t['Python: reproject_geometries'] / (t['Rust: to_mercator'] + t['extract coords (get_coordinates)']):.1f}x")

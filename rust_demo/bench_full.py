"""Stage-by-stage, like for like: Python vs Rust on 500k points (shapely geoms in -> shapely geoms out)."""
import statistics, subprocess, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, shapely
from pyproj import Transformer
from shapely import from_wkb
import merc_demo
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, crs_equal, reproject_geometries

DATA = Path(__file__).parent.parent / "starlet" / "benchmark_data" / "osm21_pois_sample.parquet"
RUNS = 7


def cpu_load():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         r"(Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 3)"
         ".CounterSamples | ForEach-Object { [math]::Round($_.CookedValue,1) }"],
        capture_output=True, text=True).stdout.split()
    return [float(x) for x in out]


def stats(fn):
    fn(); ts = []
    for _ in range(RUNS):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), min(ts), max(ts)


geoms = from_wkb(pq.read_table(DATA, columns=["geometry"])["geometry"].to_numpy(zero_copy_only=False))
coords = shapely.get_coordinates(geoms)
tr = Transformer.from_crs(WGS84_CRS, WEB_MERCATOR_CRS, always_xy=True)
xy = tr.transform(coords[:, 0], coords[:, 1])
out = merc_demo.to_mercator(coords)

load_before = cpu_load()

S = {
    "1 setup (crs_equal + Transformer)": (
        stats(lambda: (crs_equal(WGS84_CRS, WEB_MERCATOR_CRS), Transformer.from_crs(WGS84_CRS, WEB_MERCATOR_CRS, always_xy=True))),
        None),
    "2 extract coords": (stats(lambda: shapely.get_coordinates(geoms)), stats(lambda: shapely.get_coordinates(geoms))),
    "3 convert math": (stats(lambda: tr.transform(coords[:, 0], coords[:, 1])), stats(lambda: merc_demo.to_mercator(coords))),
    "4 rebuild shapely points": (stats(lambda: shapely.points(np.column_stack(xy))), stats(lambda: shapely.points(out))),
}
total_py = stats(lambda: reproject_geometries(geoms, WGS84_CRS, WEB_MERCATOR_CRS))
total_rs = stats(lambda: shapely.points(merc_demo.to_mercator(shapely.get_coordinates(geoms))))

load_after = cpu_load()

py = reproject_geometries(geoms, WGS84_CRS, WEB_MERCATOR_CRS)[0]
rs = shapely.points(merc_demo.to_mercator(shapely.get_coordinates(geoms)))
diff = np.abs(shapely.get_coordinates(py) - shapely.get_coordinates(rs)).max()

print(f"points: {len(geoms):,}   runs per stage: {RUNS} (median, [min-max]) after 1 warm-up")
print(f"CPU load before: {load_before} %   after: {load_after} %   (24 logical cores)\n")
print(f"{'stage':36s}{'Python':>26s}{'Rust path':>26s}")
f = lambda s: f"{s[0]:8.1f} [{s[1]:.0f}-{s[2]:.0f}] ms" if s else "         (none: no setup)"
for k, (p, r) in S.items():
    print(f"{k:36s}{f(p):>26s}{f(r):>26s}")
print(f"{'WHOLE, measured as one call':36s}{f(total_py):>26s}{f(total_rs):>26s}")
print(f"\nwhole-call speedup (Python / Rust path): {total_py[0] / total_rs[0]:.2f}x")
print(f"convert-stage speedup only:              {S['3 convert math'][0][0] / S['3 convert math'][1][0]:.2f}x")
print(f"max coordinate difference: {diff:.2e} m")

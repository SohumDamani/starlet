"""Starting from the stored WKB column: Python today vs Rust-parses-WKB (1..N threads) vs GeoArrow column."""
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
table = pq.read_table(DATA, columns=["geometry"])
col = table["geometry"].combine_chunks()           # what Starlet holds after reading parquet
print("WKB column type:", col.type, "| rows:", len(col), "| nulls:", col.null_count)

# ---- parity + zero-copy ----
ref_geoms = from_wkb(table["geometry"].to_numpy(zero_copy_only=False))
ref = shapely.get_coordinates(reproject_geometries(ref_geoms, WGS84_CRS, WEB_MERCATOR_CRS)[0])
out = merc_demo.wkb_to_mercator_arrow(col, 8)
got = np.column_stack([out.field("x").to_numpy(), out.field("y").to_numpy()])
print(f"parity vs reproject_geometries: max diff {np.abs(ref - got).max():.2e} m")
print("WKB buffer borrowed (not copied):", merc_demo.wkb_values_address(col) == col.buffers()[2].address)

# ---- today's Python path, in stages (as in mvt_generator.py:648-651) ----
to_np = stats(lambda: table["geometry"].to_numpy(zero_copy_only=False))
raw = table["geometry"].to_numpy(zero_copy_only=False)
decode = stats(lambda: from_wkb(raw))
reproj = stats(lambda: reproject_geometries(ref_geoms, WGS84_CRS, WEB_MERCATOR_CRS))
today = stats(lambda: reproject_geometries(from_wkb(table["geometry"].to_numpy(zero_copy_only=False)), WGS84_CRS, WEB_MERCATOR_CRS))

# ---- Rust parses WKB ----
rust = {t: stats(lambda t=t: merc_demo.wkb_to_mercator_arrow(col, t)) for t in (0, 1, 4, 8, 24)}

# ---- GeoArrow column (stored once) ----
def build():
    c = shapely.get_coordinates(from_wkb(raw))
    return pa.StructArray.from_arrays([pa.array(c[:, 0]), pa.array(c[:, 1])], names=["x", "y"])
one_time = stats(build)
ga = build()
ga_run = stats(lambda: merc_demo.to_mercator_arrow(ga))

ox, oy = out.field("x").to_numpy(), out.field("y").to_numpy()
rebuild = stats(lambda: shapely.points(ox, oy))

print(f"\npoints: {len(col):,}   runs: {RUNS} (median [min-max])\n")
print("TODAY'S PYTHON (WKB column in, shapely out)")
print(f"  {'to_numpy (bytes objects)':40s}{f(to_np)}")
print(f"  {'from_wkb decode':40s}{f(decode)}")
print(f"  {'reproject_geometries':40s}{f(reproj)}")
print(f"  {'whole':40s}{f(today)}")
print("\nRUST PARSES THE WKB COLUMN (Arrow in, Arrow out; no shapely)")
for t, s in rust.items():
    print(f"  {('1 thread (sequential)' if t == 0 else f'rayon, {t} thread' + ('s' if t > 1 else '')):40s}{f(s)}")
print("\nGEOARROW COLUMN (stored once)")
print(f"  {'one-time build from WKB':40s}{f(one_time)}")
print(f"  {'per run: Rust on Arrow column':40s}{f(ga_run)}")
print(f"\nIf shapely objects are still needed afterwards: + {f(rebuild)} rebuild")
best = min(rust.values(), key=lambda s: s[0])
print(f"\nRust-parses-WKB (best) vs today's whole:       {today[0]/best[0]:.1f}x   (with rebuild: {today[0]/(best[0]+rebuild[0]):.2f}x)")
print(f"Rust-parses-WKB (1 thread) vs today's whole:   {today[0]/rust[0][0]:.1f}x   (with rebuild: {today[0]/(rust[0][0]+rebuild[0]):.2f}x)")
print(f"GeoArrow per run vs today's whole:             {today[0]/ga_run[0]:.1f}x   (with rebuild: {today[0]/(ga_run[0]+rebuild[0]):.2f}x)")
print(f"Rust-parses-WKB 1 thread vs GeoArrow per run (both single-thread): GeoArrow is {rust[0][0]/ga_run[0]:.2f}x faster")
print(f"Rust-parses-WKB best (parallel) vs GeoArrow per run (single-thread): WKB is {ga_run[0]/best[0]:.1f}x faster")

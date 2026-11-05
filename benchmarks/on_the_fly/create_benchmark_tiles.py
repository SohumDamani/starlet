"""Create a benchmark tile set for one dataset (no timing).

1. Pick N random records (uniform over all rows, fixed seed), any geometry type; skip empty geometries.
2. For each, take a point inside its shape and find its tile at each zoom.
3. Save <name>/points.csv (one row per point) and <name>/tiles.csv (one row per unique tile).

Only the picked records are read, so it also works on very large datasets.
--from-points: skip 1-2 and reuse the points already in <name>/points.csv (lon/lat), e.g. to add zoom levels.
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import shapely

from starlet._internal.mvt.helpers import mercator_bounds_to_tile_range
from starlet._internal.mvt.intermediate_tile import feature_priority
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, geoparquet_crs, reproject_geometries

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="built dataset folder (contains parquet_tiles/)")
    ap.add_argument("--name", required=True, help="benchmark name; files go to on_the_fly/<name>/")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zmin", type=int, default=2)
    ap.add_argument("--zmax", type=int, default=10)
    ap.add_argument("--from-points", action="store_true",
                    help="reuse <name>/points.csv instead of sampling again (same points, other zoom levels)")
    args = ap.parse_args()
    out = HERE / args.name
    if args.from_points:
        with open(out / "points.csv", newline="") as f:
            points = list(csv.DictReader(f))
        lonlat = shapely.points([float(p["lon"]) for p in points], [float(p["lat"]) for p in points])
        mercator, _ = reproject_geometries(lonlat, WGS84_CRS, WEB_MERCATOR_CRS)
    else:
        points, mercator = sample_points(args)
        out.mkdir(exist_ok=True)
        with open(out / "points.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(points[0]))
            w.writeheader()
            w.writerows(points)

    tiles = defaultdict(list)
    for p, m in zip(points, mercator):
        for z in range(args.zmin, args.zmax + 1):
            x, y, _, _ = mercator_bounds_to_tile_range(z, m.x, m.y, m.x, m.y)
            tiles[(z, x, y)].append(p["point_id"])

    with open(out / "tiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z", "x", "y", "n_points", "point_ids"])
        for (z, x, y), ids in sorted(tiles.items()):
            w.writerow([z, x, y, len(ids), " ".join(ids)])

    print(f"{len(points)} points -> {len(tiles)} unique tiles")
    for z in range(args.zmin, args.zmax + 1):
        print(f"  zoom {z:2d}: {sum(1 for t in tiles if t[0] == z):3d} tiles")
    print(f"saved {out / 'tiles.csv'}")


def sample_points(args):
    """Steps 1-3: N random records and a point inside each; returns (point rows, points in Web Mercator)."""
    # 1. every row has a global position: (file, row) from the footer row counts
    files = sorted((Path(args.dataset) / "parquet_tiles").glob("*.parquet"))
    counts = [pq.read_metadata(f).num_rows for f in files]
    total = sum(counts)
    print(f"{total:,} records in {len(files)} files")

    def locate(pos):
        for f, n in zip(files, counts):
            if pos < n:
                return f, pos
            pos -= n

    # 2. random positions, read only those records; skip empty geometries until N are found
    rng = random.Random(args.seed)
    picked, tried = [], set()                     # picked: (file, row, wkb, crs)
    while len(picked) < args.n:
        pos = rng.randrange(total)                # uniform over all rows
        if pos in tried:                          # never the same row twice
            continue
        tried.add(pos)
        f, row = locate(pos)
        table = pq.ParquetFile(f).read(columns=["geometry"]).slice(row, 1)
        wkb = table["geometry"][0].as_py()
        if wkb and not shapely.from_wkb(wkb).is_empty:
            picked.append((f.name, row, wkb, geoparquet_crs(table.schema, "geometry") or WGS84_CRS))

    # 3. a point inside each shape, in lon/lat and Web Mercator
    shapes = shapely.from_wkb([p[2] for p in picked])
    inside = shapely.point_on_surface(shapes)
    mercator, _ = reproject_geometries(inside, picked[0][3], WEB_MERCATOR_CRS)

    points = [{"point_id": f"P{i + 1:03d}", "file": file, "row": row, "lon": p.x, "lat": p.y,
               "geom_type": shape.geom_type, "priority": feature_priority(wkb)}
              for i, ((file, row, wkb, _), shape, p) in enumerate(zip(picked, shapes, inside))]
    return points, mercator


if __name__ == "__main__":
    main()

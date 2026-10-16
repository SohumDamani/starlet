"""Pick benchmark tiles for on-the-fly generation (no timing).

1. Read the geometry column of every file in <dataset>/parquet_tiles/.
2. Pick N random records (fixed seed).
3. For each, take a point inside its shape and find its tile at each zoom.
4. Save points.csv (one row per point) and tiles.csv (one row per unique tile).
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
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zmin", type=int, default=2)
    ap.add_argument("--zmax", type=int, default=10)
    ap.add_argument("--polygons-only", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=HERE)
    args = ap.parse_args()

    # 1. every record = (file, row, wkb)
    records = []
    for path in sorted((Path(args.dataset) / "parquet_tiles").glob("*.parquet")):
        table = pq.read_table(path, columns=["geometry"])
        crs = geoparquet_crs(table.schema, "geometry") or WGS84_CRS
        for row, wkb in enumerate(table["geometry"].to_pylist()):
            if args.polygons_only:
                shape = shapely.from_wkb(wkb)
                if shape is None or shape.is_empty or shape.geom_type not in {"Polygon", "MultiPolygon"}:
                    continue
            records.append((path.name, row, wkb, crs))
    print(f"{len(records):,} records in {args.dataset}")

    # 2. pick N at random
    picked = random.Random(args.seed).sample(records, args.n)

    # 3. a point inside each shape, in lon/lat and Web Mercator
    shapes = shapely.from_wkb([r[2] for r in picked])
    inside = shapely.point_on_surface(shapes)
    mercator, _ = reproject_geometries(inside, picked[0][3], WEB_MERCATOR_CRS)

    points, tiles = [], defaultdict(list)
    for i, ((file, row, wkb, _), shape, p, m) in enumerate(zip(picked, shapes, inside, mercator)):
        point_id = f"P{i + 1:03d}"
        points.append({"point_id": point_id, "file": file, "row": row, "lon": p.x, "lat": p.y,
                       "geom_type": shape.geom_type, "priority": feature_priority(wkb)})
        for z in range(args.zmin, args.zmax + 1):
            x, y, _, _ = mercator_bounds_to_tile_range(z, m.x, m.y, m.x, m.y)
            tiles[(z, x, y)].append(point_id)

    # 4. save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]))
        w.writeheader()
        w.writerows(points)
    with open(args.output_dir / "tiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z", "x", "y", "n_points", "point_ids"])
        for (z, x, y), ids in sorted(tiles.items()):
            w.writerow([z, x, y, len(ids), " ".join(ids)])

    print(f"{len(points)} points -> {len(tiles)} unique tiles")
    for z in range(args.zmin, args.zmax + 1):
        print(f"  zoom {z:2d}: {sum(1 for t in tiles if t[0] == z):3d} tiles")
    print(f"saved {args.output_dir / 'points.csv'} and {args.output_dir / 'tiles.csv'}")


if __name__ == "__main__":
    main()

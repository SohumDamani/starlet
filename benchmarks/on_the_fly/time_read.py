"""Time the five on-the-fly stages (Read, Pick, Decode, Transform, Encode) for every benchmark tile.

One generation per tile. With --verify, production generate_tile_ex is also run (untimed for the
stages) and the staged tile must match it byte-for-byte; generation_ms is then its total time.
Decode includes WKB extraction/conversion and dot classification. Transform
includes projection, simplification, clipping, and cleanup. Encode includes
geometry commands, attributes, feature/dot assembly, and tile serialization.
Cold: a new index and an empty row-group cache for every tile. OS file cache: warmed once at the start.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import shapely

from starlet._internal.config import config_value
from starlet._internal.mvt.fast_tile import (
    RowGroupCache, TileFrame, select, tile_geometries, generate_tile_ex, invalidate_caches,
)
from starlet._internal.mvt.intermediate_tile import normalize_simplify_tolerance
from starlet._internal.mvt.mvt_encoder import (
    GEOM_LINE, GEOM_POLYGON, LayerEncoder, geometry_commands,
    packed_segment_dots, packed_square_dots,
)
from starlet._internal.server.tiler.parquet_index import ParquetIndex

HERE = Path(__file__).resolve().parent
EXTENT = int(config_value("mvt", "extent"))
BUFFER = int(config_value("mvt", "buffer"))
FEATURE_CAPACITY = int(config_value("mvt", "feature_capacity"))


def read_tile(index, cache, frame):
    """Read candidate rows and collect overlapping bbox/CRC arrays."""
    qx0, qy0, qx1, qy1 = frame.query_lonlat
    records_read = 0
    lon0s, lat0s, lon1s, lat1s, crcs = [], [], [], [], []
    tables, geom_cols, attr_cols, part_of, row_of = [], [], [], [], []

    for path in index.find_intersecting_files(frame.query_lonlat):
        _pf, geom_col, _has_bbox, _crs, attrs, _stats = cache.file_info(path)
        for rg in cache.candidate_row_groups(path, frame.query_lonlat):
            table, a, b, c, d, crc = cache.row_group(path, rg)
            hit = (c >= qx0) & (a <= qx1) & (d >= qy0) & (b <= qy1)
            records_read += table.num_rows
            if not hit.any():
                continue
            rows = np.flatnonzero(hit)
            tables.append(table)
            geom_cols.append(geom_col)
            attr_cols.append(attrs)
            part_of.append(np.full(len(rows), len(tables) - 1, dtype=np.int64))
            row_of.append(rows)
            lon0s.append(a[rows]); lat0s.append(b[rows]); lon1s.append(c[rows]); lat1s.append(d[rows]); crcs.append(crc[rows])

    return records_read, (lon0s, lat0s, lon1s, lat1s, crcs), (tables, geom_cols, attr_cols, part_of, row_of)


def pick_tile(frame, pieces, source):
    """Select winners and retain row references for the subsequent stages."""
    lon0s, lat0s, lon1s, lat1s, crcs = pieces
    tables, geom_cols, attr_cols, part_of, row_of = source
    if not lon0s:
        empty = np.zeros(0, dtype=np.int64)
        return 0, (empty, empty, np.zeros(0), np.zeros(0)), (tables, geom_cols, attr_cols, empty, empty)
    lon0, lat0, lon1, lat1, crc = (np.concatenate(p) for p in pieces)
    part_of_arr = np.concatenate(part_of)
    row_of_arr = np.concatenate(row_of)
    selection = select(frame, lon0, lat0, lon1, lat1, crc, FEATURE_CAPACITY)
    return len(lon0), selection, (tables, geom_cols, attr_cols, part_of_arr, row_of_arr)


def decode_tile(frame, source, selection):
    """Benchmark copy of generate_tile_ex's WKB extraction and decode block."""
    tables, geom_cols, attr_cols, part_of_arr, row_of_arr = source
    full, dots, dot_cx, dot_cy = selection

    def wkb_of(idx: np.ndarray) -> list:
        out = []
        for pi in np.unique(part_of_arr[idx]):
            m = part_of_arr[idx] == pi
            rows = row_of_arr[idx[m]]
            col = tables[pi][geom_cols[pi]].take(pa.array(rows)).to_pylist()
            out.append((idx[m], col))
        # restore idx order
        merged = [None] * len(idx)
        pos = {int(i): j for j, i in enumerate(idx)}
        for sub_idx, col in out:
            for i, w in zip(sub_idx.tolist(), col):
                merged[pos[i]] = w
        return merged

    dot_geoms = shapely.from_wkb(wkb_of(dots), on_invalid="ignore") if len(dots) else np.zeros(0, dtype=object)
    dot_tid = shapely.get_type_id(dot_geoms) if len(dots) else np.zeros(0, dtype=np.int64)
    lo, hi = -frame.buffer, frame.extent + frame.buffer
    inside = (dot_cx >= lo) & (dot_cy >= lo) & (dot_cx <= hi) & (dot_cy <= hi)
    poly_dots = ((dot_tid == 3) | (dot_tid == 6) | (dot_tid == 7)) & inside
    line_dots = ((dot_tid == 1) | (dot_tid == 5) | (dot_tid == 2)) & inside
    point_dots = ((dot_tid == 0) | (dot_tid == 4)) & inside
    indiv = np.sort(np.concatenate((full, dots[point_dots])))
    is_full = np.isin(indiv, full)
    geoms = shapely.from_wkb(wkb_of(indiv), on_invalid="ignore") if len(indiv) else np.zeros(0, dtype=object)
    return geoms, indiv, is_full, poly_dots, line_dots


def encode_tile(frame, source, selection, decoded, tgeoms):
    """Benchmark copy of generate_tile_ex's attribute and MVT encoding block."""
    tables, geom_cols, attr_cols, part_of_arr, row_of_arr = source
    full, dots, dot_cx, dot_cy = selection
    geoms, indiv, is_full, poly_dots, line_dots = decoded
    strip_point_attrs = len(dots) > max(1, int(FEATURE_CAPACITY))
    layer = LayerEncoder("layer0", frame.extent)

    def props_of(idx: np.ndarray) -> list:
        out = [None] * len(idx)
        pos = {int(i): j for j, i in enumerate(idx)}
        for pi in np.unique(part_of_arr[idx]):
            m = part_of_arr[idx] == pi
            sub = idx[m]
            rows = pa.array(row_of_arr[sub])
            cols = {name: tables[pi][name].take(rows).to_pylist() for name in attr_cols[pi]}
            for j, i in enumerate(sub.tolist()):
                out[pos[i]] = [(name, cols[name][j]) for name in attr_cols[pi]]
        return out

    cmds, types = geometry_commands(tgeoms)
    need_props = np.ones(len(indiv), dtype=bool)
    if strip_point_attrs:
        need_props &= is_full | (types != 1)
    props = props_of(indiv[need_props]) if need_props.any() else []
    pj = 0
    for j in range(len(indiv)):
        if cmds[j] is None:
            if need_props[j]:
                pj += 1
            continue
        p = None
        if need_props[j]:
            p = props[pj]
            pj += 1
        layer.add(cmds[j], int(types[j]), p)

    half = frame.cell * 0.5
    sq = packed_square_dots(np.column_stack((dot_cx[poly_dots], dot_cy[poly_dots])), half)
    if sq:
        layer.add(sq, GEOM_POLYGON, None)
    seg = packed_segment_dots(np.column_stack((dot_cx[line_dots], dot_cy[line_dots])), half)
    if seg:
        layer.add(seg, GEOM_LINE, None)
    return layer.encode_tile(), layer.feature_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="built dataset folder (contains parquet_tiles/)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tiles", type=Path, default=HERE / "tiles.csv")
    ap.add_argument("--output", type=Path, default=HERE / "results_python.csv")
    ap.add_argument("--verify", action="store_true",
                    help="also run production generate_tile_ex and require byte-identical tiles")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be at least 1")

    parquet_dir = Path(args.dataset) / "parquet_tiles"
    with args.tiles.open(newline="") as fh:
        tiles = [(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(fh)]
    for f in parquet_dir.glob("*.parquet"):          # warm the OS file cache once
        f.read_bytes()
    print(f"{len(tiles)} tiles, {args.runs} runs; CPU load {psutil.cpu_percent(interval=1.0):.0f}%", flush=True)

    out = args.output
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "z", "x", "y", "read_ms", "pick_ms",
                    "records_read", "records_in_tile", "full", "dots",
                    "generation_ms", "features", "tile_bytes",
                    "decode_ms", "transform_ms", "encode_ms"])
        for run in range(1, args.runs + 1):
            order = tiles[:]
            random.Random(run).shuffle(order)          # different, repeatable order per run
            for tile_number, (z, x, y) in enumerate(order, 1):
                index = ParquetIndex(parquet_dir)                                   # cold
                cache = RowGroupCache(index, int(config_value("mvt", "row_group_cache")))   # cold
                frame = TileFrame(z, x, y, EXTENT, BUFFER)                          # z/x/y -> rectangle
                tolerance = normalize_simplify_tolerance(None, frame.cell)

                start = time.perf_counter()
                records_read, pieces, source = read_tile(index, cache, frame)
                read_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                records_in_tile, selection, source = pick_tile(frame, pieces, source)
                pick_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                decoded = decode_tile(frame, source, selection)
                decode_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                transformed = tile_geometries(frame, decoded[0], tolerance)
                transform_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                staged_tile, staged_features = encode_tile(frame, source, selection, decoded, transformed)
                encode_ms = (time.perf_counter() - start) * 1000
                full, dots = len(selection[0]), len(selection[1])

                generation_ms = ""
                if args.verify:                                # optional: compare with production
                    del cache, index, frame, pieces, source, selection, decoded, transformed
                    invalidate_caches()
                    start = time.perf_counter()
                    tile, features = generate_tile_ex(
                        args.dataset, (z, x, y), feature_capacity=FEATURE_CAPACITY,
                        extent=EXTENT, buffer=BUFFER,
                    )
                    generation_ms = round((time.perf_counter() - start) * 1000, 6)
                    invalidate_caches()
                    if staged_tile != tile or staged_features != features:
                        raise RuntimeError(f"Benchmark output differs from production for tile {z}/{x}/{y}")

                w.writerow([run, z, x, y, round(read_ms, 6), round(pick_ms, 6),
                            records_read, records_in_tile, full, dots,
                            generation_ms, staged_features, len(staged_tile),
                            round(decode_ms, 6), round(transform_ms, 6), round(encode_ms, 6)])
                fh.flush()  # outside timers: preserve each verified result
                if tile_number == 1 or tile_number % 25 == 0:
                    print(f"run {run}: {tile_number}/{len(order)} tiles" + (" verified" if args.verify else ""), flush=True)
            print(f"run {run} done", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

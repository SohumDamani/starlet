"""Re-split a dataset's boxes into a different row-group size.

Same rows, same order, same file names, same columns and metadata; only the row groups change.
Written like starlet's writer (writer_pool.py:281-289): zstd, statistics on.

  --rows N   every row group has N rows (like starlet's fixed 16,384)
  --mb M     cut a new row group when its uncompressed data reaches M MB (rows per group vary)

Output: <out>/parquet_tiles/*.parquet (only what Read needs; *.parquet is git-ignored).
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

GEOM = "geometry"


def row_bytes(table):
    """Approximate uncompressed bytes per row: its geometry length + an equal share of the other columns."""
    geom = pc.binary_length(table[GEOM]).to_numpy(zero_copy_only=False).astype(np.float64)
    rest = table.nbytes - table[GEOM].nbytes
    return geom + rest / max(table.num_rows, 1)


def cuts_by_mb(table, mb):
    """Row positions where a new row group starts, so each group holds about `mb` MB."""
    total = np.cumsum(row_bytes(table))
    starts, limit = [0], mb * 1024 * 1024
    for i, t in enumerate(total):
        if t - (total[starts[-1] - 1] if starts[-1] else 0) >= limit and i + 1 < table.num_rows:
            starts.append(i + 1)
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="built dataset folder (contains parquet_tiles/)")
    ap.add_argument("--out", required=True, help="variant folder, e.g. rowgroups/asia/rows_1000")
    size = ap.add_mutually_exclusive_group(required=True)
    size.add_argument("--rows", type=int)
    size.add_argument("--mb", type=float)
    args = ap.parse_args()

    out_dir = Path(args.out) / "parquet_tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_groups = 0
    for f in sorted((Path(args.dataset) / "parquet_tiles").glob("*.parquet")):
        table = pq.read_table(f)
        if args.rows:
            pq.write_table(table, out_dir / f.name, compression="zstd", row_group_size=args.rows)
        else:
            starts = cuts_by_mb(table, args.mb) + [table.num_rows]
            with pq.ParquetWriter(out_dir / f.name, table.schema, compression="zstd") as w:
                for a, b in zip(starts[:-1], starts[1:]):
                    w.write_table(table.slice(a, b - a), row_group_size=b - a)   # one row group per slice
        n_groups += pq.read_metadata(out_dir / f.name).num_row_groups

    files = list(out_dir.glob("*.parquet"))
    rows = sum(pq.read_metadata(f).num_rows for f in files)
    disk = sum(f.stat().st_size for f in files)
    print(f"{args.out}: {len(files)} files, {rows:,} rows, {n_groups:,} row groups "
          f"(avg {rows / n_groups:,.0f} rows/group), {disk / 1e9:.2f} GB on disk")


if __name__ == "__main__":
    main()

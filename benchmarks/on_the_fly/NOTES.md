# On-the-fly benchmark: notes

- Benchmark tiles: 100 random asia_postal_codes records (seed 42) → their tiles at zoom 2–10 → 473 unique tiles (`pick_tiles.py`).
- Timing is **cold**: starlet's caches (row-group cache, parquet index) are cleared before every tile.
- The OS file cache stays **warm**: all dataset files are read once before timing. A true disk-cold number would need a run right after a reboot.
- Report it as: "starlet caches cold, OS file cache warm".
- `read_ms` includes more than disk reading: `RowGroupCache.row_group` also turns every WKB into a Python object and computes a crc32 priority for every row (needed later by Pick).
- Asia boxes have 1 row group each, so any overlap reads the whole file.
  (`starlet tile` writes 16,384-row row groups, `writer_pool.py:36`; asia has ~13k rows per box.)
- How each dataset is divided (measured from the Parquet footers):

  | Dataset | Records | Files | Row groups (total; per file median, max) | Rows per row group (median) | On disk | Bytes per record (disk / uncompressed) |
  |---|---|---|---|---|---|---|
  | asia_postal_codes | 184,746 | 14 | 16; 1, 2 | 12,119 | 0.73 GB | ~3,970 / ~6,110 |
  | OSM 2021 parks | 42,769,620 | 487 | 2,852; 5, 19 | 16,384 | 16.96 GB | ~400 / ~520 |

  Asia records are ~10x bigger than parks records, so read time follows bytes read more than record count.
- Later experiment: test row-group pruning, and one Rust thread per row group, on a dataset with
  many row groups per file: OSM parks (~146k rows/file → ~9 row groups) or eBird California.
- Zoom 0–7 is normally pre-generated; on-the-fly mostly serves zoom 8+. Report z8–10 separately from z2–7.
- Upstream bug: Python `fast_tile.py:252` read the wrong footer statistics for files with nested columns (asia has `geometry_bbox`), giving empty tiles. Fixed upstream in 3cf999a; same fix applied locally on 6ce4721.

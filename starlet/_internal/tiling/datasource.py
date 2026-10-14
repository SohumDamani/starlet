from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass, replace
from typing import Iterable, Optional, List, Dict, Any, Tuple
import bz2
import logging
import json
import os
from pathlib import Path
from decimal import Decimal
import math
import zipfile

import pandas as pd
import pyarrow as pa
import numpy as np
from shapely import from_wkb

from starlet._internal.executor import create_process_executor
from starlet._internal.tiling.utils_large import ensure_large_types

logger = logging.getLogger(__name__)
_GEOPARQUET_SUFFIXES = (".parquet", ".geoparquet")
_GEOJSON_SUFFIXES = (
    ".geojson",
    ".geojsonl",
    ".json",
    ".jsonl",
    ".geojson.bz2",
    ".geojsonl.bz2",
    ".json.bz2",
    ".jsonl.bz2",
)
_CSV_SUFFIXES = (".csv", ".tsv", ".txt", ".csv.bz2", ".tsv.bz2", ".txt.bz2")
_PLT_SUFFIXES = (".plt",)
_GPX_SUFFIXES = (".gpx",)
_SHAPEFILE_SUFFIXES = (".shp",)
_ZIP_SUFFIXES = (".zip",)
_GDB_SUFFIXES = (".gdb",)
_TAR_SUFFIXES = (".tar",)
_TAR_BLOCK_SIZE = 512
_TAR_SPLIT_SIZE = 32 * 1024 * 1024
_BZ2_BLOCK_MAGIC = bytes.fromhex("314159265359")
_BZ2_STREAM_HEADER_LEN = 4
_BZ2_SCAN_CHUNK_SIZE = 1024 * 1024


class DataSource:
    def schema(self) -> pa.Schema:
        raise NotImplementedError

    def create_splits(self, num_splits: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def iter_tables(self, split: Optional[Any] = None) -> Iterable[pa.Table]:
        raise NotImplementedError

    def iter_tables_for_schema_inference(
        self,
        split: Optional[Any] = None,
    ) -> Iterable[pa.Table]:
        return self.iter_tables(split)

    def set_schema(self, schema: pa.Schema) -> None:
        raise NotImplementedError

    def input_size_bytes(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class SpatialSample:
    """Source metadata prepared during the initial spatial scan."""

    sample_points: np.ndarray
    total_seen: int
    total_sampled: int
    batches_read: int
    schema: Optional[pa.Schema] = None
    gpx_field_mask: int = 0


@dataclass(frozen=True)
class TarFileSplit:
    path: str
    offset: int
    length: int


@dataclass(frozen=True)
class TarMember:
    name: str
    data: bytes


@dataclass(frozen=True)
class BZ2SplitPayload:
    payload: bytes
    owned_output_len: int
    previous_output_ended_with_newline: bool
    file_size: int


@dataclass(frozen=True)
class BZ2DecompressedBlock:
    data: bytes
    compressed_start: int
    owned: bool
    previous_output_ended_with_newline: bool
    file_size: int


@dataclass(frozen=True)
class _TarHeader:
    name: str
    size: int
    typeflag: str

    @property
    def data_size_padded(self) -> int:
        return int(math.ceil(self.size / _TAR_BLOCK_SIZE) * _TAR_BLOCK_SIZE)

    @property
    def record_size(self) -> int:
        return _TAR_BLOCK_SIZE + self.data_size_padded


# ------------------------- Helpers ------------------------- #
def _source_files(path: str, suffixes: Tuple[str, ...]) -> List[Path]:
    source_path = Path(path)
    if source_path.is_file():
        return [source_path] if str(source_path).lower().endswith(suffixes) else []
    if source_path.is_dir():
        return sorted(
            file_path
            for file_path in source_path.rglob("*")
            if file_path.is_file() and str(file_path).lower().endswith(suffixes)
        )
    raise FileNotFoundError(f"Source path does not exist: {path}")


def _iter_discoverable_files(path: str) -> Iterable[str]:
    # Entry point for lightweight source detection: yield discoverable file
    # names from a single file input or by walking a directory tree lazily.
    source_path = Path(path)
    if source_path.is_file():
        yield from _iter_discoverable_file_path(source_path, relative_name=source_path.name)
        return
    if source_path.is_dir():
        yield from _iter_discoverable_dir(source_path, root=source_path)
        return
    raise FileNotFoundError(f"Source path does not exist: {path}")


def _iter_discoverable_dir(directory: Path, *, root: Path) -> Iterable[str]:
    # Walk one directory level at a time and hand each real file off to the
    # file-level helper while preserving root-relative names for detection.
    with os.scandir(directory) as entries:
        ordered_entries = sorted(entries, key=lambda entry: (entry.name.lower(), entry.name))
    for entry in ordered_entries:
        entry_path = Path(entry.path)
        if entry.is_dir():
            yield from _iter_discoverable_dir(entry_path, root=root)
            continue
        if entry.is_file():
            relative_name = entry_path.relative_to(root).as_posix()
            yield from _iter_discoverable_file_path(entry_path, relative_name=relative_name)


def _iter_discoverable_file_path(path: Path, *, relative_name: str) -> Iterable[str]:
    # Expand archive files into their member names; otherwise just yield the
    # file name itself as a detection candidate.
    lower_name = path.name.lower()
    if lower_name.endswith(_TAR_SUFFIXES):
        yield from _iter_tar_member_names(path)
        return
    if lower_name.endswith(_ZIP_SUFFIXES):
        yield from _iter_zip_member_names(path)
        return
    yield relative_name


def _iter_zip_member_names(path: str | Path) -> Iterable[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    yield info.filename
    except zipfile.BadZipFile:
        return


def _source_tar_files(path: str, suffixes: Tuple[str, ...]) -> List[Path]:
    source_path = Path(path)
    if source_path.is_file():
        if source_path.suffix.lower() in _TAR_SUFFIXES and _tar_first_member_matches_suffixes(source_path, suffixes):
            return [source_path]
        return []
    if source_path.is_dir():
        return sorted(
            tar_path
            for tar_path in source_path.rglob("*")
            if tar_path.is_file()
            and tar_path.suffix.lower() in _TAR_SUFFIXES
            and _tar_first_member_matches_suffixes(tar_path, suffixes)
        )
    raise FileNotFoundError(f"Source path does not exist: {path}")


def _tar_first_member_matches_suffixes(path: str | Path, suffixes: Tuple[str, ...]) -> bool:
    member = _tar_first_member_name(path)
    return bool(member and member.lower().endswith(suffixes))


def _tar_first_member_name(path: str | Path) -> str | None:
    for member_name in _iter_tar_member_names(path):
        return member_name
    return None


def _iter_tar_member_names(path: str | Path) -> Iterable[str]:
    with open(path, "rb") as stream:
        file_size = Path(path).stat().st_size
        offset = 0
        while offset + _TAR_BLOCK_SIZE <= file_size:
            stream.seek(offset)
            block = stream.read(_TAR_BLOCK_SIZE)
            if len(block) < _TAR_BLOCK_SIZE or not any(block):
                break
            header = _parse_tar_header(block)
            if header is None:
                break
            if header.typeflag in {"", "0"}:
                yield header.name
            offset += header.record_size


def _tar_splits(path: str) -> List[TarFileSplit]:
    file_size = Path(path).stat().st_size
    return [
        TarFileSplit(path=path, offset=offset, length=min(_TAR_SPLIT_SIZE, file_size - offset))
        for offset in range(0, file_size, _TAR_SPLIT_SIZE)
    ]


def _iter_tar_members_for_split(
    path: str,
    *,
    offset: int,
    length: int,
    suffixes: Tuple[str, ...],
) -> Iterable[TarMember]:
    file_size = Path(path).stat().st_size
    split_end = min(offset + length, file_size)
    position = offset
    with open(path, "rb") as stream:
        while position + _TAR_BLOCK_SIZE <= file_size:
            stream.seek(position)
            block = stream.read(_TAR_BLOCK_SIZE)
            if len(block) < _TAR_BLOCK_SIZE or not any(block):
                break
            header = _parse_tar_header(block)
            if header is None:
                position += _TAR_BLOCK_SIZE
                continue
            if position >= split_end:
                break
            data_offset = position + _TAR_BLOCK_SIZE
            if header.typeflag in {"", "0"} and header.name.lower().endswith(suffixes):
                stream.seek(data_offset)
                yield TarMember(name=header.name, data=stream.read(header.size))
            position += header.record_size


def _parse_tar_header(block: bytes) -> _TarHeader | None:
    if len(block) != _TAR_BLOCK_SIZE or not any(block):
        return None
    stored_checksum = _parse_tar_octal(block[148:156])
    size = _parse_tar_octal(block[124:136])
    if stored_checksum is None or size is None:
        return None

    checksum_block = bytearray(block)
    checksum_block[148:156] = b" " * 8
    if sum(checksum_block) != stored_checksum:
        return None

    name = block[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
    if prefix:
        name = f"{prefix}/{name}" if name else prefix
    if not name:
        return None
    typeflag = block[156:157].decode("ascii", "ignore")
    return _TarHeader(name=name, size=size, typeflag=typeflag)


def _parse_tar_octal(raw: bytes) -> int | None:
    text = raw.rstrip(b"\0 ").lstrip(b" ")
    if not text:
        return 0
    try:
        return int(text, 8)
    except ValueError:
        return None


def _read_bz2_split_payload(
    path: str,
    *,
    offset: int,
    length: int,
    require_previous_output_ended_with_newline: bool = False,
    require_stream_state: bool = False,
) -> BZ2SplitPayload | None:
    payload = bytearray()
    owned_output_len = 0
    previous_output_ended_with_newline = offset == 0
    file_size = os.path.getsize(path)
    for block in _iter_bz2_decompressed_blocks(
        path,
        offset=offset,
        length=length,
        require_previous_output_ended_with_newline=require_previous_output_ended_with_newline,
        require_stream_state=require_stream_state,
    ):
        if not payload:
            previous_output_ended_with_newline = block.previous_output_ended_with_newline
        payload.extend(block.data)
        if block.owned:
            owned_output_len = len(payload)
    if not payload:
        return None
    return BZ2SplitPayload(
        payload=bytes(payload),
        owned_output_len=owned_output_len,
        previous_output_ended_with_newline=previous_output_ended_with_newline,
        file_size=file_size,
    )


def _iter_bz2_decompressed_blocks(
    path: str,
    *,
    offset: int,
    length: int,
    require_previous_output_ended_with_newline: bool = False,
    require_stream_state: bool = False,
) -> Iterable[BZ2DecompressedBlock]:
    file_size = os.path.getsize(path)
    split_end = min(offset + length, file_size)
    if length <= 0 or split_end <= offset:
        return

    with open(path, "rb") as stream:
        first_owned = _find_next_bz2_block_start(
            stream,
            max(offset, _BZ2_STREAM_HEADER_LEN),
            file_size,
        )
        if first_owned is None or first_owned >= split_end:
            return
        if require_stream_state or (require_previous_output_ended_with_newline and offset > 0):
            yield from _iter_bz2_decompressed_blocks_from_stream_start(
                path,
                offset=offset,
                length=length,
                first_owned=first_owned,
                file_size=file_size,
            )
            return

        stream.seek(0)
        header = stream.read(_BZ2_STREAM_HEADER_LEN)
        decompressor = bz2.BZ2Decompressor()
        decompressor.decompress(header)

        segment_start = first_owned
        while segment_start < file_size:
            segment_end = _find_next_bz2_block_start(
                stream,
                segment_start + len(_BZ2_BLOCK_MAGIC),
                file_size,
            )
            if segment_end is None:
                segment_end = file_size

            stream.seek(segment_start)
            chunk = stream.read(segment_end - segment_start)
            try:
                decoded = _decompress_bz2_segment(
                    stream,
                    decompressor,
                    segment_start,
                    chunk,
                )
            except OSError:
                yield from _iter_bz2_decompressed_blocks_from_stream_start(
                    path,
                    offset=offset,
                    length=length,
                    first_owned=first_owned,
                    file_size=file_size,
                )
                return
            if decoded.decompressor is not decompressor:
                decompressor = decoded.decompressor

            owned = segment_start < split_end
            if decoded.output:
                yield BZ2DecompressedBlock(
                    data=decoded.output,
                    compressed_start=segment_start,
                    owned=owned,
                    previous_output_ended_with_newline=offset == 0,
                    file_size=file_size,
                )
            if not owned:
                break

            if segment_end >= file_size:
                break
            segment_start = segment_end


def _iter_bz2_decompressed_blocks_from_stream_start(
    path: str,
    *,
    offset: int,
    length: int,
    first_owned: int,
    file_size: int,
) -> Iterable[BZ2DecompressedBlock]:
    split_end = min(offset + length, file_size)
    with open(path, "rb") as stream:
        stream.seek(0)
        header = stream.read(_BZ2_STREAM_HEADER_LEN)
        decompressor = bz2.BZ2Decompressor()
        header_output = decompressor.decompress(header)

        last_discarded_byte = header_output[-1:] if header_output else b"\n"
        yielded = False
        segment_start = _BZ2_STREAM_HEADER_LEN
        while segment_start < file_size:
            segment_end = _find_next_bz2_block_start(
                stream,
                segment_start + len(_BZ2_BLOCK_MAGIC),
                file_size,
            )
            if segment_end is None:
                segment_end = file_size

            stream.seek(segment_start)
            chunk = stream.read(segment_end - segment_start)
            decoded = _decompress_bz2_segment(
                stream,
                decompressor,
                segment_start,
                chunk,
            )
            if decoded.decompressor is not decompressor:
                decompressor = decoded.decompressor

            if segment_start >= first_owned:
                owned = segment_start < split_end
                if decoded.output:
                    yield BZ2DecompressedBlock(
                        data=decoded.output,
                        compressed_start=segment_start,
                        owned=owned,
                        previous_output_ended_with_newline=(
                            last_discarded_byte == b"\n" if not yielded else False
                        ),
                        file_size=file_size,
                    )
                    yielded = True
                if not owned:
                    break
            elif decoded.output:
                last_discarded_byte = decoded.output[-1:]

            if segment_end >= file_size:
                break
            segment_start = segment_end


@dataclass(frozen=True)
class _BZ2DecodedSegment:
    output: bytes
    decompressor: bz2.BZ2Decompressor


def _decompress_bz2_segment(
    stream,
    decompressor: bz2.BZ2Decompressor,
    segment_start: int,
    chunk: bytes,
) -> _BZ2DecodedSegment:
    try:
        return _BZ2DecodedSegment(
            output=decompressor.decompress(chunk),
            decompressor=decompressor,
        )
    except EOFError:
        if not _is_bz2_stream_block_start(stream, segment_start):
            raise
        stream.seek(segment_start - _BZ2_STREAM_HEADER_LEN)
        header = stream.read(_BZ2_STREAM_HEADER_LEN)
        restarted = bz2.BZ2Decompressor()
        restarted.decompress(header)
        return _BZ2DecodedSegment(
            output=restarted.decompress(chunk),
            decompressor=restarted,
        )


def _find_next_bz2_block_start(stream, offset: int, file_size: int) -> int | None:
    if file_size < _BZ2_STREAM_HEADER_LEN:
        return None

    position = max(_BZ2_STREAM_HEADER_LEN, offset)
    carry = b""
    while position < file_size:
        stream.seek(position)
        chunk = stream.read(min(_BZ2_SCAN_CHUNK_SIZE, file_size - position))
        if not chunk:
            return None
        haystack = carry + chunk
        found = haystack.find(_BZ2_BLOCK_MAGIC)
        if found != -1:
            return position - len(carry) + found
        carry = haystack[-(len(_BZ2_BLOCK_MAGIC) - 1):]
        position += len(chunk)
    return None


def _is_bz2_stream_block_start(stream, block_start: int) -> bool:
    if block_start < _BZ2_STREAM_HEADER_LEN:
        return False
    current = stream.tell()
    stream.seek(block_start - _BZ2_STREAM_HEADER_LEN)
    is_stream_start = stream.read(3) == b"BZh"
    stream.seek(current)
    return is_stream_start


def _source_kind(path: str) -> str:
    for discovered_name in _iter_discoverable_files(path):
        detected_type = _detect_source_type_from_name(discovered_name)
        if detected_type is not None:
            return detected_type
    raise ValueError(f"No supported geospatial files found in {path}")


def _detect_source_type_from_name(name: str) -> str | None:
    lower_name = name.lower()
    if lower_name.endswith(_GEOJSON_SUFFIXES):
        return "geojson"
    if lower_name.endswith(_GEOPARQUET_SUFFIXES):
        return "geoparquet"
    if lower_name.endswith(_CSV_SUFFIXES):
        return "csv"
    if lower_name.endswith(_PLT_SUFFIXES):
        return "plt"
    if lower_name.endswith(_GPX_SUFFIXES):
        return "gpx"
    if lower_name.endswith(_SHAPEFILE_SUFFIXES):
        return "shapefile"
    path = Path(name)
    if path.name.lower() == "gdb" or any(
        part.lower().endswith(_GDB_SUFFIXES) for part in path.parts
    ):
        return "gdb"
    return None


def source_for_path(
    path: str,
    *,
    geom_col: str = "geometry",
    csv_x_col: str | None = None,
    csv_y_col: str | None = None,
    csv_wkt_col: str | None = None,
    csv_x_index: int | None = None,
    csv_y_index: int | None = None,
    csv_wkt_index: int | None = None,
    csv_split_size: int = 32 * 1024 * 1024,
    csv_batch_rows: int | None = None,
    src_crs: str = "EPSG:4326",
    **geojson_kwargs,
) -> DataSource:
    """Create the appropriate source reader for a supported geospatial path."""
    kind = _source_kind(path)
    if kind == "geojson":
        return GeoJSONSource(path, **geojson_kwargs)
    if kind == "geoparquet":
        return GeoParquetSource(path, geom_col=geom_col)
    if kind == "csv":
        return CSVSource(
            path,
            x_col=csv_x_index if csv_x_index is not None else csv_x_col,
            y_col=csv_y_index if csv_y_index is not None else csv_y_col,
            wkt_col=csv_wkt_index if csv_wkt_index is not None else csv_wkt_col,
            split_size=csv_split_size,
            batch_rows=csv_batch_rows,
            src_crs=src_crs,
            geom_col=geom_col,
        )
    if kind == "plt":
        return PLTSource(path, geom_col=geom_col)
    if kind == "gpx":
        return GPXSource(path, geom_col=geom_col)
    if kind == "shapefile":
        return ShapefileSource(path, geom_col=geom_col)
    if kind == "gdb":
        return GDBSource(path, geom_col=geom_col)
    raise ValueError(f"Unsupported source: {path}")


def read_spatial_sample(
    path: str,
    *,
    geom_col: str = "geometry",
    csv_x_col: str | None = None,
    csv_y_col: str | None = None,
    csv_wkt_col: str | None = None,
    csv_x_index: int | None = None,
    csv_y_index: int | None = None,
    csv_wkt_index: int | None = None,
    csv_split_size: int = 32 * 1024 * 1024,
    csv_batch_rows: int | None = None,
    src_crs: str = "EPSG:4326",
    sample_cap: Optional[int] = None,
    seed: int = 42,
    geojson_workers: Optional[int] = None,
    geoparquet_workers: Optional[int] = None,
    source_workers: Optional[int] = None,
) -> SpatialSample:
    """Read a source once and return its spatial sample and inferred schema."""
    kind = _source_kind(path)
    if kind == "geojson":
        return GeoJSONSource.read_spatial_sample(
            path,
            sample_cap=sample_cap,
            seed=seed,
            workers=geojson_workers,
            src_crs=src_crs,
        )
    if kind == "geoparquet":
        return GeoParquetSource.read_spatial_sample(
            path,
            geom_col=geom_col,
            sample_cap=sample_cap,
            seed=seed,
            workers=geoparquet_workers,
        )
    if kind == "gpx":
        return GPXSource.read_spatial_sample(
            path,
            geom_col=geom_col,
            sample_cap=sample_cap,
            seed=seed,
            workers=source_workers,
        )
    if kind == "csv" and csv_wkt_col is None and csv_wkt_index is None:
        return CSVSource.read_spatial_sample(
            path,
            x_col=csv_x_index if csv_x_index is not None else csv_x_col,
            y_col=csv_y_index if csv_y_index is not None else csv_y_col,
            split_size=csv_split_size,
            batch_rows=csv_batch_rows,
            src_crs=src_crs,
            geom_col=geom_col,
            sample_cap=sample_cap,
            seed=seed,
            workers=source_workers,
        )

    source = source_for_path(
        path,
        geom_col=geom_col,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_x_index=csv_x_index,
        csv_y_index=csv_y_index,
        csv_wkt_index=csv_wkt_index,
        csv_split_size=csv_split_size,
        csv_batch_rows=csv_batch_rows,
        src_crs=src_crs,
    )
    collect_schema = isinstance(source, (CSVSource, ShapefileSource))
    if isinstance(source, GDBSource):
        source = GDBSource(path, geometry_only=True, geom_col=geom_col)
    elif isinstance(source, PLTSource):
        source = PLTSource(path, geometry_only=True, geom_col=geom_col)
    return _read_datasource_spatial_sample(
        source,
        geom_col=geom_col,
        sample_cap=sample_cap,
        seed=seed,
        source_workers=source_workers,
        collect_schema=collect_schema,
    )


def _reservoir_add(
    *,
    rng: np.random.Generator,
    sample_cap: Optional[int],
    x_sample: List[float],
    y_sample: List[float],
    n_seen: int,
    x: float,
    y: float,
) -> None:
    if sample_cap is None:
        x_sample.append(x)
        y_sample.append(y)
        return

    if sample_cap <= 0:
        return

    if n_seen <= sample_cap:
        if len(x_sample) < sample_cap:
            x_sample.append(x)
            y_sample.append(y)
        else:
            j = rng.integers(0, n_seen)
            if j < sample_cap:
                x_sample[j] = x
                y_sample[j] = y
    else:
        j = rng.integers(0, n_seen)
        if j < sample_cap:
            x_sample[j] = x
            y_sample[j] = y


def _combine_spatial_samples(parts: List[SpatialSample]) -> SpatialSample:
    logger.info("Finished the partitions ... merging")
    non_empty = [part for part in parts if part.total_seen > 0]
    if not non_empty:
        raise ValueError(
            "No geometries sampled to build RSGrove index. "
            "Provide a positive --sample-cap."
        )

    sampled = [part.sample_points for part in non_empty if part.sample_points.shape[1] > 0]
    if not sampled:
        raise ValueError(
            "No geometries sampled to build RSGrove index. "
            "Provide a positive --sample-cap."
        )

    sample_points = np.concatenate(sampled, axis=1)
    logger.info("Finished the merge")
    return SpatialSample(
        sample_points=sample_points,
        total_seen=sum(part.total_seen for part in parts),
        total_sampled=sample_points.shape[1],
        batches_read=sum(part.batches_read for part in parts),
    )


def _split_sample_cap(sample_cap: Optional[int], num_parts: int) -> List[Optional[int]]:
    if sample_cap is None:
        return [None] * num_parts

    total = max(0, int(sample_cap))
    base, remainder = divmod(total, max(1, num_parts))
    return [base + (1 if i < remainder else 0) for i in range(num_parts)]


def _spatial_sample_from_state(
    *,
    x_sample: List[float],
    y_sample: List[float],
    n_seen: int,
    batches_read: int,
    schema: Optional[pa.Schema] = None,
    gpx_field_mask: int = 0,
) -> SpatialSample:
    return SpatialSample(
        sample_points=(
            np.stack(
                [np.asarray(x_sample, dtype=np.float64), np.asarray(y_sample, dtype=np.float64)],
                axis=0,
            )
            if x_sample
            else np.empty((2, 0), dtype=np.float64)
        ),
        total_seen=n_seen,
        total_sampled=len(x_sample),
        batches_read=batches_read,
        schema=schema,
        gpx_field_mask=gpx_field_mask,
    )


_WKB_GEOMETRY_TYPES = {
    1: "Point",
    2: "LineString",
    3: "Polygon",
    4: "MultiPoint",
    5: "MultiLineString",
    6: "MultiPolygon",
    7: "GeometryCollection",
    8: "CircularString",
    9: "CompoundCurve",
    10: "CurvePolygon",
    11: "MultiCurve",
    12: "MultiSurface",
    13: "Curve",
    14: "Surface",
    15: "PolyhedralSurface",
    16: "TIN",
    17: "Triangle",
}


def _decode_wkb_geometries(values: Any, *, geom_col: str, context: str) -> Any:
    try:
        return from_wkb(values)
    except Exception:
        logger.exception(
            "Failed decoding WKB geometries (%s, geom_col=%s, wkb_geometry_types=%s)",
            context,
            geom_col,
            _wkb_geometry_type_summary(values),
        )
        raise


def _split_context(source_path: Any, split: Any) -> str:
    parts = [f"path={getattr(split, 'path', source_path)}"]
    context_attrs = (
        "layer",
        "geometry_type",
        "row_groups",
        "offset",
        "length",
        "skip_features",
        "max_features",
    )
    for attr in context_attrs:
        if hasattr(split, attr):
            parts.append(f"{attr}={getattr(split, attr)!r}")
    return " ".join(parts)


def _wkb_geometry_type_summary(values: Any, limit: int = 64) -> str:
    counts: Dict[str, int] = {}
    inspected = 0
    unavailable = 0
    for value in values:
        if inspected >= limit:
            break
        inspected += 1
        geometry_type = _wkb_geometry_type(value)
        if geometry_type is None:
            unavailable += 1
            continue
        counts[geometry_type] = counts.get(geometry_type, 0) + 1

    pieces = [f"{name}={count}" for name, count in sorted(counts.items())]
    if unavailable:
        pieces.append(f"unavailable={unavailable}")
    return ", ".join(pieces) or "<none>"


def _wkb_geometry_type(value: Any) -> str | None:
    if value is None:
        return None
    try:
        data = bytes(value)
    except Exception:
        return None
    if len(data) < 5:
        return None

    byte_order = data[0]
    if byte_order == 0:
        raw_type = int.from_bytes(data[1:5], "big")
    elif byte_order == 1:
        raw_type = int.from_bytes(data[1:5], "little")
    else:
        return None

    has_z = bool(raw_type & 0x80000000)
    has_m = bool(raw_type & 0x40000000)
    base_type = raw_type & 0x0000FFFF
    if base_type >= 3000:
        has_z = True
        has_m = True
        base_type -= 3000
    elif base_type >= 2000:
        has_m = True
        base_type -= 2000
    elif base_type >= 1000:
        has_z = True
        base_type -= 1000

    name = _WKB_GEOMETRY_TYPES.get(base_type, f"type_code_{base_type}")
    suffix = ("Z" if has_z else "") + ("M" if has_m else "")
    return f"{name}{suffix}" if suffix else name


def _read_datasource_spatial_sample(
    source: DataSource,
    *,
    geom_col: str,
    sample_cap: Optional[int],
    seed: int,
    source_workers: Optional[int],
    collect_schema: bool = False,
) -> SpatialSample:
    splits = source.create_splits()
    sample_caps = _split_sample_cap(sample_cap, len(splits))

    logger.info(
        "Reading spatial sample from %s in %d source partitions with %s process workers",
        getattr(source, "path", "<source>"),
        len(splits),
        source_workers or "auto",
    )

    with create_process_executor(
        max_workers=source_workers,
        logger=logger,
        context="datasource spatial sampling",
    ) as executor:
        futures = {
            executor.submit(
                _read_datasource_split_spatial_sample,
                source,
                split,
                geom_col,
                sample_caps[index],
                seed + index,
                collect_schema,
            ): index
            for index, split in enumerate(splits)
        }
        parts: List[SpatialSample] = []
        total_splits = len(futures)
        for completed_index, future in enumerate(as_completed(futures), start=1):
            parts.append(future.result())
            split_index = futures[future]
            print(
                f"sample split done: {completed_index}/{total_splits} (split {split_index})",
                flush=True,
            )
        sample = _combine_spatial_samples(parts)
        if not collect_schema:
            return sample
        schema = _unify_tabular_schemas(
            part.schema for part in parts if part.schema is not None
        )
        return replace(sample, schema=schema)


def _read_datasource_split_spatial_sample(
    source: DataSource,
    split: Any,
    geom_col: str,
    sample_cap: Optional[int],
    seed: int,
    collect_schema: bool = False,
) -> SpatialSample:
    rng = np.random.default_rng(seed)
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    n_batches = 0
    schemas: List[pa.Schema] = []

    tables = (
        source.iter_tables_for_schema_inference(split)
        if collect_schema
        else source.iter_tables(split)
    )
    for table in tables:
        table = table.combine_chunks()
        if table.num_rows == 0:
            continue
        n_batches += 1
        if collect_schema:
            schemas.append(table.schema)
        table = ensure_large_types(table, geom_col)
        geometries = _decode_wkb_geometries(
            table[geom_col].to_numpy(zero_copy_only=False),
            geom_col=geom_col,
            context=_split_context(getattr(source, "path", "<source>"), split),
        )

        for geom in geometries:
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid
            n_seen += 1
            _reservoir_add(
                rng=rng,
                sample_cap=sample_cap,
                x_sample=x_sample,
                y_sample=y_sample,
                n_seen=n_seen,
                x=float(centroid.x),
                y=float(centroid.y),
            )

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        n_seen=n_seen,
        batches_read=n_batches,
        schema=_unify_tabular_schemas(schemas) if schemas else None,
    )


def _attach_geoparquet_metadata(
    schema: pa.Schema,
    crs_hint: Optional[str],
    *,
    geom_col: str = "geometry",
) -> pa.Schema:
    """
    Return a copy of `schema` with a minimal GeoParquet 'geo' JSON block so
    downstream writers (WriterPool) can inject tile bbox.

    Includes:
      - version: 1.1.0
      - primary_column: the selected geometry column
      - columns.<geometry>.encoding: WKB
      - columns.<geometry>.crs: <crs_hint> (string hint if provided)
    """
    md = dict(schema.metadata or {})
    if b"geo" in md:
        try:
            geo = json.loads(md[b"geo"].decode("utf-8"))
        except Exception:
            geo = {}
    else:
        geo = {}

    geo.setdefault("version", "1.1.0")
    geo.setdefault("primary_column", geom_col)
    columns = geo.setdefault("columns", {})
    geometry_meta = columns.setdefault(geom_col, {})
    geometry_meta.setdefault("encoding", "WKB")
    if crs_hint:
        try:
            geometry_meta["crs"] = crs_hint
        except Exception:
            pass

    md[b"geo"] = json.dumps(geo, separators=(",", ":")).encode("utf-8")
    return pa.schema(schema, metadata=md)


def _normalize_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize object columns that commonly infer unstable Arrow types across
    GeoJSON batches.

    Decimal values become float64 so Arrow does not infer different
    decimal128 precision/scale. Nested JSON-like values are left intact; callers
    that build Arrow tables should use ``_properties_dataframe_to_arrow_table``
    so dynamic tag maps get a stable Arrow map type.
    """
    if df.empty:
        return df

    df = df.copy()

    def is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (dict, list)):
            return False
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    for col in df.columns:
        s = df[col]

        # Only object dtype can hold Decimal values in pandas here.
        if s.dtype != "object":
            continue

        sample = None
        for v in s:
            if not is_missing(v):
                sample = v
                break

        if isinstance(sample, Decimal):
            df[col] = s.map(lambda x: None if is_missing(x) else float(x))

    return df


def _properties_dataframe_to_arrow_table(
    df: pd.DataFrame,
    schema: pa.Schema | None = None,
) -> pa.Table:
    """Build an Arrow table for GeoJSON properties without stringifying maps."""
    if df.empty and schema is None:
        return pa.table({})

    columns = []
    fields = []

    column_fields = schema or pa.schema(
        pa.field(str(name), pa.null()) for name in df.columns
    )
    for field in column_fields:
        name = field.name
        values = (
            df[name].tolist()
            if name in df.columns
            else [None] * len(df.index)
        )
        if schema is not None:
            if pa.types.is_map(field.type):
                array = pa.array(
                    [_map_entries_or_none(value) for value in values],
                    type=field.type,
                )
            elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                array = pa.array(
                    [_stringify_json_property_value(value) for value in values],
                    type=field.type,
                )
            else:
                array = pa.array(values, type=field.type)
            columns.append(array)
            fields.append(field)
            continue

        arrow_type = _geojson_property_arrow_type(values)
        try:
            if arrow_type is None:
                array = pa.array(values)
            else:
                array = pa.array(
                    [_map_entries_or_none(value) for value in values],
                    type=arrow_type,
                )
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            array = pa.array(
                [_stringify_json_property_value(value) for value in values],
                type=pa.large_string(),
            )
        columns.append(array)
        fields.append(pa.field(str(name), array.type))

    return pa.table(columns, schema=pa.schema(fields))


def _unify_tabular_schemas(schemas: Iterable[pa.Schema]) -> pa.Schema:
    """Resolve independently inferred table schemas into one source schema."""
    schemas = list(schemas)
    fields_by_name: Dict[str, List[pa.Field]] = {}
    field_order: List[str] = []
    for schema in schemas:
        for field in schema:
            if field.name not in fields_by_name:
                fields_by_name[field.name] = []
                field_order.append(field.name)
            fields_by_name[field.name].append(field)

    fields = []
    for name in field_order:
        source_fields = fields_by_name[name]
        source_types = [field.type for field in source_fields]
        non_null_types = [field_type for field_type in source_types if not pa.types.is_null(field_type)]

        if not non_null_types:
            field_type = pa.null()
        elif all(
            pa.types.is_integer(field_type) or pa.types.is_floating(field_type)
            for field_type in non_null_types
        ):
            field_type = pa.unify_schemas(
                [pa.schema([pa.field(name, source_type)]) for source_type in non_null_types],
                promote_options="permissive",
            ).field(name).type
        elif any(
            pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
            for field_type in non_null_types
        ) and len(set(non_null_types)) > 1:
            field_type = pa.large_string()
        else:
            try:
                field_type = pa.unify_schemas(
                    [pa.schema([pa.field(name, source_type)]) for source_type in non_null_types],
                    promote_options="permissive",
                ).field(name).type
            except pa.ArrowTypeError:
                field_type = pa.large_string()

        metadata = next((field.metadata for field in source_fields if field.metadata), None)
        fields.append(pa.field(name, field_type, nullable=True, metadata=metadata))

    schema_metadata = next(
        (schema.metadata for schema in schemas if schema.metadata),
        None,
    )
    return pa.schema(fields, metadata=schema_metadata)


def _geojson_property_arrow_type(values: list[Any]) -> pa.DataType | None:
    sample = next((value for value in values if not _is_missing_property_value(value)), None)
    if not isinstance(sample, dict):
        return None
    if all(
        _is_missing_property_value(value) or _is_string_map(value)
        for value in values
    ):
        return pa.map_(pa.string(), pa.string())
    return None


def _is_missing_property_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_string_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and (item is None or isinstance(item, str))
        for key, item in value.items()
    )


def _map_entries_or_none(value: Any) -> list[tuple[str, str | None]] | None:
    if _is_missing_property_value(value):
        return None
    return list(value.items())


def _stringify_json_property_value(value: Any) -> str | None:
    if _is_missing_property_value(value):
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)

# Backward-compatible re-exports for callers importing concrete sources from
# this module.
from starlet._internal.tiling.geojson_source import GeoJSONSource, GeoJSONSplit
from starlet._internal.tiling.geoparquet_source import GeoParquetSource, GeoParquetSplit
from starlet._internal.tiling.csv_source import CSVSource, CSVSplit
from starlet._internal.tiling.plt_source import PLTSource, PLTSplit
from starlet._internal.tiling.gpx_source import GPXSource, GPXSplit
from starlet._internal.tiling.vector_source import GDBSource, ShapefileSource, VectorLayerSplit

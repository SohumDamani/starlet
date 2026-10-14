from __future__ import annotations

from concurrent.futures import as_completed
import csv
from dataclasses import dataclass
import io
import logging
import math
import re
import random
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from shapely import from_wkt, points, to_wkb

from starlet._internal.executor import create_process_executor
from starlet._internal.tiling.RSGrove import EnvelopeNDLite
from starlet._internal.tiling.datasource import (
    DataSource,
    _CSV_SUFFIXES,
    _attach_geoparquet_metadata,
    _combine_spatial_samples,
    _iter_bz2_decompressed_blocks,
    _read_bz2_split_payload,
    _spatial_sample_from_state,
    _split_sample_cap,
    _source_files,
    _unify_tabular_schemas,
    SpatialSample,
)

logger = logging.getLogger(__name__)
_HEADERLESS_COLUMN_PREFIX = "column_"


@dataclass(frozen=True)
class CSVSplit:
    path: str
    offset: int
    length: int


class CSVSource(DataSource):
    def __init__(
        self,
        path: str,
        *,
        x_col: str | int | None = None,
        y_col: str | int | None = None,
        wkt_col: str | int | None = None,
        split_size: int = 32 * 1024 * 1024,
        batch_rows: int | None = None,
        src_crs: str = "EPSG:4326",
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ) -> None:
        if bool(wkt_col) == bool(x_col and y_col):
            raise ValueError("CSV input requires either wkt_col or both x_col and y_col")
        if (x_col is None) != (y_col is None):
            raise ValueError("CSV x/y geometry requires both x_col and y_col")

        self.path = str(path)
        self.x_col = x_col
        self.y_col = y_col
        self.wkt_col = wkt_col
        self.split_size = int(batch_rows if batch_rows is not None else split_size)
        self.src_crs = src_crs
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self.has_header = _csv_uses_header(x_col=x_col, y_col=y_col, wkt_col=wkt_col)
        self._files = _source_files(self.path, _CSV_SUFFIXES)
        if not self._files:
            raise ValueError(f"No CSV files found in {self.path}")
        self._schema: pa.Schema | None = None

    def schema(self) -> pa.Schema:
        if self._schema is None:
            schemas = [
                table.schema
                for table in self.iter_tables_for_schema_inference()
            ]
            self._schema = (
                _unify_tabular_schemas(schemas)
                if schemas
                else _attach_geoparquet_metadata(
                    pa.schema([(self.geom_col, pa.binary())]),
                    self.src_crs,
                )
            )
        return self._schema

    def set_schema(self, schema: pa.Schema) -> None:
        if self.geom_col not in schema.names:
            raise ValueError(f"CSV schema must contain geometry column {self.geom_col!r}")
        self._schema = schema

    def input_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._files)

    def create_splits(self, num_splits: Optional[int] = None) -> List[CSVSplit]:
        if num_splits is None:
            target_split_size = max(1, self.split_size)
        else:
            total_bytes = max(1, self.input_size_bytes())
            target_split_size = max(1, (total_bytes + max(1, int(num_splits)) - 1) // max(1, int(num_splits)))

        splits: List[CSVSplit] = []
        for path in self._files:
            file_size = path.stat().st_size
            if file_size <= 0:
                continue
            for offset in range(0, file_size, target_split_size):
                splits.append(
                    CSVSplit(
                        path=str(path),
                        offset=offset,
                        length=min(target_split_size, file_size - offset),
                    )
                )
        return splits

    @staticmethod
    def read_spatial_sample(
        path: str,
        *,
        x_col: str | int | None = None,
        y_col: str | int | None = None,
        wkt_col: str | int | None = None,
        split_size: int = 32 * 1024 * 1024,
        batch_rows: int | None = None,
        src_crs: str = "EPSG:4326",
        geom_col: str = "geometry",
        sample_cap: Optional[int] = None,
        seed: int = 42,
        workers: Optional[int] = None,
    ) -> SpatialSample:
        source = CSVSource(
            path,
            x_col=x_col,
            y_col=y_col,
            wkt_col=wkt_col,
            split_size=split_size,
            batch_rows=batch_rows,
            src_crs=src_crs,
            geometry_only=True,
            geom_col=geom_col,
        )
        if source.wkt_col is not None:
            raise ValueError("Fast CSV spatial sampling currently requires x/y columns")

        splits = source.create_splits()
        sample_caps = _split_sample_cap(sample_cap, len(splits))
        logger.info(
            "Reading CSV spatial sample from %s in %d partitions with %s workers",
            path,
            len(splits),
            workers or "auto",
        )

        jobs = [
            (source, split, sample_caps[index], seed + index)
            for index, split in enumerate(splits)
        ]
        if workers == 1:
            parts = [_read_csv_split_spatial_sample(*job) for job in jobs]
        else:
            with create_process_executor(
                max_workers=workers,
                logger=logger,
                context="CSV spatial sampling",
            ) as executor:
                futures = [executor.submit(_read_csv_split_spatial_sample, *job) for job in jobs]
                parts = [future.result() for future in as_completed(futures)]

        sample = _combine_spatial_samples(parts)
        schema = _unify_tabular_schemas(
            part.schema for part in parts if part.schema is not None
        )
        return SpatialSample(
            sample_points=sample.sample_points,
            total_seen=sample.total_seen,
            total_sampled=sample.total_sampled,
            batches_read=sample.batches_read,
            schema=_attach_geoparquet_metadata(schema, src_crs, geom_col=geom_col),
        )

    def iter_tables(self, split: Optional[CSVSplit] = None) -> Iterable[pa.Table]:
        schema = self.schema()
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            for df in self._iter_split_dataframes(source_split):
                if df.empty:
                    continue
                yield self._dataframe_to_table(df, schema=schema)

    def iter_tables_for_schema_inference(
        self,
        split: Optional[CSVSplit] = None,
    ) -> Iterable[pa.Table]:
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            for df in self._iter_split_dataframes(source_split):
                if df.empty:
                    continue
                yield self._dataframe_to_table(df)

    def _read_split(self, split: CSVSplit) -> pd.DataFrame:
        data = _read_csv_split_bytes(split, has_header=self.has_header)
        if data is None:
            return pd.DataFrame()
        return self._read_split_bytes(split, data)

    def _iter_split_dataframes(self, split: CSVSplit) -> Iterable[pd.DataFrame]:
        if split.path.lower().endswith(".bz2"):
            for data in _iter_bz2_csv_split_byte_batches(split, has_header=self.has_header):
                yield self._read_split_bytes(split, data)
            return
        yield self._read_split(split)

    def _read_split_bytes(self, split: CSVSplit, data: bytes) -> pd.DataFrame:
        usecols = self._geometry_columns() if self.geometry_only else None
        read_kwargs = {
            "usecols": usecols,
            "dtype": str,
        }
        if not self.has_header:
            read_kwargs["header"] = None
        if split.path.lower().endswith(".txt"):
            read_kwargs["sep"] = r"\s+"
        df = pd.read_csv(io.BytesIO(data), **read_kwargs)
        if not self.has_header:
            df = df.rename(columns=lambda name: _headerless_column_name(int(name)))
        return df

    def _geometry_columns(self) -> List[str | int]:
        if self.wkt_col:
            return [self.wkt_col]
        return [self.x_col, self.y_col]  # type: ignore[list-item]

    def _dataframe_to_table(
        self,
        df: pd.DataFrame,
        schema: pa.Schema | None = None,
    ) -> pa.Table:
        wkt_column = _csv_column_name(self.wkt_col)
        x_column = _csv_column_name(self.x_col)
        y_column = _csv_column_name(self.y_col)
        if self.wkt_col:
            if wkt_column not in df.columns:
                raise ValueError(f"CSV missing WKT column {self.wkt_col!r}")
            geoms = from_wkt(df[wkt_column].astype("string").to_numpy())
        else:
            if x_column not in df.columns or y_column not in df.columns:
                raise ValueError(f"CSV missing x/y columns {self.x_col!r}, {self.y_col!r}")
            geoms = points(
                pd.to_numeric(df[x_column], errors="coerce").to_numpy(),
                pd.to_numeric(df[y_column], errors="coerce").to_numpy(),
            )

        geometry_col = pa.array(to_wkb(geoms, hex=False).tolist(), type=pa.binary())
        props_df = pd.DataFrame(index=df.index) if self.geometry_only else df.copy()
        properties_schema = (
            pa.schema([field for field in schema if field.name != self.geom_col])
            if schema is not None
            else None
        )
        props_table = _csv_dataframe_to_arrow_table(
            props_df,
            schema=properties_schema,
        )
        table = (
            pa.table([geometry_col], names=[self.geom_col])
            if props_table.num_columns == 0
            else props_table.append_column(self.geom_col, geometry_col)
        )
        schema_with_geo = _attach_geoparquet_metadata(table.schema, self.src_crs)
        table = table.replace_schema_metadata(schema_with_geo.metadata)
        if schema is not None:
            table = table.cast(schema)
        return table.combine_chunks()


_CSV_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_CSV_INTEGER_BYTES_PATTERN = re.compile(rb"^[+-]?\d+$")
_CSV_BOOLEAN_VALUES = {"true": True, "false": False}
_CSV_BOOLEAN_BYTES = {b"true", b"false"}
_CSV_TYPE_NULL = 0
_CSV_TYPE_BOOL = 1
_CSV_TYPE_INT = 2
_CSV_TYPE_FLOAT = 3
_CSV_TYPE_STRING = 4
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _csv_dataframe_to_arrow_table(
    df: pd.DataFrame,
    schema: pa.Schema | None = None,
) -> pa.Table:
    fields = list(schema) if schema is not None else [
        pa.field(str(name), _infer_csv_column_type(df[name].tolist()))
        for name in df.columns
    ]
    arrays = []
    for field in fields:
        values = df[field.name].tolist() if field.name in df.columns else [None] * len(df.index)
        arrays.append(pa.array(
            [_parse_csv_value(value, field.type) for value in values],
            type=field.type,
        ))
    return pa.table(arrays, schema=pa.schema(fields))


def _infer_csv_column_type(values: List[object]) -> pa.DataType:
    strings = [str(value) for value in values if not pd.isna(value)]
    if not strings:
        return pa.null()
    if all(value.lower() in _CSV_BOOLEAN_VALUES for value in strings):
        return pa.bool_()
    if all(_CSV_INTEGER_PATTERN.fullmatch(value) for value in strings):
        if all(_is_csv_integer(value) for value in strings):
            try:
                pa.array([int(value) for value in strings], type=pa.int64())
                return pa.int64()
            except (OverflowError, pa.ArrowInvalid):
                pass
        return pa.string()
    try:
        for value in strings:
            float(value)
        return pa.float64()
    except ValueError:
        return pa.string()


def _is_csv_integer(value: str | bytes) -> bool:
    integer_pattern = (
        _CSV_INTEGER_BYTES_PATTERN
        if isinstance(value, bytes)
        else _CSV_INTEGER_PATTERN
    )
    if not integer_pattern.fullmatch(value):
        return False
    if isinstance(value, bytes):
        digits = value.lstrip(b"+-")
        return len(digits) == 1 or not digits.startswith(b"0")
    digits = value.lstrip("+-")
    return len(digits) == 1 or not digits.startswith("0")


def _parse_csv_value(value: object, arrow_type: pa.DataType):
    if pd.isna(value):
        return None
    text = str(value)
    if pa.types.is_boolean(arrow_type):
        return _CSV_BOOLEAN_VALUES[text.lower()]
    if pa.types.is_integer(arrow_type):
        return int(text)
    if pa.types.is_floating(arrow_type):
        return float(text)
    return text


def _read_csv_split_spatial_sample(
    source: CSVSource,
    split: CSVSplit,
    sample_cap: Optional[int],
    seed: int,
) -> SpatialSample:
    data = _read_csv_split_bytes(split, has_header=source.has_header)
    if data is None:
        return _empty_csv_spatial_sample(source)

    rng = random.Random(seed)
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    requested_cap = None if sample_cap is None else max(0, int(sample_cap))
    reservoir_weight = 0.0
    reservoir_skip = 0

    names: list[str] | None = None
    x_index = y_index = -1
    type_states: list[int] = []

    for row_index, row in enumerate(_iter_csv_rows(data, split.path)):
        if row_index == 0 and source.has_header:
            names = [_csv_text(name) for name in row]
            x_index = _csv_named_column_index(names, source.x_col)
            y_index = _csv_named_column_index(names, source.y_col)
            type_states = [_CSV_TYPE_NULL] * len(names)
            _mark_csv_coordinate_columns(type_states, x_index, y_index)
            continue

        if names is None:
            names = [_headerless_column_name(index) for index in range(len(row))]
            x_index = int(source.x_col)  # type: ignore[arg-type]
            y_index = int(source.y_col)  # type: ignore[arg-type]
            type_states = [_CSV_TYPE_NULL] * len(names)
            _mark_csv_coordinate_columns(type_states, x_index, y_index)
        elif len(row) > len(names):
            names.extend(_headerless_column_name(index) for index in range(len(names), len(row)))
            type_states.extend([_CSV_TYPE_NULL] * (len(row) - len(type_states)))
            _mark_csv_coordinate_columns(type_states, x_index, y_index)

        for index, value in enumerate(row[:len(type_states)]):
            if index == x_index or index == y_index:
                continue
            type_states[index] = _update_csv_type_state(type_states[index], value)

        if x_index >= len(row) or y_index >= len(row):
            continue
        try:
            x = float(row[x_index])
            y = float(row[y_index])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        n_seen += 1

        if requested_cap is None:
            x_sample.append(x)
            y_sample.append(y)
        elif requested_cap > 0:
            if len(x_sample) < requested_cap:
                x_sample.append(x)
                y_sample.append(y)
                if len(x_sample) == requested_cap:
                    reservoir_weight = _csv_initial_reservoir_weight(rng, requested_cap)
                    reservoir_skip = _csv_next_reservoir_skip(rng, reservoir_weight)
            elif reservoir_skip:
                reservoir_skip -= 1
            else:
                slot = rng.randrange(requested_cap)
                x_sample[slot] = x
                y_sample[slot] = y
                reservoir_weight *= _csv_initial_reservoir_weight(rng, requested_cap)
                reservoir_skip = _csv_next_reservoir_skip(rng, reservoir_weight)

    if names is None:
        names = []
    schema = _csv_schema_from_states(names, type_states, source.geom_col)
    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        n_seen=n_seen,
        batches_read=1,
        schema=schema,
    )


def _empty_csv_spatial_sample(source: CSVSource) -> SpatialSample:
    return SpatialSample(
        sample_points=np.empty((2, 0), dtype=np.float64),
        total_seen=0,
        total_sampled=0,
        batches_read=0,
        schema=pa.schema([]),
    )


def _iter_csv_rows(data: bytes, path: str) -> Iterable[list[str | bytes]]:
    if path.lower().endswith(".txt"):
        for line in data.splitlines():
            if line:
                yield line.split()
        return

    if b'"' not in data:
        delimiter = b"\t" if path.lower().endswith((".tsv", ".tsv.bz2")) else b","
        for line in data.splitlines():
            if line:
                yield line.split(delimiter)
        return

    delimiter = "\t" if path.lower().endswith((".tsv", ".tsv.bz2")) else ","
    stream = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", newline="")
    yield from csv.reader(stream, delimiter=delimiter)


def _csv_named_column_index(names: list[str], column: str | int | None) -> int:
    if column is None:
        return -1
    if isinstance(column, int):
        return column
    try:
        return names.index(column)
    except ValueError as exc:
        raise ValueError(f"CSV missing geometry column {column!r}") from exc


def _mark_csv_coordinate_columns(type_states: list[int], x_index: int, y_index: int) -> None:
    if 0 <= x_index < len(type_states):
        type_states[x_index] = _CSV_TYPE_FLOAT
    if 0 <= y_index < len(type_states):
        type_states[y_index] = _CSV_TYPE_FLOAT


def _csv_text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _update_csv_type_state(state: int, value: str | bytes) -> int:
    if value == "" or value == b"":
        return state
    if state == _CSV_TYPE_STRING:
        return state
    value_type = _csv_value_type(value)
    if state == _CSV_TYPE_NULL:
        return value_type
    if state == value_type:
        return state
    if (
        state in (_CSV_TYPE_INT, _CSV_TYPE_FLOAT)
        and value_type in (_CSV_TYPE_INT, _CSV_TYPE_FLOAT)
    ):
        return _CSV_TYPE_FLOAT
    return _CSV_TYPE_STRING


def _csv_value_type(value: str | bytes) -> int:
    lower = value.lower()
    if lower in (_CSV_BOOLEAN_BYTES if isinstance(lower, bytes) else _CSV_BOOLEAN_VALUES):
        return _CSV_TYPE_BOOL
    integer_pattern = (
        _CSV_INTEGER_BYTES_PATTERN
        if isinstance(value, bytes)
        else _CSV_INTEGER_PATTERN
    )
    if integer_pattern.fullmatch(value):
        if _is_csv_integer(value):
            try:
                parsed = int(value)
                if _INT64_MIN <= parsed <= _INT64_MAX:
                    return _CSV_TYPE_INT
            except ValueError:
                pass
        return _CSV_TYPE_STRING
    try:
        float(value)
        return _CSV_TYPE_FLOAT
    except ValueError:
        return _CSV_TYPE_STRING


def _csv_schema_from_states(names: list[str], states: list[int], geom_col: str) -> pa.Schema:
    fields = [
        pa.field(name, _csv_arrow_type_from_state(state), nullable=True)
        for name, state in zip(names, states)
    ]
    fields.append(pa.field(geom_col, pa.binary(), nullable=True))
    return pa.schema(fields)


def _csv_arrow_type_from_state(state: int) -> pa.DataType:
    if state == _CSV_TYPE_BOOL:
        return pa.bool_()
    if state == _CSV_TYPE_INT:
        return pa.int64()
    if state == _CSV_TYPE_FLOAT:
        return pa.float64()
    if state == _CSV_TYPE_STRING:
        return pa.string()
    return pa.null()


def _csv_initial_reservoir_weight(rng: random.Random, capacity: int) -> float:
    random_value = rng.random()
    while random_value == 0.0:
        random_value = rng.random()
    weight = math.exp(math.log(random_value) / capacity)
    return min(weight, math.nextafter(1.0, 0.0))


def _csv_next_reservoir_skip(rng: random.Random, weight: float) -> int:
    random_value = rng.random()
    while random_value == 0.0:
        random_value = rng.random()
    return int(math.log(random_value) / math.log1p(-weight))


def _csv_uses_header(
    *,
    x_col: str | int | None,
    y_col: str | int | None,
    wkt_col: str | int | None,
) -> bool:
    refs = [ref for ref in (x_col, y_col, wkt_col) if ref is not None]
    if not refs:
        return True
    has_named = any(isinstance(ref, str) for ref in refs)
    has_indexed = any(isinstance(ref, int) for ref in refs)
    if has_named and has_indexed:
        raise ValueError("CSV geometry columns must be specified either all by name or all by index")
    return has_named


def _csv_column_name(column: str | int | None) -> str | None:
    if column is None:
        return None
    if isinstance(column, int):
        return _headerless_column_name(column)
    return column


def _headerless_column_name(index: int) -> str:
    return f"{_HEADERLESS_COLUMN_PREFIX}{index}"


def _read_csv_split_bytes(split: CSVSplit, *, has_header: bool) -> bytes | None:
    if split.path.lower().endswith(".bz2"):
        return _read_bz2_csv_split_bytes(split, has_header=has_header)

    split_start = split.offset
    split_end = split.offset + split.length
    with open(split.path, "rb") as f:
        if has_header:
            header = f.readline()
            data_start = f.tell()
        else:
            header = b""
            data_start = 0
        start = max(split_start, data_start)

        if start > data_start:
            f.seek(start - 1)
            previous = f.read(1)
            if previous != b"\n":
                f.readline()
                start = f.tell()
            else:
                f.seek(start)
        else:
            f.seek(start)

        rows = bytearray()
        while True:
            line_start = f.tell()
            if line_start >= split_end:
                break
            line = f.readline()
            if not line:
                break
            rows.extend(line)

    if not rows:
        return None
    return header + bytes(rows)


def _read_bz2_csv_split_bytes(split: CSVSplit, *, has_header: bool) -> bytes | None:
    decoded = _read_bz2_split_payload(
        split.path,
        offset=split.offset,
        length=split.length,
        require_previous_output_ended_with_newline=True,
    )
    if decoded is None:
        return None
    payload = decoded.payload
    owned_output_len = decoded.owned_output_len
    if not payload:
        return None

    if split.offset > 0 and not decoded.previous_output_ended_with_newline:
        first_newline = payload.find(b"\n")
        if first_newline == -1:
            return None
        payload = payload[first_newline + 1 :]
        owned_output_len = max(0, owned_output_len - (first_newline + 1))

    if owned_output_len <= 0:
        return None
    if payload[:owned_output_len].endswith(b"\n"):
        payload = payload[:owned_output_len]
    else:
        trailing_newline = payload.find(b"\n", owned_output_len)
        if trailing_newline == -1:
            return None
        payload = payload[: trailing_newline + 1]

    if not payload:
        return None
    if not has_header or split.offset == 0:
        return payload
    return _read_bz2_header_line(split.path) + payload


def _read_bz2_header_line(path: str) -> bytes:
    import bz2

    with bz2.open(path, "rb") as stream:
        return stream.readline()


def _iter_bz2_csv_split_byte_batches(split: CSVSplit, *, has_header: bool) -> Iterable[bytes]:
    header = _read_bz2_header_line(split.path) if has_header else b""
    current_record = bytearray()
    rows = bytearray()
    current_record_owned = False
    have_record_start = False
    skip_leading_record = False
    skip_header_record = has_header and split.offset == 0
    first_block = True

    for block in _iter_bz2_decompressed_blocks(
        split.path,
        offset=split.offset,
        length=split.length,
        require_previous_output_ended_with_newline=True,
    ):
        data = block.data
        position = 0
        if first_block:
            first_block = False
            skip_leading_record = split.offset > 0 and not block.previous_output_ended_with_newline

        while position < len(data):
            if not have_record_start:
                current_record_owned = block.owned
                have_record_start = True
                if not current_record_owned and not skip_leading_record:
                    if rows:
                        yield header + bytes(rows)
                    return

            newline = data.find(b"\n", position)
            if newline == -1:
                current_record.extend(data[position:])
                break

            current_record.extend(data[position:newline + 1])
            position = newline + 1
            if skip_leading_record:
                current_record.clear()
                skip_leading_record = False
                have_record_start = False
                continue
            if skip_header_record:
                current_record.clear()
                skip_header_record = False
                have_record_start = False
                continue
            if current_record_owned:
                rows.extend(current_record)
                current_record.clear()
                have_record_start = False
                continue

            if rows:
                yield header + bytes(rows)
            return

        if rows:
            yield header + bytes(rows)
            rows.clear()

    if current_record and current_record_owned and not skip_leading_record and not skip_header_record:
        rows.extend(current_record)
    if rows:
        yield header + bytes(rows)

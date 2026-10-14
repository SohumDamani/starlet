from __future__ import annotations

from concurrent.futures import as_completed
import json
from dataclasses import dataclass, replace
import io
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow as pa
from shapely import points, to_wkb

from starlet._internal.tiling.RSGrove import EnvelopeNDLite
from starlet._internal.executor import create_process_executor
from starlet._internal.tiling.datasource import (
    DataSource,
    TarFileSplit,
    TarMember,
        _GPX_SUFFIXES,
        _TAR_SUFFIXES,
        _iter_tar_members_for_split,
    _source_tar_files,
    _tar_splits,
    _attach_geoparquet_metadata,
    _combine_spatial_samples,
    _reservoir_add,
    _spatial_sample_from_state,
    _split_sample_cap,
    _source_files,
    SpatialSample,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPXSplit:
    """One GPX file, used as a natural source split."""

    path: str
    tar_offset: int | None = None
    tar_length: int | None = None


_GPX_VALUE_FIELDS = (
    ("filename", pa.string()),
    ("gpx_version", pa.string()),
    ("gpx_creator", pa.string()),
    ("gpx_name", pa.string()),
    ("gpx_description", pa.string()),
    ("gpx_author", pa.string()),
    ("gpx_time", pa.string()),
    ("gpx_keywords", pa.string()),
    ("gpx_bounds", pa.string()),
    ("gpx_metadata_xml", pa.string()),
    ("gpx_extensions_xml", pa.string()),
    ("point_kind", pa.string()),
    ("track_index", pa.int64()),
    ("track_name", pa.string()),
    ("track_comment", pa.string()),
    ("track_description", pa.string()),
    ("track_source", pa.string()),
    ("track_number", pa.int64()),
    ("track_type", pa.string()),
    ("track_links", pa.string()),
    ("track_extensions_xml", pa.string()),
    ("segment_index", pa.int64()),
    ("segment_extensions_xml", pa.string()),
    ("route_index", pa.int64()),
    ("route_name", pa.string()),
    ("route_comment", pa.string()),
    ("route_description", pa.string()),
    ("route_source", pa.string()),
    ("route_number", pa.int64()),
    ("route_type", pa.string()),
    ("route_links", pa.string()),
    ("route_extensions_xml", pa.string()),
    ("point_index", pa.int64()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("elevation", pa.float64()),
    ("point_time", pa.string()),
    ("point_name", pa.string()),
    ("point_comment", pa.string()),
    ("point_description", pa.string()),
    ("point_source", pa.string()),
    ("point_symbol", pa.string()),
    ("point_type", pa.string()),
    ("point_fix", pa.string()),
    ("point_satellites", pa.int64()),
    ("point_hdop", pa.float64()),
    ("point_vdop", pa.float64()),
    ("point_pdop", pa.float64()),
    ("point_age_of_dgps_data", pa.float64()),
    ("point_dgps_id", pa.int64()),
    ("point_links", pa.string()),
    ("point_extensions_xml", pa.string()),
)
_GPX_BASE_FIELDS = (
    "filename",
    "point_kind",
    "point_index",
    "latitude",
    "longitude",
)
_GPX_FIELD_INDEX = {
    name: index
    for index, (name, _field_type) in enumerate(_GPX_VALUE_FIELDS)
}
_GPX_BASE_FIELD_MASK = sum(1 << _GPX_FIELD_INDEX[name] for name in _GPX_BASE_FIELDS)
_ROOT_TEXT_FIELDS = {
    "name": "gpx_name",
    "desc": "gpx_description",
    "time": "gpx_time",
    "keywords": "gpx_keywords",
}
_TRACK_FIELDS = {
    "name": "track_name",
    "cmt": "track_comment",
    "desc": "track_description",
    "src": "track_source",
    "number": "track_number",
    "type": "track_type",
}
_ROUTE_FIELDS = {
    "name": "route_name",
    "cmt": "route_comment",
    "desc": "route_description",
    "src": "route_source",
    "number": "route_number",
    "type": "route_type",
}
_POINT_FIELDS = {
    "ele": "elevation",
    "time": "point_time",
    "name": "point_name",
    "cmt": "point_comment",
    "desc": "point_description",
    "src": "point_source",
    "sym": "point_symbol",
    "type": "point_type",
    "fix": "point_fix",
    "sat": "point_satellites",
    "hdop": "point_hdop",
    "vdop": "point_vdop",
    "pdop": "point_pdop",
    "ageofdgpsdata": "point_age_of_dgps_data",
    "dgpsid": "point_dgps_id",
}
_POINT_TAGS = {"wpt", "rtept", "trkpt"}


def _gpx_field_mask(*field_names: str) -> int:
    mask = 0
    for name in field_names:
        mask |= 1 << _GPX_FIELD_INDEX[name]
    return mask


@dataclass
class _GPXScanContainer:
    index: int
    field_mask: int
    point_index: int = -1
    segment_index: int = -1


class GPXSource(DataSource):
    """Read GPX tracks, routes, and waypoints as point records.

    Each GPX file is flattened into point rows. File-level, GPX metadata,
    track/route, segment, and point metadata are repeated on every output row
    so the original hierarchy can be reconstructed or grouped after tiling.
    """

    def __init__(
        self,
        path: str,
        *,
        batch_rows: int = 65_536,
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ) -> None:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {path}")
        if source_path.is_file() and source_path.suffix.lower() not in (*_GPX_SUFFIXES, *_TAR_SUFFIXES):
            raise ValueError(f"GPX source file must have a .gpx or .tar extension: {path}")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be greater than zero")

        self.path = str(source_path)
        self.batch_rows = int(batch_rows)
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self._files = (
            []
            if source_path.is_file() and source_path.suffix.lower() in _TAR_SUFFIXES
            else _source_files(self.path, _GPX_SUFFIXES)
        )
        self._tar_files = _source_tar_files(self.path, _GPX_SUFFIXES)
        if not self._files and not self._tar_files:
            raise ValueError(f"No GPX files found in {self.path}")
        self._schema: pa.Schema | None = (
            _schema_for_gpx_fields((), geom_col=self.geom_col)
            if self.geometry_only
            else None
        )

    @classmethod
    def read_spatial_sample(
        cls,
        path: str,
        *,
        sample_cap: Optional[int],
        seed: int,
        workers: Optional[int],
        geom_col: str = "geometry",
    ) -> SpatialSample:
        source = cls(path, geometry_only=True, geom_col=geom_col)
        splits = source.create_splits()
        sample_caps = _split_sample_cap(sample_cap, len(splits))

        with create_process_executor(
            max_workers=workers,
            logger=logger,
            context="GPX spatial sampling",
        ) as executor:
            futures = [
                executor.submit(
                    _read_gpx_file_spatial_sample,
                    split,
                    sample_caps[index],
                    seed + index,
                    geom_col,
                )
                for index, split in enumerate(splits)
            ]
            parts = [future.result() for future in as_completed(futures)]

        sample = _combine_spatial_samples(parts)
        field_mask = 0
        for part in parts:
            field_mask |= part.gpx_field_mask
        return SpatialSample(
            sample_points=sample.sample_points,
            total_seen=sample.total_seen,
            total_sampled=sample.total_sampled,
            batches_read=sample.batches_read,
            schema=_schema_for_gpx_field_mask(field_mask, geom_col=geom_col),
            gpx_field_mask=field_mask,
        )

    def schema(self) -> pa.Schema:
        if self._schema is None:
            self._schema = self._infer_schema()
        return self._schema

    def set_schema(self, schema: pa.Schema) -> None:
        if self.geom_col not in schema.names:
            raise ValueError(f"GPX schema must contain geometry column {self.geom_col!r}")
        self._schema = schema

    def input_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in [*self._files, *self._tar_files])

    def _infer_schema(self) -> pa.Schema:
        field_mask = 0
        for split in self.create_splits():
            sample = _read_gpx_file_spatial_sample(
                split,
                sample_cap=0,
                seed=0,
                geom_col=self.geom_col,
            )
            field_mask |= sample.gpx_field_mask
        return _schema_for_gpx_field_mask(field_mask, geom_col=self.geom_col)

    def create_splits(self, num_splits: Optional[int] = None) -> List[GPXSplit]:
        splits = [GPXSplit(path=str(path)) for path in self._files]
        for tar_path in self._tar_files:
            splits.extend(
                GPXSplit(path=tar_split.path, tar_offset=tar_split.offset, tar_length=tar_split.length)
                for tar_split in _tar_splits(str(tar_path))
            )
        return splits

    def iter_tables(self, split: Optional[GPXSplit] = None) -> Iterable[pa.Table]:
        schema = self.schema()
        selected_fields = set(schema.names)
        selected_fields.discard(self.geom_col)
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            if source_split.tar_offset is None or source_split.tar_length is None:
                yield from self._iter_file_tables(Path(source_split.path), selected_fields)
                continue
            for member in _iter_tar_members_for_split(
                source_split.path,
                offset=source_split.tar_offset,
                length=source_split.tar_length,
                suffixes=_GPX_SUFFIXES,
            ):
                yield from self._iter_member_tables(member, selected_fields)

    def _iter_file_tables(
        self,
        path: Path,
        selected_fields: set[str],
    ) -> Iterable[pa.Table]:
        rows: List[dict[str, Any]] = []
        for row in self._iter_file_rows(path, selected_fields):
            rows.append(row)
            if len(rows) >= self.batch_rows:
                yield self._rows_to_table(rows)
                rows = []

        if rows:
            yield self._rows_to_table(rows)

    def _iter_member_tables(
        self,
        member: TarMember,
        selected_fields: set[str],
    ) -> Iterable[pa.Table]:
        rows: List[dict[str, Any]] = []
        for row in self._iter_member_rows(member, selected_fields):
            rows.append(row)
            if len(rows) >= self.batch_rows:
                yield self._rows_to_table(rows)
                rows = []

        if rows:
            yield self._rows_to_table(rows)

    def _iter_file_rows(
        self,
        path: Path,
        selected_fields: set[str],
    ) -> Iterable[dict[str, Any]]:
        with path.open("rb") as stream:
            yield from self._iter_xml_rows(stream.read(), path.name, str(path), selected_fields)

    def _iter_member_rows(
        self,
        member: TarMember,
        selected_fields: set[str],
    ) -> Iterable[dict[str, Any]]:
        yield from self._iter_xml_rows(member.data, Path(member.name).name, member.name, selected_fields)

    def _iter_xml_rows(
        self,
        xml_bytes: bytes,
        filename: str,
        source_label: str,
        selected_fields: set[str],
    ) -> Iterable[dict[str, Any]]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid GPX XML in {source_label}: {exc}") from exc

        if _local_name(root.tag) != "gpx":
            raise ValueError(f"Invalid GPX file {source_label}: expected root <gpx>")

        base = {
            **_selected_value("filename", filename, selected_fields),
            **_gpx_metadata(root, selected_fields),
        }

        for waypoint_index, waypoint in enumerate(_children(root, "wpt")):
            yield _point_row(
                waypoint,
                path=Path(filename),
                context=f"waypoint[{waypoint_index}]",
                base=base,
                point_kind="waypoint",
                point_index=waypoint_index,
                selected_fields=selected_fields,
            )

        for route_index, route in enumerate(_children(root, "rte")):
            route_meta = _route_or_track_metadata(route, "route", selected_fields)
            for point_index, route_point in enumerate(_children(route, "rtept")):
                route_base = dict(base)
                route_base.update(route_meta)
                route_base.update(_selected_value("route_index", route_index, selected_fields))
                yield _point_row(
                    route_point,
                    path=Path(filename),
                    context=f"route[{route_index}]/point[{point_index}]",
                    base=route_base,
                    point_kind="route",
                    point_index=point_index,
                    selected_fields=selected_fields,
                )

        for track_index, track in enumerate(_children(root, "trk")):
            track_meta = _route_or_track_metadata(track, "track", selected_fields)
            for segment_index, segment in enumerate(_children(track, "trkseg")):
                segment_meta = {
                    **_selected_value("segment_index", segment_index, selected_fields),
                }
                _put_selected(
                    segment_meta,
                    "segment_extensions_xml",
                    selected_fields,
                    lambda: _child_xml(segment, "extensions"),
                )
                for point_index, track_point in enumerate(_children(segment, "trkpt")):
                    track_base = dict(base)
                    track_base.update(track_meta)
                    track_base.update(segment_meta)
                    track_base.update(_selected_value("track_index", track_index, selected_fields))
                    yield _point_row(
                        track_point,
                        path=Path(filename),
                        context=(
                            f"track[{track_index}]/segment[{segment_index}]"
                            f"/point[{point_index}]"
                        ),
                        base=track_base,
                        point_kind="track",
                        point_index=point_index,
                        selected_fields=selected_fields,
                    )

    def _rows_to_table(self, rows: Sequence[dict[str, Any]]) -> pa.Table:
        latitudes = [row["latitude"] for row in rows]
        longitudes = [row["longitude"] for row in rows]
        geometry = pa.array(
            to_wkb(points(longitudes, latitudes), hex=False).tolist(),
            type=pa.binary(),
        )

        if self.geometry_only:
            table = pa.table([geometry], schema=self.schema())
        else:
            schema = self.schema()
            columns = {
                field.name: pa.array(
                    [row.get(field.name) for row in rows],
                    type=field.type,
                )
                for field in schema
                if field.name != self.geom_col
            }
            columns[self.geom_col] = geometry
            table = pa.table(columns, schema=schema)

        table = table.replace_schema_metadata(self.schema().metadata)
        if not table.schema.equals(self.schema(), check_metadata=False):
            table = table.cast(self.schema())
        return table.combine_chunks()


def _schema_for_gpx_fields(field_names: Iterable[str], *, geom_col: str) -> pa.Schema:
    selected = set(field_names)
    fields = [
        pa.field(name, field_type)
        for name, field_type in _GPX_VALUE_FIELDS
        if name in selected
    ]
    fields.append(pa.field(geom_col, pa.binary()))
    return _attach_geoparquet_metadata(
        pa.schema(fields),
        "EPSG:4326",
        geom_col=geom_col,
    )


def _schema_for_gpx_field_mask(field_mask: int, *, geom_col: str) -> pa.Schema:
    fields = [
        pa.field(name, field_type)
        for index, (name, field_type) in enumerate(_GPX_VALUE_FIELDS)
        if field_mask & (1 << index)
    ]
    fields.append(pa.field(geom_col, pa.binary()))
    return _attach_geoparquet_metadata(
        pa.schema(fields),
        "EPSG:4326",
        geom_col=geom_col,
    )


def _gpx_metadata(root: ET.Element, selected_fields: set[str]) -> dict[str, Any]:
    metadata = _first_child(root, "metadata")
    scalar_parent = metadata if metadata is not None else root
    bounds = _first_child(scalar_parent, "bounds")
    extensions_parent = metadata if metadata is not None else root

    values = {}
    _put_selected(values, "gpx_version", selected_fields, lambda: root.attrib.get("version"))
    _put_selected(values, "gpx_creator", selected_fields, lambda: root.attrib.get("creator"))
    _put_selected(values, "gpx_name", selected_fields, lambda: _child_text(scalar_parent, "name"))
    _put_selected(
        values,
        "gpx_description",
        selected_fields,
        lambda: _child_text(scalar_parent, "desc"),
    )
    _put_selected(values, "gpx_author", selected_fields, lambda: _author_text(scalar_parent))
    _put_selected(values, "gpx_time", selected_fields, lambda: _child_text(scalar_parent, "time"))
    _put_selected(
        values,
        "gpx_keywords",
        selected_fields,
        lambda: _child_text(scalar_parent, "keywords"),
    )
    _put_selected(values, "gpx_bounds", selected_fields, lambda: _bounds_json(bounds))
    _put_selected(values, "gpx_metadata_xml", selected_fields, lambda: _element_xml(metadata))
    _put_selected(
        values,
        "gpx_extensions_xml",
        selected_fields,
        lambda: _child_xml(extensions_parent, "extensions"),
    )
    return values


def _route_or_track_metadata(
    element: ET.Element,
    prefix: str,
    selected_fields: set[str],
) -> dict[str, Any]:
    values = {}
    _put_selected(values, f"{prefix}_name", selected_fields, lambda: _child_text(element, "name"))
    _put_selected(values, f"{prefix}_comment", selected_fields, lambda: _child_text(element, "cmt"))
    _put_selected(
        values,
        f"{prefix}_description",
        selected_fields,
        lambda: _child_text(element, "desc"),
    )
    _put_selected(values, f"{prefix}_source", selected_fields, lambda: _child_text(element, "src"))
    _put_selected(values, f"{prefix}_number", selected_fields, lambda: _int_child(element, "number"))
    _put_selected(values, f"{prefix}_type", selected_fields, lambda: _child_text(element, "type"))
    _put_selected(values, f"{prefix}_links", selected_fields, lambda: _links_json(element))
    _put_selected(
        values,
        f"{prefix}_extensions_xml",
        selected_fields,
        lambda: _child_xml(element, "extensions"),
    )
    return values


def _point_row(
    element: ET.Element,
    *,
    path: Path,
    context: str,
    base: dict[str, Any],
    point_kind: str,
    point_index: int,
    selected_fields: set[str],
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "latitude": _required_float_attr(
                element,
                "lat",
                path=path,
                context=context,
            ),
            "longitude": _required_float_attr(
                element,
                "lon",
                path=path,
                context=context,
            ),
        }
    )
    row.update(_selected_value("point_kind", point_kind, selected_fields))
    row.update(_selected_value("point_index", point_index, selected_fields))
    _put_selected(row, "elevation", selected_fields, lambda: _float_child(element, "ele"))
    _put_selected(row, "point_time", selected_fields, lambda: _child_text(element, "time"))
    _put_selected(row, "point_name", selected_fields, lambda: _child_text(element, "name"))
    _put_selected(row, "point_comment", selected_fields, lambda: _child_text(element, "cmt"))
    _put_selected(
        row,
        "point_description",
        selected_fields,
        lambda: _child_text(element, "desc"),
    )
    _put_selected(row, "point_source", selected_fields, lambda: _child_text(element, "src"))
    _put_selected(row, "point_symbol", selected_fields, lambda: _child_text(element, "sym"))
    _put_selected(row, "point_type", selected_fields, lambda: _child_text(element, "type"))
    _put_selected(row, "point_fix", selected_fields, lambda: _child_text(element, "fix"))
    _put_selected(row, "point_satellites", selected_fields, lambda: _int_child(element, "sat"))
    _put_selected(row, "point_hdop", selected_fields, lambda: _float_child(element, "hdop"))
    _put_selected(row, "point_vdop", selected_fields, lambda: _float_child(element, "vdop"))
    _put_selected(row, "point_pdop", selected_fields, lambda: _float_child(element, "pdop"))
    _put_selected(
        row,
        "point_age_of_dgps_data",
        selected_fields,
        lambda: _float_child(element, "ageofdgpsdata"),
    )
    _put_selected(row, "point_dgps_id", selected_fields, lambda: _int_child(element, "dgpsid"))
    _put_selected(row, "point_links", selected_fields, lambda: _links_json(element))
    _put_selected(
        row,
        "point_extensions_xml",
        selected_fields,
        lambda: _child_xml(element, "extensions"),
    )
    return row


def _selected_value(
    name: str,
    value: Any,
    selected_fields: set[str],
) -> dict[str, Any]:
    if name not in selected_fields:
        return {}
    return {name: value}


def _put_selected(
    values: dict[str, Any],
    name: str,
    selected_fields: set[str],
    value_factory,
) -> None:
    if name in selected_fields:
        values[name] = value_factory()


def _read_gpx_file_spatial_sample(
    split: GPXSplit,
    sample_cap: Optional[int],
    seed: int,
    geom_col: str,
) -> SpatialSample:
    if split.tar_offset is None or split.tar_length is None:
        return _read_gpx_bytes_spatial_sample(
            Path(split.path).read_bytes(),
            filename=Path(split.path).name,
            source_label=split.path,
            sample_cap=sample_cap,
            seed=seed,
            geom_col=geom_col,
        )

    rng = np.random.default_rng(seed)
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    batches_read = 0
    field_mask = 0

    for member in _iter_tar_members_for_split(
        split.path,
        offset=split.tar_offset,
        length=split.tar_length,
        suffixes=_GPX_SUFFIXES,
    ):
        member_sample = _read_gpx_bytes_spatial_sample(
            member.data,
            filename=Path(member.name).name,
            source_label=member.name,
            sample_cap=sample_cap,
            seed=seed,
            geom_col=geom_col,
            rng=rng,
            x_sample=x_sample,
            y_sample=y_sample,
            n_seen_start=n_seen,
        )
        batches_read += member_sample.batches_read
        field_mask |= member_sample.gpx_field_mask
        if member_sample.total_seen <= 0:
            continue

        n_seen += member_sample.total_seen

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        n_seen=n_seen,
        batches_read=batches_read,
        gpx_field_mask=field_mask,
    )


def _read_gpx_bytes_spatial_sample(
    xml_bytes: bytes,
    *,
    filename: str,
    source_label: str,
    sample_cap: Optional[int],
    seed: int,
    geom_col: str,
    rng: np.random.Generator | None = None,
    x_sample: List[float] | None = None,
    y_sample: List[float] | None = None,
    n_seen_start: int = 0,
) -> SpatialSample:
    rng = rng if rng is not None else np.random.default_rng(seed)
    x_sample = x_sample if x_sample is not None else []
    y_sample = y_sample if y_sample is not None else []
    n_seen = 0

    emitted_field_mask = 0
    file_field_mask = 0
    stack: list[str] = []
    route_stack: list[_GPXScanContainer] = []
    track_stack: list[_GPXScanContainer] = []
    segment_stack: list[_GPXScanContainer] = []
    waypoint_index = -1
    route_index = -1
    track_index = -1

    try:
        events = ET.iterparse(io.BytesIO(xml_bytes), events=("start", "end"))
        for event, element in events:
            tag = _local_name(element.tag)
            if event == "start":
                stack.append(tag)
                parent = stack[-2] if len(stack) >= 2 else None
                if tag == "gpx":
                    if element.attrib.get("version") is not None:
                        file_field_mask |= _gpx_field_mask("gpx_version")
                    if element.attrib.get("creator") is not None:
                        file_field_mask |= _gpx_field_mask("gpx_creator")
                elif tag == "metadata" and parent == "gpx":
                    file_field_mask |= _gpx_field_mask("gpx_metadata_xml")
                elif tag == "bounds" and parent in {"gpx", "metadata"}:
                    file_field_mask |= _gpx_field_mask("gpx_bounds")
                elif tag == "author" and parent in {"gpx", "metadata"}:
                    file_field_mask |= _gpx_field_mask("gpx_author")
                elif tag == "extensions":
                    file_field_mask = _mark_extension_field(
                        file_field_mask,
                        route_stack,
                        track_stack,
                        segment_stack,
                        parent,
                    )
                elif tag in {"link", "url"}:
                    file_field_mask = _mark_link_field(
                        file_field_mask,
                        route_stack,
                        track_stack,
                        parent,
                    )
                elif tag == "rte":
                    route_index += 1
                    route_stack.append(_GPXScanContainer(index=route_index, field_mask=0))
                elif tag == "trk":
                    track_index += 1
                    track_stack.append(_GPXScanContainer(index=track_index, field_mask=0))
                elif tag == "trkseg" and track_stack:
                    track_stack[-1].segment_index += 1
                    segment_stack.append(
                        _GPXScanContainer(
                            index=track_stack[-1].segment_index,
                            field_mask=0,
                        )
                    )
                continue

            parent = stack[-2] if len(stack) >= 2 else None
            grandparent = stack[-3] if len(stack) >= 3 else None

            if parent in {"gpx", "metadata"} and tag in _ROOT_TEXT_FIELDS:
                if _direct_text(element) is not None:
                    file_field_mask |= _gpx_field_mask(_ROOT_TEXT_FIELDS[tag])
            elif parent == "author" and grandparent in {"gpx", "metadata"}:
                if _direct_text(element) is not None or element.attrib:
                    file_field_mask |= _gpx_field_mask("gpx_author")
            elif parent == "trk" and track_stack and tag in _TRACK_FIELDS:
                if _direct_text(element) is not None:
                    track_stack[-1].field_mask |= _gpx_field_mask(_TRACK_FIELDS[tag])
            elif parent == "rte" and route_stack and tag in _ROUTE_FIELDS:
                if _direct_text(element) is not None:
                    route_stack[-1].field_mask |= _gpx_field_mask(_ROUTE_FIELDS[tag])
            elif parent in _POINT_TAGS and tag in _POINT_FIELDS:
                if _direct_text(element) is not None:
                    emitted_field_mask |= _gpx_field_mask(_POINT_FIELDS[tag])
            elif parent in _POINT_TAGS and tag in {"link", "url"}:
                emitted_field_mask |= _gpx_field_mask("point_links")
            elif parent in _POINT_TAGS and tag == "extensions":
                emitted_field_mask |= _gpx_field_mask("point_extensions_xml")

            if tag in _POINT_TAGS:
                _, point_index, context, structural_fields = _point_scan_context(
                    tag,
                    Path(filename),
                    waypoint_index,
                    route_stack,
                    track_stack,
                    segment_stack,
                )
                if tag == "wpt":
                    waypoint_index = point_index
                emitted_field_mask |= _GPX_BASE_FIELD_MASK
                emitted_field_mask |= file_field_mask
                emitted_field_mask |= structural_fields

                x = _required_float_attr(
                    element,
                    "lon",
                    path=Path(filename),
                    context=context,
                )
                y = _required_float_attr(
                    element,
                    "lat",
                    path=Path(filename),
                    context=context,
                )
                n_seen += 1
                _reservoir_add(
                    rng=rng,
                    sample_cap=sample_cap,
                    x_sample=x_sample,
                    y_sample=y_sample,
                    n_seen=n_seen_start + n_seen,
                    x=x,
                    y=y,
                )

            if tag == "trkseg" and segment_stack:
                segment_stack.pop()
            elif tag == "trk" and track_stack:
                track_stack.pop()
            elif tag == "rte" and route_stack:
                route_stack.pop()

            stack.pop()
            element.clear()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid GPX XML in {source_label}: {exc}") from exc

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        n_seen=n_seen,
        batches_read=1 if n_seen else 0,
        gpx_field_mask=emitted_field_mask,
    )


def _mark_extension_field(
    file_field_mask: int,
    route_stack: list[_GPXScanContainer],
    track_stack: list[_GPXScanContainer],
    segment_stack: list[_GPXScanContainer],
    parent: str | None,
) -> int:
    if parent in {"gpx", "metadata"}:
        file_field_mask |= _gpx_field_mask("gpx_extensions_xml")
    elif parent == "rte" and route_stack:
        route_stack[-1].field_mask |= _gpx_field_mask("route_extensions_xml")
    elif parent == "trk" and track_stack:
        track_stack[-1].field_mask |= _gpx_field_mask("track_extensions_xml")
    elif parent == "trkseg" and segment_stack:
        segment_stack[-1].field_mask |= _gpx_field_mask("segment_extensions_xml")
    return file_field_mask


def _mark_link_field(
    file_field_mask: int,
    route_stack: list[_GPXScanContainer],
    track_stack: list[_GPXScanContainer],
    parent: str | None,
) -> int:
    if parent == "author":
        file_field_mask |= _gpx_field_mask("gpx_author")
    elif parent == "rte" and route_stack:
        route_stack[-1].field_mask |= _gpx_field_mask("route_links")
    elif parent == "trk" and track_stack:
        track_stack[-1].field_mask |= _gpx_field_mask("track_links")
    return file_field_mask


def _point_scan_context(
    tag: str,
    path: Path,
    waypoint_index: int,
    route_stack: list[_GPXScanContainer],
    track_stack: list[_GPXScanContainer],
    segment_stack: list[_GPXScanContainer],
) -> tuple[str, int, str, int]:
    if tag == "wpt":
        point_index = waypoint_index + 1
        return "waypoint", point_index, f"waypoint[{point_index}]", 0
    if tag == "rtept":
        if not route_stack:
            raise ValueError(f"Invalid GPX point in {path}: route point outside <rte>")
        route = route_stack[-1]
        route.point_index += 1
        return (
            "route",
            route.point_index,
            f"route[{route.index}]/point[{route.point_index}]",
            _gpx_field_mask("route_index") | route.field_mask,
        )
    if not track_stack or not segment_stack:
        raise ValueError(f"Invalid GPX point in {path}: track point outside <trkseg>")
    track = track_stack[-1]
    segment = segment_stack[-1]
    segment.point_index += 1
    return (
        "track",
        segment.point_index,
        f"track[{track.index}]/segment[{segment.index}]/point[{segment.point_index}]",
        _gpx_field_mask("track_index", "segment_index") | track.field_mask | segment.field_mask,
    )


def _required_float_attr(
    element: ET.Element,
    name: str,
    *,
    path: Path,
    context: str,
) -> float:
    value = element.attrib.get(name)
    if value is None:
        raise ValueError(f"Invalid GPX point in {path} ({context}): missing {name!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid GPX point in {path} ({context}): invalid {name!r}: {value!r}"
        ) from exc


def _float_child(element: ET.Element, name: str) -> float | None:
    value = _child_text(element, name)
    if value is None:
        return None
    return float(value)


def _int_child(element: ET.Element, name: str) -> int | None:
    value = _child_text(element, name)
    if value is None:
        return None
    return int(value)


def _author_text(parent: ET.Element) -> str | None:
    author = _first_child(parent, "author")
    if author is None:
        return None

    text = _direct_text(author)
    if text is not None:
        return text

    details = {
        "name": _child_text(author, "name"),
        "email": _email_text(_first_child(author, "email")),
        "links": _links_list(author),
    }
    compact = {key: value for key, value in details.items() if value}
    if not compact:
        return None
    return json.dumps(compact, separators=(",", ":"), sort_keys=True)


def _bounds_json(bounds: ET.Element | None) -> str | None:
    if bounds is None:
        return None
    values = {
        key: bounds.attrib.get(key)
        for key in ("minlat", "minlon", "maxlat", "maxlon")
        if bounds.attrib.get(key) is not None
    }
    if not values:
        return None
    return json.dumps(values, separators=(",", ":"), sort_keys=True)


def _links_json(element: ET.Element) -> str | None:
    links = _links_list(element)
    if not links:
        return None
    return json.dumps(links, separators=(",", ":"), sort_keys=True)


def _links_list(element: ET.Element) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for child in element:
        child_name = _local_name(child.tag)
        if child_name == "link":
            link = {
                "href": child.attrib.get("href"),
                "text": _child_text(child, "text"),
                "type": _child_text(child, "type"),
            }
        elif child_name == "url":
            link = {
                "href": _direct_text(child),
                "text": _child_text(element, "urlname"),
                "type": None,
            }
        else:
            continue

        compact = {key: value for key, value in link.items() if value is not None}
        if compact:
            links.append(compact)
    return links


def _email_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    id_part = element.attrib.get("id")
    domain_part = element.attrib.get("domain")
    if id_part and domain_part:
        return f"{id_part}@{domain_part}"
    return _direct_text(element)


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _child_text(element: ET.Element, name: str) -> str | None:
    child = _first_child(element, name)
    return _direct_text(child) if child is not None else None


def _direct_text(element: ET.Element) -> str | None:
    if element.text is None:
        return None
    text = element.text.strip()
    return text or None


def _child_xml(element: ET.Element, name: str) -> str | None:
    return _element_xml(_first_child(element, name))


def _element_xml(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return ET.tostring(element, encoding="unicode")


def _local_name(tag: Any) -> str:
    text = str(tag)
    if "}" in text:
        return text.rsplit("}", 1)[1]
    return text

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, List, Optional, Tuple
import zipfile

import pandas as pd
import pyarrow as pa
from shapely import to_wkb

from starlet._internal.config import resolve_temp_dir
from starlet._internal.tiling.datasource import (
    DataSource,
    _GDB_SUFFIXES,
    _SHAPEFILE_SUFFIXES,
    _ZIP_SUFFIXES,
    _attach_geoparquet_metadata,
    _normalize_decimal_columns,
    _unify_tabular_schemas,
    _wkb_geometry_type,
)
from starlet._internal.tiling.nonlinear_wkb import linearize_wkb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorLayerSplit:
    path: str
    layer: str | None
    skip_features: int
    max_features: int | None
    geometry_type: str | None = None


@dataclass(frozen=True)
class _VectorLayer:
    path: str
    layer: str | None
    feature_count: int | None
    geometry_type: str | None = None


class _OGRVectorSource(DataSource):
    def __init__(
        self,
        path: str,
        *,
        geometry_only: bool = False,
        geom_col: str = "geometry",
        batch_features: int = 65_536,
    ) -> None:
        self.path = str(path)
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self.batch_features = int(batch_features)
        self._layers = self._discover_layers(self.path)
        if not self._layers:
            raise ValueError(f"No vector layers found in {self.path}")
        self._schema: pa.Schema | None = None

    def schema(self) -> pa.Schema:
        if self._schema is None:
            schemas = [table.schema for table in self.iter_tables_for_schema_inference()]
            self._schema = _unify_tabular_schemas(schemas) if schemas else _attach_geoparquet_metadata(
                pa.schema([(self.geom_col, pa.binary())]),
                "EPSG:4326",
            )
        return self._schema

    def set_schema(self, schema: pa.Schema) -> None:
        if self.geom_col not in schema.names:
            raise ValueError(f"Vector schema must contain geometry column {self.geom_col!r}")
        self._schema = schema

    def input_size_bytes(self) -> int:
        total = 0
        for path in self._source_paths_for_size():
            p = Path(path)
            if p.is_file():
                total += p.stat().st_size
            elif p.is_dir():
                total += sum(child.stat().st_size for child in p.rglob("*") if child.is_file())
        return total

    def create_splits(self, num_splits: Optional[int] = None) -> List[VectorLayerSplit]:
        if num_splits is None:
            chunk_size = max(1, self.batch_features)
        else:
            known_rows = sum(layer.feature_count or 0 for layer in self._layers)
            chunk_size = max(1, (max(1, known_rows) + max(1, int(num_splits)) - 1) // max(1, int(num_splits)))

        splits: List[VectorLayerSplit] = []
        for layer in self._layers:
            if layer.feature_count is None or layer.feature_count < 0:
                splits.append(
                    VectorLayerSplit(layer.path, layer.layer, 0, None, layer.geometry_type)
                )
                continue
            for offset in range(0, layer.feature_count, chunk_size):
                splits.append(
                    VectorLayerSplit(
                        path=layer.path,
                        layer=layer.layer,
                        skip_features=offset,
                        max_features=min(chunk_size, layer.feature_count - offset),
                        geometry_type=layer.geometry_type,
                    )
                )
        return splits

    def iter_tables(self, split: Optional[VectorLayerSplit] = None) -> Iterable[pa.Table]:
        schema = self.schema()
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            try:
                table = self._read_split(source_split)
            except Exception:
                logger.exception(
                    "Failed reading vector source split (%s, %s)",
                    _vector_split_context(source_split),
                    _probe_vector_split_geometry_types(source_split),
                )
                raise
            if table.num_rows == 0:
                continue
            table = _cast_table_to_schema(table, schema)
            yield table

    def iter_tables_for_schema_inference(
        self,
        split: Optional[VectorLayerSplit] = None,
    ) -> Iterable[pa.Table]:
        if split is not None:
            table = self._read_split(split)
            if table.num_rows:
                yield table
            return

        for layer in self._layers:
            source_split = VectorLayerSplit(
                path=layer.path,
                layer=layer.layer,
                skip_features=0,
                max_features=1,
                geometry_type=layer.geometry_type,
            )
            table = self._read_split(source_split)
            if table.num_rows:
                yield table

    def _read_split(self, split: VectorLayerSplit) -> pa.Table:
        import pyogrio

        columns = [] if self.geometry_only else None
        meta, table = self._read_arrow_split(split, columns=columns)
        geometry_name = _geometry_column_name(meta, table)
        if not geometry_name:
            if columns == []:
                meta, table = self._read_arrow_split(split, columns=None)
                geometry_name = _geometry_column_name(meta, table)
            if not geometry_name:
                raise ValueError(
                    "Vector split did not contain a geometry column: "
                    f"{_vector_split_context(split)}"
                )

        fid_column = meta.get("fid_column")
        converted_geometries: List[bytes] = []
        keep_indices: List[int] = []
        skipped = 0
        linearized = 0
        geometry_values = table[geometry_name].to_pylist()
        for index, value in enumerate(geometry_values):
            try:
                converted = linearize_wkb(value)
            except Exception:
                skipped += 1
                continue
            if converted is None:
                skipped += 1
                continue
            if converted != value:
                linearized += 1
            converted_geometries.append(converted)
            keep_indices.append(index)

        if not keep_indices:
            logger.warning(
                "Skipped all records with unsupported geometries in vector source split (%s)",
                _vector_split_context(split),
            )
            return pa.table([pa.array([], type=pa.binary())], names=[self.geom_col])

        if len(keep_indices) != table.num_rows:
            table = table.take(pa.array(keep_indices, type=pa.int64()))

        drop_columns = {geometry_name}
        if fid_column:
            drop_columns.add(fid_column)
        props_table = table.drop([name for name in drop_columns if name in table.column_names])
        if self.geometry_only:
            props_table = pa.table({})

        source_crs = meta.get("crs")
        geometry_col = pa.array(converted_geometries, type=pa.binary())
        out = (
            pa.table([geometry_col], names=[self.geom_col])
            if props_table.num_columns == 0
            else props_table.append_column(self.geom_col, geometry_col)
        )
        schema_with_geo = _attach_geoparquet_metadata(
            out.schema,
            str(source_crs) if source_crs else None,
        )
        out = out.replace_schema_metadata(schema_with_geo.metadata).combine_chunks()
        if linearized or skipped:
            logger.warning(
                "Linearized nonlinear WKB in vector source split (%s, linearized_records=%d, "
                "skipped_records=%d)",
                _vector_split_context(split),
                linearized,
                skipped,
            )
        return out

    @staticmethod
    def _read_arrow_split(split: VectorLayerSplit, *, columns: List[str] | None):
        import pyogrio

        return pyogrio.read_arrow(
            split.path,
            layer=split.layer,
            columns=columns,
            read_geometry=True,
            skip_features=split.skip_features,
            max_features=split.max_features,
            return_fids=True,
        )

    def _geodataframe_to_table(self, gdf, split: VectorLayerSplit) -> pa.Table:
        try:
            geometry_col = pa.array(
                to_wkb(gdf.geometry.array, hex=False).tolist(),
                type=pa.binary(),
            )
        except Exception:
            logger.exception(
                "Failed converting vector geometries to WKB (%s, dataframe_geometry_types=%s)",
                _vector_split_context(split),
                _geodataframe_geometry_types(gdf),
            )
            raise
        props_df = pd.DataFrame(index=gdf.index) if self.geometry_only else gdf.drop(columns=gdf.geometry.name)
        props_df = _normalize_decimal_columns(props_df)
        props_table = pa.Table.from_pandas(props_df, preserve_index=False)
        table = (
            pa.table([geometry_col], names=[self.geom_col])
            if props_table.num_columns == 0
            else props_table.append_column(self.geom_col, geometry_col)
        )
        crs_hint = gdf.crs.to_json() if gdf.crs is not None else None
        schema_with_geo = _attach_geoparquet_metadata(table.schema, crs_hint)
        return table.replace_schema_metadata(schema_with_geo.metadata).combine_chunks()

    def _discover_layers(self, path: str) -> List[_VectorLayer]:
        raise NotImplementedError

    def _source_paths_for_size(self) -> List[str]:
        return sorted({_dataset_size_path(layer.path) for layer in self._layers})

    @staticmethod
    def _layers_for_dataset(path: str) -> List[_VectorLayer]:
        import pyogrio

        layers = []
        try:
            layer_info = pyogrio.list_layers(path)
        except Exception:
            layer_info = []
        layer_names = [str(row[0]) for row in layer_info] if len(layer_info) else [None]
        layer_types = {
            str(row[0]): str(row[1])
            for row in layer_info
            if len(row) > 1 and row[1] is not None
        }
        for layer_name in layer_names:
            try:
                info = pyogrio.read_info(path, layer=layer_name, force_feature_count=True)
                feature_count = info.get("features")
                geometry_type = info.get("geometry_type") or layer_types.get(str(layer_name))
            except Exception:
                feature_count = None
                geometry_type = layer_types.get(str(layer_name))
            layers.append(
                _VectorLayer(
                    path=path,
                    layer=layer_name,
                    feature_count=feature_count,
                    geometry_type=geometry_type,
                )
            )
        return layers


class ShapefileSource(_OGRVectorSource):
    def _discover_layers(self, path: str) -> List[_VectorLayer]:
        source = Path(path)
        if source.is_file():
            if source.suffix.lower() == ".zip":
                return self._layers_for_dataset(_zip_vsi_path(source))
            if source.suffix.lower() == ".shp":
                return self._layers_for_dataset(str(source))
            raise ValueError(f"Not a Shapefile source: {path}")

        if source.is_dir():
            layers: List[_VectorLayer] = []
            for shp in sorted(source.rglob("*.shp")):
                if ".gdb" not in {part.lower() for part in shp.parts}:
                    layers.extend(self._layers_for_dataset(str(shp)))
            for zip_path in sorted(source.rglob("*.zip")):
                layers.extend(self._layers_for_dataset(_zip_vsi_path(zip_path)))
            return layers

        raise FileNotFoundError(f"Source path does not exist: {path}")


class GDBSource(_OGRVectorSource):
    def _discover_layers(self, path: str) -> List[_VectorLayer]:
        source = Path(path)
        gdb_paths: List[Path]
        if source.is_file() and source.suffix.lower() in _ZIP_SUFFIXES:
            gdb_paths = _extract_zipped_gdbs(source)
        elif source.suffix.lower() in _GDB_SUFFIXES:
            gdb_paths = [source]
        elif source.is_dir():
            gdb_paths = sorted(child for child in source.rglob("*.gdb") if child.is_dir())
        else:
            raise FileNotFoundError(f"Source path does not exist: {path}")

        layers: List[_VectorLayer] = []
        for gdb_path in gdb_paths:
            layers.extend(self._layers_for_dataset(str(gdb_path)))
        return layers


def _zip_vsi_path(path: Path) -> str:
    return f"/vsizip/{path}"


def _dataset_size_path(path: str) -> str:
    if path.startswith("/vsizip/"):
        return path[len("/vsizip/") :]
    return path


def _zip_gdb_member_dirs(path: str | Path) -> List[str]:
    """Return .gdb directory names contained in a zip archive."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return []

    gdb_dirs = set()
    for name in names:
        parts = [part for part in Path(name).parts if part not in {"", "."}]
        for index, part in enumerate(parts):
            if part.lower().endswith(_GDB_SUFFIXES):
                gdb_dirs.add("/".join(parts[: index + 1]))
                break
    return sorted(gdb_dirs)


def _extract_zipped_gdbs(path: Path) -> List[Path]:
    gdb_member_dirs = _zip_gdb_member_dirs(path)
    if not gdb_member_dirs:
        raise ValueError(f"No File Geodatabase directory found in zip archive: {path}")

    stat = path.stat()
    key_material = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    cache_key = hashlib.sha256(key_material).hexdigest()[:24]
    cache_root = resolve_temp_dir(default=Path(tempfile.gettempdir())) / "starlet_gdb_zip_cache" / cache_key
    complete_marker = cache_root / ".complete"
    if complete_marker.exists():
        return [cache_root / member_dir for member_dir in gdb_member_dirs]

    cache_parent = cache_root.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"{cache_key}.", dir=str(cache_parent)))
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                member = info.filename
                if not _is_safe_zip_member(member):
                    continue
                if not any(
                    member == member_dir or member.startswith(f"{member_dir.rstrip('/')}/")
                    for member_dir in gdb_member_dirs
                ):
                    continue
                archive.extract(info, tmp_root)
        (tmp_root / ".complete").write_text("ok\n")
        try:
            os.replace(tmp_root, cache_root)
        except FileExistsError:
            shutil.rmtree(tmp_root, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    return [cache_root / member_dir for member_dir in gdb_member_dirs]


def _is_safe_zip_member(member: str) -> bool:
    path = Path(member)
    return not path.is_absolute() and ".." not in path.parts


def _vector_split_context(split: VectorLayerSplit) -> str:
    return (
        f"path={split.path} "
        f"layer={split.layer!r} "
        f"geometry_type={split.geometry_type or '<unknown>'} "
        f"skip_features={split.skip_features} "
        f"max_features={split.max_features}"
    )


def _geodataframe_geometry_types(gdf) -> str:
    try:
        counts = gdf.geometry.geom_type.value_counts(dropna=False).to_dict()
    except Exception as exc:
        return f"<unavailable: {exc}>"
    return ", ".join(f"{geom_type}={count}" for geom_type, count in counts.items()) or "<none>"


def _geometry_column_name(meta: dict, table: pa.Table) -> str | None:
    geometry_name = meta.get("geometry_name")
    if geometry_name and geometry_name in table.column_names:
        return geometry_name
    if "wkb_geometry" in table.column_names:
        return "wkb_geometry"
    geometry_columns = [
        field.name
        for field in table.schema
        if (field.metadata or {}).get(b"ARROW:extension:name") == b"geoarrow.wkb"
    ]
    return geometry_columns[0] if geometry_columns else None


def _probe_vector_split_geometry_types(split: VectorLayerSplit) -> str:
    try:
        import pyogrio

        meta, table = pyogrio.read_arrow(
            split.path,
            layer=split.layer,
            columns=[],
            read_geometry=True,
            skip_features=split.skip_features,
            max_features=split.max_features,
            return_fids=True,
        )
        geometry_name = _geometry_column_name(meta, table)
        if not geometry_name:
            return "actual_wkb_geometry_types=<unavailable: no geometry column>"

        values = table[geometry_name].to_pylist()
        fid_column = meta.get("fid_column")
        fids = table[fid_column].to_pylist() if fid_column in table.column_names else None
        counts: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        unavailable = 0
        for index, value in enumerate(values):
            geometry_type = _wkb_geometry_type(value)
            if geometry_type is None:
                unavailable += 1
                continue
            counts[geometry_type] = counts.get(geometry_type, 0) + 1
            if len(examples.setdefault(geometry_type, [])) < 5:
                feature_id = fids[index] if fids is not None else split.skip_features + index
                examples[geometry_type].append(str(feature_id))

        summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        if unavailable:
            summary = (
                f"{summary}, unavailable={unavailable}"
                if summary
                else f"unavailable={unavailable}"
            )
        examples_summary = ", ".join(
            f"{name}=[{', '.join(ids)}]" for name, ids in sorted(examples.items())
        )
        return (
            f"actual_wkb_geometry_types={summary or '<none>'} "
            f"sample_feature_ids_by_type={examples_summary or '<none>'}"
        )
    except Exception as exc:
        return f"actual_wkb_geometry_types=<unavailable: {exc}>"


def _cast_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    names = []
    for field in schema:
        if field.name in table.column_names:
            arrays.append(table[field.name])
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
        names.append(field.name)
    projected = pa.table(arrays, names=names)
    if not projected.schema.equals(schema, check_metadata=False):
        projected = projected.cast(schema)
    return projected.replace_schema_metadata(schema.metadata).combine_chunks()

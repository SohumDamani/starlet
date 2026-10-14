import json
from pathlib import Path
import tarfile

import pyarrow as pa
import pytest
from shapely import wkb
from shapely.geometry import Point

from starlet._internal.tiling.datasource import (
    GPXSource,
    GPXSplit,
    read_spatial_sample,
    source_for_path,
)


def _write_gpx(path: Path, body: str, *, version: str = "1.1") -> None:
    namespace = f"http://www.topografix.com/GPX/{version}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f"<gpx xmlns=\"{namespace}\" version=\"{version}\" creator=\"pytest\">\n"
        f"{body}\n"
        "</gpx>\n",
        encoding="utf-8",
    )


def _write_tar(archive_path: Path, members: dict[str, str]) -> None:
    with tarfile.open(archive_path, "w") as archive:
        for name, content in members.items():
            temp_path = archive_path.parent / name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(content, encoding="utf-8")
            archive.add(temp_path, arcname=name)
            temp_path.unlink()
            parent = temp_path.parent
            while parent != archive_path.parent and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent


def test_gpx_source_reads_tracks_recursively_and_preserves_hierarchy(temp_dir):
    _write_gpx(
        temp_dir / "user-a" / "track.gpx",
        """
  <metadata>
    <name>Root name</name>
    <desc>Root description</desc>
    <time>2024-01-01T00:00:00Z</time>
    <keywords>walk,test</keywords>
    <bounds minlat="40.0" minlon="-118.0" maxlat="41.0" maxlon="-117.0" />
    <extensions><dataset>alpha</dataset></extensions>
  </metadata>
  <trk>
    <name>Morning track</name>
    <cmt>track comment</cmt>
    <desc>track description</desc>
    <src>watch</src>
    <link href="https://example.com/track"><text>track link</text></link>
    <number>7</number>
    <type>run</type>
    <extensions><trackCustom>yes</trackCustom></extensions>
    <trkseg>
      <extensions><segmentCustom>one</segmentCustom></extensions>
      <trkpt lat="40.0" lon="-118.0">
        <ele>10.5</ele>
        <time>2024-01-01T00:00:01Z</time>
        <name>start</name>
        <cmt>point comment</cmt>
        <desc>point description</desc>
        <src>gps</src>
        <sym>Dot</sym>
        <type>trail</type>
        <fix>3d</fix>
        <sat>8</sat>
        <hdop>1.25</hdop>
        <vdop>2.5</vdop>
        <pdop>3.75</pdop>
        <ageofdgpsdata>4.5</ageofdgpsdata>
        <dgpsid>12</dgpsid>
        <link href="https://example.com/point"><text>point link</text></link>
        <extensions><speed>5</speed></extensions>
      </trkpt>
      <trkpt lat="41.0" lon="-117.0">
        <ele>11.5</ele>
        <time>2024-01-01T00:00:02Z</time>
      </trkpt>
    </trkseg>
  </trk>
""",
    )
    _write_gpx(
        temp_dir / "user-b" / "route.GPX",
        """
  <rte>
    <name>Route name</name>
    <number>3</number>
    <type>bike</type>
    <rtept lat="35.0" lon="-120.0">
      <ele>15.0</ele>
      <time>2024-01-02T00:00:00Z</time>
    </rtept>
  </rte>
""",
    )

    source = GPXSource(str(temp_dir), batch_rows=1)
    splits = source.create_splits(num_splits=20)
    tables = list(source.iter_tables())
    table = pa.concat_tables(tables)

    assert all(isinstance(split, GPXSplit) for split in splits)
    assert len(splits) == 2
    assert len(tables) == 3
    assert table.num_rows == 3
    assert table["filename"].to_pylist() == ["track.gpx", "track.gpx", "route.GPX"]
    assert table["gpx_version"].to_pylist() == ["1.1", "1.1", "1.1"]
    assert table["gpx_creator"].to_pylist() == ["pytest", "pytest", "pytest"]
    assert table["gpx_name"].to_pylist()[0] == "Root name"
    assert json.loads(table["gpx_bounds"].to_pylist()[0]) == {
        "maxlat": "41.0",
        "maxlon": "-117.0",
        "minlat": "40.0",
        "minlon": "-118.0",
    }
    assert table["point_kind"].to_pylist() == ["track", "track", "route"]
    assert table["track_index"].to_pylist() == [0, 0, None]
    assert table["segment_index"].to_pylist() == [0, 0, None]
    assert table["route_index"].to_pylist() == [None, None, 0]
    assert table["point_index"].to_pylist() == [0, 1, 0]
    assert table["track_name"].to_pylist()[:2] == ["Morning track", "Morning track"]
    assert table["track_number"].to_pylist()[:2] == [7, 7]
    assert table["route_name"].to_pylist()[2] == "Route name"
    assert table["route_number"].to_pylist()[2] == 3
    assert table["latitude"].to_pylist() == [40.0, 41.0, 35.0]
    assert table["longitude"].to_pylist() == [-118.0, -117.0, -120.0]
    assert table["elevation"].to_pylist() == [10.5, 11.5, 15.0]
    assert table["point_time"].to_pylist() == [
        "2024-01-01T00:00:01Z",
        "2024-01-01T00:00:02Z",
        "2024-01-02T00:00:00Z",
    ]
    assert table["point_name"].to_pylist()[0] == "start"
    assert table["point_satellites"].to_pylist()[0] == 8
    assert table["point_hdop"].to_pylist()[0] == pytest.approx(1.25)
    assert "speed" in table["point_extensions_xml"].to_pylist()[0]
    assert "segmentCustom" in table["segment_extensions_xml"].to_pylist()[0]
    assert json.loads(table["point_links"].to_pylist()[0]) == [
        {"href": "https://example.com/point", "text": "point link"}
    ]

    first_point = wkb.loads(table["geometry"][0].as_py())
    assert first_point.equals(Point(-118.0, 40.0))


def test_gpx_source_reads_waypoints_and_single_file_with_basename_id(temp_dir):
    path = temp_dir / "points.gpx"
    _write_gpx(
        path,
        """
  <wpt lat="34.0" lon="-118.25">
    <name>Standalone waypoint</name>
    <ele>100.0</ele>
  </wpt>
""",
    )

    source = source_for_path(str(path))
    table = next(source.iter_tables())

    assert isinstance(source, GPXSource)
    assert len(source.create_splits()) == 1
    assert table["filename"].to_pylist() == ["points.gpx"]
    assert table["point_kind"].to_pylist() == ["waypoint"]
    assert table["point_index"].to_pylist() == [0]
    assert table["point_name"].to_pylist() == ["Standalone waypoint"]


def test_source_for_path_detects_gpx_directory_and_custom_geometry(temp_dir):
    _write_gpx(
        temp_dir / "track.gpx",
        """
  <trk>
    <trkseg>
      <trkpt lat="40.0" lon="-118.0" />
    </trkseg>
  </trk>
""",
    )

    source = source_for_path(str(temp_dir), geom_col="geom")
    table = next(source.iter_tables())
    geo = json.loads(table.schema.metadata[b"geo"].decode("utf-8"))

    assert isinstance(source, GPXSource)
    assert table.column_names[-1] == "geom"
    assert geo["primary_column"] == "geom"
    assert geo["columns"]["geom"]["crs"] == "EPSG:4326"


def test_gpx_spatial_sample_uses_all_points_and_infers_compact_schema(temp_dir):
    _write_gpx(
        temp_dir / "track.gpx",
        """
  <trk>
    <name>Track name</name>
    <number>4</number>
    <trkseg>
      <trkpt lat="40.0" lon="-118.0">
        <ele>10.0</ele>
        <time>2024-01-01T00:00:00Z</time>
      </trkpt>
      <trkpt lat="41.0" lon="-117.0">
        <ele>20.0</ele>
        <time>2024-01-01T00:00:10Z</time>
      </trkpt>
    </trkseg>
  </trk>
""",
    )

    sample = read_spatial_sample(
        str(temp_dir),
        source_workers=1,
    )

    assert sample.total_seen == 2
    assert sample.total_sampled == 2
    assert sample.schema is not None
    assert sample.schema.names == [
        "filename",
        "gpx_version",
        "gpx_creator",
        "point_kind",
        "track_index",
        "track_name",
        "track_number",
        "segment_index",
        "point_index",
        "latitude",
        "longitude",
        "elevation",
        "point_time",
        "geometry",
    ]

    source = source_for_path(str(temp_dir))
    source.set_schema(sample.schema)
    table = next(source.iter_tables())

    assert table.column_names == sample.schema.names
    assert "gpx_name" not in table.column_names
    assert "track_comment" not in table.column_names
    assert "route_index" not in table.column_names
    assert "point_name" not in table.column_names


def test_gpx_tar_spatial_sample_applies_cap_across_members(temp_dir):
    archive_path = temp_dir / "tracks.tar"

    def gpx_member(start: int) -> str:
        points = "\n".join(
            f'<trkpt lat="{40.0 + start + index}" lon="{-118.0 - start - index}" />'
            for index in range(3)
        )
        return (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            "<gpx version=\"1.1\" creator=\"pytest\">\n"
            "<trk><trkseg>\n"
            f"{points}\n"
            "</trkseg></trk>\n"
            "</gpx>\n"
        )

    _write_tar(
        archive_path,
        {
            "a.gpx": gpx_member(0),
            "b.gpx": gpx_member(10),
            "c.gpx": gpx_member(20),
        },
    )

    sample = read_spatial_sample(
        str(archive_path),
        sample_cap=3,
        source_workers=1,
    )

    assert sample.total_seen == 9
    assert sample.total_sampled == 3
    assert sample.sample_points.shape == (2, 3)


def test_gpx_source_reports_context_for_invalid_points(temp_dir):
    _write_gpx(
        temp_dir / "broken.gpx",
        """
  <trk>
    <trkseg>
      <trkpt lat="40.0" />
    </trkseg>
  </trk>
""",
    )
    source = GPXSource(str(temp_dir))

    with pytest.raises(ValueError, match=r"broken\.gpx .*missing 'lon'"):
        list(source.iter_tables())


def test_gpx_source_reads_tar_members_across_splits(temp_dir, monkeypatch):
    import starlet._internal.tiling.datasource as datasource_module

    archive_path = temp_dir / "tracks.tar"
    members = {
        "user-a/track1.gpx": (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            "<gpx xmlns=\"http://www.topografix.com/GPX/1.1\" version=\"1.1\" creator=\"pytest\">\n"
            "<trk><trkseg>"
            + "".join(
                f"<trkpt lat=\"40.{i:06d}\" lon=\"-118.{i:06d}\"><time>2024-01-01T00:00:{i:02d}Z</time></trkpt>"
                for i in range(10)
            )
            + "</trkseg></trk></gpx>\n"
        ),
        "user-b/route2.GPX": (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            "<gpx xmlns=\"http://www.topografix.com/GPX/1.1\" version=\"1.1\" creator=\"pytest\">\n"
            "<rte>"
            + "".join(
                f"<rtept lat=\"41.{i:06d}\" lon=\"-117.{i:06d}\"><time>2024-01-02T00:00:{i:02d}Z</time></rtept>"
                for i in range(10)
            )
            + "</rte></gpx>\n"
        ),
        "user-c/points.gpx": (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            "<gpx xmlns=\"http://www.topografix.com/GPX/1.1\" version=\"1.1\" creator=\"pytest\">\n"
            + "".join(
                f"<wpt lat=\"42.{i:06d}\" lon=\"-116.{i:06d}\"><name>p{i}</name></wpt>"
                for i in range(10)
            )
            + "</gpx>\n"
        ),
    }
    _write_tar(archive_path, members)
    monkeypatch.setattr(datasource_module, "_TAR_SPLIT_SIZE", 2048)

    source = source_for_path(str(archive_path))
    table = pa.concat_tables(list(source.iter_tables()))

    assert isinstance(source, GPXSource)
    assert len(source.create_splits()) >= 2
    assert table.num_rows == 30
    assert sorted(set(table["filename"].to_pylist())) == ["points.gpx", "route2.GPX", "track1.gpx"]

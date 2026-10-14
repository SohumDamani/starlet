"""Tests for the standalone intermediate vector tile helper."""

from datetime import date, datetime, time

import mapbox_vector_tile
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from starlet._internal.mvt.intermediate_tile import IntermediateVectorTile


def _big(x, size=1_000_000.0):
    """A polygon far larger than one display pixel at z0 (cell ~ 78 km)."""
    return box(x, 0.0, x + size, size)


def _mercator_from_tile_pixel(tile, x, y):
    x_scale, _, _, y_scale, xoff, yoff = tile.affine_params
    return ((x - xoff) / x_scale, (y - yoff) / y_scale)


def test_initializes_web_mercator_to_tile_pixel_transform():
    tile = IntermediateVectorTile(0, 0, 0)

    transformed = tile.simplify_geometry(Point(0, 0))

    assert len(transformed) == 1
    assert transformed[0].x == pytest.approx(2048.0)
    assert transformed[0].y == pytest.approx(2048.0)


def test_simplify_geometry_trims_lines_to_buffered_tile_bounds():
    tile = IntermediateVectorTile(0, 0, 0)
    left = _mercator_from_tile_pixel(tile, -1000, -1000)
    right = _mercator_from_tile_pixel(tile, 5000, 5000)

    transformed = tile.simplify_geometry(LineString([left, right]))

    assert len(transformed) == 1
    assert list(transformed[0].coords) == [(-256.0, -256.0), (4352.0, 4352.0)]


def test_simplify_geometry_clips_containing_polygon_to_tile_ring():
    tile = IntermediateVectorTile(0, 0, 0)
    corners = [
        _mercator_from_tile_pixel(tile, -1000, -1000),
        _mercator_from_tile_pixel(tile, -1000, 5000),
        _mercator_from_tile_pixel(tile, 5000, 5000),
        _mercator_from_tile_pixel(tile, 5000, -1000),
        _mercator_from_tile_pixel(tile, -1000, -1000),
    ]

    transformed = tile.simplify_geometry(Polygon(corners))

    assert len(transformed) == 1
    assert transformed[0].bounds == (-256.0, -256.0, 4352.0, 4352.0)


def test_add_feature_filters_null_properties():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)

    assert tile.add_feature(Point(0, 0), {"id": 1, "name": None})

    assert tile.feature_count == 1
    assert tile._features[0].properties == {"id": 1}


def test_add_feature_delays_simplification_until_features_are_requested():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    called = False
    original_simplify_geometry = tile.simplify_geometry

    def fail_if_called(geometry):
        nonlocal called
        called = True
        raise AssertionError("add_feature should only sample raw geometries")

    tile.simplify_geometry = fail_if_called
    assert tile.add_feature(Point(0, 0), {"id": 1})
    assert tile._features[0].properties == {"id": 1}
    assert not called

    tile.simplify_geometry = original_simplify_geometry
    assert mapbox_vector_tile.decode(tile.encode())["layer0"]["features"][0]["properties"] == {"id": 1}


def test_encode_flattens_nested_properties_for_mvt_only():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    properties = {
        "id": 1,
        "tagsMap": {"name": "Oak Hill", "landuse": "cemetery"},
        "metadata": {"source": "osm", "nested": {"rank": 2}},
    }

    assert tile.add_feature(Point(0, 0), properties)
    assert tile._features[0].properties == properties

    decoded = mapbox_vector_tile.decode(tile.encode())["layer0"]["features"][0]["properties"]
    assert decoded == {
        "id": 1,
        "tagsMap.name": "Oak Hill",
        "tagsMap.landuse": "cemetery",
        "metadata.source": "osm",
        "metadata.nested.rank": 2,
    }


def test_encode_flattens_loaded_arrow_map_properties_for_mvt_only(tmp_path):
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    properties = {
        "tagsMap": [("name", "Oak Hill"), ("landuse", "cemetery")],
    }

    assert tile.add_feature(Point(0, 0), properties)
    path = tmp_path / "tile.arrow"
    tile.write_features(path)

    loaded = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    loaded.load_features(path)
    assert loaded._features[0].properties == {
        "tagsMap": [["name", "Oak Hill"], ["landuse", "cemetery"]]
    }

    decoded = mapbox_vector_tile.decode(loaded.encode())["layer0"]["features"][0]["properties"]
    assert decoded == {
        "tagsMap.name": "Oak Hill",
        "tagsMap.landuse": "cemetery",
    }


def test_feature_arrow_roundtrip_serializes_date_like_properties(tmp_path):
    path = tmp_path / "0-0-0.pyarrow"
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    tile.add_feature(
        Point(0, 0),
        {
            "created": date(2026, 7, 26),
            "updated": datetime(2026, 7, 26, 16, 34, 22),
            "clock": time(16, 34, 22),
        },
        priority=9,
    )

    tile.write_features(path)

    loaded = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    loaded.load_features(path)

    assert loaded._features[0].properties == {
        "created": "2026-07-26",
        "updated": "2026-07-26T16:34:22",
        "clock": "16:34:22",
    }



def test_feature_can_be_skipped_by_lower_priority():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    assert tile.add_feature(_big(0), {"id": 1}, priority=10)

    called = False

    def fail_if_called(geometry):
        nonlocal called
        called = True
        raise AssertionError("unsampled feature should skip processing")

    tile.simplify_geometry = fail_if_called

    # a lower-priority feature does not enter the top-k; it survives only as
    # a dot, and nothing is simplified at offer time
    assert tile.add_feature(_big(3e6), {"id": 2}, priority=5)
    assert [f.properties["id"] for f in tile.full_features] == [1]
    assert [f.properties["id"] for f in tile.dot_features] == [2]
    assert not called


def test_feature_capacity_evicts_lowest_priority_to_make_room():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=2)

    assert tile.add_feature(_big(-3e6), {"id": 1}, priority=1)
    assert tile.add_feature(_big(0), {"id": 2}, priority=5)
    assert tile.add_feature(_big(3e6), {"id": 3}, priority=9)

    # the two best stay in full; the evicted one survives as a dot
    assert {feature.properties["id"] for feature in tile.full_features} == {2, 3}
    assert {feature.properties["id"] for feature in tile.dot_features} == {1}
    assert tile.feature_count == 3
    assert tile._features_seen == 3


def test_merge_combines_same_tile_without_simplifying_again():
    left = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    right = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    left.add_feature(_big(-3e6), {"id": 1}, priority=1)
    right.add_feature(_big(0), {"id": 2}, priority=5)
    right.add_feature(_big(3e6), {"id": 3}, priority=9)

    def fail_if_called(geometry):
        raise AssertionError("merge should not simplify geometries")

    original_simplify_geometry = left.simplify_geometry
    original_add_feature = left.add_feature
    left.simplify_geometry = fail_if_called
    left.add_feature = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("merge should combine feature lists directly")
    )

    left.merge(right)

    left.simplify_geometry = original_simplify_geometry
    left.add_feature = original_add_feature
    assert {feature.properties["id"] for feature in left.full_features} == {2, 3}
    assert {feature.properties["id"] for feature in left.dot_features} == {1}
    assert left._features_seen == 3


def test_merge_rejects_different_tile_ids():
    left = IntermediateVectorTile(0, 0, 0)
    right = IntermediateVectorTile(1, 0, 0)

    with pytest.raises(ValueError):
        left.merge(right)


def test_encode_returns_valid_mvt_binary():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    tile.add_feature(Point(0, 0), {"id": 1})

    decoded = mapbox_vector_tile.decode(tile.encode())

    assert "layer0" in decoded
    assert len(decoded["layer0"]["features"]) == 1
    assert decoded["layer0"]["features"][0]["properties"]["id"] == 1


def test_encode_accepts_layer_name():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    tile.add_feature(Point(0, 0), {"id": 1})

    decoded = mapbox_vector_tile.decode(tile.encode(layer_name="custom"))

    assert "custom" in decoded


def test_feature_arrow_roundtrip_populates_tile_state(tmp_path):
    path = tmp_path / "0-0-0.pyarrow"
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    tile.add_feature(_big(0), {"id": 1}, priority=9)
    tile.add_feature(_big(3e6), {"id": 2}, priority=1)
    tile.write_features(path)

    loaded = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    loaded.load_features(path)

    # both survive (one in full, the evicted one as a dot) and reload as such
    assert loaded.feature_count == 2
    assert loaded._features_seen == 2
    assert [f.properties for f in loaded.full_features] == [{"id": 1}, {"id": 2}]
    # Priorities survive the roundtrip so reduce-side merges rank correctly.
    assert max(p for p, _, _ in loaded._heap) & 0xFFFFFFFF == 9


def test_same_geometry_gets_same_default_priority_in_every_tile():
    """A geometry offered to two different tiles must win or lose in both."""
    from starlet._internal.mvt.intermediate_tile import feature_priority

    shared = Point(30, 30).buffer(10.0)
    assert feature_priority(shared.wkb) == feature_priority(shared.wkb)

    left = IntermediateVectorTile(1, 0, 0, feature_capacity=1)
    right = IntermediateVectorTile(1, 1, 0, feature_capacity=1)
    for tile in (left, right):
        tile.add_feature(shared, {"id": "shared"})

    competitor = Point(31, 31).buffer(5.0)
    left_kept = left.add_feature(competitor, {"id": "competitor"})
    right_kept = right.add_feature(competitor, {"id": "competitor"})

    # Identical candidates -> identical decision in both tiles (no seams).
    assert left_kept == right_kept
    left_ids = {feature.properties["id"] for feature in left._features}
    right_ids = {feature.properties["id"] for feature in right._features}
    assert left_ids == right_ids


def test_merge_is_order_independent():
    a = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    b = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    c = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    a.add_feature(_big(-3e6), {"id": 1}, priority=3)
    b.add_feature(_big(0), {"id": 2}, priority=7)
    c.add_feature(_big(3e6), {"id": 3}, priority=5)

    def merged_ids(order):
        first = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
        for src in order:
            clone = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
            for prio, _, feat, _ in src._entries():
                clone.add_feature(feat.geometry, dict(feat.properties), priority=prio)
            first.merge(clone)
        return ({f.properties["id"] for f in first.full_features}, {f.properties["id"] for f in first.dot_features})

    assert merged_ids([a, b, c]) == merged_ids([c, b, a]) == ({2, 3}, {1})


def test_sub_pixel_features_keep_one_per_pixel_cell_regardless_of_capacity():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    # three points inside one display pixel (cell ~ 78 km at z0; the origin
    # itself sits on a cell boundary, so stay clear of it): best priority wins
    assert tile.add_feature(Point(1000, 1000), {"id": 1}, priority=3)
    assert tile.add_feature(Point(1100, 1100), {"id": 2}, priority=9)
    assert not tile.add_feature(Point(1200, 1200), {"id": 3}, priority=5)
    assert {f.properties["id"] for f in tile._features} == {2}
    # points in other pixels are kept even though the capacity is 1
    assert tile.add_feature(Point(3e6, 0), {"id": 4}, priority=1)
    assert tile.add_feature(Point(-3e6, 0), {"id": 5}, priority=1)
    assert tile.feature_count == 3
    # merge keeps the best of each cell
    other = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    other.add_feature(Point(1050, 1050), {"id": 6}, priority=20)
    tile.merge(other)
    assert {f.properties["id"] for f in tile._features} == {6, 4, 5}


def test_sub_pixel_polygon_encodes_as_one_pixel_square_without_attributes():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    tile.add_feature(Point(1e6, 1e6).buffer(10.0), {"id": 1})   # 20 m across
    tile.add_feature(LineString([(2e6, 0), (2e6 + 10, 10)]), {"id": 2})
    tile.add_feature(_big(-5e6), {"id": 3})
    decoded = mapbox_vector_tile.decode(tile.encode())["layer0"]["features"]
    by_type = {f["geometry"]["type"]: f for f in decoded}
    assert set(by_type) == {"Polygon", "LineString"} or len(decoded) == 3
    small = [f for f in decoded if f["properties"] == {}]
    assert len(small) == 2  # the dots carry no attributes (one packed feature per kind)
    kinds = sorted(f["geometry"]["type"] for f in small)
    assert kinds == ["LineString", "Polygon"]  # ... but keep their geometry type
    sq = next(f for f in small if f["geometry"]["type"] == "Polygon")
    xs = [c[0] for c in sq["geometry"]["coordinates"][0]]
    assert max(xs) - min(xs) == tile.cell
    big = next(f for f in decoded if f["properties"])
    assert big["properties"] == {"id": 3}


def test_dense_sub_pixel_points_drop_attributes_beyond_capacity():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    tile.add_feature(Point(0, 0), {"id": 1})
    tile.add_feature(Point(3e6, 0), {"id": 2})
    decoded = mapbox_vector_tile.decode(tile.encode())["layer0"]["features"]
    assert len(decoded) == 2 and all(f["properties"] == {} for f in decoded)
    roomy = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    roomy.add_feature(Point(0, 0), {"id": 1})
    roomy.add_feature(Point(3e6, 0), {"id": 2})
    decoded = mapbox_vector_tile.decode(roomy.encode())["layer0"]["features"]
    assert sorted(f["properties"]["id"] for f in decoded) == [1, 2]


def test_tile_attributes_policy_filters_encoded_properties():
    from starlet._internal.mvt.intermediate_tile import normalize_tile_attributes

    assert normalize_tile_attributes(None) is None and normalize_tile_attributes("all") is None
    assert normalize_tile_attributes("none") == [] and normalize_tile_attributes("a, b") == ["a", "b"]
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10, tile_attributes="name")
    tile.add_feature(_big(0), {"id": 1, "name": "x"})
    decoded = mapbox_vector_tile.decode(tile.encode())["layer0"]["features"]
    assert decoded[0]["properties"] == {"name": "x"}


def test_size_outranks_hash_for_features_larger_than_a_pixel():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    assert tile.add_feature(_big(0, size=5e5), {"id": "small"}, priority=2**32 - 1)
    assert tile.add_feature(_big(3e6, size=5e6), {"id": "big"}, priority=0)
    assert [f.properties["id"] for f in tile.full_features] == ["big"]
    assert [f.properties["id"] for f in tile.dot_features] == ["small"]
    decoded = mapbox_vector_tile.decode(tile.encode())["layer0"]["features"]
    dot = next(f for f in decoded if not f["properties"])
    xs = [c[0] for c in dot["geometry"]["coordinates"][0]]
    assert max(xs) - min(xs) == tile.cell  # demoted feature drawn as a one-pixel square


def test_dots_are_packed_into_one_feature_per_kind():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    for i in range(5):
        tile.add_feature(Point(1e6 + i * 3e5, 1e6).buffer(10.0), {"id": i})
    decoded = mapbox_vector_tile.decode(tile.encode())["layer0"]["features"]
    assert len(decoded) == 1
    assert decoded[0]["geometry"]["type"] == "MultiPolygon"
    assert len(decoded[0]["geometry"]["coordinates"]) == 5

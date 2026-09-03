"""Tests for the standalone intermediate vector tile helper."""

from pathlib import Path

import mapbox_vector_tile
import pytest
from shapely.geometry import LineString, Point, Polygon

from starlet._internal.mvt.intermediate_tile import (
    DEFAULT_SPARSIFY_BUDGET_BYTES,
    IntermediateVectorTile,
    SolverResult,
)
from starlet._internal.mvt.mvt_generator import _iter_web_mercator_features
from starlet._internal.tiling.geojson_source import GeoJSONSource
from starlet._internal.tiling.geoparquet_source import GeoParquetSource

_BENCHMARK_DATA = Path(__file__).resolve().parents[2] / "benchmark_data"


def _load_parquet_sample(filename: str, n: int):
    """First ``n`` rows of a real benchmark_data parquet file, extracted via
    the same ``_iter_web_mercator_features`` the production map-phase uses
    (mvt_generator.py) -- CRS detection, WKB decode, reprojection to Web
    Mercator, and attribute typing are all real pipeline code, not
    reimplemented here. If that extraction logic changes, this test tracks
    it automatically instead of silently drifting out of sync.
    """
    source = GeoParquetSource(str(_BENCHMARK_DATA / filename))
    table = next(iter(source.iter_tables())).slice(0, n)
    return list(_iter_web_mercator_features(table, source.geom_col))


def _load_asia_postal_codes_sample(n: int):
    return _load_parquet_sample("asia_postal_codes.parquet", n)


def _load_geojson_sample(filename: str, n: int):
    """First ``n`` features of a real benchmark_data GeoJSON file, via the
    same real-pipeline functions as ``_load_parquet_sample`` -- GeoJSONSource
    always uses geom_col="geometry" (see geojson_source.py).
    """
    source = GeoJSONSource(str(_BENCHMARK_DATA / filename))
    features = []
    for table in source.iter_tables():
        for geom, attrs, priority in _iter_web_mercator_features(table, "geometry"):
            features.append((geom, attrs, priority))
            if len(features) >= n:
                return features
    return features


@pytest.fixture
def real_postal_codes_tile():
    """A tile pre-loaded with 300 real asia_postal_codes features, capacity
    set well above 300 so nothing gets evicted -- isolates triage() from the
    separate admission-sampling behavior tested elsewhere in this file.
    Uses the real per-feature priority from the extraction pipeline (not
    that it matters here: capacity >> feature count, so nothing competes).
    """
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1000)
    for geom, props, priority in _load_asia_postal_codes_sample(300):
        tile.add_feature(geom, props, priority=priority)
    return tile


def _build_real_tile(dataset: str, n: int) -> IntermediateVectorTile:
    """Build a tile from the first ``n`` real features of any benchmark_data
    file, parquet or GeoJSON, auto-detected by extension. Dataset and scale
    are both plain arguments here (not baked into a fixture) so solver
    comparisons across datasets/sizes -- the whole point of Phase 1 onward --
    are just a matter of passing different arguments, not writing new setup.
    Capacity is exactly n: add_feature() only ever evicts when the heap is
    already full AND a higher-priority feature arrives, so loading exactly n
    features into a capacity-n tile never triggers eviction -- every feature
    lands while the heap still has room. No extra buffer is needed.
    """
    if dataset.endswith(".parquet"):
        sample = _load_parquet_sample(dataset, n)
    else:
        sample = _load_geojson_sample(dataset, n)
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=n)
    for geom, props, priority in sample:
        tile.add_feature(geom, props, priority=priority)
    return tile


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


def test_feature_can_be_skipped_by_lower_priority():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    assert tile.add_feature(Point(0, 0), {"id": 1}, priority=10)

    called = False

    def fail_if_called(geometry):
        nonlocal called
        called = True
        raise AssertionError("unsampled feature should skip processing")

    tile.simplify_geometry = fail_if_called

    assert not tile.add_feature(Point(1000, 0), {"id": 2}, priority=5)
    assert not called


def test_feature_capacity_evicts_lowest_priority_to_make_room():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=2)

    assert tile.add_feature(Point(-1000, 0), {"id": 1}, priority=1)
    assert tile.add_feature(Point(0, 0), {"id": 2}, priority=5)
    assert tile.add_feature(Point(1000, 0), {"id": 3}, priority=9)

    retained_ids = {feature.properties["id"] for feature in tile._features}
    assert retained_ids == {2, 3}
    assert tile.feature_count == 2
    assert tile._features_seen == 3


def test_merge_combines_same_tile_without_simplifying_again():
    left = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    right = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    left.add_feature(Point(-1000, 0), {"id": 1}, priority=1)
    right.add_feature(Point(0, 0), {"id": 2}, priority=5)
    right.add_feature(Point(1000, 0), {"id": 3}, priority=9)

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
    retained_ids = {feature.properties["id"] for feature in left._features}
    assert retained_ids == {2, 3}
    assert left.feature_count == 2
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
    tile.add_feature(Point(0, 0), {"id": 1}, priority=9)
    tile.add_feature(Point(1, 1), {"id": 2}, priority=1)
    tile.write_features(path)

    loaded = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    loaded.load_features(path)

    assert loaded.feature_count == 1
    assert loaded._features_seen == 2
    assert loaded._features[0].properties == {"id": 1}
    # Priorities survive the roundtrip so reduce-side merges rank correctly.
    assert loaded._heap[0][0] == 9


def test_same_geometry_gets_same_default_priority_in_every_tile():
    """A geometry offered to two different tiles must win or lose in both."""
    from starlet._internal.mvt.intermediate_tile import feature_priority

    shared = Point(0, 0).buffer(10.0)
    assert feature_priority(shared.wkb) == feature_priority(shared.wkb)

    left = IntermediateVectorTile(1, 0, 0, feature_capacity=1)
    right = IntermediateVectorTile(1, 1, 0, feature_capacity=1)
    for tile in (left, right):
        tile.add_feature(shared, {"id": "shared"})

    competitor = Point(1, 1).buffer(5.0)
    left_kept = left.add_feature(competitor, {"id": "competitor"})
    right_kept = right.add_feature(competitor, {"id": "competitor"})

    # Identical candidates -> identical decision in both tiles (no seams).
    assert left_kept == right_kept
    left_ids = {feature.properties["id"] for feature in left._features}
    right_ids = {feature.properties["id"] for feature in right._features}
    assert left_ids == right_ids


def test_triage_numeric_quantization_on_real_data(real_postal_codes_tile):
    """Live understanding-test for HiFIVE Sec 6.2.1 (numeric quantization),
    run against a real slice of asia_postal_codes.parquet.

    Isolates the numeric half of triage() via numeric_only=True so the
    string column (uname) passes through untouched -- this test is only
    about what happens to version/timestamp/changeSetId/uid.
    """
    tile = real_postal_codes_tile
    assert tile.feature_count == 300  # capacity > n, nothing evicted

    numeric_cols = ["version", "timestamp", "changeSetId", "uid"]

    def snapshot():
        stats = tile._column_stats(tile._heap)
        distinct = {
            col: len({feature.properties[col] for _, _, feature in tile._heap})
            for col in numeric_cols
        }
        samples = {
            col: [feature.properties[col] for _, _, feature in list(tile._heap)[:8]]
            for col in numeric_cols
        }
        geoms = [feature.geometry.wkb for _, _, feature in tile._heap]
        return stats, distinct, samples, geoms

    before_stats, before_distinct, before_samples, before_geoms = snapshot()

    tile.triage(numeric_only=True)

    after_stats, after_distinct, after_samples, after_geoms = snapshot()

    print("\n=== Numeric quantization on real data (asia_postal_codes, n=300) ===")
    print(f"{'column':<14}{'distinct before':<18}{'distinct after':<16}"
          f"{'dict bytes before':<20}{'dict bytes after'}")
    for col in numeric_cols:
        print(
            f"{col:<14}{before_distinct[col]:<18}{after_distinct[col]:<16}"
            f"{before_stats[col]['dict_bytes']:<20.1f}{after_stats[col]['dict_bytes']:.1f}"
        )

    print("\nSample uid values, first 8 features:")
    print("  before:", before_samples["uid"])
    print("  after: ", after_samples["uid"])
    print("\nSample timestamp values, first 8 features:")
    print("  before:", before_samples["timestamp"])
    print("  after: ", after_samples["timestamp"])

    # feature count and geometries must be untouched -- triage only rewrites
    # attribute values, never drops records or moves geometry (that's sparsify's job).
    assert tile.feature_count == 300
    assert after_geoms == before_geoms

    # uname (string) must be untouched by numeric_only=True.
    assert before_stats["uname"]["dict_bytes"] == after_stats["uname"]["dict_bytes"]

    for col in numeric_cols:
        # After equal-width binning into <=10 bins, distinct values and dict
        # bytes should shrink (unless the column already had <=10 distinct
        # values, in which case triage leaves it alone).
        assert after_distinct[col] <= max(10, before_distinct[col])
        assert after_stats[col]["dict_bytes"] <= before_stats[col]["dict_bytes"]


def test_triage_string_prefix_on_real_data(real_postal_codes_tile):
    """Live understanding-test for HiFIVE Sec 6.2.2 (string prefix triage),
    run against the same real slice of asia_postal_codes.parquet.

    Isolates the string half of triage() via string_only=True so the numeric
    columns (control group) must come out byte-identical.
    """
    tile = real_postal_codes_tile

    numeric_cols = ["version", "timestamp", "changeSetId", "uid"]

    def snapshot():
        stats = tile._column_stats(tile._heap)
        uname_distinct = {feature.properties["uname"] for _, _, feature in tile._heap}
        return stats, uname_distinct

    before_stats, before_unames = snapshot()

    tile.triage(string_only=True)

    after_stats, after_unames = snapshot()

    print("\n=== String prefix triage on real data (asia_postal_codes, n=300) ===")
    print(f"distinct uname before: {len(before_unames)}   after: {len(after_unames)}")
    print(f"dict bytes before: {before_stats['uname']['dict_bytes']:.1f}   "
          f"after: {after_stats['uname']['dict_bytes']:.1f}")

    # NOTE: distinct-count barely moved (see docstring) -- most of the byte
    # saving here comes from shortening every value's stored length, not
    # from merging values together. Only true collisions (values that now
    # share a prefix) actually reduce the dictionary's entry count.
    true_merges = len(before_unames) - len(after_unames)
    print(f"\nTrue merges (distinct values that became identical): {true_merges}")
    print(f"Remaining distinct prefixes after triage: {len(after_unames)}")

    # feature count and geometries untouched.
    assert tile.feature_count == 300

    # Numeric columns are the control group: string_only=True must leave
    # them byte-identical.
    for col in numeric_cols:
        assert before_stats[col]["dict_bytes"] == after_stats[col]["dict_bytes"]

    # uname must have shrunk (or stayed the same if no safe merge existed).
    assert after_stats["uname"]["dict_bytes"] <= before_stats["uname"]["dict_bytes"]
    assert len(after_unames) <= len(before_unames)


def test_triage_string_prefix_on_non_json_tag_blob():
    """Edge-case test for HiFIVE Sec 6.2.2's JSON-column guard (_is_json_column).

    OSM2015_parks.parquet's '$2' column stores tags as '[key#value,...]' --
    real attribute redundancy (many rows share exact values like
    '[landuse#forest]'), but NOT valid JSON, so _is_json_column's json.loads()
    check does not skip it (unlike asia_postal_codes' tagsMap, which IS valid
    JSON and correctly gets skipped -- see the numeric/string tests above).

    This means string-prefix triage runs on it like an ordinary text column.
    Live-computed prediction (see conversation): conditional entropy does not
    drop under the default 0.1-bit threshold until prefix length 90, and
    values run up to 343 characters -- so triage DOES truncate here, cutting
    long structured tag lists mid-key/mid-value rather than at a semantic
    boundary. This is the same class of corruption commit 47b6d1f fixed for
    JSON, just for a serialization format the current guard doesn't recognize.
    """
    features = _load_parquet_sample("OSM2015_parks.parquet", 2000)

    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=5000)
    for geom, props, priority in features:
        tile.add_feature(geom, props, priority=priority)
    assert tile.feature_count == 2000

    def distinct_and_dict_bytes():
        return (
            {feature.properties.get("$2") for _, _, feature in tile._heap},
            tile._column_stats(tile._heap),
        )

    before_values, before_stats = distinct_and_dict_bytes()

    # Track ONE specific feature (by priority, which triage never changes)
    # across the before/after snapshots, so we compare the same record's
    # value to itself rather than independently re-picking "the longest"
    # from each snapshot (which could pick two different features if
    # multiple long values happen to tie after truncation).
    target_priority, _, target_feature = max(
        tile._heap, key=lambda entry: len(entry[2].properties.get("$2", ""))
    )
    before_longest = target_feature.properties["$2"]

    tile.triage(string_only=True)

    after_values, after_stats = distinct_and_dict_bytes()
    after_longest = next(
        feature.properties["$2"]
        for priority, _, feature in tile._heap
        if priority == target_priority
    )

    print("\n=== String prefix triage on OSM2015_parks '$2' tag blob (n=2000) ===")
    print(f"distinct before: {len(before_values)}   after: {len(after_values)}")
    print(f"dict bytes before: {before_stats['$2']['dict_bytes']:.1f}   "
          f"after: {after_stats['$2']['dict_bytes']:.1f}")
    print(f"\nLongest original value ({len(before_longest)} chars):")
    print(" ", before_longest)
    print(f"Same feature's value after triage ({len(after_longest)} chars):")
    print(" ", after_longest)

    assert tile.feature_count == 2000

    # The whole point: this long value should have been cut mid-content, not
    # left alone -- proving the JSON guard's blind spot actually engages
    # triage's truncation, unlike tagsMap in the other tests.
    assert len(before_longest) > 90
    assert len(after_longest) <= 90
    assert after_longest != before_longest
    # And it's a real prefix cut, not a clean re-encoding -- the tail content
    # (whatever came after char 90) is simply gone.
    assert before_longest.startswith(after_longest)

    assert after_stats["$2"]["dict_bytes"] <= before_stats["$2"]["dict_bytes"]


def test_triage_budget_aware_skips_when_already_under_budget(real_postal_codes_tile):
    """Live understanding-test for the new budget_bytes param (HiFIVE Sec 6.2.3).

    Real unreduced size for this 300-feature asia_postal_codes tile is
    ~114,653 bytes (measured live before writing this test). Giving triage()
    a generous 200,000-byte budget -- well above that -- should make it do
    NOTHING at all: this is the actual gap-B bug fix, since the old
    unconditional triage() would still fully quantize/truncate every column
    regardless of whether the tile needed it.
    """
    tile = real_postal_codes_tile
    before_size = tile._estimate_size_bytes()
    before_props = [dict(feature.properties) for _, _, feature in tile._heap]

    tile.triage(budget_bytes=200_000)

    after_size = tile._estimate_size_bytes()
    after_props = [dict(feature.properties) for _, _, feature in tile._heap]

    print(f"\n=== Budget-aware triage, generous budget (n=300, budget=200,000) ===")
    print(f"size before: {before_size:.0f}   size after: {after_size:.0f}   "
          f"(tile was already under budget: {before_size <= 200_000})")

    assert before_size <= 200_000  # sanity: this scenario really is "already fits"
    assert after_props == before_props  # not a single value was touched
    assert after_size == before_size


def test_triage_budget_aware_stops_once_under_budget(real_postal_codes_tile):
    """Live understanding-test: with a real budget the tile doesn't already
    meet, budget-aware triage should touch FEWER columns than unconditional
    triage() would -- stopping the moment the running estimate drops under
    budget, rather than fully quantizing/truncating every candidate.
    """
    budget_bytes = 95_000

    tile_budget_aware = real_postal_codes_tile
    before_size = tile_budget_aware._estimate_size_bytes()
    assert before_size > budget_bytes  # sanity: real data actually needs reduction here

    tile_budget_aware.triage(budget_bytes=budget_bytes)
    after_size = tile_budget_aware._estimate_size_bytes()

    # A second, independent tile over the same real features, triaged the
    # OLD unconditional way (no budget), to compare how much MORE it touches.
    tile_unconditional = IntermediateVectorTile(0, 0, 0, feature_capacity=1000)
    for geom, props, priority in _load_asia_postal_codes_sample(300):
        tile_unconditional.add_feature(geom, props, priority=priority)
    tile_unconditional.triage()  # budget_bytes=None -- old unconditional behavior

    print(f"\n=== Budget-aware vs unconditional triage (n=300, budget={budget_bytes}) ===")
    print(f"budget-aware final size: {after_size:.0f}   budget: {budget_bytes}")

    assert after_size <= budget_bytes  # the budget was actually respected

    # Prove it stopped early rather than doing everything: compare the count
    # of *columns whose value set changed* relative to the untouched real
    # sample, per tile.
    original_sample = _load_asia_postal_codes_sample(300)
    original_values = {}
    for geom, props, priority in original_sample:
        for col, val in props.items():
            original_values.setdefault(col, set()).add(val)

    def columns_actually_modified(triaged_tile):
        modified = set()
        for col in original_values:
            current_values = {
                feature.properties[col]
                for _, _, feature in triaged_tile._heap
                if col in feature.properties
            }
            if current_values != original_values[col]:
                modified.add(col)
        return modified

    modified_budget_aware = columns_actually_modified(tile_budget_aware)
    modified_unconditional = columns_actually_modified(tile_unconditional)

    print(f"columns modified, budget-aware:   {sorted(modified_budget_aware)}")
    print(f"columns modified, unconditional:  {sorted(modified_unconditional)}")

    assert len(modified_budget_aware) < len(modified_unconditional)
    assert modified_budget_aware <= modified_unconditional


@pytest.mark.parametrize(
    "dataset, n",
    [
        ("asia_postal_codes.parquet", 2000),
        ("NE_states_provinces.geojson", 500),
    ],
)
def test_sparsify_scipy_adapter_returns_solver_result(dataset, n):
    """Live understanding-test for Phase 1's SolverResult refactor, at the
    HiFIVE paper's own default budget (256KB, Table 6).

    Parametrized over (dataset, n) rather than a fixed fixture -- adding a
    new dataset or scale to this test later is one line in the list above,
    not new setup code. Measured live for the default case (2000 real
    asia_postal_codes features): unreduced size ~1,070KB, about 4x the
    256KB default budget, so sparsify() has real, meaningful work to do.

    Confirms: the scipy adapter returns the shared SolverResult shape (not
    scipy's raw OptimizeResult), the reported variable/constraint counts
    match the paper's own Sec 5 "Solver cost" formula (N+d+non-null cells
    variables, 2*non-null cells+1 constraints), and sparsify() end-to-end
    still applies the decision correctly through the new shape -- including
    using the new default budget with no argument at all.
    """
    tile = _build_real_tile(dataset, n)
    before_count = tile.feature_count
    before_size = tile._estimate_size_bytes()
    assert before_size > DEFAULT_SPARSIFY_BUDGET_BYTES  # sanity: real work needed here

    problem = tile._build_sparsify_problem(DEFAULT_SPARSIFY_BUDGET_BYTES)
    result = tile._solve_sparsify_problem_scipy(problem)

    print(f"\n=== scipy adapter on real data ({dataset}, n={n}, "
          f"budget={DEFAULT_SPARSIFY_BUDGET_BYTES:,}) ===")
    print(f"unreduced size estimate: {before_size:,.0f} bytes")
    print(f"objective_value: {result.objective_value:.2f}")
    print(f"wall_time_seconds: {result.wall_time_seconds:.3f}")
    print(f"status: {result.status}")
    print(f"num_variables: {result.num_variables}   num_constraints: {result.num_constraints}")

    assert isinstance(result, SolverResult)
    assert result.status == "optimal"
    assert result.objective_value > 0
    assert result.wall_time_seconds > 0
    assert len(result.x) == result.num_variables

    # Cross-check against the paper's own Sec 5 "Solver cost" formula:
    # N + d + non-null cells variables, 2*non-null cells + 1 constraints.
    non_null_cells = len(problem.x_index)
    assert result.num_variables == len(problem.entries) + len(problem.columns) + non_null_cells
    assert result.num_constraints == 2 * non_null_cells + 1

    # End-to-end: sparsify() itself, called with NO budget argument at all,
    # must use the new 256KB default and still apply the decision correctly.
    tile.sparsify()
    after_count = tile.feature_count
    print(f"feature_count: {before_count} -> {after_count}")
    assert after_count < before_count
    assert tile._estimate_size_bytes() <= DEFAULT_SPARSIFY_BUDGET_BYTES


def _true_objective_and_bytes(problem, x):
    """Recompute the exact (unrounded) objective and byte cost for a given
    0/1 decision vector, using the ORIGINAL problem arrays -- never whatever
    rounded/scaled math a specific solver adapter used internally.

    This is the honesty check for any solver that has to approximate (like
    CP-SAT, which requires integers): does its real-world answer still hold
    up once its own internal rounding is stripped away?

    In plain terms: a solver like CP-SAT can only work with whole numbers,
    so before it starts, our real (fractional) scores and byte costs get
    rounded. That means whatever "score" the solver reports afterward is
    based on those rounded numbers, not the real ones. This function takes
    the solver's actual final choice (which things it decided to keep) and
    plugs that exact choice back into the ORIGINAL, un-rounded formulas --
    like re-adding up a receipt by hand instead of trusting a rounded total
    at the bottom. That tells us how much the rounding actually mattered.
    """
    objective = sum(problem.c[i] * x[i] for i in range(len(x)))
    last_row = problem.A_ub.shape[0] - 1
    A_ub_csr = problem.A_ub.tocsr()
    start, end = A_ub_csr.indptr[last_row], A_ub_csr.indptr[last_row + 1]
    bytes_used = sum(A_ub_csr.data[k] * x[A_ub_csr.indices[k]] for k in range(start, end))
    return objective, bytes_used


@pytest.mark.parametrize(
    "dataset, n",
    [
        ("asia_postal_codes.parquet", 2000),
    ],
)
def test_sparsify_cpsat_adapter_matches_scipy_closely(dataset, n):
    """Live understanding-test for the CP-SAT adapter (Phase 2).

    Solves the SAME real problem instance with both scipy and CP-SAT, then
    runs CP-SAT's answer through the honesty check above: its own internal
    rounding must not meaningfully distort the achieved score, and must
    never cause a REAL budget violation, even though CP-SAT solved against
    rounded byte costs internally.

    In plain terms: we give the exact same real-world problem to two
    different solver libraries and see how they each answer it. Then, for
    CP-SAT specifically, we double-check its answer using the real,
    un-rounded numbers (via _true_objective_and_bytes) to make sure the
    rounding it needed to even run didn't secretly cost us much accuracy,
    and -- most importantly -- didn't let it sneak past the real byte
    budget while only appearing to respect a rounded version of it.
    """
    tile = _build_real_tile(dataset, n)
    problem = tile._build_sparsify_problem(DEFAULT_SPARSIFY_BUDGET_BYTES)

    scipy_result = tile._solve_sparsify_problem_scipy(problem)
    cpsat_result = tile._solve_sparsify_problem_cpsat(problem)

    print(f"\n=== scipy vs CP-SAT on real data ({dataset}, n={n}, "
          f"budget={DEFAULT_SPARSIFY_BUDGET_BYTES:,}) ===")
    print(f"scipy:  objective={scipy_result.objective_value:.3f}  "
          f"time={scipy_result.wall_time_seconds:.3f}s  status={scipy_result.status}")
    print(f"cpsat:  objective={cpsat_result.objective_value:.3f}  "
          f"time={cpsat_result.wall_time_seconds:.3f}s  status={cpsat_result.status}")

    assert scipy_result.num_variables == cpsat_result.num_variables
    assert scipy_result.num_constraints == cpsat_result.num_constraints
    assert cpsat_result.status in ("optimal", "feasible")

    x_cpsat = [round(v) for v in cpsat_result.x]
    true_objective, true_bytes_used = _true_objective_and_bytes(problem, x_cpsat)

    print(f"\nCP-SAT reported objective (its own scaled/rounded math): "
          f"{cpsat_result.objective_value:.3f}")
    print(f"TRUE objective (exact floats, same decisions):          {true_objective:.3f}")
    print(f"TRUE byte cost used: {true_bytes_used:,.1f}   budget: {DEFAULT_SPARSIFY_BUDGET_BYTES:,}")

    # CP-SAT's internal rounding must not meaningfully distort the score...
    assert abs(cpsat_result.objective_value - true_objective) < 1.0
    # ...and must never cause a REAL budget violation, even though CP-SAT
    # solved against rounded byte costs internally.
    assert true_bytes_used <= DEFAULT_SPARSIFY_BUDGET_BYTES
    # And the two solvers' true achieved objectives should be close to each
    # other -- not identical (different rounding), but close.
    assert abs(scipy_result.objective_value - true_objective) / scipy_result.objective_value < 0.01


def test_merge_is_order_independent():
    a = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    b = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    c = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    a.add_feature(Point(-1000, 0), {"id": 1}, priority=3)
    b.add_feature(Point(0, 0), {"id": 2}, priority=7)
    c.add_feature(Point(1000, 0), {"id": 3}, priority=5)

    def merged_ids(order):
        first = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
        for src in order:
            clone = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
            for prio, _, feat in src._heap:
                clone.add_feature(feat.geometry, dict(feat.properties), priority=prio)
            first.merge(clone)
        return {feature.properties["id"] for feature in first._features}

    assert merged_ids([a, b, c]) == merged_ids([c, b, a]) == {2, 3}

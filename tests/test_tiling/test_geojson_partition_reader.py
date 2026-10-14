import json

from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader


def test_geojson_partition_reader_reads_features_across_variable_blocks():
    feature_count = 50
    payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i, "name": f"feature-{i}"},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(i), float(i * 2)],
                    },
                }
                for i in range(feature_count)
            ],
        },
        separators=(",", ":"),
    )

    for block_size in range(49, 76):
        payload_bytes = payload.encode("utf-8")
        blocks = (
            payload_bytes[offset:offset + block_size]
            for offset in range(0, len(payload_bytes), block_size)
        )

        features = [json.loads(feature) for feature in GeoJSONPartitionReader(blocks)]

        assert len(features) == feature_count
        assert [feature["properties"]["id"] for feature in features] == list(range(feature_count))
        assert all(feature["type"] == "Feature" for feature in features)


def test_geojson_partition_reader_reads_two_features_from_single_block():
    payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"id": 2},
                    "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
                },
            ],
        },
        separators=(",", ":"),
    )

    features = [
        json.loads(feature)
        for feature in GeoJSONPartitionReader(iter([payload.encode("utf-8")]))
    ]

    assert len(features) == 2
    assert [feature["properties"]["id"] for feature in features] == [1, 2]


def test_geojson_partition_reader_recovers_when_first_block_starts_inside_string():
    payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                }
            ],
        },
        separators=(",", ":"),
    )
    text = f'this is the tail of an earlier string"{payload}'.encode("utf-8")
    blocks = (text[offset:offset + 13] for offset in range(0, len(text), 13))

    features = [
        json.loads(feature)
        for feature in GeoJSONPartitionReader(blocks)
    ]

    assert len(features) == 1
    assert features[0]["properties"]["id"] == 1


def test_geojson_partition_reader_handles_string_tokens_split_across_blocks():
    payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "id": 1,
                        "name": 'a string with an escaped " quote',
                    },
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                }
            ],
        },
        separators=(",", ":"),
    )
    payload_bytes = payload.encode("utf-8")
    blocks = (
        payload_bytes[offset:offset + 3]
        for offset in range(0, len(payload_bytes), 3)
    )

    features = [json.loads(feature) for feature in GeoJSONPartitionReader(blocks)]

    assert len(features) == 1
    assert features[0]["properties"]["name"] == 'a string with an escaped " quote'


def test_geojson_partition_reader_can_read_partitioned_geojson_once():
    feature_count = 80
    block_size = 97
    payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "id": i,
                        "name": f"partition-feature-{i}",
                        "description": f"long enough to cross block boundaries {i}",
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(i), float(i * 3)],
                    },
                }
                for i in range(feature_count)
            ],
        },
        separators=(",", ":"),
    )
    payload_bytes = payload.encode("utf-8")
    blocks = [
        payload_bytes[offset:offset + block_size]
        for offset in range(0, len(payload_bytes), block_size)
    ]

    class TrackedBlockIterator:
        def __init__(self, start_block: int):
            self.next_block = start_block
            self.max_requested_block = start_block - 1
            self.sent_initial_block = False
            self.sent_partition_end = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.sent_initial_block and not self.sent_partition_end:
                self.sent_partition_end = True
                return b"\0"
            if self.next_block >= len(blocks):
                raise StopIteration
            block_index = self.next_block
            self.next_block += 1
            self.max_requested_block = block_index
            self.sent_initial_block = True
            return blocks[block_index]

    ids = []
    for partition_index in range(len(blocks)):
        tracked_blocks = TrackedBlockIterator(partition_index)
        for feature_text in GeoJSONPartitionReader(tracked_blocks):
            feature = json.loads(feature_text)
            ids.append(feature["properties"]["id"])
            if tracked_blocks.max_requested_block > partition_index:
                break

    assert sorted(ids) == list(range(feature_count))
    assert len(ids) == feature_count

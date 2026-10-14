"""Feature-aligned GeoJSON reader over byte blocks."""

from __future__ import annotations

from typing import Iterable, Iterator
import logging
import re

logger = logging.getLogger(__name__)

WHITESPACE = b" \t\r\n"
QUOTE = ord('"')
BACKSLASH = ord("\\")
OPEN_BRACE = ord("{")
CLOSE_BRACE = ord("}")
COMMA = ord(",")
# A Feature's required type member can appear anywhere in its object. The
# match identifies that member; _object_start_before finds the enclosing `{`.
FEATURE_TYPE = re.compile(rb'"type"\s*:\s*"Feature"', re.IGNORECASE)
PARTITION_END_BLOCK = b"\0"
MARKER_TAIL_KEEP = 1024


class GeoJSONPartitionReader:
    """Yield complete GeoJSON Feature strings from byte blocks.

    ``blocks`` is an iterator of bytes. A block containing exactly ``b"\0"``
    marks the nominal end of the split: Features whose object starts before
    that marker are emitted, and parsing continues beyond the marker only as
    far as needed to finish the last owned Feature.
    """

    def __init__(self, blocks: Iterable[bytes]):
        self._blocks = iter(blocks)

    def __iter__(self) -> Iterator[str]:
        for feature in self.iter_bytes():
            yield feature.decode("utf-8")

    def iter_bytes(self) -> Iterator[bytes]:
        """Yield Feature objects without paying for UTF-8 decoding."""
        buffer = bytearray()
        search_from = 0
        partition_end = -1
        stream_done = False

        def read_next_block() -> bool:
            nonlocal partition_end, stream_done
            while True:
                try:
                    block = next(self._blocks)
                except StopIteration:
                    stream_done = True
                    return False
                if not block:
                    continue
                block = bytes(block)
                if block == PARTITION_END_BLOCK:
                    if partition_end == -1:
                        partition_end = len(buffer)
                    continue
                buffer.extend(block)
                return True

        def drop_consumed(end: int) -> None:
            nonlocal search_from, partition_end
            if end <= 0:
                return
            del buffer[:end]
            search_from = max(0, search_from - end)
            if partition_end != -1:
                partition_end = max(0, partition_end - end)

        def trim_if_needed() -> None:
            nonlocal search_from, partition_end
            if search_from <= len(buffer) // 2:
                return
            drop_count = len(buffer) // 2
            if drop_count <= 0:
                return
            del buffer[:drop_count]
            search_from = 0
            if partition_end != -1:
                partition_end = max(0, partition_end - drop_count)

        def find_feature_start() -> int | None:
            nonlocal search_from
            start = self._next_feature_start(buffer, search_from, search_from)
            while start is None:
                search_from = max(search_from, max(0, len(buffer) - MARKER_TAIL_KEEP))
                trim_if_needed()
                if partition_end != -1 and search_from >= partition_end:
                    return None
                if stream_done or not read_next_block():
                    return None
                start = self._next_feature_start(buffer, search_from, search_from)
            return start

        current_start = find_feature_start()
        if current_start is None:
            return

        while True:
            if partition_end != -1 and current_start >= partition_end:
                return

            next_start = self._next_feature_start(
                buffer,
                current_start + 1,
                current_start + 1,
            )
            if next_start is None and not stream_done:
                # Emit a Feature that is already complete in the buffer before
                # reading ahead, so consumers get it without pulling the next block.
                try:
                    current_end = self._find_json_object_end(buffer, current_start)
                except ValueError:
                    current_end = None
                if current_end is not None:
                    yield bytes(buffer[current_start:current_end])
                    drop_consumed(current_end)
                    search_from = 0
                    current_start = find_feature_start()
                    if current_start is None:
                        return
                    continue
            while next_start is None and not stream_done:
                if not read_next_block():
                    break
                next_start = self._next_feature_start(
                    buffer,
                    current_start + 1,
                    current_start + 1,
                )

            if next_start is None:
                try:
                    current_end = self._find_json_object_end(buffer, current_start)
                except ValueError:
                    logger.warning("Unterminated GeoJSON Feature object")
                    return
                stop_after_current = True
            else:
                current_end = self._trim_feature_end(buffer, next_start)
                stop_after_current = partition_end != -1 and next_start >= partition_end

            yield bytes(buffer[current_start:current_end])

            if stop_after_current:
                return

            drop_consumed(current_end)
            current_start = next_start - current_end

    @staticmethod
    def _next_feature_start(
        data: bytearray | bytes,
        search_from: int,
        minimum_start: int,
    ) -> int | None:
        """Find the next Feature object beginning at or after ``minimum_start``."""
        candidate = FEATURE_TYPE.search(data, search_from)
        while candidate is not None:
            object_start = GeoJSONPartitionReader._object_start_before(
                data,
                candidate.start(),
            )
            if object_start is not None and object_start >= minimum_start:
                return object_start
            candidate = FEATURE_TYPE.search(data, candidate.end())
        return None

    @staticmethod
    def _object_start_before(data: bytearray | bytes, position: int) -> int | None:
        """Return the opening brace of the JSON object containing ``position``."""
        nesting = 0
        in_string = False
        position -= 1
        while position >= 0:
            byte = data[position]
            if byte == QUOTE and not GeoJSONPartitionReader._is_escaped(data, position):
                in_string = not in_string
            elif not in_string:
                if byte == CLOSE_BRACE:
                    nesting += 1
                elif byte == OPEN_BRACE:
                    if nesting == 0:
                        return position
                    nesting -= 1
            position -= 1
        return None

    @staticmethod
    def _is_escaped(data: bytearray | bytes, position: int) -> bool:
        backslashes = 0
        position -= 1
        while position >= 0 and data[position] == BACKSLASH:
            backslashes += 1
            position -= 1
        return backslashes % 2 == 1

    @staticmethod
    def _trim_feature_end(data: bytearray | bytes, end: int) -> int:
        """Remove the comma and whitespace separating adjacent features."""
        while end > 0 and data[end - 1] in WHITESPACE:
            end -= 1
        if end > 0 and data[end - 1] == COMMA:
            end -= 1
        while end > 0 and data[end - 1] in WHITESPACE:
            end -= 1
        return end

    @staticmethod
    def _find_json_object_end(data: bytearray | bytes, start: int) -> int:
        """Return the exclusive end of the JSON object beginning at ``start``."""
        depth = 0
        in_string = False
        position = start
        while position < len(data):
            byte = data[position]
            if byte == QUOTE and not GeoJSONPartitionReader._is_escaped(data, position):
                in_string = not in_string
            elif not in_string:
                if byte == OPEN_BRACE:
                    depth += 1
                elif byte == CLOSE_BRACE:
                    depth -= 1
                    if depth == 0:
                        return position + 1
            position += 1
        raise ValueError(f"Unterminated JSON object beginning at byte {start}")

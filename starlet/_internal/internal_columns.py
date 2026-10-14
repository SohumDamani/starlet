"""Shared names for Starlet-managed columns."""

FEATURE_ID_COL = "_id"
TILE_ID_COL = "_tile_id"
BBOX_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")
QUERY_INTERNAL_COLS = (FEATURE_ID_COL, TILE_ID_COL, *BBOX_COLS)
MVT_EXCLUDED_ATTRIBUTE_COLS = (TILE_ID_COL, *BBOX_COLS)

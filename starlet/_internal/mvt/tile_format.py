# Single control point for tile compression format.
# Set to True to write/serve gzip-compressed .mvt.gz tiles.
# Set to False for raw .mvt tiles.
# Changing this one variable switches the entire pipeline.
TILE_GZIP: bool = True

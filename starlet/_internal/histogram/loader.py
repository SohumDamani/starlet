import logging
from pathlib import Path

from starlet._internal.histogram.io import load_prefix_histogram, resolve_histogram_path

logger = logging.getLogger("bucket_mvt")


class HistogramLoader:
    def __init__(self, hist_path):
        self.hist_path = Path(hist_path)
        self.prefix = None

    def load(self):
        self.hist_path = resolve_histogram_path(self.hist_path)
        logger.info("Loading histogram from %s", self.hist_path)
        self.prefix = load_prefix_histogram(self.hist_path.parent)
        return self.prefix

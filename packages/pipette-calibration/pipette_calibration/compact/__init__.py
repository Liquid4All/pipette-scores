"""Self-contained dataset-compaction pipeline.

Independent of calibration.representative — owns its own scoring so we can
experiment with filter chains without touching the production representative-
selection code path.

Public entry point: `compact_dataset(...)`.
"""

from pipette_calibration.compact.compact import compact_dataset

__all__ = ["compact_dataset"]

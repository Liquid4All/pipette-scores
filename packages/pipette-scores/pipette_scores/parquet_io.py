"""Single-file parquet reading (robust across PyArrow versions)."""

import pathlib

import pyarrow.parquet as pq


def read_parquet(path: pathlib.Path) -> list[dict]:
    # Read single parquet files directly to avoid dataset-layer incompatibilities
    # across PyArrow versions in CI.
    return pq.ParquetFile(path).read().to_pylist()

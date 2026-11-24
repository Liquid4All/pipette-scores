import pytest

from pipette_calibration.slurm.models import MAX_UNSHARDED_SAMPLES, ModelSpec, effective_shards


@pytest.mark.parametrize(
    ("base_shards", "n_samples", "expected"),
    [
        (8, 200, 1),
        (8, MAX_UNSHARDED_SAMPLES, 1),
        (8, MAX_UNSHARDED_SAMPLES + 1, 8),
        (1, 10_000, 1),
    ],
)
def test_effective_shards(base_shards: int, n_samples: int, expected: int) -> None:
    assert effective_shards(ModelSpec("org/model", "16G", base_shards), n_samples) == expected

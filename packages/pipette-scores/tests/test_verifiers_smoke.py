"""Smoke test: each eval's in-repo verifier imports from its scoring package."""

from pipette_scores.scoring.gpqa_diamond.mcq import score_mcq
from pipette_scores.scoring.math_500.math_generic import score_math_generic


def test_verifier_imports():
    assert score_mcq
    assert score_math_generic

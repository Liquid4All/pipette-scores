import pytest

from pipette_scores.repeats import expand, logical_id
from pipette_scores.types import EvalSample


def _sample(id):
    return EvalSample(id=id, messages=[])


def test_logical_id_strips_suffix():
    assert logical_id("abc#0") == "abc"
    assert logical_id("abc#4") == "abc"


def test_logical_id_without_suffix_is_self():
    assert logical_id("abc") == "abc"


def test_expand_salts_n_unique_ids_per_sample():
    expanded = expand([_sample("a"), _sample("b")], 3)
    ids = [s.id for s in expanded]
    assert ids == ["a#0", "a#1", "a#2", "b#0", "b#1", "b#2"]
    assert len(set(ids)) == len(ids)
    # logical ids round-trip back to the originals.
    assert {logical_id(i) for i in ids} == {"a", "b"}


def test_expand_repeats_one_is_noop():
    samples = [_sample("a"), _sample("b")]
    expanded = expand(samples, 1)
    assert [s.id for s in expanded] == ["a", "b"]


def test_expand_returns_new_list_repeats_one():
    samples = [_sample("a")]
    expanded = expand(samples, 1)
    assert expanded is not samples


def test_expand_preserves_payload():
    original = EvalSample(id="a", messages=[{"role": "user", "content": "hi"}])
    expanded = expand([original], 2)
    assert all(s.messages == original.messages for s in expanded)


def test_expand_rejects_zero_repeats():
    with pytest.raises(ValueError):
        expand([_sample("a")], 0)

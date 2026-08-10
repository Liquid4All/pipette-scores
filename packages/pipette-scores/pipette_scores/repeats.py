"""Shared ``#k`` repeat-expansion helpers.

IFBench runs each sample R times (pass@1 over R, loose). Per PIP-171 (Option A),
repeats are realized as *salted ids* — ``<id>#0 .. #(R-1)`` — keeping exactly one
completion per id, so the management/client contracts (unique ids, one
completion per id) are untouched. IFStruct uses ``repeats: 1``, which flows
through these helpers as a no-op.

Both the dataset catalog (producer: writes ``#k`` ids onto served prompts and
scoring ground-truth) and the scorers (consumer: groups scored ids back by
logical id) import ``expand``/``logical_id``, so the producer and consumer of
the ``#k`` convention cannot drift. ``read_repeats`` is the shared config reader
both loaders use to decide R.
"""

import json
import pathlib
from typing import TypeVar

from pydantic import BaseModel

_SUFFIX_SEP = "#"

S = TypeVar("S", bound=BaseModel)


def read_repeats(dataset_dir: pathlib.Path) -> int:
    """Read ``metadata.repeats`` for a dataset (defaults to 1 when absent).

    Repeats salt each id into ``#k`` attempts (see ``expand``). IFBench runs each
    sample R times (pass@1 over R); IFStruct uses 1. Both loaders read this so the
    ``repeats`` contract is honored uniformly — a dataset declaring ``repeats`` is
    never silently ignored.
    """
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        return 1
    repeats = json.loads(meta_path.read_text()).get("repeats", 1)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"metadata.repeats must be a positive integer, got {repeats!r} in {meta_path}")
    return repeats


def logical_id(sample_id: str) -> str:
    """Strip a ``#k`` repeat suffix to recover the logical sample id.

    Ids without a ``#`` are their own logical id, so ``repeats: 1`` (no salting)
    is transparent. Logical ids are content hashes (``short_hash``), which never
    contain ``#``, so splitting on the last ``#`` is unambiguous.
    """
    return sample_id.rsplit(_SUFFIX_SEP, 1)[0]


def expand(samples: list[S], repeats: int) -> list[S]:
    """Return ``samples`` with each id salted into ``repeats`` unique ``#k`` ids.

    ``repeats == 1`` returns the samples unchanged (no suffix) so id stability —
    and the management "missing = incorrect" contract — is preserved for evals
    that don't repeat. Each logical sample expands into ``#0 .. #(repeats-1)``,
    grouped consecutively.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if repeats == 1:
        return list(samples)
    return [
        sample.model_copy(update={"id": f"{sample.id}{_SUFFIX_SEP}{k}"}) for sample in samples for k in range(repeats)
    ]

"""Stable content hashing for sample identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import TypeAdapter

from pipette_scores.types import EvalSample

# A sample as `/samples` writes it: through the declared `EvalSample`, as the
# response model does, so a subclass's own fields are dropped here as there.
_AS_SERVED = TypeAdapter(EvalSample)


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def samples_digest(samples: Iterable[EvalSample]) -> str:
    """``sha256:<hex>`` of prompt samples as ``/samples`` serves them.

    Each sample is hashed as its JSON with keys sorted, no whitespace and
    non-ASCII written as UTF-8, followed by a newline, in served order. It
    changes when anything served changes (an id, a prompt's wording, an image,
    the order) and nothing else, so a pin taken from it holds across restarts.
    Hashing one sample at a time keeps memory to one sample's JSON.
    """
    h = hashlib.sha256()
    for sample in samples:
        h.update(canonical_sample(sample))
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"


def canonical_sample(sample: EvalSample) -> bytes:
    """One sample's canonical JSON, the bytes `samples_digest` hashes: keys
    sorted, no whitespace, non-ASCII as UTF-8. It is also valid JSON for the
    sample, so a streamed ``/samples`` body can be written from it."""
    return json.dumps(
        _AS_SERVED.dump_python(sample, mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()

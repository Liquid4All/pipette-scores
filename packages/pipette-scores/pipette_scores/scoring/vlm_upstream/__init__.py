"""Deterministic answer extraction + metrics vendored from VLMEvalKit @ 8eca8e5.

Each module cites the exact upstream file it was ported from. All logic is
pure string/number manipulation — no pandas, no file IO, and NO LLM judge
(unparseable answers deterministically count as wrong).

See PORTING_NOTES.md for the per-benchmark raw_doc / prompt contract.
"""

from . import matching_util, multiple_choice, ocrbench, yorn

__all__ = ["matching_util", "multiple_choice", "ocrbench", "yorn"]

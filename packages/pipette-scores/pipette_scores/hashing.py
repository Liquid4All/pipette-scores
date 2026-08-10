"""Stable content hashing for sample identity."""

import hashlib


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]

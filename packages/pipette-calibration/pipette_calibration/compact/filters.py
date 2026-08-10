"""Sample-filtering strategies that operate on pass-vector matrices.

Each filter takes `pass_vectors: dict[sample_id -> tuple[bool, ...]]` and
returns a reduced dict. Pure functions, no parameters beyond the input.
"""


def unanimous(pv: dict[str, tuple[bool, ...]]) -> dict[str, tuple[bool, ...]]:
    """Drop samples where every model agrees (all pass or all fail).

    Unanimous samples give no signal about which model is better than which,
    so they're dead weight for calibration.
    """
    return {sid: p for sid, p in pv.items() if any(p) and not all(p)}


def dedup_behavior(pv: dict[str, tuple[bool, ...]]) -> dict[str, tuple[bool, ...]]:
    """Keep one sample per distinct pass-vector equivalence class.

    Two samples with the same pass pattern across all models contribute
    identically to any per-model accuracy measurement, so deduping is
    lossless for calibration.
    """
    first_of: dict[tuple[bool, ...], str] = {}
    for sid, pattern in pv.items():
        first_of.setdefault(pattern, sid)
    return {sid: pv[sid] for sid in first_of.values()}

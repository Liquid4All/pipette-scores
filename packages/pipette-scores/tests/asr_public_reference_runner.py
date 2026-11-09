"""Answer a corpus with the PUBLIC reference implementations (run by the reference interpreter).

Not a test module: tests/asr_public_reference.py runs it with the Python of the
environment scripts/setup_asr_reference_env.sh builds, as

    <env>/venv/bin/python asr_public_reference_runner.py <env> <request.json> <answer.json>

Each input carries a ``name``, which its answer echoes (the caller pairs them
by name). Everything computed here is the reference code's own, unchanged:

- ``open_asr``: huggingface/open_asr_leaderboard ``normalizer/data_utils``
  (``normalizer``, ``is_target_text_in_range``: the ``prepare_data`` filter) and
  ``normalizer/eval_utils`` (``merge_chunked_manifest``, ``align_manifests``,
  ``score_manifests``, ``_align_pair``), English, ``num_workers=1``;
- ``fleurs``: the same leaderboard code as its multilingual scoring runs it
  (``score_results(multilingual=True, language=<code>)``): ``eval_utils._normalize``
  (its English normalizer for "en", else ``data_utils.ml_normalizer`` with the
  language), ``_align_pair`` (``normalize_compound_pairs``, then kaldialign) and
  ``align_manifests``; the language code read from the leaderboard config name
  (``fleurs_<code>``) as its ``run_eval_ml.py`` reads it;
- ``audio``: ``datasets.Audio(sampling_rate=16000).decode_example`` of datasets
  2.19.0 (soundfile, librosa).
"""

from __future__ import annotations

import base64
import json
import pathlib
import re
import sys


def _open_asr(manifests: list[dict]) -> list[dict]:
    from normalizer import data_utils, eval_utils

    answers = []
    for named in manifests:
        manifest = named["rows"]
        kept = []
        for row in manifest:
            # prepare_data: norm_text = normalizer(text), filtered by is_target_text_in_range.
            if not data_utils.is_target_text_in_range(data_utils.normalizer(row["text"])):
                continue
            kept.append({"duration": None, "time": None, **row})
        merged = eval_utils.merge_chunked_manifest(kept)
        [totals] = eval_utils.align_manifests([merged], language="en", num_workers=1)
        [score] = eval_utils.score_manifests([kept], language="en", num_workers=1)
        pairs = [eval_utils._align_pair((d["text"], d["pred_text"], "en", False)) for d in merged]
        answers.append(
            {
                "name": named["name"],
                "kept": len(kept),
                "units": len(merged),
                "totals": {k: totals[k] for k in ("ins", "del", "sub", "total", "ref_len", "err_rate")},
                "wer": score["wer"],
                "pairs": [[p["ins"], p["del"], p["sub"], p["ref_len"]] for p in pairs],
            }
        )
    return answers


def _fleurs(datasets: list[dict]) -> list[dict]:
    from normalizer import eval_utils

    answers = []
    for dataset in datasets:
        # run_eval_ml.py: the language from the config name (fleurs_de -> de).
        match = re.search(r"_([a-z]{2})(?:_test)?$", dataset["config"])
        if match is None:
            raise SystemExit(f"no language code in config {dataset['config']!r}")
        language = match.group(1)
        manifest = [{"text": r, "pred_text": h} for r, h in dataset["pairs"]]
        [totals] = eval_utils.align_manifests([manifest], language=language, multilingual=True, num_workers=1)
        pairs = [eval_utils._align_pair((r, h, language, True)) for r, h in dataset["pairs"]]
        answers.append(
            {
                "name": dataset["name"],
                "language": language,
                "references": [eval_utils._normalize(r, language) for r, _ in dataset["pairs"]],
                "hypotheses": [eval_utils._normalize(h, language) for _, h in dataset["pairs"]],
                "totals": {k: totals[k] for k in ("ins", "del", "sub", "total", "ref_len", "err_rate")},
                "pairs": [[p["ins"], p["del"], p["sub"], p["ref_len"]] for p in pairs],
            }
        )
    return answers


def _audio(clips: list[dict]) -> list[dict]:
    from datasets import Audio

    feature = Audio(sampling_rate=16000)
    answers = []
    for clip in clips:
        decoded = feature.decode_example({"path": None, "bytes": base64.b64decode(clip["bytes"])})
        array = decoded["array"]
        answers.append(
            {
                "name": clip["name"],
                "sampling_rate": decoded["sampling_rate"],
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "data": base64.b64encode(array.tobytes()).decode("ascii"),
            }
        )
    return answers


def main() -> None:
    env, request_path, answer_path = (pathlib.Path(a) for a in sys.argv[1:4])
    sys.path.insert(0, str(env / "open_asr_leaderboard"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    import datasets
    import kaldialign  # noqa: F401
    import num2words  # noqa: F401
    from importlib.metadata import version

    answer = {
        "versions": {"datasets": datasets.__version__, "num2words": version("num2words")},
        "open_asr": _open_asr(request.get("open_asr", [])),
        "fleurs": _fleurs(request.get("fleurs", [])),
        "audio": _audio(request.get("audio", [])),
    }
    answer_path.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

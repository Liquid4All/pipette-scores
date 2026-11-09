"""The corpus the ASR scoring and audio are checked on, and the call into the public references.

Not a test module (tests/test_asr_public_reference.py uses it). The corpus:

- a synthetic part, generated here deterministically: English sentences rich in
  what the normalizers rewrite (numbers, currency, ordinals, abbreviations,
  contractions, British spellings, fillers, bracketed noise markers, compounds,
  Unicode punctuation and diacritics), sentences in every FLEURS language
  evaluated (with digits, number words, compounds and bracketed spans), and
  their hypotheses as perturbed references (words dropped, inserted,
  substituted, compounds split and joined, neighbouring words cut at another
  point, numbers spelled out or written as digits, case and punctuation changed,
  empty answers, Earnings22-style chunked calls);
- a real part: every test transcript of each FLEURS locale (google/fleurs'
  ``transcription``, the reference, and ``raw_transcription``, at the pinned
  revision), and, when built datasets are found under
  ``PIPETTE_SCORES_DATA_DIR``, every reference of each, with the same
  perturbations, and a few of its clips.

`run_reference` answers it with the reference environment's code
(asr_public_reference_runner.py); every input carries a name, which the answer
echoes, and the answers must be exactly the inputs' names in order.
``python tests/asr_public_reference.py --write-fixture`` (with
``PIPETTE_ASR_REFERENCE_ENV`` set) records the answers on the synthetic part in
fixtures/asr_public_reference.json, which the default test run compares against
without the reference environment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import tempfile
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
RUNNER = HERE / "asr_public_reference_runner.py"
FIXTURE = HERE / "fixtures" / "asr_public_reference.json"

# --- synthetic English -------------------------------------------------------

_ENGLISH = [
    "Mr. Smith won't go, um, at 10:00 a.m. on the 3rd of May.",
    "It costs twenty five dollars, or $25.50 with tax, about 3.5% more.",
    "The colour of the Centre's theatre was grey; they organised a travelling show.",
    "Dr. O'Neil's 2nd favourite is the jewellery box from the U.S.A., e.g. the 1990s model.",
    "Uh, hmm, mm, I think, uh-huh, that's right.",
    "[noise] We'll see you (laughs) <unk> at the {breath} meeting.",
    "IGNORE TIME SEGMENT IN SCORING",
    "<unk> [noise]",
    "",
    "The football white paper was e-mailed to the re-elected board.",
    "Revenue grew five percent this quarter to $1.2 billion, up from $1,150,000,000.",
    "She ran 5km in 21 minutes and lifted 100kg; a 5mm gap remained.",
    "One hundred and five people, 1/2 of them under 18, met on 7 July 2021.",
    "Call 555-0199 or e-mail info@example.com before 5 p.m.",
    "“Café” and “naïve” are spelled with diacritics — Œuvre, too…",
    "We'd've gone if y'all hadn't said it's 11 o'clock.",
    "The St. Louis Cardinals won 4-3 in the 9th inning, Gov. Brown said.",
    "Our EBITDA margin was 23.4 percent, versus twenty-two point one percent last year.",
    "Thirty-three thousand three hundred and thirty-three, 33,333 and 33333.",
    'I said, "Don\'t!" — but he did; it was okay, OK, ok.',
    "First, second, 21st, 102nd and the 1,000th customer.",
    "£20, €5, ¥300 and 15 cents.",
    "Mmm, mhm, mm-hmm.",
    "The analyse, analyze; catalogue, catalog; programme, program question.",
    "Q3 2024 results: EPS of $0.87, guidance of $3.40 to $3.50.",
]
_FILLERS = ["um", "uh", "the", "and", "really", "twenty", "a", "of"]
_NUMBER_WORDS = {
    "5": "five",
    "25": "twenty five",
    "3rd": "third",
    "10:00": "ten o'clock",
    "100kg": "one hundred kilograms",
    "1/2": "half",
    "18": "eighteen",
    "2nd": "second",
    "4-3": "four three",
}


def _perturb(text: str, rng: random.Random) -> str:
    words = text.split()
    op = rng.randrange(11)
    if op == 0 or not words:
        return text
    if op == 1:
        del words[rng.randrange(len(words))]
    elif op == 2:
        words.insert(rng.randrange(len(words) + 1), rng.choice(_FILLERS))
    elif op == 3:
        words[rng.randrange(len(words))] = rng.choice(_FILLERS)
    elif op == 4:
        # Split a long word: a compound split the leaderboard's merge_compounds forgives.
        long = [i for i, w in enumerate(words) if len(w) > 6]
        if long:
            i = rng.choice(long)
            cut = rng.randrange(3, len(words[i]) - 2)
            words[i : i + 1] = [words[i][:cut], words[i][cut:]]
    elif op == 5 and len(words) > 1:
        # Join two words.
        i = rng.randrange(len(words) - 1)
        words[i : i + 2] = [words[i] + words[i + 1]]
    elif op == 6:
        words = [_NUMBER_WORDS.get(w.strip(".,;"), w) for w in words]
    elif op == 7:
        return text.lower() if rng.random() < 0.5 else text.upper()
    elif op == 8:
        return "".join(c for c in text if c.isalnum() or c.isspace())
    elif op == 9:
        return ""
    elif op == 10:
        i = rng.randrange(len(words))
        words = words[: i + 1] + words[: i + 1] + words[i + 1 :]
    return " ".join(words)


def synthetic_english(n: int = 1200, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs = []
    for k in range(n):
        reference = _ENGLISH[k % len(_ENGLISH)]
        if k >= len(_ENGLISH):
            # Recombine sentences so lengths and contexts vary.
            reference = " ".join(rng.sample(_ENGLISH, rng.randrange(1, 4)))
        hypothesis = reference
        for _ in range(rng.randrange(0, 4)):
            hypothesis = _perturb(hypothesis, rng)
        pairs.append((reference, hypothesis))
    return pairs


def synthetic_calls(n_calls: int = 6, seed: int = 1) -> list[dict[str, Any]]:
    """Earnings22-style rows: each chunk carries its call's transcript, the call
    (parent_id) and its chunk_index; chunks out of order, some answers blank."""
    rng = random.Random(seed)
    rows = []
    for call in range(n_calls):
        transcript = " ".join(rng.choice(_ENGLISH) for _ in range(rng.randrange(8, 20)))
        answer = transcript
        for _ in range(3):
            answer = _perturb(answer, rng)
        words = answer.split()
        n = rng.randrange(2, 7)
        cuts = sorted(rng.sample(range(1, max(len(words), n + 1)), n - 1)) if len(words) > n else []
        pieces = [" ".join(words[a:b]) for a, b in zip([0, *cuts], [*cuts, len(words)])]
        pieces += [""] * (n - len(pieces))
        chunks = [
            {
                "text": transcript,
                "pred_text": f"  {piece} " if rng.random() < 0.9 else "   ",
                "parent_id": f"call-{call}",
                "chunk_index": i,
            }
            for i, piece in enumerate(pieces)
        ]
        rng.shuffle(chunks)
        rows += chunks
    return rows


# --- synthetic FLEURS --------------------------------------------------------

_FLEURS_TEXT: dict[str, list[str]] = {
    "eng": _ENGLISH,
    "spa": [
        "El niño comió 25 manzanas, ¿verdad?",
        "Mañana, a las 10:30, llegará el tren de Málaga.",
        "En 1999 había 10 000 habitantes (según el censo) y veinte escuelas.",
    ],
    "por": [
        "A população é de 1.234.567 habitantes; não é?",
        "Ela disse: «Obrigada!» às três horas.",
        "O gabinete afirmou que 19 dos feridos eram agentes policiais [ruído] em 2006.",
    ],
    "deu": [
        "Löwenrudel agieren ähnlich wie Wolfs- oder Hunderudel.",
        "Die Donaudampfschifffahrtsgesellschaft fährt um 9 Uhr ab — pünktlich!",
        "Rund 1000000 Menschen sahen am 3. Oktober zwanzig Schiffe (laut Polizei).",
    ],
    "fra": [
        "L'été dernier, j'ai payé 12,50 € pour un café à Noël.",
        "Où est-ce qu'il a mis les clés ?",
        "En 1900, vingt-cinq des 25 voyageurs avaient « environ » 30 ans.",
    ],
    "ita": [
        "Perché l'università è chiusa il 25 aprile?",
        "È così: più o meno 3 ore.",
        "Nel dopoguerra, a partire dal 2009, venti visti (separati) bastavano.",
    ],
    "nld": [
        "Het is één uur 's nachts, zei de buurman.",
        "De fietsenmaker repareerde twee-en-twintig banden.",
        "Volgens Angel (2006) kunnen 10 000 organisaties 11e of 12e eeuwse modellen gebruiken.",
    ],
}

# Number words a hypothesis may write for a reference's digits, or the reverse
# (fixed here, so that the corpus does not depend on num2words).
_FLEURS_NUMBERS: dict[str, dict[str, str]] = {
    "eng": {"25": "twenty five", "3rd": "third", "18": "eighteen"},
    "spa": {"25": "veinticinco", "1999": "mil novecientos noventa y nueve", "veinte": "20"},
    "por": {"19": "dezenove", "2006": "dois mil e seis", "três": "3"},
    "deu": {"9": "neun", "zwanzig": "20", "3.": "dritten"},
    "fra": {"25": "vingt-cinq", "1900": "mille neuf cents", "30": "trente", "vingt-cinq": "25"},
    "ita": {"25": "venticinque", "2009": "duemilanove", "venti": "20", "3": "tre"},
    "nld": {"2006": "tweeduizend zes", "één": "1", "10": "tien"},
}


def _spell_numbers(text: str, language: str, rng: random.Random) -> str:
    """Some of the text's numbers written the other way: digits as words, or words as digits."""
    words = _FLEURS_NUMBERS[language]
    return " ".join(
        words[w.strip(".,;")] if w.strip(".,;") in words and rng.random() < 0.7 else w for w in text.split()
    )


def _resegment(text: str, rng: random.Random) -> str:
    """Two neighbouring words cut at another point ("zum beispiel" as "zumbei
    spiel"): what the leaderboard's normalize_compound_pairs forgives."""
    words = text.split()
    pairs = [i for i in range(len(words) - 1) if len(words[i]) + len(words[i + 1]) > 4]
    if not pairs:
        return text
    i = rng.choice(pairs)
    joined = words[i] + words[i + 1]
    cut = rng.choice([c for c in range(1, len(joined)) if c != len(words[i])])
    words[i : i + 2] = [joined[:cut], joined[cut:]]
    return " ".join(words)


def fleurs_config(eval_id: str) -> str:
    """The leaderboard's config name for a FLEURS eval's language (``fleurs_<code>``),
    which the reference reads the language code from, as its run_eval_ml.py does."""
    from pipette_scores.dataset_catalog import asr_datasets
    from pipette_scores.scoring.asr.normalize import OPEN_ASR_LANGUAGE_CODES
    from pipette_scores.types import EvalId

    return f"fleurs_{OPEN_ASR_LANGUAGE_CODES[asr_datasets.ASR_DATASETS[EvalId(eval_id)].language]}"


def synthetic_fleurs(seed: int = 2, per_language: int = 60) -> list[dict[str, Any]]:
    from pipette_scores.dataset_catalog import asr_datasets

    rng = random.Random(seed)
    datasets = []
    for locale, language, _ in asr_datasets.FLEURS_LOCALES:
        texts = _FLEURS_TEXT[language]
        pairs = []
        for k in range(per_language):
            reference = texts[k % len(texts)] if k < len(texts) else " ".join(rng.sample(texts, len(texts)))
            hypothesis = _spell_numbers(reference, language, rng) if rng.random() < 0.5 else reference
            for _ in range(rng.randrange(0, 4)):
                hypothesis = _perturb(hypothesis, rng)
            if rng.random() < 0.3:
                hypothesis = _resegment(hypothesis, rng)
            pairs.append((reference, hypothesis))
        eval_id = f"asr_fleurs_{locale}"
        datasets.append({"eval_id": eval_id, "config": fleurs_config(eval_id), "language": language, "pairs": pairs})
    return datasets


# --- synthetic audio ---------------------------------------------------------


def synthetic_clips() -> list[bytes]:
    """Clips at the rates, channel counts, subtypes and containers a source may
    ship: 8 to 48 kHz, mono to three channels, 16/24-bit, float, WAV and FLAC,
    odd lengths. Their samples are deterministic (their bytes are not: a float
    WAV written by libsndfile carries a timestamp)."""
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(3)
    clips = []
    for rate, channels, subtype, fmt, seconds in [
        (16000, 1, "PCM_16", "WAV", 1.0),
        (16000, 2, "PCM_16", "WAV", 0.7),
        (44100, 1, "PCM_16", "WAV", 2.0),
        (44100, 1, "PCM_16", "WAV", 0.50001),
        (44100, 1, "PCM_16", "WAV", 0.50003),  # 22,051 frames: 8000.36 at 16 kHz, so the length rule matters
        (44100, 2, "FLOAT", "WAV", 1.3),
        (48000, 2, "PCM_24", "WAV", 0.9),
        (48000, 3, "PCM_16", "WAV", 0.4),
        (22050, 1, "PCM_16", "FLAC", 1.1),
        (24000, 1, "PCM_16", "FLAC", 0.6),
        (32000, 1, "FLOAT", "WAV", 0.33),
        (8000, 1, "PCM_16", "WAV", 1.7),
        (11025, 2, "PCM_16", "FLAC", 0.8),
    ]:
        n = int(seconds * rate)
        t = np.arange(n) / rate
        tone = 0.2 * np.sin(2 * np.pi * 440 * t)[:, None] * (1 + np.arange(channels))[None, :] / channels
        data = np.clip(tone + 0.05 * rng.standard_normal((n, channels)), -1, 1)
        buffer = io.BytesIO()
        sf.write(buffer, data, rate, format=fmt, subtype=subtype)
        clips.append(buffer.getvalue())
    return clips


# --- real data ---------------------------------------------------------------


def built_datasets(root: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """The built ASR datasets under a datasets root: (eval_id, the latest
    version's directory), refusing a root that is missing, holds none, or holds
    an ``asr_*`` directory that is no catalog eval or has no built version."""
    from pipette_scores.dataset_catalog import asr_datasets
    from pipette_scores.types import EvalId

    if not root.is_dir():
        raise ValueError(f"PIPETTE_SCORES_DATA_DIR: {root} is not a directory")
    found = []
    for eval_dir in sorted(root.glob("asr_*")):
        try:
            asr_datasets.ASR_DATASETS[EvalId(eval_dir.name)]
        except (ValueError, KeyError):
            raise ValueError(f"PIPETTE_SCORES_DATA_DIR: {eval_dir} is no ASR eval of the catalog") from None
        versions = sorted(v for v in eval_dir.iterdir() if (v / "metadata.json").is_file())
        if not versions:
            raise ValueError(f"PIPETTE_SCORES_DATA_DIR: {eval_dir} holds no built version (no metadata.json)")
        found.append((eval_dir.name, versions[-1]))
    if not found:
        raise ValueError(f"PIPETTE_SCORES_DATA_DIR: {root} holds no built ASR dataset (asr_*/<version>/)")
    return found


def data_roots(value: str | None) -> list[pathlib.Path]:
    """``PIPETTE_SCORES_DATA_DIR``'s roots (several joined with os.pathsep),
    each a data dir holding ``datasets/`` or a datasets root. Unset or empty:
    none. Set: every root must hold at least one built ASR dataset
    (`built_datasets`), so that a mistyped path fails instead of checking
    nothing."""
    roots = [pathlib.Path(v) for v in (value or "").split(os.pathsep) if v]
    roots = [r / "datasets" if (r / "datasets").is_dir() else r for r in roots]
    for root in roots:
        built_datasets(root)
    return roots


def real_corpus(data_dir: pathlib.Path, clips_per_dataset: int = 24) -> dict[str, Any]:
    """Every reference of each built dataset under ``data_dir`` (its latest
    version), perturbed into hypotheses, and a few clips of each, all named
    ``<data_dir>:<eval_id>[:<clip>]``."""
    import pyarrow.parquet as pq

    from pipette_scores.dataset_catalog import _asr_common, asr_datasets
    from pipette_scores.types import EvalId

    english: dict[str, list[dict[str, Any]]] = {}
    fleurs: dict[str, dict[str, Any]] = {}
    clips: dict[str, bytes] = {}
    rng = random.Random(4)
    for eval_id, version in built_datasets(data_dir):
        dataset = asr_datasets.ASR_DATASETS[EvalId(eval_id)]
        name = f"{data_dir}:{eval_id}"
        files = _asr_common.parquet_files(version)
        columns = ["reference", "session", "chunk_index"]
        rows = [r for f in files for r in pq.read_table(f, columns=columns).to_pylist()]
        if not rows:
            raise ValueError(f"{version}: no rows")
        if dataset.family == "open_asr_leaderboard":
            manifest = []
            sessions: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                if row["session"] is None:
                    hypothesis = row["reference"]
                    for _ in range(rng.randrange(0, 3)):
                        hypothesis = _perturb(hypothesis, rng)
                    manifest.append({"text": row["reference"], "pred_text": hypothesis})
                else:
                    sessions.setdefault(row["session"], []).append(row)
            for session, chunks in sessions.items():
                answer = chunks[0]["reference"]
                for _ in range(8):
                    answer = _perturb(answer, rng)
                words = answer.split()
                size = -(-len(words) // len(chunks))
                for chunk in chunks:
                    k = chunk["chunk_index"]
                    piece = " ".join(words[k * size : (k + 1) * size])
                    manifest.append(
                        {"text": chunk["reference"], "pred_text": piece, "parent_id": session, "chunk_index": k}
                    )
            english[name] = manifest
        else:
            pairs = []
            for row in rows:
                hypothesis = row["reference"]
                for _ in range(rng.randrange(0, 3)):
                    hypothesis = _perturb(hypothesis, rng)
                pairs.append((row["reference"], hypothesis))
            fleurs[name] = {
                "eval_id": eval_id,
                "config": fleurs_config(eval_id),
                "language": dataset.language,
                "pairs": pairs,
            }
        audio = pq.ParquetFile(files[0]).read_row_group(0, columns=["audio"]).column("audio").to_pylist()
        clips.update({f"{name}:{k}": clip for k, clip in enumerate(audio[:clips_per_dataset])})
    return {"open_asr": english, "fleurs": fleurs, "clips": clips}


def fleurs_transcripts(seed: int = 5) -> dict[str, dict[str, Any]]:
    """Every test-split transcript of each FLEURS locale evaluated, from the
    pinned google/fleurs revision's ``data/<locale>/test.tsv``: its column 3,
    ``transcription`` (the reference), named ``fleurs_tsv:<eval_id>``, and its
    column 2, ``raw_transcription`` (cased, punctuated, with its parentheses),
    named ``fleurs_tsv_raw:<eval_id>``; each perturbed into hypotheses (numbers
    among them written the other way)."""
    from huggingface_hub import hf_hub_download

    from pipette_scores.dataset_catalog import asr_datasets

    rng = random.Random(seed)
    datasets = {}
    for locale, language, expected in asr_datasets.FLEURS_LOCALES:
        dataset = asr_datasets.ASR_DATASETS[asr_datasets.EvalId(f"asr_fleurs_{locale}")]
        assert dataset.reference_column == "transcription", dataset.eval_id
        path = hf_hub_download(
            asr_datasets.FLEURS_REPO, f"data/{locale}/test.tsv", repo_type="dataset", revision=dataset.revision
        )
        rows = [line.split("\t") for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == expected, (locale, len(rows))
        for prefix, column in (("fleurs_tsv", 3), ("fleurs_tsv_raw", 2)):
            pairs = []
            for row in rows:
                hypothesis = _spell_numbers(row[column], language, rng)
                for _ in range(rng.randrange(0, 3)):
                    hypothesis = _perturb(hypothesis, rng)
                pairs.append((row[column], hypothesis))
            datasets[f"{prefix}:{dataset.eval_id.value}"] = {
                "eval_id": dataset.eval_id.value,
                "config": fleurs_config(dataset.eval_id.value),
                "language": language,
                "pairs": pairs,
            }
    return datasets


def synthetic_inputs() -> dict[str, dict[str, Any]]:
    """The synthetic part, named: open_asr ``synthetic`` (the English pairs) and
    ``synthetic_calls``, fleurs ``synthetic:<eval_id>``, clips ``synthetic:<k>``."""
    return {
        "open_asr": {
            "synthetic": [{"text": r, "pred_text": h} for r, h in synthetic_english()],
            "synthetic_calls": synthetic_calls(),
        },
        "fleurs": {f"synthetic:{d['eval_id']}": d for d in synthetic_fleurs()},
        "clips": {f"synthetic:{k}": clip for k, clip in enumerate(synthetic_clips())},
    }


# --- the reference call ------------------------------------------------------


def reference_env() -> pathlib.Path | None:
    value = os.environ.get("PIPETTE_ASR_REFERENCE_ENV")
    return pathlib.Path(value) if value else None


def by_name(names: list[str], answers: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    """The reference's answers keyed by the names they echo, which must be
    exactly ``names``, in order: one answer per input, none lost or added."""
    if len(answers) != len(names):
        raise AssertionError(f"{kind}: {len(answers)} answers for {len(names)} inputs")
    echoed = [a.get("name") for a in answers]
    if echoed != names:
        raise AssertionError(f"{kind}: the answers are for {echoed}, not {names}")
    return {a["name"]: a for a in answers}


def run_reference(
    env: pathlib.Path,
    open_asr: dict[str, list[dict[str, Any]]],
    fleurs: dict[str, dict[str, Any]],
    clips: dict[str, bytes],
) -> dict[str, Any]:
    """The reference's answers, each keyed by its input's name."""
    request = {
        "open_asr": [{"name": name, "rows": rows} for name, rows in open_asr.items()],
        "fleurs": [{"name": name, **{k: d[k] for k in ("config", "language", "pairs")}} for name, d in fleurs.items()],
        "audio": [{"name": name, "bytes": base64.b64encode(c).decode("ascii")} for name, c in clips.items()],
    }
    with tempfile.TemporaryDirectory() as tmp:
        request_path, answer_path = pathlib.Path(tmp) / "request.json", pathlib.Path(tmp) / "answer.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [str(env / "venv" / "bin" / "python"), str(RUNNER), str(env), str(request_path), str(answer_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
    return {
        "versions": answer["versions"],
        "open_asr": by_name(list(open_asr), answer["open_asr"], "open_asr"),
        "fleurs": by_name(list(fleurs), answer["fleurs"], "fleurs"),
        "audio": by_name(list(clips), answer["audio"], "audio"),
    }


def reference_array(answer: dict[str, Any]) -> Any:
    """The samples of one audio answer, as the reference decoded them."""
    import numpy as np

    return np.frombuffer(base64.b64decode(answer["data"]), dtype=answer["dtype"]).reshape(answer["shape"])


def audio_summary(array: Any) -> dict[str, Any]:
    """What the fixture records of a decoded clip: its dtype, its shape and four
    fixed +-1 projections of its samples (each correctly rounded, math.fsum), so
    that equal samples give equal numbers on any platform, and a resampler that
    is off anywhere in the clip moves them far past floating-point rounding."""
    import numpy as np

    weights = np.random.default_rng(6).integers(0, 2, size=(4, array.shape[-1])) * 2 - 1
    samples = np.asarray(array, dtype=np.float64)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "projections": [math.fsum(row * samples) for row in weights.astype(np.float64)],
    }


def digest(texts: list[str]) -> str:
    return hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()


def synthetic_fixture(answer: dict[str, Any]) -> dict[str, Any]:
    """What fixtures/asr_public_reference.json records of the reference's
    answers (keyed by name) on the synthetic corpus."""
    inputs = synthetic_inputs()
    english, calls = answer["open_asr"]["synthetic"], answer["open_asr"]["synthetic_calls"]
    return {
        "note": (
            "Answers of the public reference implementations on the synthetic corpus of "
            "tests/asr_public_reference.py; regenerate with `python tests/asr_public_reference.py --write-fixture`."
        ),
        "corpus_sha256": corpus_digest(),
        "open_asr_english": {k: english[k] for k in ("kept", "units", "totals", "wer")},
        "open_asr_calls": {k: calls[k] for k in ("kept", "units", "totals", "wer")},
        "fleurs": {
            d["eval_id"]: {
                "totals": answer["fleurs"][name]["totals"],
                "references_sha256": digest(answer["fleurs"][name]["references"]),
                "hypotheses_sha256": digest(answer["fleurs"][name]["hypotheses"]),
            }
            for name, d in inputs["fleurs"].items()
        },
        "audio": {name: audio_summary(reference_array(answer["audio"][name])) for name in inputs["clips"]},
    }


def corpus_digest() -> str:
    english = synthetic_english()
    calls = synthetic_calls()
    fleurs = synthetic_fleurs()
    return hashlib.sha256(json.dumps([english, calls, fleurs], ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    env = reference_env()
    if env is None:
        raise SystemExit("set PIPETTE_ASR_REFERENCE_ENV (scripts/setup_asr_reference_env.sh)")
    inputs = synthetic_inputs()
    answer = run_reference(env, inputs["open_asr"], inputs["fleurs"], inputs["clips"])
    fixture = synthetic_fixture(answer)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        json.dump(fixture, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

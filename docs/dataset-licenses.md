# VLM benchmark dataset licenses

A license review (2026-07) of the 7 VLM benchmarks supported by
`pipette-scores` concluded that **none of them may be redistributed in
transformed form** (e.g. the parquet files our builders produce) from this
repository. All 7 are therefore **builder-only**: they are materialized
locally / at deploy by the one-shot scripts
`packages/pipette-scores/scripts/build_<eval>_dataset.py` from VLMEvalKit's
pinned upstream TSVs, and the `datasets/<eval>/` directories are gitignored.
This matches the existing repo practice for third-party text evals (ifbench,
gpqa_diamond, math_500).

## Summary

| Benchmark | Source (pinned TSV) | License | Images provenance | Redistribution verdict |
|---|---|---|---|---|
| mme | VLMEvalKit `MME.tsv` | none declared; authors state academic-research-only / non-commercial terms | Benchmark authors' collection | Builder-only. Users must self-assess the authors' terms, especially for commercial use |
| mmstar | VLMEvalKit `MMStar.tsv` | none declared | Aggregates source benchmarks incl. SEED-Bench (CC-BY-NC) and ScienceQA (CC-BY-NC-SA), whose terms it inherits | Builder-only. Users must self-assess the inherited non-commercial terms |
| ocrbench | VLMEvalKit `OCRBench.tsv` | MIT (benchmark code) | ~29 aggregated OCR datasets with mixed / partly research-only terms | Builder-only. Users must self-assess per-source terms |
| mmmu_val | VLMEvalKit `MMMU_DEV_VAL.tsv` (VAL rows) | Apache-2.0 (dataset card) | Textbooks and online sources, rights uncleared; card carries a takedown clause | Builder-only |
| realworldqa | VLMEvalKit `RealWorldQA.tsv` | **CC-BY-ND-4.0** | xAI-collected real-world images | Builder-only. The built parquet is Adapted Material under CC-BY-ND and **must not be redistributed in any form** |
| mmbench_en_dev | VLMEvalKit `MMBench_DEV_EN_V11.tsv` | Apache-2.0 (repository) | Public datasets and the internet, rights uncleared | Builder-only |
| pope | VLMEvalKit `POPE.tsv` | MIT (annotations) | COCO val2014 photos under Flickr per-photo terms; no redistribution right | Builder-only |

Notes:

- The pinned TSV URLs and MD5s come from VLMEvalKit (commit `8eca8e5`,
  `vlmeval/dataset/image_mcq.py`, `image_yorn.py`, `image_vqa.py`); each
  builder records the URL and MD5 it used in the dataset's `metadata.json`,
  together with `license` / `license_notes` fields and prints a license
  notice on completion.
- The lmms-lab / community Hugging Face mirrors of these benchmarks declare
  no license of their own; a mirror is not a rights grant.

## Distribution surfaces

There are two ways this data could leave a machine, and both count as
redistribution:

1. **The repository.** Built datasets must never be committed. The
   `datasets/<eval>/` directories for all 7 benchmarks are gitignored, and
   builders write only to those locations.
2. **A hosted pipette-mgmt server.** When pipette-mgmt serves a VLM eval, it
   pushes the prompt samples — including base64-encoded benchmark images —
   to registered clients. Serving these benchmarks to clients outside your
   organization is distribution of the benchmark content. Operators should
   obtain legal sign-off before hosting these datasets on a publicly
   reachable pipette-mgmt instance.

## Private storage

Keep built VLM datasets only in the organization's private storage, under
`<eval>/<version>/`. At deploy, an init container overlays them onto
`PIPETTE_SCORES_DATA_DIR`, the same flow as the gated gpqa_diamond dataset.
The service itself reads only its local data directory, never a bucket.
Private storage does not lift the terms above: the copies remain
internal-use only. In particular, do not share the RealWorldQA (CC-BY-ND)
build onward publicly in any form.

## ASR datasets (Open ASR Leaderboard English, FLEURS)

Built by `packages/pipette-scores/scripts/build_asr_datasets.py` from pinned
Hugging Face sources; **builder-only** like the VLM sets: never committed to
this repository or a public one, nor baked into the image. The builder refuses
an output inside this checkout, and `datasets/asr_*/` is gitignored and
dockerignored as a backstop. Internal copies are kept and deployed as the VLM
sets are (see [Private storage](#private-storage)): the scores server reads
them from its private data directory, and pipette-mgmt's dataset cache builds
each set's served corpus from the server's samples stream into private storage,
from which an organization's devices download it for a run. Nothing of an ASR
set is committed to this repository or a public one. Each `metadata.json`
records `license`, `license_notes` and `redistribution`. Licenses are as each
dataset card (or the Open ASR Leaderboard's About table) states them. **Every
ASR set is for internal evaluation only**: pipette uses the sets only for
internal, non-commercial benchmarking of its models, keeps them org-private,
and never redistributes them. SPGISpeech and URGENT24 are licensed for
non-commercial use only, and GigaSpeech is gated; that internal,
non-commercial benchmarking is their only use here.

| Datasets | Source | License | Verdict |
|---|---|---|---|
| `asr_esb_ami_cleaned` | AMI Meeting Corpus via `hf-audio/open-asr-leaderboard` | CC-BY-4.0 | Builder-only, internal evaluation use only |
| `asr_esb_earnings22_cleaned_aa_chunked` | Earnings-22 (Rev.com; CC-BY-SA-4.0) cleaned and chunked by Artificial Analysis, `ArtificialAnalysis/Earnings22-Cleaned-AA(-chunked)` | Apache-2.0 (the Artificial Analysis cards) | Builder-only, internal evaluation use only |
| `asr_esb_gigaspeech_cleaned` | GigaSpeech via `hf-audio/open-asr-leaderboard` | Apache-2.0, access gated behind the SpeechColab terms | Builder-only, internal evaluation use only |
| `asr_esb_librispeech_test_clean`, `_other` | LibriSpeech via `hf-audio/open-asr-leaderboard` | CC-BY-4.0 | Builder-only, internal evaluation use only |
| `asr_esb_spgispeech` | SPGISpeech via `hf-audio/open-asr-leaderboard` | Kensho user agreement, **non-commercial** | Builder-only, internal evaluation use only |
| `asr_esb_urgent2024` | URGENT 2024 Challenge (noisy) via `hf-audio/open-asr-leaderboard` | mixed, **non-commercial**: speech CC0 / CC-BY-4.0 / ODC-BY / Creative Commons, noise including WHAM! (CC-BY-NC-4.0) and TUT Urban Acoustic Scenes 2018 (non-commercial, research only) | Builder-only, internal evaluation use only |
| `asr_esb_monsoon_en_in` | `VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN` | CC-BY-4.0 (its card) | Builder-only, internal evaluation use only |
| `asr_esb_voxpopuli_cleaned_aa` | VoxPopuli (CC0) with references cleaned by Artificial Analysis, `ArtificialAnalysis/VoxPopuli-Cleaned-AA` via `hf-audio/open-asr-leaderboard` | Apache-2.0 (the Artificial Analysis card) | Builder-only, internal evaluation use only |
| `asr_fleurs_*` (`en_us`, `es_419`, `pt_br`, `de_de`, `fr_fr`, `it_it`, `nl_nl`) | `google/fleurs` | CC-BY-4.0 | Builder-only, internal evaluation use only |

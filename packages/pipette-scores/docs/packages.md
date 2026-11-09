# Eval packages

`pipette-scores packages` packs every prompt set `/samples` serves for a VLM or ASR eval into the tar pipette-mgmt hands a device, so mgmt serves the tar without fetching or packing it. The text evals have no packages.

## 1. Command

```bash
pipette-scores packages --datasets s3://<bucket>/scores-datasets --packages s3://<bucket>/scores-packages
pipette-scores packages --datasets "$PIPETTE_SCORES_DATA_DIR" --packages ./packages --eval mme pope --version 2026.11.1-0-ga1b2c3d4ab
```

| Option | Description |
|--------|-------------|
| `--datasets` | `s3://<bucket>/<prefix>` holding `<eval_id>/<dataset_version>/`, each dataset downloaded to a temporary directory to be packed; or a data root holding `datasets/<eval_id>/<dataset_version>/` (as `PIPETTE_SCORES_DATA_DIR`), read in place |
| `--packages` | `s3://<bucket>/<prefix>` or a local directory |
| `--eval` | Only these evals (default: every VLM and ASR eval) |
| `--version` | The scores release the index is written for (default: `PIPETTE_SCORES_VERSION`); refused unset or `dev` |

A dataset whose index for that version exists is skipped. Any other is packed for every profile `/samples` takes: a VLM set for its own prompts and each prompt profile, one profile at a time; an ASR set for each ASR profile, in one pass over its audio (the pass the server's digest makes). Before a tar is stored, every sample is rebuilt from its line and media files, and the rebuilt samples must hash to the samples digest again. A tar already stored is not written again. Each package logs one line: eval, dataset, profile, identity, digest, bytes, and `uploaded` or `already present`. A dataset that fails is logged and the others are still packed; the run then exits non-zero. A run holds one dataset on disk at a time, with the media of its packages.

## 2. Layout

The store is write-once, like `scores-datasets/`: every write is conditional (S3 `If-None-Match: *`), and a 412 means another run wrote the key first.

```
<packages>/
  tars/<tar sha256>.tar
  index/<scores version>/<eval_id>/<dataset_version>.json
```

A package is an uncompressed ustar of `manifest.json`, `media/<sha256>.<ext>` (each image or clip once, however many samples share it) and `samples.jsonl`, sorted by path, every header field fixed. Its bytes are the ones pipette-mgmt's dataset-cache build packed from `/samples` (`vlm-eval-corpus/1`, `asr-eval-corpus/1`), so a catalog pin keeps its digest; `tests/test_packages.py` pins a test vector mgmt shares.

## 3. Index

Written once per scores version and dataset, after all its packages are stored:

```json
{
  "eval_id": "asr_fleurs_nl_nl",
  "dataset_version": "2026.10.1",
  "scores_version": "2026.11.1-0-ga1b2c3d4ab",
  "profiles": [
    {
      "prompt_profile": "gemma4",
      "samples_digest": "sha256:804f31798e0a7ef3213062619c57d9846960cef1ce271b1f863518c61320cba0",
      "sample_count": 364,
      "package": {"format": "asr-eval-corpus/1", "digest": "dc3c5314d753b2cfd04cd7160d0edfaa5605e31e70958e4ba4b439f8b8f29642", "bytes": 223719936, "media": 364}
    }
  ]
}
```

`prompt_profile` is `null` for a VLM set's own prompts. A catalog entry is pinned from an index entry: `[dataset]` `format`, `adapter = "evals-server@1"`, `digest`, `bytes`; `[dataset.upstream]` `source = "evals_server"`, `eval_id`, `dataset_version`, `prompt_profile`, and `identity`, the `samples_digest` without `sha256:`.

With `PIPETTE_SCORES_PACKAGES` set to the store, the server reads `index/<PIPETTE_SCORES_VERSION>/<eval_id>/<dataset_version>.json` on the first digest request for a set it holds and takes every profile's digest from it; a set without an index file has its digest computed from its samples ([httpapi.md §3.3](httpapi.md#33-get-evalseval_iddatasetsdataset_namesamplesdigest)). A dataset version missing from the server's data directory is not found, even when the index lists it.

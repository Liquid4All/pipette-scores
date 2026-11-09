# IFBench checkers

The IFBench instruction checkers that `pipette-scores` scores with, from
[allenai/IFBench](https://github.com/allenai/IFBench) @
`1a902f9e824259f044b6554f6ccb95e2bb322e7b` (2025-11-13).

That is the revision Liquid4All/liquid_evals scores its `ifbench_aa` task with,
the key that the internal eval scheduler runs through liquid_evals. A completion gets
the same verdict here as it does there, with the exceptions listed below.
liquid_evals runs its own copy of this revision with two small guards: a
guarded spacy import, and `False` for a response with no trigrams. Changes 2
and 4 below cover the same cases. The checkers were reworked after this
revision, which is why they are vendored at a fixed commit rather than tracked
through the `vendor/ifbench` submodule. That submodule is still the source of
the IFBench **test set**; only the checker code comes from here.

Git blob ids of the upstream files at `1a902f9`. The `vendor/ifbench`
submodule history contains that commit, so
`git -C vendor/ifbench ls-tree 1a902f9` lists them:

| File | Blob |
|---|---|
| `evaluation_lib.py` | `a0db9e728d067672eb116dd3e769774ae728c916` |
| `instructions.py` | `f32ff486907490e88b77b014fa3ce5fe35c357c3` |
| `instructions_registry.py` | `4bdbdf52e776a6592dc60285954d6eb771dbd952` |
| `instructions_util.py` | `218247ae86d070ac0e9248d6a0a35a92453a6e4e` |
| `LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |

## Local changes

The edited files carry a header notice, and each added block is commented
`pipette-scores:`. The first two changes alter no verdicts. The last two only
affect inputs on which liquid_evals gives no stable verdict.

1. **No network at scoring time** (`instructions.py`, `instructions_util.py`).
   Upstream calls `nltk.download()` from eleven `build_description()` methods
   and from every `count_stopwords()` call; those calls are removed. Instead,
   the resources the checkers use (`punkt_tab`, `stopwords`,
   `averaged_perceptron_tagger_eng`) are fetched once at import, only if
   missing. So is `punkt`, which upstream fetched at import although no
   checker uses it. If a resource is still missing after that, the import
   fails rather than scoring without it. The Docker image pre-downloads all
   four.
2. **No spacy** (`instructions.py`). Upstream runs
   `spacy.cli.download('en_core_web_sm')` on every import. No checker uses
   spacy, so the import and the download are dropped.
3. **Deterministic `repeat:repeat_span`** (`instructions.py`). Upstream treats
   `n_start=0` as unset (`if not n_start`) and picks a random start on every
   call. The one dataset row with `n_start=0` (key 286) therefore gets a
   random verdict, and a correct answer passes there about 1 time in 42. Here
   `0` is used as given, and `n_end` gets the same `is None` check. On that
   row this scorer can report up to 5 of 1500 attempts (0.33 pp) more than
   liquid_evals.
4. **A checker that raises counts as not followed** (`evaluation_lib.py`).
   Some checkers raise on degenerate responses, for example `IndexError` on
   `Wow!! Wow is great.` for `words:last_first`, or `ZeroDivisionError` on
   `...` for `ratio:stop_words`. Upstream lets that abort the run with no
   score. Here the check counts as not followed for that response. In loose
   mode the other response variants are still tried, so the instruction can
   still pass. The trigram-overlap checker raises `ZeroDivisionError` when a
   response, or a loose variant of it, is shorter than three characters.
   Catching it gives the `False` that liquid_evals' guard returns, and the
   remaining variants are tried on both sides. Missing NLTK data
   (`LookupError`) is not caught; it still raises.

Verdicts also depend on the nltk version: 3.9.4 splits dashes (U+2012–U+2015)
into their own tokens, which changes `word_tokenize` for four checkers.
`pipette-scores` pins nltk 3.9.4, the version that the internal liquid_evals
environments lock.

## License

Apache-2.0, (c) Allen Institute for AI and The Google Research Authors; full
text in `LICENSE`. See the repository's `NOTICE` and `THIRD_PARTY_LICENSES.md`.

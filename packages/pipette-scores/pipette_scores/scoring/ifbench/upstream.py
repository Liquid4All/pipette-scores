"""Adapter that exposes the vendored IFBench checkers as a package.

The checkers are allenai/IFBench at the revision Liquid4All/liquid_evals
scores its ``ifbench_aa`` task with (``vendor/ifbench-checkers``; see its
README for the commit and the local changes). They ship four top-level modules
(``evaluation_lib``, ``instructions``, ``instructions_registry``,
``instructions_util``) that import each other by bare name. We don't want to
rewrite those imports, so the bootstrap below:

1. Temporarily prepends the checker dir to ``sys.path`` so the bare
   imports resolve.
2. Imports the four modules in dependency order (primes ``sys.modules``).
3. Rehomes them under ``pipette_scores.scoring.ifbench._upstream.*`` so
   the bare top-level names (``instructions``, etc.) are not visible to
   anything else in the interpreter.
4. Cleans up ``sys.path``.

After import, the ifbench scorer can use the re-exported symbols as if
IFBench were a normal Python package.
"""

import importlib
import pathlib
import sys

# `_upstream/` is NOT in the source tree — it is generated at wheel-build time by
# the `[tool.hatch.build.targets.wheel.force-include]` block in
# packages/pipette-scores/pyproject.toml, which copies the checker files in
# `vendor/ifbench-checkers` (+ LICENSE) into the wheel. So:
#   - installed wheel  -> bundled copy at _upstream/          (_BUNDLED)
#   - editable/source  -> fall back to vendor/ifbench-checkers (_REPO)
# CI exercises the bundled path via the `wheel-smoke` job (see .github/workflows).
_BUNDLED = pathlib.Path(__file__).resolve().parent / "_upstream"
# parents climb from .../pipette_scores/scoring/ifbench/upstream.py:
# ifbench[0] scoring[1] pipette_scores[2] pipette-scores[3] packages[4] <repo>[5].
_REPO = pathlib.Path(__file__).resolve().parents[5] / "vendor" / "ifbench-checkers"

if (_BUNDLED / "evaluation_lib.py").exists():
    _SRC = _BUNDLED
elif (_REPO / "evaluation_lib.py").exists():
    _SRC = _REPO
else:
    raise RuntimeError(
        "IFBench checkers not found. Expected either a bundled copy at "
        f"{_BUNDLED} (installed wheel) or the vendored copy at {_REPO} "
        "(editable install)."
    )

_MODULES = ("instructions_util", "instructions", "instructions_registry", "evaluation_lib")

sys.path.insert(0, str(_SRC))
try:
    for _name in _MODULES:
        importlib.import_module(_name)
finally:
    sys.path.remove(str(_SRC))

# Rehome under the ifbench *package* (`__package__`), not this adapter module —
# the bundled source physically lives at the package's `_upstream/` subdir.
for _name in _MODULES:
    sys.modules[f"{__package__}._upstream.{_name}"] = sys.modules.pop(_name)

from pipette_scores.scoring.ifbench._upstream.evaluation_lib import (  # noqa: E402
    InputExample,
    OutputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

__all__ = [
    "InputExample",
    "OutputExample",
    "test_instruction_following_loose",
    "test_instruction_following_strict",
]

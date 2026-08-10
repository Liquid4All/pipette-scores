"""IFBench upstream-adapter bootstrap contract (scoring/ifbench/upstream.py).

The scorer tests exercise the adapter *functionally*, but they don't pin *how* the
upstream modules are rehomed. The regression where they were registered under the
adapter module (`upstream._upstream`) instead of the ifbench package
(`ifbench._upstream`) passed every scorer test and surfaced only in the wheel,
where `_upstream/` is imported as the package's namespace subpackage. These
assertions catch that class of bug in plain pytest.
"""

import sys

import pipette_scores.scoring.ifbench.upstream as upstream

_UPSTREAM_MODULES = ("instructions_util", "instructions", "instructions_registry", "evaluation_lib")


def test_reexports_available():
    for name in (
        "InputExample",
        "OutputExample",
        "test_instruction_following_loose",
        "test_instruction_following_strict",
    ):
        assert hasattr(upstream, name), f"adapter missing re-export {name!r}"


def test_upstream_rehomed_under_ifbench_package():
    # Must hang off the ifbench *package* (`_upstream`), not the adapter module
    # (`upstream._upstream`): only the former resolves as the bundled namespace
    # subpackage in an installed wheel.
    for name in _UPSTREAM_MODULES:
        key = f"pipette_scores.scoring.ifbench._upstream.{name}"
        assert key in sys.modules, f"{key} not rehomed into sys.modules"


def test_bare_upstream_names_not_leaked():
    # The bootstrap's whole purpose: upstream's bare top-level imports
    # (e.g. `import instructions_registry`) must not leak into sys.modules.
    for name in _UPSTREAM_MODULES:
        assert name not in sys.modules, f"bare upstream name {name!r} leaked into sys.modules"

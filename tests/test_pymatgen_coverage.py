"""How much of pymatgen matverse reaches, measured rather than estimated.

The number in this file has been wrong before. "15 of 48 analysis modules" was
quoted against a denominator that counted only the top level of
``pymatgen.analysis``, where the real tree has 110 leaves; and three pymatgen
add-ons were installed, counted toward coverage, and never wired into a single
code path.

So the map is checked against the code rather than trusted:

- every public pymatgen module is classified, and an unclassified one fails
- a module claimed as WRAPPED must actually be imported by matverse
- a matverse function named in the map must actually be registered
- coverage may not fall below the recorded floor

Re-export shims are resolved to the module holding the implementation, because
pymatgen moves modules between subpackages between releases — twenty-two of them
in the version this was written against — and counting a shim as covered while
its target sits in the gap list is the same error one level down.
"""

from __future__ import annotations

import glob
import re
import warnings

import pytest

import matverse as mv
from matverse import _coverage

warnings.filterwarnings("ignore")

#: Modules covered on this branch. Raise it as batches land; a drop is a
#: regression. A count rather than a fraction, because the denominator moves
#: when pymatgen adds modules and a ratchet that slips on rounding is not one.
COVERED_FLOOR = 92

#: The pymatgen the floor was recorded against. matverse supports two, and they
#: ship different module trees — 162 modules in scope against 134 — so a count
#: from one is not a ratchet for the other. The number is reported on both and
#: asserted on this one.
FLOOR_PYMATGEN = "2026."


@pytest.fixture(scope="module")
def report():
    return _coverage.report()


@pytest.fixture(scope="module")
def matverse_source() -> str:
    text = ""
    for path in sorted(glob.glob(
            _coverage.__file__.replace("_coverage.py", "*.py"))):
        if path.endswith("_coverage.py"):
            continue
        text += open(path, encoding="utf-8").read() + "\n"
    return re.sub(r"\\\s*\n\s*", "", text)


class TestTheMapIsHonest:
    def test_every_module_is_classified(self, report):
        """TODO is a classification. Being absent from the file is not."""
        assert report["total"] == sum(report["buckets"].values())

    def test_a_wrapped_module_is_actually_imported(self):
        """A coverage claim has to be backed by an import somewhere in the
        package. This is the check that would have caught three add-ons being
        installed for the count and never used."""
        alias_map = _coverage.aliases()
        present = set(_coverage.public_modules())
        reached = {_coverage.canonical(m, alias_map)
                   for m in _coverage.reached_modules()}
        # A key naming a module this pymatgen does not ship is not a false
        # claim; the two supported versions lay the same code out differently
        # and the map lists both names.
        unbacked = []
        for module in _coverage.WRAPPED:
            if module in _coverage.TRANSITIVE:
                continue          # reached through a returned object; see below
            group = {_coverage.canonical(n, alias_map)
                     for n in _coverage.equivalents(module)}
            if group & present and not (group & reached):
                unbacked.append(module)
        assert not unbacked, (
            "claimed as WRAPPED but never imported by matverse:\n  "
            + "\n  ".join(unbacked))

    def test_every_named_function_exists(self):
        """The map names matverse functions; a rename must not leave the map
        pointing at nothing."""
        missing = []
        for module, functions in _coverage.WRAPPED.items():
            for name in functions:
                if mv.registry.get(name) is None:
                    missing.append(f"{module} -> {name}")
        assert not missing, (
            "the coverage map names functions that are not registered:\n  "
            + "\n  ".join(missing))

    def test_transitive_reach_names_what_hands_the_object_over(self):
        """A module reached through an object rather than an import is still
        used, but the import is not there to find. Saying so beats weakening
        the check that caught it."""
        for module, reason in _coverage.TRANSITIVE.items():
            assert module in _coverage.WRAPPED, module
            assert reason.strip(), module

    def test_a_blocked_gap_names_what_blocks_it(self, report):
        """A gap that cannot be closed here is still a gap — BLOCKED entries
        stay in scope and uncovered. What they add is the distinction between
        a backlog and a wish."""
        for module, reason in _coverage.BLOCKED.items():
            assert reason.strip(), module
        assert set(report["blocked"]) <= set(_coverage.BLOCKED)
        assert not (set(report["open"]) & set(_coverage.BLOCKED))

    def test_blocked_and_open_account_for_every_gap(self, report):
        assert len(report["blocked"]) + len(report["open"]) == \
            report["buckets"].get("TODO", 0)

    def test_every_exemption_carries_a_reason(self):
        for mapping in (_coverage.NATIVE, _coverage.NOT_A_GOAL):
            for module, reason in mapping.items():
                assert reason.strip(), f"{module} is exempted without a reason"

    def test_a_module_lands_in_exactly_one_bucket(self):
        overlap = set(_coverage.WRAPPED) & set(_coverage.NATIVE)
        assert not overlap, f"in both WRAPPED and NATIVE: {sorted(overlap)}"


class TestShims:
    def test_shims_are_resolved_rather_than_counted(self):
        """pymatgen.analysis.local_env is three lines re-exporting
        pymatgen.core.local_env. Counting both doubles the denominator and puts
        the real module in the gap list while the stub reads as covered."""
        alias_map = _coverage.aliases()
        if not alias_map:
            pytest.skip("this pymatgen predates the move into core/")
        target = _coverage.canonical("analysis.local_env", alias_map)
        assert target == "core.local_env"
        assert "analysis.local_env" not in _coverage.public_modules()

    def test_a_shim_and_its_target_are_the_same_capability(self):
        """Whichever name the map uses, the module counts once and as covered."""
        alias_map = _coverage.aliases()
        if not alias_map:
            pytest.skip("this pymatgen predates the move into core/")
        assert _coverage.classify("core.local_env", alias_map) == "WRAPPED"
        assert _coverage.classify("analysis.local_env", alias_map) == "WRAPPED"


class TestCoverage:
    def test_the_number_is_reported(self, report):
        print("\n" + _coverage.summary())
        assert report["in_scope"] > 0

    def test_coverage_has_not_regressed(self, report):
        import pymatgen.core
        if not pymatgen.core.__version__.startswith(FLOOR_PYMATGEN):
            pytest.skip(f"floor recorded against pymatgen {FLOOR_PYMATGEN}x, "
                        f"this is {pymatgen.core.__version__}")
        assert report["covered"] >= COVERED_FLOOR, (
            f"coverage fell to {report['covered']} modules from a floor of "
            f"{COVERED_FLOOR}")

    def test_the_floor_is_kept_close_to_the_truth(self, report):
        """A floor far below the real number stops being a ratchet."""
        import pymatgen.core
        if not pymatgen.core.__version__.startswith(FLOOR_PYMATGEN):
            pytest.skip("floor is recorded against another pymatgen")
        assert report["covered"] - COVERED_FLOOR < 5, (
            f"{report['covered']} modules are covered but the floor is still "
            f"{COVERED_FLOOR}; raise COVERED_FLOOR")

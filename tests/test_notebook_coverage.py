"""Every registered function has to appear in a tutorial, and be called.

`build_notebooks.py` already refuses to build if a registered function appears
in no notebook at all. That check is satisfied by a mention in prose, which is
weaker than it sounds: a function can be described in a paragraph, never
executed, and drift out of working without anything noticing — which is exactly
how mv.elec.transport came to advertise three output columns it did not
produce.

So this asks for more. A registered function must be **called in a code cell**,
or be on PROSE_ONLY with a reason it cannot be. The list is short on purpose:
everything on it needs a network credential or a file from a real
first-principles run, and neither belongs in a tutorial that has to execute in
CI.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest

import matverse as mv

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / (
    "matverse_guide/docs/_scripts")

#: Functions a tutorial can describe but not run, and why. Anything added here
#: needs a reason of the same kind — an external credential or a real
#: calculation's output — not "it was awkward".
PROSE_ONLY: dict[str, str] = {
    "mv.data.from_mp": "needs a Materials Project API key and the network",
    "mv.thermo.references_from_mp": "needs a Materials Project API key",
    "mv.elec.read_bands": "parses vasprun.xml from a real VASP run",
    "mv.elec.cohp": "parses ICOHPLIST.lobster from a real LOBSTER run",
    "mv.elec.dos_fingerprint": "needs a density of states deposited by "
                               "mv.dft.read_dos, which parses a real "
                               "vasprun.xml",
}


def _cells():
    """Every (kind, source) pair across the tutorial sources."""
    for path in sorted(SCRIPTS.glob("nb_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for kind, source in module.CELLS:
            yield path.stem, kind, source


def _mentions(source: str) -> set[str]:
    return {f"mv.{namespace}.{name}"
            for namespace, name in re.findall(r"\bmv\.(\w+)\.(\w+)", source)}


@pytest.fixture(scope="module")
def coverage():
    called, described = set(), set()
    for _name, kind, source in _cells():
        (called if kind == "code" else described).update(_mentions(source))
    registered = {name.replace("matverse.", "mv.")
                  for name in set(mv.registry._index.values())}
    return registered, called, described


def test_every_registered_function_is_called_or_excused(coverage):
    registered, called, described = coverage
    uncalled = sorted(registered - called - set(PROSE_ONLY))
    assert not uncalled, (
        f"registered but never called in a tutorial code cell: {uncalled}. "
        f"Add a cell that runs it — guarded with try/except ImportError if it "
        f"needs an optional extra — or add it to PROSE_ONLY with a reason it "
        f"cannot be run.")


def test_nothing_registered_is_missing_altogether(coverage):
    """The check build_notebooks.py makes, kept here so a failure names the
    function rather than only failing the docs build."""
    registered, called, described = coverage
    missing = sorted(registered - called - described)
    assert not missing, f"absent from every tutorial: {missing}"


def test_the_excuse_list_stays_honest(coverage):
    """Two ways this list rots: an entry for a function that no longer exists,
    and an entry for one that is now called anyway."""
    registered, called, _described = coverage
    unknown = sorted(set(PROSE_ONLY) - registered)
    assert not unknown, f"PROSE_ONLY names unregistered functions: {unknown}"
    redundant = sorted(set(PROSE_ONLY) & called)
    assert not redundant, (
        f"these are called in a code cell and no longer need an excuse: "
        f"{redundant}")


def test_every_excuse_names_an_external_dependency(coverage):
    """The bar for the list: a credential or a real calculation's output.
    'It was awkward' is not a reason, and this makes adding one deliberate."""
    allowed = ("api key", "network", "vasp", "lobster", "vasprun")
    for name, reason in PROSE_ONLY.items():
        assert any(word in reason.lower() for word in allowed), (
            f"{name}'s reason does not name an external dependency: {reason!r}")

"""Shared fixtures.

Structures are restricted to elements EMT is parameterised for — Al, Cu, Ni, Pd,
Ag, Au, Pt, plus H C N O — so that the calculator path is exercised for real
rather than mocked. EMT is not a good potential; it is a real one, which is what
a pipeline test needs.
"""

from __future__ import annotations

import pytest


def _fcc(symbol: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def _l12(host: str, guest: str, a: float):
    """A guest-on-corner ordered fcc alloy, guest:host = 1:3."""
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def _b2(a_sym: str, b_sym: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [a_sym, b_sym],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def structures():
    """A small Al-Cu-Ni library: three elementals and three ordered alloys."""
    return [
        _fcc("Al", 4.05),
        _fcc("Cu", 3.61),
        _fcc("Ni", 3.52),
        _l12("Al", "Cu", 3.90),      # Cu Al3
        _l12("Cu", "Al", 3.70),      # Al Cu3
        _b2("Al", "Ni", 2.89),       # AlNi
    ]


@pytest.fixture
def make_md(structures):
    """A factory, because the contract probes are destructive."""
    import matverse as mv

    def _make():
        return mv.data.from_structures(list(structures))
    return _make


@pytest.fixture
def md(make_md):
    return make_md()

"""Starting datasets for matverse-bench.

Every fixture uses elements the EMT calculator is parameterised for — Al, Cu,
Ag, Au, Ni, Pd, Pt, H, C, N, O — so a benchmark run needs no downloaded model
and no network. That constrains the chemistry to metals, which constrains what
the tasks can ask about, and the constraint is worth naming: a benchmark whose
tasks only run on one machine measures that machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fcc(symbol: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def _l12(host: str, guest: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def _b2(a_sym: str, b_sym: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [a_sym, b_sym],
                     [[0, 0, 0], [.5, .5, .5]])


def _base() -> list:
    return [_fcc("Al", 4.05), _fcc("Cu", 3.61), _fcc("Ni", 3.52),
            _l12("Al", "Cu", 3.90), _l12("Cu", "Al", 3.70),
            _b2("Al", "Ni", 2.89)]


def alcuni():
    import matverse as mv
    return mv.data.from_structures(_base())


def alcuni_broken():
    from pymatgen.core import Lattice, Structure

    import matverse as mv
    broken = Structure(Lattice.cubic(4.0), ["Cu", "Cu"],
                       [[0, 0, 0], [0.02, 0, 0]])       # 0.08 A apart
    return mv.data.from_structures(_base() + [broken])


def alcuni_duplicated():
    import matverse as mv
    structures = _base()
    return mv.data.from_structures(structures + [structures[0].copy()])


def two_databases():
    """The same four compositions from two databases, one carrying an offset.

    The offset is linear in composition — Al is shifted by +0.30 eV/atom and Cu
    by -0.20 — which is what a per-element reference correction is for and what
    a single constant offset cannot absorb.
    """
    import matverse as mv

    structures = [_fcc("Al", 4.05), _fcc("Cu", 3.61),
                  _l12("Al", "Cu", 3.90), _l12("Cu", "Al", 3.70)]
    reference = np.array([-3.0, -3.5, -3.2, -3.4])
    fractions = np.array([[1.0, 0.0], [0.0, 1.0], [0.75, 0.25], [0.25, 0.75]])
    shifted = reference + fractions @ np.array([0.30, -0.20])

    obs = pd.DataFrame({
        "database": ["mp"] * 4 + ["oqmd"] * 4,
        "energy_per_atom_dft": np.concatenate([reference, shifted]),
    })
    return mv.data.from_structures(
        structures + [s.copy() for s in structures], obs=obs)


BUILDERS = {
    "alcuni": alcuni,
    "alcuni_broken": alcuni_broken,
    "alcuni_duplicated": alcuni_duplicated,
    "two_databases": two_databases,
}


def build(name: str):
    if name not in BUILDERS:
        raise KeyError(f"no fixture {name!r}; have {sorted(BUILDERS)}")
    return BUILDERS[name]()


__all__ = ["build", "BUILDERS"]

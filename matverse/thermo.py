"""``mv.thermo`` — thermodynamic stability.

``e_above_hull`` is only meaningful within one level of theory. Every function
here takes a ``level`` and refuses to mix, because a hull built from a
surrogate potential's energies and DFT's is not a hull of anything.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import record, structures


def hull(md: AnnData, level: str = "emt", source: str = "input") -> None:
    """Convex hull over the dataset's own compositions.

    Built from `pymatgen`'s ``PhaseDiagram``, so the result is a real hull and
    not a per-formula minimum. The hull is over *this dataset only* — without
    the elemental references it is a relative statement, which
    ``uns['phase_diagram']['closed_system']`` records rather than hides.

    requires: obs['energy_<level>']
    produces: obs['e_above_hull_<level>'], obs['is_stable_<level>'],
              uns['phase_diagram']
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    key = f"energy_{level}"
    if key not in md.obs:
        raise ValueError(f"obs[{key!r}] absent; run mv.calc.energy(level={level!r}) "
                         f"or mv.calc.relax(level={level!r}) first")
    S = structures(md, source)
    energies = md.obs[key].to_numpy(dtype=float)
    entries, idx = [], []
    for i, (s, e) in enumerate(zip(S, energies)):
        if e == e:                                   # skip NaN
            entries.append(ComputedEntry(s.composition, float(e)))
            idx.append(i)

    above = np.full(len(S), np.nan)
    stable = np.zeros(len(S), dtype=bool)
    elements = sorted({str(el) for s in S for el in s.composition.elements})
    try:
        pd_ = PhaseDiagram(entries)
        for j, i in enumerate(idx):
            above[i] = float(pd_.get_e_above_hull(entries[j]))
            stable[i] = entries[j] in pd_.stable_entries
        ok, why = True, None
    except Exception as exc:
        ok, why = False, f"{type(exc).__name__}: {exc}"

    md.obs[f"e_above_hull_{level}"] = above
    md.obs[f"is_stable_{level}"] = stable
    md.uns["phase_diagram"] = {"level": level, "elements": elements,
                               "n_entries": len(entries), "closed_system": True,
                               "built": ok, "error": why}
    record(md, f"thermo.hull(level={level}, source={source})")


__all__ = ["hull"]

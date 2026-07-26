"""``mv.calc`` — energies and relaxation, tagged by level of theory.

The ``level`` argument is the whole point. A formation energy computed with
PBE, with HSE06 and with a machine-learned potential are three different
quantities; in a flat table they are three columns whose names happen to
differ. Here the level of theory is the slot name and its settings live in
``uns['calc'][level]``, so comparing a surrogate against DFT requires naming
both rather than silently averaging them.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import deposit_structures, record, structures

_CALCULATORS = {}


def register_calculator(name: str, factory, **meta) -> None:
    """Make a calculator available to ``relax`` / ``energy`` under ``name``."""
    _CALCULATORS[name] = (factory, meta)


def _get(name: str):
    if name in _CALCULATORS:
        return _CALCULATORS[name]
    if name == "emt":
        from ase.calculators.emt import EMT
        return (EMT, {"method": "EMT",
                      "note": "ASE effective-medium theory; fast, and only "
                              "parameterised for Al Cu Ag Au Ni Pd Pt H C N O"})
    if name in ("mace", "mace-mp"):                # pragma: no cover
        from mace.calculators import mace_mp
        return (lambda: mace_mp(default_dtype="float64"),
                {"method": "MACE-MP-0", "surrogate": True})
    if name in ("chgnet",):                        # pragma: no cover
        from chgnet.model.dynamics import CHGNetCalculator
        return (CHGNetCalculator, {"method": "CHGNet", "surrogate": True})
    raise KeyError(f"unknown calculator {name!r}; have {sorted(_CALCULATORS) + ['emt','mace','chgnet']}")


def energy(md: AnnData, level: str = "emt", source: str = "input", **params) -> None:
    """Single-point energy for every structure.

    produces: obs['energy_<level>'], obs['energy_per_atom_<level>'],
              uns['calc'][level]
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    factory, meta = _get(level)
    ad = AseAtomsAdaptor()
    e, epa, failed = [], [], 0
    for s in structures(md, source):
        try:
            atoms = ad.get_atoms(s)
            atoms.calc = factory()
            val = float(atoms.get_potential_energy())
        except Exception:
            val, failed = np.nan, failed + 1
        e.append(val)
        epa.append(val / len(s) if val == val else np.nan)
    md.obs[f"energy_{level}"] = e
    md.obs[f"energy_per_atom_{level}"] = epa
    md.uns["calc"][level] = {**meta, **params, "source": source, "n_failed": failed}
    record(md, f"calc.energy(level={level}, source={source})")


def relax(md: AnnData, level: str = "emt", source: str = "input",
          fmax: float = 0.05, steps: int = 200, **params) -> None:
    """Relax every structure and deposit the result as its own variant.

    produces: uns['structures']['relaxed_<level>'],
              obs['energy_<level>'], obs['energy_per_atom_<level>'],
              obs['relax_converged_<level>'], uns['calc'][level]
    """
    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor
    factory, meta = _get(level)
    ad = AseAtomsAdaptor()
    out, e, epa, conv = [], [], [], []
    for s in structures(md, source):
        try:
            atoms = ad.get_atoms(s)
            atoms.calc = factory()
            opt = BFGS(atoms, logfile=None)
            ok = bool(opt.run(fmax=fmax, steps=steps))
            val = float(atoms.get_potential_energy())
            out.append(ad.get_structure(atoms))
        except Exception:
            ok, val = False, np.nan
            out.append(s)
        e.append(val)
        epa.append(val / len(s) if val == val else np.nan)
        conv.append(ok)
    deposit_structures(md, f"relaxed_{level}", out)
    md.obs[f"energy_{level}"] = e
    md.obs[f"energy_per_atom_{level}"] = epa
    md.obs[f"relax_converged_{level}"] = conv
    md.uns["calc"][level] = {**meta, **params, "fmax": fmax, "steps": steps,
                             "source": source}
    record(md, f"calc.relax(level={level}, source={source}, fmax={fmax})")


__all__ = ["energy", "relax", "register_calculator"]

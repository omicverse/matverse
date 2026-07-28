"""``mv.calc`` — energies and relaxation, tagged by level of theory.

The ``level`` argument is the whole point. An energy from PBE, from a model
trained to reproduce PBE+U, and from one trained on r2SCAN are three different
quantities; in a flat table they are three columns whose names happen to differ.
Here the level is the slot name and its record lives in ``uns['levels'][level]``,
so comparing a surrogate against DFT means naming both.

Choosing a level
----------------
``mv.calc.available()`` lists what this installation can run. matverse ships no
default beyond ``emt`` — the machine-learned potential leaderboard reorders every
few months and its current leaders are separated by less than the spread between
seeds, so a library that hardcodes "the best model" is stale on arrival and
wrong in a way its users cannot see. Register what you have; name it when you
call.

Three things are recorded that a bare calculator does not tell you:

``reference``
    what the level reproduces. A model trained on OMat24 targets PBE+U; one
    trained on MatPES targets r2SCAN. Mixing them is the same class of error as
    mixing PBE with HSE06 one level up, and ``surrogate: True`` alone no longer
    distinguishes them.

``license``
    weights are not uniformly open. MACE-MP and MACE-MPA are MIT; MACE-OMAT and
    MACE-MATPES are ASL and forbid commercial use; UMA's licence excludes
    several countries. :func:`check_licenses` reads it back off the object.

``uncertainty``
    where ``obs['energy_<level>_std']`` came from, when a level produces one.
    Active learning is unbuildable without it, and screening pipelines drop it
    constantly.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import (check_commercial_use, deposit_structures, record, set_level,
                    structures)
from ._registry import register_function

#: name -> (factory, metadata). Populated by :func:`register_calculator`.
_CALCULATORS: dict[str, tuple] = {}

#: Levels matverse knows how to build, if their backend happens to be installed.
BUILTIN_LEVELS = ("emt", "lj", "mace-mpa", "mace-omat", "sevennet", "chgnet")


@register_function(
    aliases=["register calculator", "add calculator", "custom potential",
             "register level of theory"],
    category="calc",
    description="Make a calculator available to mv.calc under a level name, "
                "together with the metadata every result computed at that level "
                "will carry.",
    examples=["mv.calc.register_calculator('myff', MyCalculator, kind='mlip', "
              "method='MyFF', reference='PBE', surrogate=True)"],
    related=["mv.calc.available", "mv.calc.energy"],
)
def register_calculator(name: str, factory, *, kind: str = "mlip",
                        method: str | None = None, reference: str | None = None,
                        surrogate: bool = True, license: str | None = None,
                        uncertainty: str | None = None, **extra) -> None:
    """Register a level of theory.

    ``factory`` is called with no arguments and must return an ASE calculator.
    Deferring construction matters: a foundation-model checkpoint costs seconds
    and hundreds of megabytes to instantiate, and registering should not.
    """
    _CALCULATORS[name] = (factory, {
        "kind": kind, "method": method or name, "reference": reference,
        "surrogate": bool(surrogate), "license": license,
        "uncertainty": uncertainty, **extra})


def _builtin(name: str):
    """Levels matverse can build, each importing its backend inside its branch.

    Optional dependencies are imported here rather than at module scope so that
    importing matverse costs nothing and a missing backend fails with an
    instruction instead of at import time.
    """
    if name == "emt":
        from ase.calculators.emt import EMT
        return (EMT, {
            "kind": "classical", "method": "EMT", "reference": None,
            "surrogate": True, "license": "LGPL-2.1", "uncertainty": None,
            "note": "ASE effective-medium theory. Parameterised only for Al Cu "
                    "Ag Au Ni Pd Pt H C N O — enough to exercise a pipeline, "
                    "not enough to screen with."})
    if name == "lj":
        from ase.calculators.lj import LennardJones
        return (LennardJones, {
            "kind": "classical", "method": "Lennard-Jones", "reference": None,
            "surrogate": True, "license": "LGPL-2.1", "uncertainty": None})
    if name in ("mace-mpa", "mace-mpa-0"):
        from mace.calculators import mace_mp
        return (lambda: mace_mp(model="medium-mpa-0", default_dtype="float64"),
                {"kind": "mlip", "method": "MACE-MPA-0",
                 "reference": "PBE+U (MPtrj + sAlex)", "surrogate": True,
                 "license": "MIT", "uncertainty": None})
    if name in ("mace-omat", "mace-omat-0"):
        from mace.calculators import mace_mp
        return (lambda: mace_mp(model="medium-omat-0", default_dtype="float64"),
                {"kind": "mlip", "method": "MACE-OMAT-0",
                 "reference": "PBE+U (OMat24)", "surrogate": True,
                 "license": "ASL", "uncertainty": None,
                 "note": "ASL forbids commercial use; MACE-MPA-0 is MIT."})
    if name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return (SevenNetCalculator, {
            "kind": "mlip", "method": "SevenNet", "reference": "PBE+U",
            "surrogate": True, "license": "GPL-3.0", "uncertainty": None})
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return (CHGNetCalculator, {
            "kind": "mlip", "method": "CHGNet", "reference": "PBE+U (MPtrj)",
            "surrogate": True, "license": "BSD-3-Clause", "uncertainty": None,
            "note": "Superseded on Matbench Discovery by OMat24-trained models; "
                    "kept because a great deal of published work used it."})
    raise KeyError(
        f"unknown level {name!r}. Runnable here: "
        f"{sorted(available(check_imports=False))}. Register your own with "
        f"mv.calc.register_calculator({name!r}, ...).")


def _get(level: str):
    if level in _CALCULATORS:
        return _CALCULATORS[level]
    return _builtin(level)


@register_function(
    aliases=["available calculators", "which levels", "list levels",
             "available levels of theory"],
    category="calc",
    description="List the levels of theory this installation can actually run, "
                "with the metadata each would record.",
    examples=["mv.calc.available()"],
    related=["mv.calc.register_calculator", "mv.calc.energy"],
)
def available(check_imports: bool = True) -> dict:
    """Levels runnable here, mapped to their metadata.

    With ``check_imports`` the built-in levels are constructed far enough to see
    whether their backend is installed — slower, and truthful. A level whose
    backend is missing appears with an ``unavailable`` key rather than silently
    vanishing, so the reason is visible.
    """
    out = {name: dict(meta) for name, (_, meta) in _CALCULATORS.items()}
    for name in BUILTIN_LEVELS:
        if name in out:
            continue
        try:
            _, meta = _builtin(name)
            out[name] = dict(meta)
        except Exception as exc:
            if check_imports:
                out[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return out


@register_function(
    aliases=["check licenses", "licence check", "commercial use",
             "non-commercial weights"],
    category="calc",
    description="Report which levels used in this object carry a licence that "
                "forbids commercial use.",
    requires={"levels": ["{level}"]},
    examples=["mv.calc.check_licenses(md)"],
    related=["mv.calc.energy", "mv.calc.available"],
)
def check_licenses(md: AnnData) -> list[str]:
    """Levels in this object whose licence forbids commercial use."""
    return check_commercial_use(md)


@register_function(
    aliases=["single point energy", "energy", "compute energy",
             "potential energy", "static calculation"],
    category="calc",
    description="Compute the single-point energy of every structure at one "
                "level of theory, leaving the geometry untouched.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["energy_{level}", "energy_per_atom_{level}"],
              "levels": ["{level}"]},
    dispatch="level= selects the calculator; each registered level is a "
             "separate route with its own reference method and licence",
    examples=["mv.calc.energy(md, level='emt')",
              "mv.calc.energy(md, level='mace-mpa')"],
    related=["mv.calc.relax", "mv.thermo.hull", "mv.calc.available"],
)
def energy(md: AnnData, level: str = "emt", source: str = "input",
           **params) -> None:
    """Single-point energy for every structure, in eV per cell."""
    from pymatgen.io.ase import AseAtomsAdaptor

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    calc = factory()

    e, epa, failed = [], [], 0
    for s in structures(md, source):
        try:
            atoms = adaptor.get_atoms(s)
            atoms.calc = calc
            val = float(atoms.get_potential_energy())
        except Exception:
            val, failed = float("nan"), failed + 1
        e.append(val)
        epa.append(val / len(s) if val == val else float("nan"))

    md.obs[f"energy_{level}"] = e
    md.obs[f"energy_per_atom_{level}"] = epa
    set_level(md, level, **meta, source=source, n_failed=failed, **params)
    record(md, "calc.energy", level=level, source=source)


@register_function(
    aliases=["relax", "geometry optimization", "geometry optimisation",
             "structure relaxation", "optimize structure", "ionic relaxation"],
    category="calc",
    description="Relax every structure at one level of theory, depositing the "
                "relaxed geometry as its own structure variant alongside the "
                "final energy and whether the optimiser converged.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{key_added}"],
              "obs": ["energy_{level}", "energy_per_atom_{level}",
                      "relax_converged_{level}", "max_force_{level}"],
              "levels": ["{level}"]},
    dispatch="level= selects the calculator, as for mv.calc.energy",
    examples=["mv.calc.relax(md, level='emt')",
              "mv.calc.relax(md, level='emt', source='hop_final', "
              "key_added='relaxed_final')"],
    related=["mv.calc.energy", "mv.thermo.hull"],
    notes="The relaxed geometry becomes a named variant rather than replacing "
          "the input, so 'which structure was this energy computed on' stays "
          "answerable from the object alone.\n\n"
          "key_added defaults to 'relaxed_<level>', which is the same name "
          "every time — so relaxing two variants at one level in sequence "
          "overwrites the first. Anything needing two relaxed geometries, such "
          "as the endpoints of an NEB, must name them apart.",
)
def relax(md: AnnData, level: str = "emt", source: str = "input",
          fmax: float = 0.05, steps: int = 200,
          key_added: str | None = None, **params) -> None:
    """Relax every structure and deposit the result as its own variant.

    ``key_added`` names the output variant, which matters whenever more than one
    variant is relaxed at the same level: the default ``relaxed_<level>`` is the
    same name each time, so relaxing two variants in sequence silently
    overwrites the first. Anything needing two relaxed geometries — an NEB
    between endpoints, a slab against its bulk — must name them apart.
    """
    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    calc = factory()

    out, e, epa, conv, maxf, failed = [], [], [], [], [], 0
    for s in structures(md, source):
        try:
            atoms = adaptor.get_atoms(s)
            atoms.calc = calc
            opt = BFGS(atoms, logfile=None)
            ok = bool(opt.run(fmax=fmax, steps=steps))
            val = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            maxf.append(float(np.sqrt((forces ** 2).sum(axis=1)).max())
                        if len(forces) else 0.0)
            out.append(adaptor.get_structure(atoms))
        except Exception:
            ok, val, failed = False, float("nan"), failed + 1
            maxf.append(float("nan"))
            out.append(s)
        e.append(val)
        epa.append(val / len(s) if val == val else float("nan"))
        conv.append(ok)

    variant = key_added or f"relaxed_{level}"
    deposit_structures(md, variant, out)
    md.obs[f"energy_{level}"] = e
    md.obs[f"energy_per_atom_{level}"] = epa
    md.obs[f"relax_converged_{level}"] = conv
    md.obs[f"max_force_{level}"] = maxf
    set_level(md, level, **meta, source=source, fmax=fmax, steps=steps,
              n_failed=failed, **params)
    record(md, "calc.relax", level=level, source=source, fmax=fmax,
           key_added=variant)


@register_function(
    aliases=["forces", "atomic forces", "per site forces", "force on each atom",
             "site forces"],
    category="calc",
    description="Compute the force on every atom at one level of theory and "
                "deposit it onto the sites axis, where per-atom results are a "
                "matrix rather than ragged records.",
    requires={"structures": ["{source}"]},
    produces={"levels": ["{level}"]},
    prerequisites=["mv.multi.sites"],
    dispatch="level= selects the calculator, as for mv.calc.energy",
    examples=["sites = mv.multi.sites(md); mv.calc.forces(md, sites, "
              "level='emt')"],
    related=["mv.multi.sites", "mv.multi.aggregate", "mv.calc.relax"],
    notes="Takes both objects because forces need the structures, which live on "
          "the material axis, and produce one row per atom, which lives on the "
          "sites axis. Writes sites.obsm['forces_{level}'] and "
          "sites.obs['force_magnitude_{level}'] — slots on the object passed "
          "as `sites`, which the contract fields cannot name because they "
          "describe one object only.",
)
def forces(md: AnnData, sites: AnnData, level: str = "emt",
           source: str = "input", **params) -> None:
    """Per-atom forces, in eV/angstrom, onto the sites axis."""
    from pymatgen.io.ase import AseAtomsAdaptor

    from .multi import AXIS_KEY

    if sites.uns.get(AXIS_KEY) != "sites":
        raise ValueError("second argument must be a sites object from "
                         "mv.multi.sites(md)")
    if sites.uns.get("parent", {}).get("source") != source:
        raise ValueError(
            f"the sites object was built from {sites.uns.get('parent', {}).get('source')!r} "
            f"but forces were asked for on {source!r}; the atoms would not "
            f"line up. Rebuild with mv.multi.sites(md, source={source!r}).")

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    calc = factory()

    rows, failed = [], 0
    for s in structures(md, source):
        try:
            atoms = adaptor.get_atoms(s)
            atoms.calc = calc
            rows.append(np.asarray(atoms.get_forces(), dtype=float))
        except Exception:
            rows.append(np.full((len(s), 3), np.nan))
            failed += 1

    stacked = np.vstack(rows) if rows else np.zeros((0, 3))
    if len(stacked) != sites.n_obs:
        raise ValueError(
            f"computed {len(stacked)} force vectors for {sites.n_obs} sites; "
            f"the sites object is out of date with the structures")

    sites.obsm[f"forces_{level}"] = stacked
    sites.obs[f"force_magnitude_{level}"] = np.sqrt((stacked ** 2).sum(axis=1))
    set_level(md, level, **meta, source=source, n_failed=failed, **params)
    set_level(sites, level, **meta, source=source, n_failed=failed, **params)
    record(md, "calc.forces", level=level, source=source)
    record(sites, "calc.forces", level=level, source=source)


@register_function(
    aliases=["committee", "ensemble uncertainty", "model uncertainty",
             "predictive variance", "error bar"],
    category="calc",
    description="Combine several levels into a consensus level, recording the "
                "mean energy and the spread across the committee as an "
                "uncertainty for active learning.",
    requires={"obs": ["energy_per_atom_{level}"]},
    produces={"obs": ["energy_per_atom_{key}", "energy_per_atom_{key}_std"],
              "levels": ["{key}"]},
    prerequisites=["mv.calc.energy"],
    examples=["mv.calc.committee(md, ['mace-mpa', 'chgnet'], key='ensemble')"],
    related=["mv.calc.energy", "mv.screen.rank"],
    notes="Committee spread is a proxy for error, not a calibrated one. It is "
          "useful for ranking what to compute next and should not be reported "
          "as an error bar without calibration.",
)
def committee(md: AnnData, levels: list[str], key: str = "ensemble") -> None:
    """Mean and spread of energy per atom across several levels."""
    cols = [f"energy_per_atom_{lv}" for lv in levels]
    missing = [c for c in cols if c not in md.obs]
    if missing:
        raise ValueError(
            f"obs column(s) {missing} absent; run mv.calc.energy for each of "
            f"{list(levels)} first")
    M = np.column_stack([md.obs[c].to_numpy(dtype=float) for c in cols])
    with np.errstate(invalid="ignore"):
        md.obs[f"energy_per_atom_{key}"] = np.nanmean(M, axis=1)
        md.obs[f"energy_per_atom_{key}_std"] = np.nanstd(M, axis=1, ddof=0)
    set_level(md, key, kind="model", method=f"committee of {len(levels)}",
              reference=None, surrogate=True, license=None,
              uncertainty="committee spread (uncalibrated)",
              members=list(levels))
    record(md, "calc.committee", levels=list(levels), key=key)


__all__ = ["energy", "relax", "forces", "committee", "register_calculator",
           "available", "check_licenses"]

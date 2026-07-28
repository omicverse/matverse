"""``mv.mag`` — magnetic order, and why a hull needs it.

Every energy computed so far in matverse assumed one thing without saying it:
that the magnetic configuration handed to the calculator is the right one. For
anything containing iron, cobalt, nickel, manganese or chromium that assumption
is usually wrong, and it is wrong by an amount that matters.

The ferromagnetic and antiferromagnetic states of a transition-metal compound
can differ by hundreds of meV per atom. A hull built from whichever ordering the
input file happened to carry is a hull of the wrong quantity, and the error does
not average out — it is systematic in the direction of whichever ordering was
guessed.

So the workflow is: enumerate the plausible orderings, compute all of them,
keep the lowest, and record how far apart they were. That last part is the one
people skip, and it is the one that tells you whether the guess would have
mattered.

    orderings = mv.mag.orderings(md)              # more rows than parents
    mv.calc.relax(orderings, level='mace-mpa')
    mv.mag.ground_state(orderings, md, level='mace-mpa')
    md.obs['magnetic_spread_mace-mpa']            # how much the choice was worth

Collinear only
--------------
Spins are up or down along one axis. Non-collinear order, spin-orbit coupling
and spin spirals are a different calculation with a different cost, and
pretending otherwise by enumerating collinear states and calling the lowest one
"the ground state" would overclaim. What this gives you is the best collinear
guess, which is what a screen wants and what almost every high-throughput study
actually uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function

#: Elements whose magnetic ordering is usually worth enumerating. Not a physical
#: law — a screening heuristic, and :func:`orderings` will happily run on
#: anything you point it at.
MAGNETIC_ELEMENTS = frozenset({
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "V", "Ti",
    "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "U", "Np", "Pu",
})


@register_function(
    aliases=["magnetic orderings", "enumerate magnetic", "spin configurations",
             "ferromagnetic", "antiferromagnetic", "magnetic states",
             "spin ordering"],
    category="mag",
    description="Enumerate the plausible collinear magnetic orderings of every "
                "structure — ferromagnetic, antiferromagnetic and ferrimagnetic "
                "— and return them as a new dataset for the calculator to rank.",
    requires={"structures": ["{source}"]},
    examples=["orderings = mv.mag.orderings(md)",
              "orderings = mv.mag.orderings(md, strategies="
              "('ferromagnetic', 'antiferromagnetic', 'ferrimagnetic_by_motif'))"],
    related=["mv.mag.ground_state", "mv.calc.relax", "mv.thermo.hull"],
    notes="Returns rather than deposits, because one structure gives many "
          "orderings. obs['parent'] points back at the material and "
          "obs['ordering'] names the state, so the whole set can be relaxed at "
          "once and collapsed afterwards with mv.mag.ground_state.\n\n"
          "Structures with no magnetic element are passed through as a single "
          "non-magnetic row rather than dropped, so the returned dataset covers "
          "every input and nothing needs re-joining by hand.",
)
def orderings(md: AnnData, source: str = "input",
              strategies=("ferromagnetic", "antiferromagnetic"),
              max_orderings: int = 8,
              default_magmoms: dict | None = None) -> AnnData:
    """Enumerate collinear magnetic orderings. Returns a new dataset."""
    from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

    from .data import from_structures

    built, rows, failed = [], [], []
    for name, structure in zip(md.obs_names, structures(md, source)):
        symbols = {str(el) for el in structure.composition.elements}
        if not (symbols & MAGNETIC_ELEMENTS):
            built.append(structure)
            rows.append({"parent": str(name), "ordering": "nonmagnetic",
                         "ordering_index": 0, "total_magmom": 0.0,
                         "is_magnetic": False})
            continue

        candidates, labels, why = _enumerate(
            structure, strategies, default_magmoms, max_orderings)
        if why:
            failed.append(f"{name}: {why}")
        if not candidates:
            built.append(structure)
            rows.append({"parent": str(name), "ordering": "unenumerated",
                         "ordering_index": 0, "total_magmom": np.nan,
                         "is_magnetic": True})
            continue

        for j, (candidate, label) in enumerate(zip(candidates[:max_orderings],
                                                   labels[:max_orderings])):
            built.append(candidate)
            rows.append({"parent": str(name),
                         "ordering": str(label or f"ordering_{j}"),
                         "ordering_index": j,
                         "total_magmom": _total_magmom(candidate),
                         "is_magnetic": True})

    if not built:
        raise ValueError(f"no ordering was generated; {len(failed)} structures "
                         f"failed: {failed[:3]}")

    out = from_structures(built, pd.DataFrame(rows))
    out.uns["magnetic_orderings"] = {
        "source": source, "strategies": list(strategies),
        "max_orderings": max_orderings, "n_parents": int(md.n_obs),
        "collinear": True, "errors": failed,
    }
    record(out, "mag.orderings", source=source, strategies=list(strategies),
           n_parents=int(md.n_obs))
    return out


def _enumerate(structure, strategies, default_magmoms, max_orderings):
    """Orderings for one structure, degrading rather than failing.

    pymatgen's enumerator is tried first and is the better tool, but its
    antiferromagnetic strategies call out to ``enumlib`` — Fortran executables
    that are not pip-installable and are absent from most environments. When
    they are missing it raises rather than returning the ferromagnetic state it
    could have produced.

    So the fallback generates sign assignments on the magnetic sublattice
    directly: ferromagnetic, the balanced up/down splits that cover Neel-type
    order, and the unbalanced ones that cover ferrimagnetic order. Coarser than
    enumlib, dependency-free, and enough for a screen to notice that the choice
    of ordering matters.
    """
    from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

    try:
        enumerator = MagneticStructureEnumerator(
            structure, default_magmoms=default_magmoms,
            strategies=tuple(strategies), automatic=True)
        candidates = list(enumerator.ordered_structures)
        labels = [str(v) for v in getattr(
            enumerator, "ordered_structure_origins", [""] * len(candidates))]
        if candidates:
            return candidates[:max_orderings], labels[:max_orderings], ""
        why = "enumerator returned nothing"
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}".split("\n")[0]

    candidates, labels = _sign_assignments(structure, max_orderings)
    if candidates:
        return candidates, labels, f"{why}; used the built-in fallback"
    return [], [], why


def _sign_assignments(structure, max_orderings: int):
    """Ferromagnetic, antiferromagnetic and ferrimagnetic sign assignments.

    Deduplicated by structure matching, so two assignments that are the same
    state seen from different labellings count once.
    """
    from itertools import combinations

    from pymatgen.analysis.structure_matcher import StructureMatcher

    sites = [i for i, site in enumerate(structure)
             if str(site.specie.symbol) in MAGNETIC_ELEMENTS]
    if not sites:
        return [], []

    moments = [_default_moment(structure[i].specie.symbol) for i in sites]
    patterns: list[tuple[str, list[float]]] = [
        ("fm", [+m for m in moments])]

    n = len(sites)
    for k in range(1, n):
        for flipped in combinations(range(n), k):
            signs = [-1.0 if j in flipped else 1.0 for j in range(n)]
            total = sum(s * m for s, m in zip(signs, moments))
            kind = "afm" if abs(total) < 1e-6 else "fim"
            patterns.append((kind, [s * m for s, m in zip(signs, moments)]))
        if len(patterns) > 4 * max_orderings:
            break

    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False)
    kept, labels = [], []
    for kind, assignment in patterns:
        candidate = structure.copy()
        magmoms = [0.0] * len(candidate)
        for site_index, value in zip(sites, assignment):
            magmoms[site_index] = float(value)
        candidate.add_site_property("magmom", magmoms)

        if any(matcher.fit(candidate, seen) and
               _same_moments(candidate, seen) for seen in kept):
            continue
        kept.append(candidate)
        labels.append(kind)
        if len(kept) >= max_orderings:
            break
    return kept, labels


def _same_moments(a, b) -> bool:
    left = np.asarray(a.site_properties.get("magmom", []), dtype=float)
    right = np.asarray(b.site_properties.get("magmom", []), dtype=float)
    return left.shape == right.shape and np.allclose(np.sort(left),
                                                     np.sort(right))


def _default_moment(symbol) -> float:
    """A starting moment for a magnetic species, in Bohr magnetons.

    These are initial guesses for the calculator to relax, not predictions. The
    values follow pymatgen's own defaults where it has them.
    """
    defaults = {"Fe": 5.0, "Co": 5.0, "Ni": 5.0, "Mn": 5.0, "Cr": 5.0,
                "V": 5.0, "Ti": 5.0, "Cu": 1.0}
    return defaults.get(str(symbol), 5.0)


def _total_magmom(structure) -> float:
    """Net moment per cell, from the site magmoms the enumerator assigned."""
    moments = structure.site_properties.get("magmom")
    if moments is None:
        return float("nan")
    return float(np.sum(np.asarray(moments, dtype=float)))


@register_function(
    aliases=["magnetic ground state", "ground state", "lowest ordering",
             "collapse orderings", "pick magnetic state"],
    category="mag",
    description="Pick the lowest-energy ordering for each material, copy its "
                "energy back onto the parent, and record how far apart the "
                "orderings were.",
    requires={"obs": ["energy_per_atom_{level}", "parent", "ordering"]},
    produces={"obs": ["magnetic_ordering_{level}",
                      "magnetic_spread_{level}",
                      "energy_per_atom_{level}", "total_magmom_{level}",
                      "is_ground_state_{level}"]},
    prerequisites=["mv.mag.orderings", "mv.calc.relax"],
    examples=["mv.mag.ground_state(orderings, md, level='emt')"],
    related=["mv.mag.orderings", "mv.thermo.hull"],
    notes="obs['magnetic_spread'] is the gap between the best and worst "
          "ordering, and it is the number worth looking at before trusting any "
          "hull. A spread of 5 meV/atom means the choice did not matter; one of "
          "300 meV/atom means a hull built from the wrong ordering is a hull of "
          "a different material.\n\n"
          "Writes the winning energy onto the parent under the same "
          "energy_per_atom_<level> name the calculator would have produced, so "
          "mv.thermo.hull needs no special case — it sees an ordinary column "
          "that happens to be the magnetic ground state.",
)
def ground_state(orderings_: AnnData, md: AnnData, level: str = "emt",
                 source: str = "input") -> None:
    """Collapse enumerated orderings onto their parents, keeping the lowest."""
    energy_key = f"energy_per_atom_{level}"
    if energy_key not in orderings_.obs:
        raise ValueError(
            f"obs[{energy_key!r}] absent on the orderings; run "
            f"mv.calc.relax(orderings, level={level!r}) first")
    for column in ("parent", "ordering"):
        if column not in orderings_.obs:
            raise ValueError(f"obs[{column!r}] absent; these did not come from "
                             f"mv.mag.orderings")

    energies = orderings_.obs[energy_key].to_numpy(dtype=float)
    parents = orderings_.obs["parent"].astype(str).to_numpy()
    labels = orderings_.obs["ordering"].astype(str).to_numpy()
    moments = (orderings_.obs["total_magmom"].to_numpy(dtype=float)
               if "total_magmom" in orderings_.obs
               else np.full(len(energies), np.nan))

    best_energy = np.full(md.n_obs, np.nan)
    best_label = [""] * md.n_obs
    best_moment = np.full(md.n_obs, np.nan)
    spread = np.full(md.n_obs, np.nan)
    is_best = np.zeros(orderings_.n_obs, dtype=bool)

    for i, name in enumerate(map(str, md.obs_names)):
        rows = np.where((parents == name) & np.isfinite(energies))[0]
        if not len(rows):
            continue
        winner = rows[int(np.argmin(energies[rows]))]
        is_best[winner] = True
        best_energy[i] = float(energies[winner])
        best_label[i] = str(labels[winner])
        best_moment[i] = float(moments[winner])
        if len(rows) > 1:
            spread[i] = float(energies[rows].max() - energies[rows].min())

    md.obs[energy_key] = best_energy
    md.obs[f"magnetic_ordering_{level}"] = best_label
    md.obs[f"magnetic_spread_{level}"] = spread
    md.obs[f"total_magmom_{level}"] = best_moment
    orderings_.obs[f"is_ground_state_{level}"] = is_best

    md.uns.setdefault("magnetic", {})[level] = {
        "n_with_alternatives": int(np.isfinite(spread).sum()),
        "max_spread": float(np.nanmax(spread)) if np.isfinite(spread).any()
        else None,
        "collinear": True,
        "note": "magnetic_spread is the gap between the best and worst "
                "ordering; a large one means the hull depends on this choice",
    }
    record(md, "mag.ground_state", level=level,
           n_orderings=int(orderings_.n_obs))
    record(orderings_, "mag.ground_state", level=level)


@register_function(
    aliases=["magnetic analysis", "is magnetic", "magnetic moments",
             "describe magnetism", "spin analysis"],
    category="mag",
    description="Record the magnetic character of every structure — total and "
                "absolute moment, whether the ordering is ferromagnetic or "
                "antiferromagnetic, and how many magnetic species it contains.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["total_magmom", "absolute_magmom", "magnetic_order",
                      "n_magnetic_species"]},
    examples=["mv.mag.describe(md)"],
    related=["mv.mag.orderings"],
    notes="Reads the magnetic moments already on the structure — from an "
          "enumerator, from a DFT run, or from a magnetism-aware potential. It "
          "does not invent them: a structure carrying no moments is recorded as "
          "having none rather than being guessed at.",
)
def describe(md: AnnData, source: str = "input") -> None:
    """Magnetic character from the moments a structure already carries."""
    from pymatgen.analysis.magnetism.analyzer import (
        CollinearMagneticStructureAnalyzer)

    total, absolute, order, n_species = [], [], [], []
    for structure in structures(md, source):
        moments = structure.site_properties.get("magmom")
        if moments is None:
            total.append(np.nan)
            absolute.append(np.nan)
            order.append("unknown")
        else:
            values = np.asarray(moments, dtype=float)
            total.append(float(values.sum()))
            absolute.append(float(np.abs(values).sum()))
            try:
                order.append(str(CollinearMagneticStructureAnalyzer(
                    structure).ordering.value))
            except Exception:
                order.append(_order_from_moments(values))
        n_species.append(int(len({str(el) for el
                                  in structure.composition.elements}
                                 & MAGNETIC_ELEMENTS)))

    md.obs["total_magmom"] = total
    md.obs["absolute_magmom"] = absolute
    md.obs["magnetic_order"] = order
    md.obs["n_magnetic_species"] = n_species
    record(md, "mag.describe", source=source)


def _order_from_moments(values: np.ndarray) -> str:
    """A fallback classification when pymatgen's analyser cannot decide."""
    significant = values[np.abs(values) > 0.05]
    if not len(significant):
        return "NM"
    if (significant > 0).all() or (significant < 0).all():
        return "FM"
    return "FiM" if abs(significant.sum()) > 0.05 else "AFM"


__all__ = ["orderings", "ground_state", "describe", "MAGNETIC_ELEMENTS"]

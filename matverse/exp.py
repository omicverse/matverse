"""``mv.exp`` — measured data, on the same footing as computed data.

Experiment is a level of theory. That sounds like a slogan and is actually the
whole implementation: ``uns['levels']['experiment']`` records an instrument
instead of a functional, a measured pattern lands in ``obsm['xrd_experiment']``
next to the computed ``obsm['xrd_calc']``, and every comparison written for two
computed levels works unchanged on a computed one against a measured one.

Nothing new was needed to make that true, which is the argument for having typed
the level of theory in the first place.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import deposit_grid, grid_of, record, set_level
from ._registry import register_function


@register_function(
    aliases=["attach measurement", "add measured spectrum", "measured data",
             "experimental pattern", "attach experiment"],
    category="exp",
    description="Attach a measured curve for every material onto the same grid "
                "as its computed counterpart, recording the instrument as a "
                "level of theory.",
    produces={"obsm": ["{quantity}_{level}"], "levels": ["{level}"]},
    prerequisites=["mv.prop.xrd"],
    examples=["mv.exp.attach(md, 'xrd', patterns, two_theta, "
              "instrument='Bruker D8')"],
    related=["mv.prop.compare_grids", "mv.exp.match_xrd"],
    notes="Resampled onto the existing grid rather than stored on its own, "
          "because two curves on different grids cannot be subtracted and a "
          "resample done once is better than one done in every comparison. "
          "Attach a measurement at its own resolution or better: diffraction "
          "peaks are narrow, and resampling onto a grid coarser than the peak "
          "width discards them permanently. Points outside the measured range "
          "become NaN rather than zero, and comparisons use the overlap.\n\n"
          "Claims no requires. It used to claim uns['grids'], and probing "
          "deleted it: when the quantity has no grid yet the measured one "
          "becomes it, so a measurement can be attached before anything was "
          "computed.",
)
def attach(md: AnnData, quantity: str, values, grid, level: str = "experiment",
           instrument: str | None = None, unit: str = "",
           **meta) -> None:
    """Attach measured curves, resampled onto the quantity's existing grid."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    grid = np.asarray(grid, dtype=float)
    if values.shape[0] != md.n_obs:
        raise ValueError(
            f"got {values.shape[0]} measured curves for {md.n_obs} materials; "
            f"pass one row per material, or use mv.exp.match_xrd for a single "
            f"pattern you want ranked against every candidate")

    target = _target_grid(md, quantity, grid, unit)
    resampled = np.vstack([np.interp(target, grid, row, left=np.nan,
                                     right=np.nan) for row in values])

    deposit_grid(md, quantity, level, resampled, target, unit=unit)
    set_level(md, level, kind="experiment", method=instrument or "measurement",
              reference=None, surrogate=False, license=None,
              uncertainty="measured", instrument=instrument, **meta)
    record(md, "exp.attach", quantity=quantity, level=level,
           instrument=instrument)


def _target_grid(md: AnnData, quantity: str, grid: np.ndarray, unit: str):
    """The grid to resample onto — the existing one, or this one if it is new."""
    try:
        return grid_of(md, quantity)
    except KeyError:
        return grid


@register_function(
    aliases=["match xrd", "phase identification", "identify phase",
             "search match", "which phase is this", "index a pattern"],
    category="exp",
    description="Rank every candidate in the object by how well its simulated "
                "diffraction pattern matches one measured pattern — phase "
                "identification against the candidate set you already have.",
    requires={"obsm": ["xrd_{level}"]},
    produces={"obs": ["xrd_match", "xrd_match_rank"], "uns": ["xrd_match"]},
    prerequisites=["mv.prop.xrd"],
    examples=["mv.exp.match_xrd(md, intensity, two_theta)"],
    related=["mv.prop.xrd", "mv.screen.rank"],
    notes="Scores against the candidates in this object and nothing else. A "
          "high score means 'the best of what you gave it', not 'identified' — "
          "the true phase can be absent from the library entirely.",
)
def match_xrd(md: AnnData, intensity, two_theta, level: str = "calc",
              key_added: str = "xrd_match") -> None:
    """Rank candidates against one measured powder pattern.

    Both patterns are baseline-shifted to zero and unit-normalised before the
    dot product, so the score reflects peak positions and relative heights
    rather than absolute counts or exposure time.
    """
    key = f"xrd_{level}"
    if key not in md.obsm:
        raise ValueError(f"obsm[{key!r}] absent; run mv.prop.xrd(md, "
                         f"level={level!r}) first")

    grid = grid_of(md, "xrd")
    measured = np.interp(grid, np.asarray(two_theta, dtype=float),
                         np.asarray(intensity, dtype=float),
                         left=0.0, right=0.0)
    computed = np.asarray(md.obsm[key], dtype=float)

    m = _unit(measured - np.nanmin(measured))
    C = np.vstack([_unit(row - np.nanmin(row)) if np.isfinite(row).all()
                   else np.full(len(grid), np.nan) for row in computed])
    with np.errstate(invalid="ignore"):
        score = C @ m

    md.obs[key_added] = score
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    ranks[np.isnan(score)] = np.nan
    md.obs[f"{key_added}_rank"] = ranks

    best = int(order[0]) if len(order) else None
    md.uns[key_added] = {
        "level": level,
        "n_candidates": int(md.n_obs),
        "best": str(md.obs_names[best]) if best is not None else None,
        "best_score": float(score[best]) if best is not None else None,
        "scored_against": "this object's candidates only",
    }
    record(md, "exp.match_xrd", level=level, key_added=key_added)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


@register_function(
    aliases=["measured property", "attach measurement column",
             "experimental value", "measured scalar"],
    category="exp",
    description="Attach a measured scalar property as its own level of theory, "
                "so it sits beside the computed values of the same quantity and "
                "can be compared with mv.compare_levels.",
    produces={"obs": ["{quantity}_{level}"], "levels": ["{level}"]},
    examples=["mv.exp.measure(md, 'band_gap', values, instrument='UV-Vis')"],
    related=["mv.compare_levels", "mv.exp.attach"],
    notes="The point of the naming convention. Once a measurement is a level, "
          "mv.compare_levels(md, 'band_gap') puts PBE, HSE06 and the "
          "spectrometer in one table without anyone deciding which is the "
          "band gap.",
)
def measure(md: AnnData, quantity: str, values, level: str = "experiment",
            instrument: str | None = None, uncertainty=None, **meta) -> None:
    """Attach a measured scalar, tagged as an experimental level."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] != md.n_obs:
        raise ValueError(f"got {values.shape[0]} values for {md.n_obs} "
                         f"materials")
    md.obs[f"{quantity}_{level}"] = values
    if uncertainty is not None:
        md.obs[f"{quantity}_{level}_std"] = np.asarray(uncertainty, dtype=float)
    set_level(md, level, kind="experiment", method=instrument or "measurement",
              reference=None, surrogate=False, license=None,
              uncertainty="measured" if uncertainty is not None else None,
              instrument=instrument, **meta)
    record(md, "exp.measure", quantity=quantity, level=level,
           instrument=instrument)


__all__ = ["attach", "match_xrd", "measure"]


#: kJ/mol -> eV, the conversion ExpEntry does not do.
_KJ_PER_MOL_TO_EV = 1.0 / 96.48533212331

#: What a formation enthalpy may be quoted in, as a multiplier to eV.
_ENTHALPY_UNITS = {
    "kJ/mol": _KJ_PER_MOL_TO_EV,
    "kcal/mol": 4.184 * _KJ_PER_MOL_TO_EV,
    "eV": 1.0,
    "eV/atom": None,          # handled separately: already intensive
    "meV/atom": None,
}


@register_function(
    aliases=["experimental hull", "measured stability", "formation enthalpy "
             "hull", "experimental phase diagram", "exp entries",
             "is it really stable", "thermochemical hull"],
    category="exp",
    description="Build a convex hull out of measured formation enthalpies, so "
                "stability from experiment sits on the same object, and the "
                "same axis, as stability from calculation.",
    requires={"obs": ["{column}"], "structures": ["{source}"]},
    produces={"obs": ["e_above_hull_{level}", "is_stable_{level}",
                      "formation_energy_{level}"]},
    prerequisites=["mv.exp.measure"],
    examples=["mv.exp.formation_hull(md, 'dHf', unit='kJ/mol')",
              "mv.exp.formation_hull(md, 'dHf_ev_per_atom', "
              "unit='eV/atom', level='janaf')"],
    related=["mv.thermo.hull", "mv.exp.measure", "mv.compare_levels"],
    notes="**The unit is a required argument and it is not guessed.** A "
          "formation enthalpy from a thermochemical table is in kJ/mol of "
          "formula unit; a hull is in eV per atom; the two differ by 96.485 "
          "and by the number of atoms in the formula. Getting that wrong "
          "produces a hull that still ranks, still plots and is wrong by two "
          "orders of magnitude, which is exactly what pymatgen's ExpEntry "
          "does — it hands ThermoData.value straight to PDEntry as an eV "
          "energy and ThermoData carries no unit to check against. Hence "
          "this, rather than a wrapper.\n\n"
          "Elements are the reference and are pinned to zero formation "
          "enthalpy whether or not you supply a value for them, which is what "
          "makes the numbers formation enthalpies in the first place. They do "
          "not need to be rows: a hull over an oxide needs an O2 reservoir, "
          "and ExpEntry refuses to hold one because it rejects any phase "
          "marked gas or liquid.\n\n"
          "What this is for is the comparison. Run it beside mv.thermo.hull "
          "at a computed level and mv.compare_levels will put measurement and "
          "calculation side by side on the same rows — which is the only "
          "honest way to find out whether a functional is right about "
          "stability, as opposed to self-consistent.",
)
def formation_hull(md: AnnData, column: str, unit: str,
                   source: str = "input", level: str = "experiment",
                   key_added: str | None = None) -> None:
    """A convex hull from measured formation enthalpies. Deposits ``None``."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition, Element

    from ._core import structures

    if column not in md.obs:
        raise ValueError(f"obs[{column!r}] absent; attach the measured "
                         f"enthalpies with mv.exp.measure first")
    if unit not in _ENTHALPY_UNITS:
        raise ValueError(f"unit must be one of {sorted(_ENTHALPY_UNITS)}, got "
                         f"{unit!r}; a formation enthalpy without a unit is "
                         f"not a number")

    values = md.obs[column].to_numpy(dtype=float)
    compositions = [s.composition for s in structures(md, source)]

    # Everything becomes eV per atom, which is what a hull is in.
    per_atom = np.full(md.n_obs, np.nan)
    for row, composition in enumerate(compositions):
        value = values[row]
        if not np.isfinite(value):
            continue
        if unit == "eV/atom":
            per_atom[row] = value
        elif unit == "meV/atom":
            per_atom[row] = value / 1000.0
        else:
            # Per formula unit, so divide by the atoms in the formula the
            # enthalpy was quoted for - the reduced formula, not the cell.
            reduced, factor = composition.get_reduced_composition_and_factor()
            per_atom[row] = (value * _ENTHALPY_UNITS[unit]
                             / float(reduced.num_atoms))

    entries, rows = [], []
    for row, composition in enumerate(compositions):
        if not np.isfinite(per_atom[row]):
            continue
        reduced, _ = composition.get_reduced_composition_and_factor()
        entries.append(PDEntry(reduced,
                               per_atom[row] * float(reduced.num_atoms)))
        rows.append(row)

    if not entries:
        raise ValueError(f"no finite value in obs[{column!r}]; nothing to "
                         f"build a hull from")

    # The elemental references. A formation enthalpy is measured against them,
    # so they are zero by definition and are added whether or not the caller
    # supplied a row for them - an oxide hull needs its O2 corner and there is
    # rarely a row for oxygen gas.
    elements = {element for entry in entries
                for element in entry.composition.elements}
    supplied = {next(iter(e.composition.elements)) for e in entries
                if e.composition.is_element}
    for element in sorted(elements - supplied, key=str):
        entries.append(PDEntry(Composition({Element(str(element)): 1}), 0.0))

    diagram = PhaseDiagram(entries)

    name = key_added or level
    above = np.full(md.n_obs, np.nan)
    for entry, row in zip(entries, rows):
        above[row] = float(diagram.get_e_above_hull(entry))

    md.obs[f"formation_energy_{name}"] = per_atom
    md.obs[f"e_above_hull_{name}"] = above
    md.obs[f"is_stable_{name}"] = np.where(
        np.isfinite(above), above <= 1e-9, False)
    set_level(md, name, kind="experiment",
              method="measured formation enthalpy",
              note=f"convex hull built from obs[{column!r}], quoted in "
                   f"{unit}, converted to eV/atom")
    md.uns.setdefault("experimental_hull", {})[name] = {
        "column": column, "unit": unit,
        "n_entries": len(entries), "n_rows": len(rows),
        "elements": sorted(str(e) for e in elements),
        "stable": sorted(e.composition.reduced_formula
                         for e in diagram.stable_entries),
    }
    record(md, "exp.formation_hull", column=column, unit=unit, level=name)

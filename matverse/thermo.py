"""``mv.thermo`` — thermodynamic stability.

``e_above_hull`` is only meaningful within one level of theory, and only
absolute when the hull includes the competing phases the material could decay
into. Both conditions are enforced here rather than assumed.

Closed versus referenced hulls
------------------------------
A hull built only from the candidates in one dataset is a *relative* statement:
it says which of these is lowest, not whether any of them is stable. Screening
40 generated oxides against each other will happily report several as "on the
hull" when all 40 decompose. Pass ``references=`` to make it absolute, and read
``uns['phase_diagram']['closed_system']`` to know which kind of number you have.

Mixing levels
-------------
Reference entries from Materials Project are PBE+U with fitted corrections.
Candidate energies from a machine-learned potential are whatever that model was
trained to reproduce. Putting them on one hull is the error the level system
exists to catch, so :func:`hull` compares ``uns['levels'][level]['reference']``
against the references' own and refuses by default when they disagree.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function


class LevelMismatch(ValueError):
    """Candidate energies and reference energies are not the same quantity."""


@register_function(
    aliases=["convex hull", "energy above hull", "thermodynamic stability",
             "e above hull", "is stable", "phase diagram"],
    category="thermo",
    description="Build the convex hull of energies at one level of theory and "
                "record each material's distance above it, together with what "
                "it would decompose into.",
    requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
    produces={"obs": ["e_above_hull_{level}", "is_stable_{level}",
                      "formation_energy_{level}", "decomposes_to_{level}"],
              "uns": ["phase_diagram"]},
    prerequisites=["mv.calc.energy"],
    dispatch="references= chooses the hull's scope: None is a closed hull over "
             "this dataset only; a list of entries or another matverse object "
             "makes it absolute",
    examples=["mv.thermo.hull(md, level='emt')",
              "mv.thermo.hull(md, level='mace-mpa', references=known_phases)"],
    related=["mv.calc.energy", "mv.screen.filter", "mv.thermo.references_from_mp"],
    notes="A closed hull is a relative statement and is recorded as "
          "uns['phase_diagram']['closed_system'] rather than hidden. Without "
          "elemental references, formation energies are not computed at all. "
          "A claim on uns['levels'][level] was probed and deleted: the level "
          "record is only read when references= is given, so the hull does not "
          "require it on the default path.",
)
def hull(md: AnnData, level: str = "emt", source: str = "input",
         references=None, allow_level_mismatch: bool = False) -> None:
    """Distance above the convex hull, at one level of theory.

    ``references`` may be a list of pymatgen ``ComputedEntry``, another matverse
    object carrying energies at the same level, or ``None`` for a hull closed
    over this dataset.
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    key = f"energy_{level}"
    if key not in md.obs:
        raise ValueError(
            f"obs[{key!r}] absent; run mv.calc.energy(md, level={level!r}) or "
            f"mv.calc.relax(md, level={level!r}) first")

    S = structures(md, source)
    energies = md.obs[key].to_numpy(dtype=float)

    own, own_index = [], []
    for i, (s, e) in enumerate(zip(S, energies)):
        if e == e:                                        # skip NaN
            own.append(ComputedEntry(s.composition, float(e),
                                     entry_id=str(md.obs_names[i])))
            own_index.append(i)

    ref_entries, ref_meta = _reference_entries(md, level, references,
                                               allow_level_mismatch)
    entries = own + ref_entries

    elements = sorted({str(el) for s in S for el in s.composition.elements})
    have_elemental = _elemental_coverage(entries, elements)

    above = np.full(len(S), np.nan)
    stable = np.zeros(len(S), dtype=bool)
    formation = np.full(len(S), np.nan)
    decomp = [""] * len(S)
    built, why = True, None

    try:
        pd_ = PhaseDiagram(entries)
        for j, i in enumerate(own_index):
            entry = own[j]
            above[i] = float(pd_.get_e_above_hull(entry))
            stable[i] = entry in pd_.stable_entries
            if have_elemental:
                formation[i] = float(pd_.get_form_energy_per_atom(entry))
            decomp[i] = _decomposition_label(pd_, entry)
    except Exception as exc:
        built, why = False, f"{type(exc).__name__}: {exc}"

    md.obs[f"e_above_hull_{level}"] = above
    md.obs[f"is_stable_{level}"] = stable
    md.obs[f"formation_energy_{level}"] = formation
    md.obs[f"decomposes_to_{level}"] = decomp
    md.uns["phase_diagram"] = {
        "level": level,
        "elements": elements,
        "n_candidates": len(own),
        "n_references": len(ref_entries),
        "closed_system": not ref_entries,
        "has_elemental_references": bool(have_elemental),
        "built": built,
        "error": why,
        **ref_meta,
    }
    if not ref_entries:
        warnings.warn(
            f"hull for level {level!r} was built over this dataset's own "
            f"compositions only, so e_above_hull is relative to these "
            f"candidates and not to the known phase diagram. Pass references= "
            f"to make it absolute; see uns['phase_diagram']['closed_system'].",
            stacklevel=2)
    record(md, "thermo.hull", level=level, source=source,
           n_references=len(ref_entries))


def _reference_entries(md: AnnData, level: str, references,
                       allow_mismatch: bool):
    """Normalise ``references`` to a list of entries, checking level agreement."""
    if references is None:
        return [], {}

    if isinstance(references, AnnData):
        entries, ref_level = _entries_from_object(references, level)
        _check_levels(md, level, references, ref_level, allow_mismatch)
        return entries, {"reference_source": "matverse object",
                         "reference_level": ref_level}

    entries = list(references)
    if entries and not hasattr(entries[0], "composition"):
        raise TypeError(
            "references must be pymatgen ComputedEntry objects, a matverse "
            f"AnnData, or None; got {type(entries[0]).__name__}")
    return entries, {"reference_source": "entries"}


def _entries_from_object(ref: AnnData, level: str):
    """Pull entries out of another matverse object at the same level."""
    from pymatgen.entries.computed_entries import ComputedEntry
    from ._core import structures as _structures

    key = f"energy_{level}"
    if key not in ref.obs:
        available = [c for c in ref.obs.columns if c.startswith("energy_")]
        raise ValueError(
            f"reference object has no obs[{key!r}]; it has {available}. A hull "
            f"must be built from one level of theory.")
    S = _structures(ref, "input")
    energies = ref.obs[key].to_numpy(dtype=float)
    entries = [ComputedEntry(s.composition, float(e), entry_id=f"ref:{name}")
               for s, e, name in zip(S, energies, ref.obs_names) if e == e]
    ref_level = (ref.uns.get("levels", {}).get(level, {}) or {}).get("reference")
    return entries, ref_level


def _check_levels(md: AnnData, level: str, ref: AnnData, ref_reference,
                  allow_mismatch: bool) -> None:
    """Refuse to build a hull from two incompatible references."""
    own = (md.uns.get("levels", {}).get(level, {}) or {}).get("reference")
    if own == ref_reference or allow_mismatch:
        return
    raise LevelMismatch(
        f"candidate energies at level {level!r} reproduce {own!r} but the "
        f"reference entries reproduce {ref_reference!r}. A hull mixing them is "
        f"not a hull of anything. Recompute one side at the other's level, or "
        f"pass allow_level_mismatch=True if you know why this is acceptable.")


def _elemental_coverage(entries, elements) -> bool:
    """Whether every element has an elemental reference phase in the hull.

    Without one, pymatgen still builds a hull but formation energies are
    measured from an arbitrary origin, so they are not reported.
    """
    have = set()
    for entry in entries:
        comp = entry.composition
        if len(comp.elements) == 1:
            have.add(str(comp.elements[0]))
    return bool(elements) and set(elements).issubset(have)


def _decomposition_label(pd_, entry) -> str:
    """What this entry decomposes into, as a readable formula string."""
    try:
        decomp = pd_.get_decomposition(entry.composition)
    except Exception:
        return ""
    parts = []
    for phase, amount in sorted(decomp.items(), key=lambda kv: -kv[1]):
        if amount > 1e-6:
            parts.append(f"{phase.composition.reduced_formula}:{amount:.2f}")
    return " + ".join(parts)


@register_function(
    aliases=["reaction energy", "reaction", "balance reaction",
             "synthesis energy", "will it react"],
    category="thermo",
    description="Balance a reaction between compositions present in this "
                "dataset and compute its energy at one level of theory.",
    requires={"obs": ["energy_{level}"], "structures": ["input"]},
    produces={"uns": ["reactions"]},
    prerequisites=["mv.calc.energy"],
    examples=["mv.thermo.reaction(md, ['Al', 'Ni'], ['AlNi'], level='emt')"],
    related=["mv.thermo.hull", "mv.thermo.chempot_limits"],
    notes="Uses the lowest energy found in this dataset for each formula, so "
          "the answer is about the polymorphs you have. A reaction energy is "
          "not a synthesis route: it says a product is downhill, not that "
          "anything gets there.",
)
def reaction(md: AnnData, reactants: list, products: list, level: str = "emt",
             source: str = "input", name: str | None = None) -> dict:
    """Balance and evaluate a reaction. Returns the result and records it."""
    from pymatgen.analysis.reaction_calculator import ComputedReaction
    from pymatgen.core.composition import Composition

    entries = _lowest_entries(md, level, source)
    try:
        left = [_entry_for(entries, Composition(f)) for f in reactants]
        right = [_entry_for(entries, Composition(f)) for f in products]
        computed = ComputedReaction(left, right)
        energy = float(computed.calculated_reaction_energy)
        equation = str(computed)
    except Exception as exc:
        raise ValueError(
            f"could not balance {reactants} -> {products} from this dataset: "
            f"{type(exc).__name__}: {exc}. Every formula must be present at "
            f"level {level!r}; this dataset has "
            f"{sorted(entries)}.") from exc

    result = {
        "reactants": list(reactants), "products": list(products),
        "level": level, "equation": equation,
        "energy": energy,
        "energy_per_atom": energy / max(sum(
            Composition(f).num_atoms for f in products), 1.0),
        "favourable": bool(energy < 0),
    }
    md.uns.setdefault("reactions", {})[
        name or f"{'+'.join(reactants)}->{'+'.join(products)}"] = result
    record(md, "thermo.reaction", reactants=list(reactants),
           products=list(products), level=level)
    return result


def _lowest_entries(md: AnnData, level: str, source: str) -> dict:
    """One entry per reduced formula — the lowest energy this dataset has."""
    from pymatgen.entries.computed_entries import ComputedEntry

    key = f"energy_{level}"
    if key not in md.obs:
        raise ValueError(f"obs[{key!r}] absent; run mv.calc.energy(md, "
                         f"level={level!r}) first")
    energies = md.obs[key].to_numpy(dtype=float)
    best: dict[str, ComputedEntry] = {}
    for structure, energy in zip(structures(md, source), energies):
        if energy != energy:
            continue
        formula = structure.composition.reduced_formula
        per_atom = energy / len(structure)
        current = best.get(formula)
        if current is None or per_atom < current.energy / current.composition.num_atoms:
            best[formula] = ComputedEntry(structure.composition, float(energy))
    return best


def _entry_for(entries: dict, composition):
    formula = composition.reduced_formula
    if formula not in entries:
        raise KeyError(f"{formula} is not in this dataset")
    return entries[formula]


@register_function(
    aliases=["chemical potential", "chempot", "stability window",
             "chemical potential limits", "growth conditions"],
    category="thermo",
    description="Report the range of elemental chemical potentials over which "
                "each stable phase remains on the hull — the conditions a phase "
                "could be grown under.",
    requires={"obs": ["energy_{level}"]},
    produces={"uns": ["chempot_limits"]},
    prerequisites=["mv.thermo.hull"],
    examples=["mv.thermo.chempot_limits(md, level='emt')"],
    related=["mv.thermo.hull", "mv.thermo.reaction"],
    notes="Only meaningful when the hull includes the competing phases; on a "
          "hull closed over one dataset the window is bounded by the dataset "
          "rather than by chemistry, and this says so.",
)
def chempot_limits(md: AnnData, level: str = "emt", source: str = "input",
                   references=None) -> dict:
    """Chemical potential window for each stable phase, in eV."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    entries = list(_lowest_entries(md, level, source).values())
    if references is not None:
        extra, _ = _reference_entries(md, level, references, True)
        entries = entries + extra

    diagram = PhaseDiagram(entries)
    out = {}
    for entry in diagram.stable_entries:
        try:
            ranges = diagram.get_chempot_range_stability_phase(
                entry.composition, entry.composition.elements[0])
        except Exception:
            ranges = None
        out[entry.composition.reduced_formula] = {
            str(element): [float(v) for v in values]
            for element, values in (ranges or {}).items()
        }

    md.uns["chempot_limits"] = {
        "level": level,
        "closed_system": references is None,
        "limits": out,
        "note": "bounded by this dataset rather than by chemistry"
                if references is None else "",
    }
    record(md, "thermo.chempot_limits", level=level,
           n_stable=len(diagram.stable_entries))
    return out


@register_function(
    aliases=["defect formation energy", "defect thermodynamics", "charge "
             "transition level", "does the defect form", "formation energy "
             "diagram", "charged defect"],
    category="thermo",
    description="Compute defect formation energies as a function of Fermi "
                "level and charge state, and find the charge transition levels "
                "where the stable charge changes.",
    requires={"obs": ["energy_{level}", "parent", "defect"]},
    dispatch="dielectric= adds the image-charge term; locpots= adds the "
             "potential-alignment term on top of it",
    produces={"obs": ["defect_formation_energy_{level}",
                      "stable_charge_{level}"],
              "obsm": ["formation_vs_fermi_{level}"],
              "uns": ["grids", "defect_thermodynamics"]},
    prerequisites=["mv.pp.defects", "mv.calc.relax"],
    examples=["mv.thermo.defect_formation(defective, host=md, level='pbe', "
              "chempot={'Al': -3.7}, band_gap=1.2)",
              "mv.thermo.defect_formation(defective, host=md, level='pbe', "
              "chempot={'Al': -3.7}, band_gap=1.2, dielectric=9.1)"],
    related=["mv.pp.defects", "mv.dft.read_dos"],
    notes="Enumerating a defect and knowing whether it forms are different "
          "questions, and this is the second one. The formation energy depends "
          "on where the Fermi level sits and on the chemical potential of "
          "whatever was added or removed, so it is a line rather than a number "
          "— stored on the grid axis against the Fermi level, with the lowest "
          "charge state at each point giving the stable charge.\n\n"
          "**Pass dielectric= to correct for the image charge.** A charged "
          "defect in a periodic cell interacts with its own periodic images, "
          "which lowers its energy spuriously by tenths of an eV in a small "
          "supercell — enough to change which charge state looks stable. With "
          "a dielectric constant this applies the electrostatic half of the "
          "Freysoldt correction, which needs only the cell, the charge and "
          "epsilon: it scales as q^2, as 1/epsilon, and as 1/L, so it matters "
          "most exactly where supercells are smallest.\n\n"
          "**Pass locpots= as well for the other half.** Freysoldt is an "
          "image-charge term plus a potential-alignment term, and the second "
          "needs the planar-averaged electrostatic potential — a LOCPOT — "
          "from both the defective and the pristine run. Give it a "
          "{row name: path or Locpot} mapping covering the defect rows and "
          "the host rows they name in obs['parent'], and both terms are "
          "applied. Which ones were is recorded in "
          "uns['defect_thermodynamics']['correction_terms'], and "
          "['potential_alignment'] is a plain boolean.\n\n"
          "The alignment enters as q * dV, so it is exactly linear in a rigid "
          "shift of either potential and vanishes at q = 0 — which is what "
          "makes it testable without a real LOCPOT.\n\n"
          "The defect site is found by comparing the two cells rather than "
          "with pymatgen's DefectSiteFinder, which fits a SOAP descriptor and "
          "so needs dscribe — a package that cannot import at all on numpy "
          ">= 2.5. For a vacancy, an interstitial or a substitution the "
          "correspondence between the cells is one-to-one apart from the "
          "defect, so matching sites by position answers it directly. "
          "obs['defect_a', 'defect_b', 'defect_c'] from mv.pp.locate_defect "
          "are used instead when present.\n\n"
          "Without dielectric= nothing is corrected and the uns flag says so, "
          "which is the older behaviour and still the right one for ranking "
          "neutral defects.",
)
def defect_formation(defective: AnnData, host: AnnData, level: str = "emt",
                     chempot: dict | None = None, band_gap: float = 2.0,
                     charges=(-2, -1, 0, 1, 2), n_points: int = 200,
                     dielectric: float | None = None,
                     locpots: dict | None = None) -> None:
    """Defect formation energy against the Fermi level, per charge state."""
    from ._core import deposit_grid

    energy_key = f"energy_{level}"
    for obj, label in ((defective, "defects"), (host, "host")):
        if energy_key not in obj.obs:
            raise ValueError(f"obs[{energy_key!r}] absent on the {label}; run "
                             f"mv.calc.relax(..., level={level!r}) on both")
    for column in ("parent", "defect"):
        if column not in defective.obs:
            raise ValueError(f"obs[{column!r}] absent; these did not come from "
                             f"mv.pp.defects")

    chempot = dict(chempot or {})
    fermi = np.linspace(0.0, float(band_gap), n_points)
    host_energy = dict(zip(map(str, host.obs_names),
                           host.obs[energy_key].to_numpy(dtype=float)))

    defect_energy = defective.obs[energy_key].to_numpy(dtype=float)
    parents = defective.obs["parent"].astype(str).to_numpy()
    removed = defective.obs.get("removed", pd.Series([""] * defective.n_obs))
    added = defective.obs.get("added", pd.Series([""] * defective.n_obs))
    removed = removed.astype(str).to_numpy()
    added = added.astype(str).to_numpy()

    curves = np.full((defective.n_obs, n_points), np.nan)
    neutral = np.full(defective.n_obs, np.nan)
    stable = [""] * defective.n_obs
    missing_chempot: set[str] = set()
    cells = structures(defective, "input") if dielectric is not None else None
    corrections = np.zeros((defective.n_obs, len(charges)))
    correction_error = ""

    if locpots is not None and dielectric is None:
        raise ValueError(
            "locpots= given without dielectric=; the alignment term is only "
            "half of Freysoldt and applying it alone would be worse than "
            "applying neither. Pass the bulk dielectric constant too")

    alignment = np.zeros((defective.n_obs, len(charges)))
    aligned = False

    if dielectric is not None:
        try:
            from pymatgen.analysis.defects.corrections.freysoldt import (
                perform_es_corr)
            from pymatgen.analysis.defects.utils import QModel
        except ImportError as exc:                         # pragma: no cover
            raise ImportError(
                f"dielectric= needs pymatgen-analysis-defects for the "
                f"Freysoldt image-charge term. Install it with `pip install "
                f"pymatgen-analysis-defects`, or leave dielectric unset for "
                f"the uncorrected energies. ({exc})") from exc
        model = QModel()
        for i, cell in enumerate(cells):
            for j, q in enumerate(charges):
                if int(q) == 0:
                    continue
                try:
                    corrections[i, j] = float(perform_es_corr(
                        cell.lattice, q=int(q), dielectric=float(dielectric),
                        q_model=model))
                except Exception as exc:                   # pragma: no cover
                    correction_error = f"{type(exc).__name__}: {exc}"
                    corrections[i, j] = np.nan

    if locpots is not None:
        aligned, failures = _potential_alignment(
            defective, parents, charges, float(dielectric), locpots,
            alignment)
        if failures:
            correction_error = "; ".join(failures[:3])

    for i in range(defective.n_obs):
        bulk = host_energy.get(parents[i], np.nan)
        if not (np.isfinite(defect_energy[i]) and np.isfinite(bulk)):
            continue

        exchange = 0.0
        ok = True
        for symbol, sign in ((removed[i], +1.0), (added[i], -1.0)):
            if not symbol:
                continue
            if symbol not in chempot:
                missing_chempot.add(symbol)
                ok = False
                break
            exchange += sign * float(chempot[symbol])
        if not ok:
            continue

        base = float(defect_energy[i]) - float(bulk) + exchange
        neutral[i] = base
        # E_f(q, E_F) = base + q * E_F + E_img(q) + q * dV, where E_img
        # removes the spurious interaction of the charged defect with its own
        # periodic images and q * dV realigns the two calculations' potential
        # zeros. E_img is positive and zero at q = 0; both terms are zero
        # throughout when no dielectric constant was given, and the alignment
        # is zero unless locpots= supplied the potentials it needs.
        per_charge = np.vstack([base + q * fermi + corrections[i, j]
                                + alignment[i, j]
                                for j, q in enumerate(charges)])
        curves[i] = per_charge.min(axis=0)
        stable[i] = str(charges[int(np.argmin(per_charge[:, 0]))])

    defective.obs[f"defect_formation_energy_{level}"] = neutral
    defective.obs[f"stable_charge_{level}"] = stable
    deposit_grid(defective, "formation_vs_fermi", level, curves, fermi,
                 unit="eV above the valence band maximum")
    defective.uns["defect_thermodynamics"] = {
        "level": level,
        "band_gap": float(band_gap),
        "charges": [int(q) for q in charges],
        "chempot": chempot,
        "missing_chempot": sorted(missing_chempot),
        "image_charge_correction": dielectric is not None,
        "dielectric": None if dielectric is None else float(dielectric),
        "potential_alignment": bool(aligned),
        "correction_terms": (
            None if dielectric is None else
            "Freysoldt, both terms: electrostatic (image-charge) and "
            "potential alignment from the supplied LOCPOTs" if aligned else
            "Freysoldt electrostatic (image-charge) only; the "
            "potential-alignment term needs locpots="),
        "correction_error": correction_error or None,
        "note": ("fully corrected in the Freysoldt sense" if aligned else
                 "uncorrected for periodic image interaction; pass "
                 "dielectric= and locpots= for the full Freysoldt correction"
                 if dielectric is None else
                 "image-charge corrected; pass locpots= for the alignment "
                 "term as well"),
    }
    if missing_chempot:
        warnings.warn(
            f"no chemical potential given for {sorted(missing_chempot)}, so "
            f"defects exchanging those species have no formation energy. A "
            f"defect creates or destroys atoms, and what they cost is not "
            f"derivable from the defective cell alone — pass chempot=.",
            stacklevel=2)
    record(defective, "thermo.defect_formation", level=level,
           band_gap=band_gap, n_charges=len(charges))


@register_function(
    aliases=["pourbaix", "aqueous stability", "corrosion", "electrochemical "
             "stability", "water stability", "ph potential"],
    category="thermo",
    description="Compute how far each material sits from aqueous stability at "
                "a given pH and applied potential, which is what decides "
                "whether it survives in water.",
    requires={"structures": ["input"]},
    produces={"obs": ["pourbaix_decomposition"], "uns": ["pourbaix"]},
    examples=["mv.thermo.pourbaix(md, ph=7.0, potential=0.0)"],
    related=["mv.thermo.hull"],
    notes="Needs mp-api and an MP_API_KEY: aqueous stability is measured "
          "against the ion energies Materials Project fits, and there is no "
          "way to compute it from a candidate set alone. A material on the "
          "solid-state hull can still dissolve, which is why this is a "
          "separate question rather than a column of the same one.",
)
def pourbaix(md: AnnData, ph: float = 7.0, potential: float = 0.0,
             api_key: str | None = None) -> None:
    """Distance from aqueous stability, in eV/atom, at one pH and potential."""
    import os

    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram
    except ImportError as exc:                            # pragma: no cover
        raise ImportError("mv.thermo.pourbaix needs `pip install "
                          "matverse[mp]`") from exc

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("set MP_API_KEY or pass api_key=; aqueous stability "
                         "is measured against Materials Project's fitted ion "
                         "energies and cannot be computed from candidates alone")

    S = structures(md, "input")
    distances, failures = [], []
    with MPRester(key) as mpr:
        for structure in S:
            elements = sorted({str(el)
                               for el in structure.composition.elements})
            try:
                entries = mpr.get_pourbaix_entries(elements)
                diagram = PourbaixDiagram(entries)
                entry = min(
                    (e for e in entries
                     if e.composition.reduced_formula
                     == structure.composition.reduced_formula),
                    key=lambda e: e.energy_per_atom, default=None)
                if entry is None:
                    raise KeyError("no matching Pourbaix entry")
                distances.append(float(diagram.get_decomposition_energy(
                    entry, pH=ph, V=potential)))
            except Exception as exc:
                distances.append(np.nan)
                failures.append(f"{structure.composition.reduced_formula}: "
                                f"{type(exc).__name__}: {exc}")

    md.obs["pourbaix_decomposition"] = distances
    md.uns["pourbaix"] = {"ph": float(ph), "potential": float(potential),
                          "n_failed": len(failures), "errors": failures[:10]}
    record(md, "thermo.pourbaix", ph=ph, potential=potential)


@register_function(
    aliases=["reference phases", "competing phases", "get mp entries",
             "elemental references", "known phases"],
    category="thermo",
    description="Fetch the known competing phases spanning a set of elements "
                "from Materials Project, for use as hull references.",
    produces={"files": []},
    examples=["refs = mv.thermo.references_from_mp(['Fe', 'O'])",
              "mv.thermo.hull(md, level='pbe', references=refs)"],
    related=["mv.thermo.hull"],
    notes="Returns PBE+U entries with Materials Project's fitted corrections "
          "applied. Putting them on a hull with a machine-learned potential's "
          "energies is the mistake mv.thermo.hull refuses by default.",
)
def references_from_mp(elements, api_key: str | None = None):
    """Competing phases across a chemical system, from Materials Project.

    Needs ``mp-api`` and an ``MP_API_KEY``. Kept thin: ``mp-api`` owns the query
    language and duplicating it here would mean tracking their schema forever.
    """
    import os

    try:
        from mp_api.client import MPRester
    except ImportError as exc:                            # pragma: no cover
        raise ImportError(
            "mv.thermo.references_from_mp needs `pip install matverse[mp]`"
        ) from exc

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("set MP_API_KEY or pass api_key=")
    with MPRester(key) as mpr:
        return mpr.get_entries_in_chemsys([str(e) for e in elements])


__all__ = ["hull", "reaction", "chempot_limits", "pourbaix",
           "defect_formation",
           "references_from_mp",
           "corrections", "CORRECTION_SCHEMES", "chempot_diagram",
           "LevelMismatch"]


#: The correction schemes pymatgen ships, by the name matverse dispatches on.
CORRECTION_SCHEMES = {
    "mp2020": ("pymatgen.entries.compatibility",
               "MaterialsProject2020Compatibility",
               "Materials Project 2020 corrections"),
    "mp2020_aqueous": ("pymatgen.entries.compatibility",
                       "MaterialsProjectAqueousCompatibility",
                       "MP 2020 corrections with the aqueous reference"),
    "mp_legacy": ("pymatgen.entries.compatibility",
                  "MaterialsProjectCompatibility",
                  "the pre-2020 Materials Project corrections"),
    "mit": ("pymatgen.entries.compatibility", "MITCompatibility",
            "the MIT parameter set"),
}


@register_function(
    aliases=["corrections", "energy corrections", "mp2020", "anion correction",
             "hubbard correction", "compatibility", "correct energies",
             "mixing scheme"],
    category="thermo",
    description="Apply the Materials Project energy corrections to a computed "
                "energy and deposit the result as its own level of theory, so "
                "a corrected and an uncorrected energy are never the same "
                "column.",
    requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
    produces={"obs": ["energy_{level}-{scheme}",
                      "energy_per_atom_{level}-{scheme}",
                      "correction_{level}-{scheme}",
                      "correction_per_atom_{level}-{scheme}",
                      "run_type_{level}-{scheme}"],
              "levels": ["{level}-{scheme}"], "uns": ["corrections"]},
    prerequisites=["mv.calc.energy"],
    dispatch="scheme= selects the correction set; see mv.thermo."
             "CORRECTION_SCHEMES",
    examples=["mv.thermo.corrections(md, level='pbe')",
              "mv.thermo.corrections(md, level='pbe', run_type='GGA+U')",
              "mv.thermo.corrections(md, level='pbe', scheme='mp_legacy')"],
    related=["mv.thermo.hull", "mv.pp.harmonize", "mv.calc.energy"],
    notes="A raw GGA energy and a corrected one are different quantities, so "
          "the corrected energy becomes a new **level** — 'pbe' in, "
          "'pbe-mp2020' out — rather than overwriting the column it came from. "
          "That is the same rule that keeps emt and pbe apart, applied one step "
          "further along, and it means a hull built on the wrong one is a "
          "visible mistake rather than a silent one.\n\n"
          "The corrections are large enough that ignoring them is not a small "
          "error: Fe2O3 moves by 6.6 eV per formula unit, of which 2.1 eV is "
          "the oxide anion correction and 4.5 eV the +U correction on iron. "
          "Energies fetched through mv.data.from_mp arrive **already "
          "corrected**; energies you computed yourself and read back through "
          "mv.dft.read_outputs do not. Mixing the two on one hull is the error "
          "this function exists to make impossible to make quietly.\n\n"
          "run_type decides whether the +U corrections apply at all. Left "
          "unset it is inferred as GGA+U when the structure contains an "
          "element MP applies a U to together with oxygen or fluorine, which "
          "is MP's own rule, and the inference is recorded next to the result."
          "\n\n"
          "The produces slots interpolate level and scheme rather than "
          "key_added, because those are the two parameters the default output "
          "name is built from and a template naming key_added resolves to "
          "nothing on the call everybody makes. Passing key_added overrides "
          "the name and the claim with it.",
)
def corrections(md: AnnData, level: str = "pbe", scheme: str = "mp2020",
                source: str = "input", run_type: str | None = None,
                hubbards: dict | None = None,
                key_added: str | None = None) -> None:
    """Corrected energies as a new level of theory. Deposits; returns ``None``."""
    import importlib

    from ._core import set_level
    from pymatgen.entries.computed_entries import ComputedEntry

    key = str(scheme).strip().lower()
    if key not in CORRECTION_SCHEMES:
        raise ValueError(
            f"unknown scheme {scheme!r}; known: {sorted(CORRECTION_SCHEMES)}. "
            f"Each is described in mv.thermo.CORRECTION_SCHEMES.")
    module_name, class_name, description = CORRECTION_SCHEMES[key]
    energy_key = f"energy_{level}"
    if energy_key not in md.obs:
        raise ValueError(
            f"obs[{energy_key!r}] absent; there is nothing to correct. Run "
            f"mv.calc.energy or mv.dft.read_outputs at level={level!r} first.")

    target = key_added or f"{level}-{key}"
    compatibility = getattr(importlib.import_module(module_name), class_name)(
        check_potcar=False)

    raw = md.obs[energy_key].to_numpy(dtype=float)
    corrected = np.full(md.n_obs, np.nan)
    delta = np.full(md.n_obs, np.nan)
    per_atom = np.full(md.n_obs, np.nan)
    delta_per_atom = np.full(md.n_obs, np.nan)
    inferred = np.empty(md.n_obs, dtype=object)
    failed = []

    for i, structure in enumerate(structures(md, source)):
        if not np.isfinite(raw[i]):
            inferred[i] = ""
            continue
        kind = run_type or _infer_run_type(structure)
        u_values = (dict(hubbards) if hubbards is not None
                    else _default_hubbards(structure))
        inferred[i] = kind
        entry = ComputedEntry(
            structure.composition, float(raw[i]),
            parameters={"run_type": kind,
                        "is_hubbard": kind.endswith("+U"),
                        "hubbards": u_values if kind.endswith("+U") else {},
                        "potcar_symbols": []})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed = compatibility.process_entries(
                    [entry], clean=True, on_error="raise")
        except Exception as exc:
            failed.append(f"{structure.composition.reduced_formula}: {exc}")
            continue
        if not processed:
            failed.append(f"{structure.composition.reduced_formula}: the "
                          f"scheme rejected this entry")
            continue
        n = len(structure)
        corrected[i] = float(processed[0].energy)
        delta[i] = float(processed[0].correction)
        per_atom[i] = corrected[i] / n
        delta_per_atom[i] = delta[i] / n

    md.obs[f"energy_{target}"] = corrected
    md.obs[f"energy_per_atom_{target}"] = per_atom
    md.obs[f"correction_{target}"] = delta
    md.obs[f"correction_per_atom_{target}"] = delta_per_atom
    md.obs[f"run_type_{target}"] = inferred.astype(str)
    md.uns.setdefault("corrections", {})[target] = {
        "scheme": key, "from_level": level, "source": source,
        "n_corrected": int(np.isfinite(delta).sum()),
        "n_failed": len(failed),
        "failures": failed[:20],
    }
    set_level(md, target, kind="corrected", method=description,
              reference=level, surrogate=False, license=None,
              uncertainty=None, scheme=key, corrected_from=level)
    record(md, "thermo.corrections", level=level, scheme=key,
           key_added=target)


#: Elements the Materials Project applies a Hubbard U to, and with what value,
#: read from MP's own VASP input set rather than copied.
def _mp_hubbards() -> dict:
    from pymatgen.io.vasp.sets import MPRelaxSet
    return MPRelaxSet.CONFIG.get("INCAR", {}).get("LDAUU", {})


def _default_hubbards(structure) -> dict:
    """MP's U values for the elements in this structure, by its own rule.

    MP applies +U only when an oxide or fluoride anion is present, and only to
    the transition metals in its table. Guessing differently from MP means the
    corrections are applied to a calculation MP would not have run.
    """
    table = _mp_hubbards()
    symbols = {site.specie.symbol for site in structure
               if hasattr(site, "specie")}
    for anion in ("O", "F"):
        if anion in symbols and anion in table:
            return {s: table[anion][s] for s in symbols if s in table[anion]}
    return {}


def _infer_run_type(structure) -> str:
    return "GGA+U" if _default_hubbards(structure) else "GGA"


@register_function(
    aliases=["chemical potential diagram", "chempot diagram",
             "chemical potential domains", "phase stability region",
             "where is each phase stable", "synthesis window per phase"],
    category="thermo",
    description="The region of chemical potential space in which each phase is "
                "stable, and how wide that region is — which is how hard the "
                "phase is to make.",
    requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
    produces={"obs": ["chempot_stable_{level}", "chempot_window_{level}"],
              "uns": ["chempot_diagram"]},
    prerequisites=["mv.calc.energy"],
    examples=["mv.thermo.chempot_diagram(md, level='emt')",
              "mv.thermo.chempot_diagram(md, level='pbe-mp2020')"],
    related=["mv.thermo.chempot_limits", "mv.thermo.hull",
             "mv.thermo.defect_formation"],
    notes="mv.thermo.hull says whether a phase is stable. This says **under "
          "what conditions**, which is the question that decides whether it "
          "can be synthesised: a phase stable only in a sliver of chemical "
          "potential space needs the growth atmosphere controlled to match, "
          "and one with a wide domain will form over a range of "
          "conditions.\\n\\n"
          "obs['chempot_window'] is the extent of that domain, summed over the "
          "elements. It is a comparative number rather than an absolute one — "
          "use it to rank candidates in one chemical system, not across "
          "systems, because the axes are different chemical potentials.\\n\\n"
          "The domains are exactly consistent with the formation energies they "
          "came from, which is what makes them checkable: on an Al-Ni system "
          "where Al3Ni forms at -1.8 eV, its domain reaches mu_Ni = -1.8 at "
          "mu_Al = 0, and its boundary with AlNi sits where "
          "mu_Al + mu_Ni = -1.4.\\n\\n"
          "A phase that is not on the hull has no domain at all and gets a "
          "window of zero rather than NaN, because 'never stable' is an "
          "answer. An elemental reference gets zero for the opposite reason: "
          "its domain is **open**, running to the artificial floor along the "
          "other axes, so a width there would be a plotting choice rather than "
          "a physical one. uns records which domains are open.\n\n"
          "The bare alias 'stability window' belongs to "
          "mv.thermo.chempot_limits, which computes exactly that for one "
          "target phase. This computes every phase's region at once, so it is "
          "reached as 'chemical potential diagram'.",
)
def chempot_diagram(md: AnnData, level: str = "emt", source: str = "input",
                    default_min_limit: float = -50.0) -> None:
    """Chemical potential domains per phase. Deposits; returns ``None``."""
    from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    energy_key = f"energy_{level}"
    if energy_key not in md.obs:
        raise ValueError(
            f"obs[{energy_key!r}] absent; a chemical potential diagram is built "
            f"from energies. Run mv.calc.energy(md, level={level!r}) first.")

    energies = md.obs[energy_key].to_numpy(dtype=float)
    S = structures(md, source)
    entries, rows = [], []
    for i, (structure, energy) in enumerate(zip(S, energies)):
        if not np.isfinite(energy):
            continue
        entries.append(ComputedEntry(structure.composition, float(energy)))
        rows.append(i)
    if len(entries) < 2:
        raise ValueError(
            f"need at least two finite energies to build a diagram; got "
            f"{len(entries)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagram = ChemicalPotentialDiagram(
            entries, default_min_limit=default_min_limit)

    stable = np.zeros(md.n_obs, dtype=bool)
    window = np.zeros(md.n_obs, dtype=float)
    domains = {}
    for formula, vertices in diagram.domains.items():
        array = np.asarray(vertices, dtype=float)
        # Ignore the artificial floor: a domain that runs to the default limit
        # is open, and its extent along that axis is a plotting choice.
        touches_floor = bool(array.size
                             and np.any(array <= default_min_limit + 1e-9))
        finite = array[np.all(array > default_min_limit + 1e-9, axis=1)] \
            if array.size else array
        extent = float(np.sum(finite.max(axis=0) - finite.min(axis=0))) \
            if finite.size else 0.0
        domains[str(formula)] = {
            "vertices": array.tolist(),
            # Open along at least one axis: the domain runs to the artificial
            # floor, so its extent there is a plotting choice rather than a
            # physical width. The elemental references are always open.
            "extent": extent,
            "open": touches_floor,
        }

    for i, structure in enumerate(S):
        name = structure.composition.reduced_formula
        if name in domains:
            stable[i] = True
            window[i] = domains[name]["extent"]

    md.obs[f"chempot_stable_{level}"] = stable
    md.obs[f"chempot_window_{level}"] = window
    md.uns["chempot_diagram"] = {
        "level": level, "source": source,
        "elements": [str(e) for e in diagram.elements],
        "domains": domains,
        "n_entries": len(entries),
        "note": "a phase absent from the diagram is never stable and gets a "
                "window of zero, which is an answer rather than a gap",
    }
    record(md, "thermo.chempot_diagram", level=level, source=source)


@register_function(
    aliases=["fit corrections", "derive corrections", "correction calculator",
             "calibrate against experiment", "my own mp2020",
             "fit anion correction", "refit corrections"],
    category="thermo",
    description="Fit your own energy corrections by regressing computed "
                "formation energies against measured ones, the way MP2020 "
                "was derived.",
    requires={"obs": ["energy_{level}", "e_above_hull_{level}",
                      "{experimental}"], "structures": ["{source}"]},
    produces={"uns": ["fitted_corrections"],
              "obs": ["correction_{level}", "energy_corrected_{level}"]},
    prerequisites=["mv.thermo.hull", "mv.exp.measure"],
    examples=["mv.thermo.fit_corrections(md, 'dHf_exp', level='pbe')",
              "mv.thermo.fit_corrections(md, 'dHf_exp', level='r2scan', "
              "species=('oxide', 'sulfide'))"],
    related=["mv.thermo.corrections", "mv.thermo.hull",
             "mv.exp.formation_hull"],
    notes="mv.thermo.corrections applies the Materials Project's corrections. "
          "This one derives corrections of your own, which is what you need "
          "the moment you are not using MP's functional or MP's settings — "
          "the MP2020 anion corrections are calibrated to PBE+U at MP's "
          "cutoffs and do not transfer to r2SCAN, to a different pseudo"
          "potential set, or to a machine-learned potential.\\n\\n"
          "The fit is a least-squares regression of (measured minus computed) "
          "formation energy onto the count of each correction species, both "
          "divided by the atoms in the formula unit, so a correction comes "
          "out in eV per atom of that species. Verified by construction: "
          "inject a known offset per oxygen into the measured energies and "
          "the fit returns it to four decimals.\\n\\n"
          "**The measured column is a formation energy per formula unit**, "
          "not per atom, and getting that wrong scales every correction by "
          "the formula size without failing. mv.exp.measure is where such a "
          "column comes from.\\n\\n"
          "Needs e_above_hull, so run mv.thermo.hull first: the fit excludes "
          "compounds too far above the hull, on the reasoning that a "
          "measurement of a phase the calculation says is unstable is "
          "probably not a measurement of the same thing.\\n\\n"
          "A correction fitted to six compounds is a number with an error "
          "bar, and the error bar is returned beside it. MP2020 used "
          "thousands.",
)
def fit_corrections(md: AnnData, experimental: str, level: str = "pbe",
                    source: str = "input", species=("oxide",),
                    max_error: float = 0.1, allow_unstable=0.1,
                    key_added: str | None = None) -> None:
    """Fit energy corrections against measured formation energies."""
    try:
        from pymatgen.analysis.compatibility.correction_calculator import (
            CorrectionCalculator)
    except ImportError:                    # pymatgen <= 2025.10 kept it here
        from pymatgen.entries.correction_calculator import (
            CorrectionCalculator)
    from pymatgen.analysis.structure_analyzer import oxide_type
    from pymatgen.entries.computed_entries import ComputedEntry

    energy_key = f"energy_{level}"
    hull_key = f"e_above_hull_{level}"
    for column, hint in ((energy_key, f"run mv.calc.energy(md, level="
                                      f"{level!r})"),
                         (hull_key, f"run mv.thermo.hull(md, level={level!r})"),
                         (experimental, "attach it with mv.exp.measure")):
        if column not in md.obs:
            raise ValueError(f"obs[{column!r}] absent; {hint}")

    energies = md.obs[energy_key].to_numpy(dtype=float)
    above = md.obs[hull_key].to_numpy(dtype=float)
    measured = md.obs[experimental].to_numpy(dtype=float)
    cells = structures(md, source)

    exp_entries, calc_entries = [], {}
    skipped: list[str] = []
    for row, structure in enumerate(cells):
        name = str(md.obs_names[row])
        composition = structure.composition.reduced_composition
        formula = composition.reduced_formula
        if not np.isfinite(energies[row]):
            skipped.append(f"{name}: no {energy_key}")
            continue

        entry = ComputedEntry(composition, float(energies[row]))
        entry.parameters["run_type"] = "GGA"
        try:
            kind = oxide_type(structure)
        except Exception:
            kind = "None"
        entry.data.update({
            "e_above_hull": float(above[row]) if np.isfinite(above[row]) else 0.0,
            "oxide_type": kind, "sulfide_type": None})
        calc_entries[formula] = entry

        if composition.is_element:
            continue                       # a reference, not a fitting point
        if not np.isfinite(measured[row]):
            skipped.append(f"{name}: no measured value")
            continue
        exp_entries.append({"formula": formula,
                            "exp energy": float(measured[row]),
                            "uncertainty": 0.01})

    if len(exp_entries) < 2:
        raise ValueError(f"only {len(exp_entries)} compound(s) have both a "
                         f"computed and a measured energy; a regression needs "
                         f"more than that")

    # Formation energies are measured against the elements, so the elements
    # have to be in the dataset at the same level of theory. Nothing can be
    # inferred here: an elemental energy from a different functional would
    # shift every formation energy and so every correction.
    needed = {str(element)
              for entry in calc_entries.values()
              for element in entry.composition.elements}
    have = {str(next(iter(entry.composition.elements)))
            for entry in calc_entries.values()
            if entry.composition.is_element}
    missing = sorted(needed - have)
    if missing:
        raise ValueError(
            f"no elemental reference for {', '.join(missing)}. A formation "
            f"energy is measured against the elements, so they must be rows "
            f"of this dataset with an {energy_key} from the same level of "
            f"theory — add them and rerun.")

    calculator = CorrectionCalculator(species=list(species),
                                      max_error=float(max_error),
                                      allow_unstable=allow_unstable)
    fitted = calculator.compute_corrections(exp_entries, calc_entries)

    # Apply what was fitted, so the corrected energies sit beside the raw ones
    # rather than requiring the caller to redo the bookkeeping.
    name = key_added or level
    per_row = np.zeros(md.n_obs)
    for row, structure in enumerate(cells):
        composition = structure.composition.reduced_composition
        total = 0.0
        for label, (value, _err) in fitted.items():
            if label == "oxide":
                total += float(value) * composition.get("O", 0.0)
            elif label in {str(el) for el in composition.elements}:
                total += float(value) * composition.get(label, 0.0)
        per_row[row] = total

    md.obs[f"correction_{name}"] = per_row
    md.obs[f"energy_corrected_{name}"] = energies + per_row
    md.uns.setdefault("fitted_corrections", {})[name] = {
        "corrections": {k: [float(v[0]), float(v[1])]
                        for k, v in fitted.items()},
        "unit": "eV per atom of the species",
        "species": list(species), "n_compounds": len(exp_entries),
        "measured_column": experimental,
        "skipped": skipped,
        "note": "fitted here, not the Materials Project's — MP2020 is "
                "calibrated to PBE+U at MP's settings and does not transfer",
    }
    record(md, "thermo.fit_corrections", level=level,
           experimental=experimental, species=list(species), key_added=name)


def _potential_alignment(defective, parents, charges, dielectric, locpots,
                         out) -> tuple[bool, list]:
    """Fill ``out`` with the q * dV term of Freysoldt. Returns (did_any, errors).

    ``locpots`` is keyed by row name, and a defect row's bulk reference is
    found through obs['parent'] — the same link mv.pp.defects already records,
    so nothing new has to be threaded through.
    """
    from pymatgen.analysis.defects.corrections.freysoldt import (
        get_freysoldt_correction, perform_es_corr)
    from pymatgen.analysis.defects.utils import QModel
    from pymatgen.io.vasp.outputs import Locpot

    def load(value):
        return value if not isinstance(value, (str, Path)) else \
            Locpot.from_file(str(value))

    coordinate_columns = ("defect_a", "defect_b", "defect_c")
    has_site = all(c in defective.obs for c in coordinate_columns)
    defect_cells = structures(defective, "input")

    model, done, errors = QModel(), False, []
    for i, name in enumerate(map(str, defective.obs_names)):
        parent = str(parents[i])
        if name not in locpots or parent not in locpots:
            errors.append(f"{name}: no LOCPOT for the defect or for its "
                          f"parent {parent!r}")
            continue
        try:
            defect_locpot, bulk_locpot = load(locpots[name]), load(
                locpots[parent])
            coords = ([float(defective.obs[c].iloc[i])
                       for c in coordinate_columns] if has_site
                      else _locate(defect_cells[i], bulk_locpot.structure))
            for j, q in enumerate(charges):
                if int(q) == 0:
                    continue
                total = get_freysoldt_correction(
                    q=int(q), dielectric=dielectric,
                    defect_locpot=defect_locpot, bulk_locpot=bulk_locpot,
                    defect_frac_coords=coords).correction_energy
                # get_freysoldt_correction returns both terms and the
                # electrostatic one is already in `corrections`; subtracting
                # the same perform_es_corr call leaves the alignment alone,
                # rather than double-counting a term computed twice.
                out[i, j] = float(total) - float(perform_es_corr(
                    defect_locpot.structure.lattice, q=int(q),
                    dielectric=dielectric, q_model=model))
            done = True
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            out[i, :] = np.nan
    return done, errors


def _locate(defect, bulk):
    """Fractional coordinates of the defect, by comparing the two cells.

    pymatgen's own fallback is DefectSiteFinder, which fits a SOAP descriptor
    and so needs dscribe - a package that cannot import on numpy >= 2.5 at
    all, and which for a defect matverse itself generated is answering a
    harder question than the one being asked. The two structures are already
    in hand and the correspondence is one-to-one apart from the defect, so
    matching sites by position says directly which one it is.

    Returns ``None`` if the two cells disagree by more than one site, which is
    not a point defect and not something to guess at.
    """
    lattice = bulk.lattice
    defect_frac = np.asarray(defect.frac_coords, dtype=float)
    bulk_frac = np.asarray(bulk.frac_coords, dtype=float)
    if not len(defect_frac) or not len(bulk_frac):
        return None
    distances = lattice.get_all_distances(bulk_frac, defect_frac)

    if len(defect) == len(bulk) - 1:                       # vacancy
        return list(bulk_frac[int(np.argmax(distances.min(axis=1)))])
    if len(defect) == len(bulk) + 1:                       # interstitial
        return list(defect_frac[int(np.argmax(distances.min(axis=0)))])
    if len(defect) == len(bulk):                           # substitution
        partner = distances.argmin(axis=1)
        changed = [j for i, j in enumerate(partner)
                   if bulk[i].specie.symbol != defect[int(j)].specie.symbol]
        if len(changed) == 1:
            return list(defect_frac[int(changed[0])])
        # Nothing swapped: the cells differ only by relaxation, so the caller
        # is asking where a defect is in a structure that has none.
        return None
    return None


@register_function(
    aliases=["calphad", "assessed phase diagram", "assessed thermodynamics",
             "which phases coexist", "phase equilibrium", "tdb file",
             "finite temperature phases", "solidus", "liquidus", "eutectic"],
    category="thermo",
    description="Equilibrium phases at temperature from an assessed CALPHAD "
                "database — which phases coexist for each composition, and in "
                "what fraction.",
    requires={"obs": ["formula"]},
    produces={"obs": ["calphad_phases", "calphad_n_phases",
                      "calphad_major_phase", "calphad_major_fraction"],
              "uns": ["calphad"]},
    examples=["mv.thermo.calphad(md, 'pbsn.tdb', temperature=450)",
              "mv.thermo.calphad(md, database, temperature=800, "
              "elements=['AL', 'NI'])"],
    related=["mv.thermo.hull", "mv.prop.quasiharmonic",
             "mv.disorder.monte_carlo"],
    notes="A different kind of answer from mv.thermo.hull, and worth keeping "
          "separate from it. The hull is computed here, at 0 K, from energies "
          "this library produced; CALPHAD reads parameters that were fitted "
          "to measured phase boundaries, and gives multicomponent equilibrium "
          "at temperature including liquids and solution phases. Neither "
          "supersedes the other: the hull extends to compositions nobody has "
          "assessed, and CALPHAD is right about the ones somebody has.\\n\\n"
          "**The database is the whole calculation.** matverse ships none — "
          "assessed databases are the product of years of work and are mostly "
          "licensed — so a path or a Database object must be supplied, and "
          "which one was used is recorded in uns['calphad']. Two databases "
          "for the same system will disagree, and that disagreement is the "
          "honest error bar on any number here.\\n\\n"
          "Composition comes from obs['formula'], normalised to mole "
          "fractions of the database's own elements. A material containing an "
          "element the database does not assess is skipped and counted rather "
          "than silently projected onto the elements that remain, which would "
          "return an answer to a different question.\\n\\n"
          "Checked against the Pb-Sn eutectic: at x(Sn)=0.739 this gives two "
          "solid phases at 450 K and a single liquid at 455 K, bracketing the "
          "measured 456 K.",
)
def calphad(md: AnnData, database, temperature: float = 300.0,
            elements=None, phases=None, pressure: float = 101325.0) -> None:
    """Equilibrium phases from a CALPHAD database. Deposits; returns ``None``."""
    try:
        from pycalphad import Database, equilibrium
        from pycalphad import variables as v
    except ImportError as exc:
        raise ImportError(
            f"mv.thermo.calphad needs pycalphad: `pip install "
            f"matverse[calphad]` or `pip install pycalphad`. ({exc})") from exc

    from pymatgen.core.composition import Composition

    if "formula" not in md.obs:
        raise ValueError(
            "obs['formula'] absent; run mv.pp.describe(md), or build the "
            "object with mv.data.from_compositions")

    db = database if isinstance(database, Database) else Database(str(database))
    assessed = sorted(e for e in db.elements if e not in ("/-", "VA"))
    wanted = ([str(e).upper() for e in elements] if elements is not None
              else assessed)
    missing = sorted(set(wanted) - set(assessed))
    if missing:
        raise ValueError(
            f"the database does not assess {missing}; it covers {assessed}")
    active = list(phases) if phases is not None else list(db.phases)

    names = np.empty(md.n_obs, dtype=object)
    counts = np.full(md.n_obs, np.nan)
    major = np.empty(md.n_obs, dtype=object)
    major_fraction = np.full(md.n_obs, np.nan)
    skipped = []

    for i, formula in enumerate(md.obs["formula"].astype(str)):
        names[i] = ""
        major[i] = ""
        try:
            composition = Composition(formula).fractional_composition
            present = {str(el).upper(): float(amount)
                       for el, amount in composition.get_el_amt_dict().items()}
        except Exception as exc:
            skipped.append(f"{md.obs_names[i]}: {type(exc).__name__}: {exc}")
            continue

        outside = sorted(set(present) - set(wanted))
        if outside:
            # Dropping the unassessed elements and renormalising would answer
            # a question about a different material.
            skipped.append(
                f"{md.obs_names[i]} ({formula}): contains {outside}, which "
                f"this database does not assess")
            continue

        # pycalphad fixes the last element by difference, so only the others
        # are named as conditions.
        conditions = {v.T: float(temperature), v.P: float(pressure), v.N: 1}
        for element in wanted[:-1]:
            conditions[v.X(element)] = max(present.get(element, 0.0), 1e-9)
        try:
            result = equilibrium(db, wanted + ["VA"], active, conditions)
            found = [str(p).strip()
                     for p in np.asarray(result.Phase.squeeze()).ravel()
                     if str(p).strip()]
            fractions = [float(f) for f, p in
                         zip(np.asarray(result.NP.squeeze()).ravel(),
                             np.asarray(result.Phase.squeeze()).ravel())
                         if str(p).strip()]
        except Exception as exc:
            skipped.append(f"{md.obs_names[i]}: {type(exc).__name__}: {exc}")
            continue

        if not found:
            skipped.append(f"{md.obs_names[i]}: equilibrium returned no phase")
            continue
        order = int(np.argmax(fractions))
        names[i] = " + ".join(found)
        counts[i] = len(found)
        major[i] = found[order]
        major_fraction[i] = fractions[order]

    md.obs["calphad_phases"] = names.astype(str)
    md.obs["calphad_n_phases"] = counts
    md.obs["calphad_major_phase"] = major.astype(str)
    md.obs["calphad_major_fraction"] = major_fraction
    md.uns["calphad"] = {
        "database": str(getattr(database, "__class__", type(database)))
        if isinstance(database, Database) else str(database),
        "elements": wanted,
        "assessed_elements": assessed,
        "phases": active,
        "temperature": float(temperature),
        "pressure": float(pressure),
        "n_skipped": len(skipped),
        "errors": skipped[:10],
        "note": "assessed parameters fitted to measured phase boundaries, "
                "not energies computed here; a different database for the "
                "same system will give a different answer, and that spread "
                "is the error bar",
    }
    if skipped:
        warnings.warn(
            f"{len(skipped)} of {md.n_obs} materials produced no equilibrium; "
            f"see uns['calphad']['errors']. First: {skipped[0]}",
            RuntimeWarning, stacklevel=2)
    record(md, "thermo.calphad", temperature=temperature, elements=wanted)


@register_function(
    aliases=["voltage", "intercalation voltage", "electrode", "battery",
             "average voltage", "capacity", "cathode", "what voltage does "
             "this give", "energy density"],
    category="thermo",
    description="Average intercalation voltage and capacity for an electrode, "
                "from the energies of its lithiated and delithiated forms.",
    requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
    produces={"obs": ["voltage_{level}", "capacity_gravimetric_{level}",
                      "capacity_volumetric_{level}",
                      "energy_density_{level}", "volume_change_{level}"],
              "uns": ["electrode"]},
    prerequisites=["mv.calc.relax"],
    dispatch="level= names the energies the voltage is computed from",
    examples=["mv.thermo.voltage(md, working_ion='Li', level='chgnet')",
              "mv.thermo.voltage(md, working_ion='Na', level='pbe', "
              "reference=metal_na)"],
    related=["mv.thermo.hull", "mv.neb.barrier", "mv.md.conductivity",
             "mv.prop.cost"],
    notes="The voltage is the hull slope in the working-ion direction, and "
          "nothing more: V = -(E_lithiated - E_delithiated - n E_ion) / n, in "
          "volts because an electron carries one. Everything needed was "
          "already here — mv.thermo.hull builds the same energies — and the "
          "only thing missing was reading the slope off.\\n\\n"
          "**The reference matters more than the cathode.** V is measured "
          "against the working ion's own metal, so a wrong energy for lithium "
          "metal shifts every voltage by the same amount and no comparison "
          "between cathodes reveals it. Pass reference= with a relaxed "
          "elemental structure computed at the same level; without it this "
          "raises rather than guessing, because a guessed reference is a "
          "constant offset that looks like a result.\\n\\n"
          "Checked against LiFePO4, whose plateau is measured at 3.4-3.5 V "
          "with a theoretical capacity of 170 mAh/g. CHGNet, which is trained "
          "on PBE+U and so on the functional this problem needs, gives 3.497 "
          "V and 169.9 mAh/g. Plain PBE gives closer to 3.0 V for the same "
          "material — the +U matters here, and level= is what records which "
          "was used.\\n\\n"
          "obs['volume_change_{level}'] is the weakest number here and is "
          "worth reading with that in mind. On the LiFePO4 check it comes out "
          "at -14%% against a measured -6.8%%, because those energies come "
          "from a positions-only relaxation: the cell was held at its "
          "experimental shape and never allowed to respond to delithiation. "
          "Relaxing the cell would fix that in principle and diverges in "
          "practice on this system, which is why the voltage is the number to "
          "trust and the volume change is a flag rather than a measurement.\\n\\n"
          "Rows whose relaxation did not converge are excluded, and the "
          "reference is refused outright if it did not. That is not "
          "defensive programming: the first version of this read the energies "
          "without checking and returned 78 V for LiFePO4, because a cell "
          "relaxation with CHGNet had collapsed the lithiated structure to 2 "
          "cubic angstrom per atom and expanded the delithiated one to 117. "
          "obs['relax_converged_{level}'] was False for both the whole time. "
          "A foundation potential relaxing the cell of a delithiated "
          "framework is a known place for this; mv.calc.relax(cell=False) "
          "relaxes positions only and is the usual way round it.\\n\\n"
          "This is an *average* voltage between two states, not a curve. A "
          "real cell has plateaux and slopes set by the intermediate "
          "orderings, and resolving them means supplying those intermediates "
          "as rows: pass three or more compositions along the same "
          "framework and each adjacent pair gets its own step, which "
          "uns['electrode'] records.",
)
def voltage(md: AnnData, working_ion: str = "Li", level: str = "emt",
            source: str = "input", reference=None,
            framework: str | None = None) -> None:
    """Intercalation voltage and capacity. Deposits; returns ``None``."""
    from pymatgen.apps.battery.insertion_battery import InsertionElectrode
    from pymatgen.core import Composition, Element
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    energy_key = f"energy_{level}"
    if energy_key not in md.obs:
        raise ValueError(
            f"obs[{energy_key!r}] absent; run mv.calc.relax(md, "
            f"level={level!r}) first — a voltage is a difference of relaxed "
            f"energies and an unrelaxed one is not a state of the material")
    ion = Element(str(working_ion))

    if reference is None:
        raise ValueError(
            f"reference= is required: the voltage is measured against "
            f"{ion.symbol} metal, so it needs that metal's energy at the same "
            f"level. Pass a one-row object holding a relaxed elemental "
            f"{ion.symbol} structure with obs[{energy_key!r}]. Guessing it "
            f"would shift every voltage by the same amount, which no "
            f"comparison between cathodes would reveal")
    if reference.n_obs != 1 or energy_key not in reference.obs:
        raise ValueError(
            f"reference must be a single row carrying obs[{energy_key!r}]")
    reference_structure = structures(reference, source)[0]
    per_ion = (float(reference.obs[energy_key].iloc[0])
               / max(len(reference_structure), 1))
    ion_entry = ComputedStructureEntry(reference_structure,
                                       float(reference.obs[energy_key].iloc[0]))

    # A voltage is a difference of relaxed energies, so an unconverged
    # relaxation makes it meaningless — and mv.calc.relax already says which
    # rows those are. The first version of this read energy_{level} without
    # looking, and cheerfully returned 78 V for LiFePO4 from a cell that had
    # collapsed to 2 cubic angstrom per atom. The diagnostic existed; the
    # function did not consult it.
    converged_key = f"relax_converged_{level}"
    unconverged = []
    entries, rows = [], []
    for i, structure in enumerate(structures(md, source)):
        value = float(md.obs[energy_key].iloc[i])
        if not np.isfinite(value):
            continue
        if converged_key in md.obs and not bool(md.obs[converged_key].iloc[i]):
            unconverged.append(str(md.obs_names[i]))
            continue
        entries.append(ComputedStructureEntry(structure, value))
        rows.append(i)

    if converged_key in reference.obs and not bool(
            reference.obs[converged_key].iloc[0]):
        raise ValueError(
            f"the {ion.symbol} reference did not converge; every voltage "
            f"would be shifted by the same amount and no comparison between "
            f"cathodes would reveal it")

    # Group by the framework left when the working ion is removed, so two
    # unrelated cathodes in one object do not get averaged into one electrode.
    groups: dict = {}
    for entry, i in zip(entries, rows):
        composition = entry.composition.copy()
        stripped = Composition({el: n for el, n in composition.items()
                                if el.symbol != ion.symbol})
        label = (framework if framework is not None
                 else stripped.reduced_formula or "empty")
        groups.setdefault(label, []).append((entry, i))

    if unconverged:
        warnings.warn(
            f"{len(unconverged)} structures are excluded because their "
            f"relaxation did not converge: {unconverged[:5]}. A voltage from "
            f"an unrelaxed cell is not a voltage. Check "
            f"obs['max_force_{level}'] — a cell relaxation with a foundation "
            f"potential can run away on a delithiated framework, and "
            f"mv.calc.relax(cell=False) is the usual way round it",
            RuntimeWarning, stacklevel=2)

    volts = np.full(md.n_obs, np.nan)
    grav = np.full(md.n_obs, np.nan)
    vol = np.full(md.n_obs, np.nan)
    density = np.full(md.n_obs, np.nan)
    swelling = np.full(md.n_obs, np.nan)
    recorded, failures = {}, []

    for label, members in groups.items():
        if len(members) < 2:
            failures.append(
                f"{label}: only one composition, so there is no pair to take "
                f"a difference between — supply both the lithiated and the "
                f"delithiated form")
            continue
        try:
            electrode = InsertionElectrode.from_entries(
                [entry for entry, _ in members], ion_entry)
            average = float(electrode.get_average_voltage())
            gravimetric = float(electrode.get_capacity_grav())
            volumetric = float(electrode.get_capacity_vol())
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue

        amounts = [entry.composition.get(ion.symbol, 0.0)
                   for entry, _ in members]
        volumes = [entry.structure.volume / len(entry.structure)
                   for entry, _ in members]
        order = int(np.argmax(amounts)), int(np.argmin(amounts))
        change = (100.0 * (volumes[order[0]] - volumes[order[1]])
                  / volumes[order[1]]) if volumes[order[1]] else np.nan

        for _, i in members:
            volts[i] = average
            grav[i] = gravimetric
            vol[i] = volumetric
            density[i] = average * gravimetric      # V * mAh/g = mWh/g
            swelling[i] = change
        recorded[label] = {
            "average_voltage": average,
            "capacity_gravimetric_mAh_per_g": gravimetric,
            "capacity_volumetric_mAh_per_cc": volumetric,
            "volume_change_percent": float(change),
            "n_states": len(members),
        }

    md.obs[f"voltage_{level}"] = volts
    md.obs[f"capacity_gravimetric_{level}"] = grav
    md.obs[f"capacity_volumetric_{level}"] = vol
    md.obs[f"energy_density_{level}"] = density
    md.obs[f"volume_change_{level}"] = swelling
    md.uns.setdefault("electrode", {})[level] = {
        "working_ion": ion.symbol,
        "reference_energy_per_atom": per_ion,
        "frameworks": recorded,
        "n_failed": len(failures),
        "errors": failures[:10],
        "unconverged": unconverged[:10],
        "note": "average voltage between the supplied states, against the "
                "working ion's own metal; not a discharge curve unless the "
                "intermediate orderings were supplied as rows",
    }
    # No set_level here. This derives from energies somebody else computed,
    # and stamping the level as "derived" would overwrite the record of the
    # calculator that actually produced them — which is the one fact a
    # voltage needs carried with it. mv.thermo.hull does not stamp one
    # either, for the same reason.
    if failures:
        warnings.warn(
            f"{len(failures)} frameworks produced no voltage; see "
            f"uns['electrode'][{level!r}]['errors']. First: {failures[0]}",
            RuntimeWarning, stacklevel=2)
    record(md, "thermo.voltage", working_ion=ion.symbol, level=level,
           source=source)


@register_function(
    aliases=["theoretical capacity", "maximum capacity", "capacity limit",
             "how much lithium can it hold", "ion removal", "ion insertion"],
    category="thermo",
    description="The capacity a structure could give up or take in, from its "
                "oxidation states alone — no delithiated structure needed.",
    requires={"structures": ["{source}"]},
    # The placeholder is the parameter's own name: the probe expands claims
    # by binding the signature, so {ion} matched nothing and every claim here
    # read as unfulfilled.
    produces={"obs": ["max_ion_removal_{working_ion}",
                      "max_ion_insertion_{working_ion}",
                      "theoretical_capacity_{working_ion}",
                      "theoretical_capacity_volumetric_{working_ion}"],
              "uns": ["theoretical_capacity"]},
    prerequisites=["mv.pp.describe"],
    examples=["mv.thermo.theoretical_capacity(md)",
              "mv.thermo.theoretical_capacity(md, working_ion='Na')"],
    related=["mv.thermo.voltage", "mv.transform.oxidation_states",
             "mv.prop.cost"],
    notes="mv.thermo.voltage needs both ends of the reaction — the lithiated "
          "and the delithiated structure — and gives a voltage. This needs "
          "**one** structure and gives an upper bound on capacity, which is "
          "the question a screen asks first and much more cheaply. Ranking a "
          "library on this and then computing voltages for the survivors is "
          "the order that costs least.\\n\\n"
          "It is a bound and not a prediction. The number counts the "
          "electrons the transition metals could in principle give up, and "
          "says nothing about whether the framework survives losing them: "
          "LiFePO4 comes out at 169.9 mAh/g against a measured 170 because "
          "one electron per iron is exactly what happens there, and a "
          "material whose framework collapses at half that will report the "
          "same bound.\\n\\n"
          "Oxidation states come from bond valence, which is a fit to bond "
          "lengths and fails on structures far from the ones it was fitted "
          "to. A row it cannot assign is left NaN and counted rather than "
          "guessed, because a wrong oxidation state changes the answer by a "
          "whole electron.",
)
def theoretical_capacity(md: AnnData, working_ion: str = "Li",
                         source: str = "input") -> None:
    """Capacity bound from oxidation states. Deposits; returns ``None``."""
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.apps.battery.analyzer import BatteryAnalyzer

    ion = str(working_ion)
    removal = np.full(md.n_obs, np.nan)
    insertion = np.full(md.n_obs, np.nan)
    gravimetric = np.full(md.n_obs, np.nan)
    volumetric = np.full(md.n_obs, np.nan)
    failures = []

    analyzer = BVAnalyzer()
    for i, structure in enumerate(structures(md, source)):
        try:
            decorated = analyzer.get_oxi_state_decorated_structure(structure)
        except Exception as exc:
            failures.append(
                f"{md.obs_names[i]}: bond valence could not assign oxidation "
                f"states ({type(exc).__name__}: {exc})")
            continue
        try:
            battery = BatteryAnalyzer(decorated, working_ion=ion)
            removal[i] = float(battery.max_ion_removal)
            insertion[i] = float(battery.max_ion_insertion)
            gravimetric[i] = float(battery.get_max_capgrav())
            volumetric[i] = float(battery.get_max_capvol())
        except Exception as exc:
            failures.append(f"{md.obs_names[i]}: {type(exc).__name__}: {exc}")

    md.obs[f"max_ion_removal_{ion}"] = removal
    md.obs[f"max_ion_insertion_{ion}"] = insertion
    md.obs[f"theoretical_capacity_{ion}"] = gravimetric
    md.obs[f"theoretical_capacity_volumetric_{ion}"] = volumetric
    md.uns.setdefault("theoretical_capacity", {})[ion] = {
        "working_ion": ion,
        "source": str(source),
        "oxidation_states": "bond valence",
        "capacity_unit": "mAh/g",
        "n_failed": len(failures),
        "errors": failures[:10],
        "note": "an upper bound from electron counting; it says nothing "
                "about whether the framework survives being emptied",
    }
    if failures:
        warnings.warn(
            f"{len(failures)} of {md.n_obs} structures got no capacity; see "
            f"uns['theoretical_capacity'][{ion!r}]['errors']. First: "
            f"{failures[0]}", RuntimeWarning, stacklevel=2)
    record(md, "thermo.theoretical_capacity", working_ion=ion, source=source)

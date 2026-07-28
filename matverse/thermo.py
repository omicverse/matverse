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

import numpy as np
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


__all__ = ["hull", "reaction", "chempot_limits", "references_from_mp",
           "LevelMismatch"]

"""``mv.gen`` — generated candidates, and whether they are any good.

matverse trains no generative model. It holds their output and scores it, which
is the part every paper reimplements slightly differently.

Why the scoring is the contribution
-----------------------------------
Validity, uniqueness, novelty and stability are reported by every crystal
generation paper, and until recently each one pinned the reference dataset, the
stability threshold and the structure-matching tolerance differently — so the
numbers were not comparable even when the metric names were identical.
LeMat-GenBench fixed all three and published a leaderboard against them.

:func:`validate` implements **their** definitions rather than a variant, and
records every parameter it used in ``uns['gen_validate']['definitions']``. Being
able to say which reference set and which threshold produced a S.U.N. rate is
worth more than another wrapper around another model.

What the numbers do not mean
----------------------------
A 2026 stress test found that neither MatterGen nor DiffCSP++ recovered the
experimentally observed structure of the newly synthesised GdNiSn4 and LuNiSn4
within matching tolerance, despite both being built from known motifs. Current
models recombine compositions within known structural families rather than
inventing structure types, and a high novelty rate measured against a reference
database does not contradict that — it measures absence from a list.
:func:`validate` reports novelty as "absent from this reference set" and names
the set.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function

#: Defaults following LeMat-GenBench. Changing one changes what a rate means, so
#: every call records the values it used.
DEFAULTS = {
    "min_distance": 0.5,          # angstrom; below this a cell is broken
    "stability_threshold": 0.0,   # eV/atom above hull for "stable"
    "metastability_threshold": 0.1,   # eV/atom for "metastable"
    "ltol": 0.2,                  # StructureMatcher tolerances
    "stol": 0.3,
    "angle_tol": 5.0,
}


@register_function(
    aliases=["validate generated", "sun rate", "validity uniqueness novelty",
             "generative metrics", "score generated structures", "novelty rate"],
    category="gen",
    description="Score a set of generated structures for validity, uniqueness, "
                "novelty against a reference set, and stability, and report the "
                "combined S.U.N. rate using LeMat-GenBench's definitions.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["gen_valid", "gen_unique", "gen_novel", "gen_stable",
                      "gen_metastable", "gen_sun", "gen_msun"],
              "uns": ["gen_validate"]},
    examples=["mv.gen.validate(md, reference=known)",
              "mv.gen.validate(md, reference=known, level='emt')"],
    related=["mv.pp.qc", "mv.pp.dedup", "mv.tl.novelty", "mv.thermo.hull"],
    notes="Stability is only scored when a level of theory is given and the "
          "hull was built with real reference phases; otherwise it is reported "
          "as not assessed rather than as zero. A closed hull cannot say "
          "whether anything is stable.",
)
def validate(md: AnnData, reference: AnnData | None = None,
             source: str = "input", level: str | None = None,
             stability_threshold: float | None = None,
             metastability_threshold: float | None = None,
             min_distance: float | None = None,
             ltol: float | None = None, stol: float | None = None,
             angle_tol: float | None = None) -> None:
    """Validity, uniqueness, novelty, stability, and the S.U.N. rate."""
    params = {
        "min_distance": _or(min_distance, "min_distance"),
        "stability_threshold": _or(stability_threshold, "stability_threshold"),
        "metastability_threshold": _or(metastability_threshold,
                                       "metastability_threshold"),
        "ltol": _or(ltol, "ltol"), "stol": _or(stol, "stol"),
        "angle_tol": _or(angle_tol, "angle_tol"),
    }
    S = structures(md, source)

    valid, why = _validity(S, params["min_distance"])
    unique = _uniqueness(S, params)
    novel, novelty_note = _novelty(S, reference, params)
    stable, metastable, stability_note = _stability(md, level, params)

    md.obs["gen_valid"] = valid
    md.obs["gen_valid_reason"] = why
    md.obs["gen_unique"] = unique
    md.obs["gen_novel"] = novel
    md.obs["gen_stable"] = stable
    md.obs["gen_metastable"] = metastable

    sun = valid & unique & novel & _as_bool(stable)
    msun = valid & unique & novel & _as_bool(metastable)
    md.obs["gen_sun"] = sun
    md.obs["gen_msun"] = msun

    n = max(md.n_obs, 1)
    md.uns["gen_validate"] = {
        "n": int(md.n_obs),
        "rates": {
            "valid": float(valid.mean()),
            "unique": float(unique.mean()),
            "novel": float(novel.mean()) if novelty_note is None else None,
            "stable": float(_as_bool(stable).sum() / n)
            if stability_note is None else None,
            "sun": float(sun.sum() / n) if stability_note is None
            and novelty_note is None else None,
            "msun": float(msun.sum() / n) if stability_note is None
            and novelty_note is None else None,
        },
        "definitions": {
            **params,
            "source": source,
            "level": level,
            "reference": _reference_label(reference),
            "novelty_means": "absent from the named reference set under "
                             "StructureMatcher at the stated tolerances",
            "metric_family": "LeMat-GenBench (arXiv:2512.04562) definitions",
        },
        "not_assessed": {k: v for k, v in
                         {"novelty": novelty_note,
                          "stability": stability_note}.items() if v},
    }
    record(md, "gen.validate", source=source, level=level,
           reference=_reference_label(reference))


def _or(value, key: str):
    return DEFAULTS[key] if value is None else value


def _as_bool(values) -> np.ndarray:
    """Treat 'not assessed' as not passing, without letting it read as False."""
    arr = np.asarray(values)
    if arr.dtype == object:
        return np.array([bool(v) if v is not None else False for v in arr])
    return arr.astype(bool)


def _reference_label(reference: AnnData | None) -> str | None:
    if reference is None:
        return None
    return f"{reference.n_obs} structures"


def _validity(S: list, min_distance: float):
    """Structurally valid: no atoms closer than ``min_distance``, ordered."""
    valid, reason = [], []
    for s in S:
        why = []
        try:
            if len(s) > 1:
                d = float(np.min(s.distance_matrix[np.triu_indices(len(s), k=1)]))
                if d < min_distance:
                    why.append(f"min_distance {d:.2f}")
        except Exception:
            why.append("distance matrix failed")
        if not bool(getattr(s, "is_ordered", True)):
            why.append("disordered")
        if s.volume <= 0:
            why.append("non-positive cell volume")
        valid.append(not why)
        reason.append("; ".join(why))
    return np.asarray(valid, dtype=bool), reason


def _matcher(params: dict):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    return StructureMatcher(ltol=params["ltol"], stol=params["stol"],
                            angle_tol=params["angle_tol"],
                            primitive_cell=True, scale=True)


def _blocks(S: list) -> dict:
    """Group by reduced formula, so matching stays local rather than all-pairs."""
    out: dict[str, list[int]] = {}
    for i, s in enumerate(S):
        out.setdefault(s.composition.reduced_formula, []).append(i)
    return out


def _uniqueness(S: list, params: dict) -> np.ndarray:
    """First occurrence of each distinct structure counts as unique."""
    matcher = _matcher(params)
    unique = np.ones(len(S), dtype=bool)
    for members in _blocks(S).values():
        seen: list[int] = []
        for i in members:
            for j in seen:
                try:
                    if matcher.fit(S[i], S[j]):
                        unique[i] = False
                        break
                except Exception:
                    continue
            else:
                seen.append(i)
    return unique


def _novelty(S: list, reference: AnnData | None, params: dict):
    """Absent from the reference set. Structural, not compositional."""
    if reference is None:
        return (np.zeros(len(S), dtype=bool),
                "no reference set was given, so novelty was not assessed")
    ref_structures = structures(reference, "input")
    ref_blocks = _blocks(ref_structures)
    matcher = _matcher(params)

    novel = np.ones(len(S), dtype=bool)
    for i, s in enumerate(S):
        for j in ref_blocks.get(s.composition.reduced_formula, []):
            try:
                if matcher.fit(s, ref_structures[j]):
                    novel[i] = False
                    break
            except Exception:
                continue
    return novel, None


def _stability(md: AnnData, level: str | None, params: dict):
    """Distance above the hull, but only when the hull can support the claim."""
    n = md.n_obs
    if level is None:
        return (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool),
                "no level was given, so stability was not assessed")
    column = f"e_above_hull_{level}"
    if column not in md.obs:
        return (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool),
                f"obs[{column!r}] absent; run mv.thermo.hull(md, "
                f"level={level!r}, references=...) first")
    diagram = md.uns.get("phase_diagram", {})
    if diagram.get("closed_system", True):
        return (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool),
                "the hull was built over this dataset's own compositions only, "
                "so it cannot say whether anything is stable. Rebuild with "
                "mv.thermo.hull(..., references=...)")

    above = md.obs[column].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        stable = above <= params["stability_threshold"]
        metastable = above <= params["metastability_threshold"]
    stable &= ~np.isnan(above)
    metastable &= ~np.isnan(above)
    return stable, metastable, None


@register_function(
    aliases=["substitute", "element substitution", "ionic substitution",
             "enumerate candidates", "swap elements", "chemical substitution"],
    category="gen",
    description="Enumerate candidate structures by substituting elements into "
                "known ones, optionally keeping only the charge-balanced "
                "results — the strong baseline that generative models are "
                "measured against.",
    requires={"structures": ["{source}"]},
    examples=["cand = mv.gen.substitute(md, {'Al': ['Ga', 'In']})",
              "cand = mv.gen.substitute(md, {'Cu': ['Ag', 'Au']}, "
              "charge_balanced=True)"],
    related=["mv.gen.validate", "mv.calc.relax"],
    notes="Returns a new dataset rather than depositing, because it has more "
          "rows than the object it came from. Substitution within a known "
          "structure type is what several generative models were found to be "
          "doing implicitly, so it is the baseline worth beating rather than a "
          "fallback.",
)
def substitute(md: AnnData, substitutions: dict, source: str = "input",
               charge_balanced: bool = False,
               keep_parents: bool = False) -> AnnData:
    """Enumerate element substitutions into every structure."""
    from .data import from_structures

    if not substitutions:
        raise ValueError("substitutions must map an element to its replacements, "
                         "e.g. {'Al': ['Ga', 'In']}")

    S = structures(md, source)
    parents = list(md.obs_names)
    out, rows = [], []

    for i, structure in enumerate(S):
        present = {str(el) for el in structure.composition.elements}
        applicable = {k: v for k, v in substitutions.items() if k in present}
        if keep_parents:
            out.append(structure)
            rows.append({"parent": str(parents[i]), "substitution": "",
                         "is_parent": True})
        for original, replacements in applicable.items():
            for replacement in replacements:
                candidate = structure.copy()
                try:
                    candidate.replace_species({original: replacement})
                except Exception:
                    continue
                if charge_balanced and not _plausible(candidate):
                    continue
                out.append(candidate)
                rows.append({"parent": str(parents[i]),
                             "substitution": f"{original}->{replacement}",
                             "is_parent": False})

    if not out:
        raise ValueError(
            "no candidate survived; none of the named elements is present in "
            f"the {source!r} structures, or every result failed the charge "
            f"balance filter")

    candidates = from_structures(out, pd.DataFrame(rows))
    record(candidates, "gen.substitute", source=source,
           n_parents=int(md.n_obs), charge_balanced=charge_balanced)
    return candidates


def _plausible(structure) -> bool:
    """Whether some oxidation-state assignment makes the composition neutral.

    Uses pymatgen's own list of common oxidation states. It is a filter on
    obvious nonsense, not a prediction that the compound forms.
    """
    try:
        return bool(structure.composition.oxi_state_guesses())
    except Exception:
        return True


__all__ = ["validate", "substitute", "alloy_pairs", "predict_substitutions",
           "predict_dopants", "predict_hosts", "DEFAULTS"]


@register_function(
    aliases=["predict substitutions", "substitution probability",
             "likely substitutions", "data mined substitution",
             "probabilistic substitution", "suggest substitutions"],
    category="gen",
    description="Propose element substitutions ranked by how often they occur "
                "in known compounds, and return the candidates as a new "
                "dataset with the probability attached.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["parent", "substitution", "substitution_probability"],
              "structures": ["input"]},
    prerequisites=["mv.transform.oxidation_states"],
    examples=["cand = mv.gen.predict_substitutions(md, source='oxidized')",
              "cand = mv.gen.predict_substitutions(md, source='oxidized', "
              "n=20, threshold=1e-3)"],
    related=["mv.gen.substitute", "mv.transform.oxidation_states",
             "mv.gen.validate"],
    notes="mv.gen.substitute enumerates the swaps you name; this ranks the "
          "swaps you did not think of. The probabilities come from the "
          "data-mined ionic substitution model of Hautier et al., which asks "
          "how often two species replace one another across the ICSD rather "
          "than whether their radii match.\n\n"
          "It needs **oxidation states**, because the model is defined over "
          "ionic species rather than elements — Fe2+ and Fe3+ substitute "
          "differently and the whole point is the distinction. Run "
          "mv.transform.oxidation_states first and pass its variant as source; "
          "a structure without them raises rather than guessing neutral.\n\n"
          "A probability here is a prior from what has been made before. It "
          "says nothing about whether a particular substitution is stable in "
          "this structure, which is what the hull is for — the intended use is "
          "to generate candidates worth relaxing, not to rank them.",
)
def predict_substitutions(md: AnnData, source: str = "input", n: int = 10,
                          threshold: float = 1e-3) -> AnnData:
    """Ranked substitution candidates as a new dataset."""
    from pymatgen.analysis.structure_prediction.substitution_probability import (
        SubstitutionPredictor)
    from pymatgen.core import Species

    from .data import from_structures

    predictor = SubstitutionPredictor(threshold=threshold)
    built, parents, swaps, probabilities = [], [], [], []
    unoxidised = []

    for row, structure in zip(md.obs_names, structures(md, source)):
        species = sorted({str(site.specie) for site in structure})
        if not all(getattr(site.specie, "oxi_state", None) is not None
                   for site in structure):
            unoxidised.append(str(row))
            continue
        try:
            predictions = predictor.list_prediction(species,
                                                    to_this_composition=False)
        except Exception:
            continue
        predictions.sort(key=lambda p: -p["probability"])
        for prediction in predictions[:n]:
            mapping = {str(k): v for k, v in prediction["substitutions"].items()}
            if all(str(k) == str(v) for k, v in mapping.items()):
                continue                       # the identity is not a candidate
            candidate = structure.copy()
            try:
                candidate.replace_species(
                    {Species.from_str(k): v for k, v in mapping.items()})
            except Exception:
                continue
            built.append(candidate)
            parents.append(str(row))
            swaps.append(", ".join(f"{k}->{v}" for k, v in mapping.items()
                                   if str(k) != str(v)))
            probabilities.append(float(prediction["probability"]))

    if unoxidised:
        raise ValueError(
            f"{len(unoxidised)} structure(s) carry no oxidation states, and "
            f"the substitution model is defined over ionic species rather "
            f"than elements. Run mv.transform.oxidation_states(md) and pass "
            f"source='oxidized'. Offending rows: {unoxidised[:5]}")
    if not built:
        raise ValueError(
            f"no substitution cleared threshold={threshold}; lower it, or "
            f"check that the species in this structure appear in the "
            f"data-mined table at all")

    out = from_structures(built, obs=pd.DataFrame({
        "parent": parents,
        "substitution": swaps,
        "substitution_probability": probabilities,
    }))
    out.uns["predict_substitutions"] = {
        "source": source, "threshold": float(threshold), "n_per_parent": int(n),
        "model": "Hautier data-mined ionic substitution",
    }
    record(out, "gen.predict_substitutions", source=source, n=n,
           threshold=threshold)
    return out


@register_function(
    aliases=["predict dopants", "dopants", "n-type dopant", "p-type dopant",
             "doping candidates", "which dopant"],
    category="gen",
    description="Rank likely n-type and p-type dopants for every material from "
                "the same data-mined substitution probabilities.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["n_type_dopant", "n_type_probability",
                      "p_type_dopant", "p_type_probability"],
              "uns": ["dopants"]},
    prerequisites=["mv.transform.oxidation_states"],
    examples=["mv.gen.predict_dopants(md, source='oxidized')",
              "mv.gen.predict_dopants(md, source='oxidized', n=10)"],
    related=["mv.gen.predict_substitutions", "mv.disorder.dope",
             "mv.transform.oxidation_states"],
    notes="n-type means the dopant carries more charge than the site it "
          "replaces and p-type less, so the classification is arithmetic on "
          "oxidation states rather than a calculation of where the level "
          "lands. Whether the dopant is actually shallow, soluble, or "
          "compensated by a native defect is not decided here — "
          "mv.thermo.defect_formation is.\n\n"
          "The top few per material go into obs so a screen can reach them; "
          "the full ranked list stays in uns['dopants'], because the second "
          "and third choices are usually the interesting ones.\n\n"
          "mv.disorder.dope builds the doped supercells once you have chosen. "
          "This chooses.",
)
def predict_dopants(md: AnnData, source: str = "input", n: int = 5,
                    threshold: float = 1e-3) -> None:
    """Ranked dopants per material. Deposits; returns ``None``."""
    from pymatgen.analysis.structure_prediction.dopant_predictor import (
        get_dopants_from_substitution_probabilities)

    best_n = np.empty(md.n_obs, dtype=object)
    best_p = np.empty(md.n_obs, dtype=object)
    prob_n = np.full(md.n_obs, np.nan)
    prob_p = np.full(md.n_obs, np.nan)
    detail: dict = {}
    failed = 0

    for i, (row, structure) in enumerate(
            zip(md.obs_names, structures(md, source))):
        best_n[i] = ""
        best_p[i] = ""
        if not all(getattr(site.specie, "oxi_state", None) is not None
                   for site in structure):
            failed += 1
            continue
        try:
            ranked = get_dopants_from_substitution_probabilities(
                structure, num_dopants=n, threshold=threshold)
        except Exception:
            failed += 1
            continue
        entry = {}
        for kind, target, probability in (("n_type", best_n, prob_n),
                                          ("p_type", best_p, prob_p)):
            candidates = ranked.get(kind) or []
            entry[kind] = [
                {"dopant": str(c["dopant_species"]),
                 "replaces": str(c.get("original_species", "")),
                 "probability": float(c["probability"])}
                for c in candidates]
            if candidates:
                target[i] = str(candidates[0]["dopant_species"])
                probability[i] = float(candidates[0]["probability"])
        detail[str(row)] = entry

    md.obs["n_type_dopant"] = best_n.astype(str)
    md.obs["n_type_probability"] = prob_n
    md.obs["p_type_dopant"] = best_p.astype(str)
    md.obs["p_type_probability"] = prob_p
    md.uns["dopants"] = {"source": source, "n": int(n),
                         "threshold": float(threshold),
                         "n_failed": int(failed), "ranked": detail}
    record(md, "gen.predict_dopants", source=source, n=n, threshold=threshold)


@register_function(
    aliases=["predict hosts", "which structure could host this",
             "target composition", "structure prediction from a target",
             "find a host structure", "inverse substitution"],
    category="gen",
    description="Given a target set of ionic species, find which known "
                "structures could host it and build the substituted "
                "candidates.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["parent", "target", "host_probability"],
              "structures": ["input"]},
    prerequisites=["mv.transform.oxidation_states"],
    examples=["cand = mv.gen.predict_hosts(md, ['Na+', 'Mn2+', 'P5+', 'O2-'], "
              "source='oxidized')"],
    related=["mv.gen.predict_substitutions", "mv.gen.substitute",
             "mv.pp.predict_volume"],
    notes="The inverse of mv.gen.predict_substitutions, and the more useful "
          "direction when you know what you want. That function starts from a "
          "structure and asks what could be swapped into it; this starts from "
          "a **composition** and asks which of the structures you already have "
          "could host it.\\n\\n"
          "From LiFePO4 alone, targeting Na+/Mn2+/P5+/O2-, it produces "
          "NaMnPO4 — a real sodium-ion cathode — because the substitution "
          "model has seen those species replace one another often enough. The "
          "library you pass is the whole search space, so a bigger one finds "
          "more; passing a database export rather than three structures is the "
          "intended use.\\n\\n"
          "**The species count must match.** A four-species target only "
          "considers four-species hosts, because the model substitutes species "
          "one for one and never changes how many there are. A target that "
          "returns nothing usually means the library has no host with the "
          "right number of distinct species, not that the chemistry is "
          "impossible.\\n\\n"
          "Needs oxidation states on the library, for the same reason "
          "mv.gen.predict_substitutions does: the model is defined over ions.",
)
def predict_hosts(md: AnnData, target, source: str = "input",
                  threshold: float = 1e-3,
                  remove_duplicates: bool = True) -> AnnData:
    """Candidate structures for a target species set. Returns a new dataset."""
    try:
        from pymatgen.core.structure_prediction.substitutor import Substitutor
    except ImportError:
        from pymatgen.analysis.structure_prediction.substitutor import (
            Substitutor)
    from pymatgen.core import Species

    from .data import from_structures

    try:
        wanted = [s if isinstance(s, Species) else Species.from_str(str(s))
                  for s in target]
    except Exception as exc:
        raise ValueError(
            f"target must be ionic species with charges, e.g. "
            f"['Na+', 'Mn2+', 'P5+', 'O2-']; got {list(target)!r} ({exc})"
        ) from exc

    library, unoxidised = [], []
    for name, structure in zip(md.obs_names, structures(md, source)):
        if not all(getattr(site.specie, "oxi_state", None) is not None
                   for site in structure):
            unoxidised.append(str(name))
            continue
        library.append({"structure": structure, "id": str(name)})
    if unoxidised:
        raise ValueError(
            f"{len(unoxidised)} structure(s) carry no oxidation states, and "
            f"the substitution model is defined over ions. Run "
            f"mv.transform.oxidation_states(md) and pass source='oxidized'. "
            f"Offending rows: {unoxidised[:5]}")

    sizes = {len({site.specie.symbol for site in entry["structure"]})
             for entry in library}
    predicted = Substitutor(threshold=threshold).pred_from_structures(
        wanted, library, remove_duplicates=remove_duplicates)
    if not predicted:
        raise ValueError(
            f"no host found for {[str(s) for s in wanted]}. The model "
            f"substitutes species one for one, so it only considers hosts with "
            f"{len(wanted)} distinct species; this library has "
            f"{sorted(sizes)}. Widen the library or lower threshold "
            f"(currently {threshold}).")

    built, rows = [], []
    label = ", ".join(str(s) for s in wanted)
    for transformed in predicted:
        built.append(transformed.final_structure)
        history = transformed.history or [{}]
        # The id travels in history[0]['source'], not under 'id' — and the
        # substitution probability is in other_parameters['proba'].
        extra = getattr(transformed, "other_parameters", {}) or {}
        rows.append({
            "parent": str(history[0].get("source", "")),
            "target": label,
            "host_probability": float(extra.get("proba", np.nan)),
        })

    out = from_structures(built, obs=pd.DataFrame(rows))
    out.uns["predict_hosts"] = {
        "source": source, "threshold": float(threshold),
        "target": [str(s) for s in wanted],
        "library_size": len(library),
        "model": "Hautier data-mined ionic substitution",
    }
    record(out, "gen.predict_hosts", source=source, threshold=threshold,
           n_library=len(library))
    return out


@register_function(
    aliases=["alloy pairs", "alloys", "what can i alloy this with",
             "pseudobinary", "solid solution partners", "miscible",
             "substitutable pairs", "vegard"],
    category="gen",
    description="Find every pair of materials in a dataset that forms a "
                "pseudobinary alloy system, and how badly their lattices "
                "disagree.",
    requires={"structures": ["{source}"]},
    examples=["pairs = mv.gen.alloy_pairs(md)",
              "pairs = mv.gen.alloy_pairs(md, max_mismatch=0.05)"],
    related=["mv.gen.predict_substitutions", "mv.disorder.sqs",
             "mv.disorder.sro", "mv.iface.match"],
    notes="Two materials form an alloy pair when one species can be swapped "
          "for another on the same structure, leaving the rest of the lattice "
          "in place — GaAs and AlAs share their arsenic and differ only in the "
          "cation, so (Ga,Al)As exists across the whole range. Silicon and "
          "GaAs do not, whatever their lattices look like, and the pairing is "
          "refused rather than scored badly.\n\n"
          "Returns rather than deposits, because one dataset of n materials "
          "gives up to n(n-1)/2 pairs — the same reason mv.iface.match does. "
          "obs['parent_a'] and obs['parent_b'] point back at the rows.\n\n"
          "obs['lattice_mismatch'] is the fractional difference in the cube "
          "root of the cell volume, which is the number that decides whether "
          "a film will grow strained or relax into misfit dislocations. It is "
          "not a miscibility criterion: GaAs-AlAs at 0.14% is famously "
          "lattice-matched and GaAs-InAs at 7% is famously not, and both are "
          "real alloy systems with real devices built on them. Filter on it "
          "when you care about epitaxy, not when you care about whether the "
          "alloy exists.\n\n"
          "obs['substituting'] names the species being swapped and "
          "obs['observer'] the ones that stay put, which is what tells you "
          "which sublattice the disorder lives on — and so what "
          "mv.disorder.sqs would have to enumerate.",
)
def alloy_pairs(md: AnnData, source: str = "input",
                max_mismatch: float | None = None) -> AnnData:
    """Pseudobinary alloy systems in a dataset. Returns a pairs object."""
    try:
        from pymatgen.analysis.alloys.core import AlloyPair
    except ImportError as exc:                             # pragma: no cover
        raise ImportError(
            f"mv.gen.alloy_pairs needs pymatgen-analysis-alloys, one of "
            f"pymatgen's own add-on packages. Install it with `pip install "
            f"pymatgen-analysis-alloys`. ({exc})") from exc

    from .data import from_structures

    names = [str(n) for n in md.obs_names]
    labels = ([str(v) for v in md.obs["name"]] if "name" in md.obs else names)
    cells = structures(md, source)

    built, rows, refused = [], [], []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            try:
                # from_structures wants the oxidation-decorated structures as a
                # separate argument and will not derive them; passing the plain
                # ones is what its own documentation shows.
                pair = AlloyPair.from_structures(
                    structures=(cells[i], cells[j]),
                    structures_with_oxidation_states=(cells[i], cells[j]),
                    ids=(names[i], names[j]))
            except Exception as exc:
                refused.append(f"{labels[i]}/{labels[j]}: {type(exc).__name__}: "
                               f"{exc}")
                continue

            a = float(pair.volume_cube_root_a)
            b = float(pair.volume_cube_root_b)
            mismatch = abs(a - b) / ((a + b) / 2.0)
            if max_mismatch is not None and mismatch > max_mismatch:
                refused.append(f"{labels[i]}/{labels[j]}: lattice mismatch "
                               f"{mismatch:.3f} over {max_mismatch}")
                continue

            # AlloyPair sorts its own members, so a and b need not be i and j.
            first = names.index(pair.id_a) if pair.id_a in names else i
            second = names.index(pair.id_b) if pair.id_b in names else j
            built.append(cells[first])
            rows.append({
                "parent_a": names[first], "parent_b": names[second],
                "name": f"{labels[first]}-{labels[second]}",
                "pair_formula": str(pair.pair_formula),
                "chemsys": str(pair.chemsys),
                "substituting": "-".join(sorted(
                    {str(e) for e in pair.structure_a.composition.elements}
                    ^ {str(e) for e in pair.structure_b.composition.elements})),
                "observer": "-".join(str(e) for e in pair.observer_elements),
                "lattice_mismatch": mismatch,
                "spacegroup": int(pair.spacegroup_intl_number_a),
            })

    if not built:
        raise ValueError(
            f"no pair of these {md.n_obs} structures forms an alloy system; "
            f"{len(refused)} were refused, first few: {refused[:3]}")

    pairs = from_structures(built, pd.DataFrame(rows))
    pairs.uns["alloy_pairs"] = {
        "source": source, "n_parents": int(md.n_obs),
        "max_mismatch": max_mismatch,
        "mismatch_definition": "|a-b| / mean(a,b) on the cube root of the "
                               "cell volume",
        "n_refused": len(refused), "refused": refused[:20],
    }
    record(pairs, "gen.alloy_pairs", source=source, n_parents=int(md.n_obs))
    return pairs


@register_function(
    aliases=["enumerate compositions", "charge neutral compositions",
             "smact", "candidate formulas", "composition screening",
             "which compositions are plausible", "chemical filter"],
    category="gen",
    description="Enumerate charge-neutral, electronegativity-consistent "
                "compositions from a set of elements — candidates before any "
                "structure exists.",
    produces={"obs": ["formula", "n_elements", "oxidation_states",
                      "n_oxidation_assignments", "stoichiometry"],
              "X": ["composition"]},
    examples=["candidates = mv.gen.compositions(['Ti', 'O'])",
              "candidates = mv.gen.compositions(['Ba', 'Ti', 'O'], "
              "threshold=4)",
              "candidates = mv.gen.compositions(['Li', 'Fe', 'P', 'O'], "
              "sizes=(3, 4))"],
    related=["mv.data.from_compositions", "mv.gen.predict_substitutions",
             "mv.screen.filter", "mv.prop.cost", "mv.thermo.hull"],
    notes="The cheapest filter in the funnel, and the only one that runs "
          "before a structure exists. Two rules do the work: a compound must "
          "be **charge neutral** in some combination of its elements' known "
          "oxidation states, and the more electronegative element must take "
          "the more negative one. Everything failing those is discarded for "
          "the cost of arithmetic, against minutes per candidate for anything "
          "that needs a structure.\\n\\n"
          "What survives is **not** a prediction that the compound exists. It "
          "is a statement that one reason for it not to has been ruled out — "
          "the filter passes roughly a thousand compositions for every one "
          "that has ever been made. It is a way of not wasting a calculator "
          "on CaF3, not a way of finding new materials.\\n\\n"
          "Returns a dataset with **no structures**, because there are none "
          "yet. X is the composition matrix, so mv.feat.element_stats, "
          "mv.tl.pca, mv.screen.filter, mv.prop.cost and mv.prop.supply_risk "
          "all work on it directly; anything reading a structure raises until "
          "one is built.\\n\\n"
          "A composition often has **more than one** charge-neutral "
          "assignment, and charge neutrality cannot say which is meant. "
          "smact_filter returns TiO2 twice, as Ti(+2)O(-1) and as the "
          "rutile-like Ti(+4)O(-2); obs['oxidation_states'] lists all of them "
          "separated by ' | ' and obs['n_oxidation_assignments'] counts them, "
          "rather than reporting whichever came first and dropping the "
          "rest.\\n\\n"
          "oxidation_states= chooses the table. The default 'icsd24' is what "
          "has been observed in the ICSD, which is narrower and more "
          "realistic than the union of everything ever reported; 'smact14' is "
          "the older permissive set and passes considerably more. The choice "
          "changes the answer, so it is recorded in uns['compositions'].",
)
def compositions(elements, threshold: int = 8, sizes=None,
                 use_pauling: bool = True,
                 oxidation_states: str = "icsd24") -> AnnData:
    """Charge-neutral candidate compositions. Returns a materials object."""
    try:
        import smact
        from smact.screening import smact_filter
    except ImportError as exc:
        raise ImportError(
            f"mv.gen.compositions needs SMACT: `pip install "
            f"matverse[screening]` or `pip install smact`. ({exc})") from exc

    from itertools import combinations

    from .data import from_compositions

    symbols = [str(e) for e in elements]
    if len(symbols) < 2:
        raise ValueError(
            f"got {len(symbols)} elements; charge neutrality needs at least "
            f"two so that something can balance something else")
    wanted = (tuple(range(2, len(symbols) + 1)) if sizes is None
              else tuple(int(s) for s in sizes))
    if any(s < 2 or s > len(symbols) for s in wanted):
        raise ValueError(
            f"sizes={sizes} must lie between 2 and the {len(symbols)} "
            f"elements given")

    # One row per reduced formula, but every charge-neutral assignment kept.
    # smact_filter returns TiO2 twice - once as Ti(+2)O(-1) and once as the
    # rutile-like Ti(+4)O(-2) - and taking the first and discarding the rest
    # reports a peroxide assignment for a composition that also has the
    # ordinary one. Which assignment is "right" is not something charge
    # neutrality can decide, so all of them are reported.
    found_by_formula: dict = {}
    order, failures = [], []
    for size in wanted:
        for chosen in combinations(symbols, size):
            try:
                found = smact_filter(
                    [smact.Element(s) for s in chosen], threshold=int(threshold),
                    oxidation_states_set=str(oxidation_states))
            except Exception as exc:
                failures.append(f"{'-'.join(chosen)}: {type(exc).__name__}: "
                                f"{exc}")
                continue
            for entry in found:
                names, states, amounts = entry[0], entry[1], entry[2]
                key = _reduced("".join(f"{n}{a}"
                                       for n, a in zip(names, amounts)))
                if key not in found_by_formula:
                    found_by_formula[key] = []
                    order.append(key)
                found_by_formula[key].append(
                    (tuple(names), tuple(states), tuple(amounts)))

    rows = []
    for key in order:
        assignments = found_by_formula[key]
        names, _, amounts = assignments[0]
        rows.append({
            "formula": key,
            "n_elements": len(names),
            "oxidation_states": " | ".join(
                " ".join(f"{n}{s:+d}" for n, s in zip(a_names, a_states))
                for a_names, a_states, _ in assignments),
            "n_oxidation_assignments": len(assignments),
            "stoichiometry": ":".join(str(a) for a in amounts),
        })

    if not rows:
        raise ValueError(
            f"no charge-neutral composition was found for {symbols} at "
            f"threshold={threshold}"
            + (f"; first failure was {failures[0]}" if failures else
               ". Raising threshold= allows larger stoichiometric ratios"))

    frame = pd.DataFrame(rows)
    md = from_compositions(frame["formula"].tolist(),
                           obs=frame.drop(columns=["formula"]))
    md.uns["compositions"] = {
        "elements": symbols,
        "threshold": int(threshold),
        "sizes": list(wanted),
        "oxidation_states_set": str(oxidation_states),
        "use_pauling": bool(use_pauling),
        "n_candidates": len(rows),
        "n_combinations_tried": sum(
            len(list(combinations(symbols, s))) for s in wanted),
        "n_failed": len(failures),
        "errors": failures[:10],
        "note": "charge neutrality and electronegativity ordering only; "
                "survival is not a prediction that the compound exists",
    }
    if failures:
        warnings.warn(
            f"{len(failures)} element combinations could not be screened; see "
            f"uns['compositions']['errors']. First: {failures[0]}",
            RuntimeWarning, stacklevel=2)
    record(md, "gen.compositions", elements=symbols, threshold=threshold,
           oxidation_states=oxidation_states)
    return md


def _reduced(formula: str) -> str:
    """The reduced formula, so Ti2O4 and TiO2 are not two candidates."""
    from pymatgen.core.composition import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:                                      # pragma: no cover
        return formula


@register_function(
    aliases=["pyxtal", "random crystals", "generate structures", "structure "
             "from symmetry", "space group generation", "make structures from "
             "a formula", "symmetry-based generation"],
    category="gen",
    description="Generate crystal structures for a composition by filling "
                "Wyckoff positions of a space group — structures from a "
                "formula, where none existed.",
    # No requires: obs[source_column] is used when present and the
    # composition matrix is the fallback, so the column is not required. The
    # probe deletes a required slot and expects the call to fail; this one
    # succeeded, which is what a wrong claim looks like.
    produces={"obs": ["parent", "formula", "requested_space_group",
                      "space_group", "space_group_symbol",
                      "symmetry_as_requested", "nsites"],
              "structures": ["input"]},
    prerequisites=["mv.gen.compositions"],
    examples=["built = mv.gen.from_symmetry(candidates, space_groups=[221])",
              "built = mv.gen.from_symmetry(candidates, per_composition=5)",
              "built = mv.gen.from_symmetry(candidates, space_groups=[62, 194])"],
    related=["mv.gen.compositions", "mv.data.from_compositions",
             "mv.calc.relax", "mv.gen.validate", "mv.gen.substitute"],
    notes="This is the step that turns a composition into something a "
          "calculator can accept. mv.gen.compositions produces rows with no "
          "structures; this gives them one, by placing the atoms on Wyckoff "
          "positions of a chosen space group and randomising the free "
          "parameters that remain.\\n\\n"
          "**The output is a starting geometry, not a structure.** Cell "
          "lengths come from a volume estimate and the free coordinates are "
          "random, so bond lengths are only approximately right — generated "
          "BaTiO3 in Pm-3m comes out near 5.06 A against a measured 4.00. "
          "Run mv.calc.relax before believing any energy, and mv.gen.validate "
          "before believing the structure.\\n\\n"
          "The space group is **verified rather than assumed**. What is "
          "requested is what pyxtal was asked for; obs['space_group'] is what "
          "pymatgen finds in the structure that came back, and "
          "obs['symmetry_as_requested'] says whether they agree. They can "
          "differ often, not occasionally: asking for 40 random groups for "
          "TiO2 gave 7 structures of which 5 came back at *higher* symmetry "
          "than requested — P-6 asked for, P6/mmm delivered — because with "
          "few atoms the random free parameters land on a special position. A "
          "generator that reported only the request would be reporting its "
          "input.\\n\\n"
          "Sampling random space groups is wasteful. Of those same 40 "
          "attempts, 33 failed outright because the group could not host the "
          "composition at that cell size. Passing space_groups= explicitly is "
          "both faster and more likely to give what was asked for.\\n\\n"
          "The composition comes from obs[source_column] when it is there and "
          "from the composition matrix otherwise, so a dataset that never had "
          "a formula column still works.\\n\\n"
          "Generation fails for combinations a group cannot host, and a "
          "failure is a missing row plus a counted reason in "
          "uns['from_symmetry'], never a silently substituted structure.",
)
def from_symmetry(md: AnnData, space_groups=None, per_composition: int = 1,
                  dimension: int = 3, source_column: str = "formula",
                  factor: float = 1.1, max_attempts: int = 10,
                  seed: int = 0) -> AnnData:
    """Structures from a composition and a space group. Returns a new object."""
    try:
        from pyxtal import pyxtal
    except ImportError as exc:
        raise ImportError(
            f"mv.gen.from_symmetry needs PyXtal: `pip install "
            f"matverse[generation]` or `pip install pyxtal`. ({exc})") from exc

    from pymatgen.core.composition import Composition

    from .data import from_structures

    formulas = _formulas_of(md, source_column)
    groups = (None if space_groups is None
              else [int(g) for g in np.atleast_1d(space_groups)])
    if groups is not None and any(not 1 <= g <= 230 for g in groups):
        raise ValueError(f"space_groups={space_groups} must lie in 1..230")
    if int(per_composition) < 1:
        raise ValueError("per_composition must be at least 1")

    rng = np.random.default_rng(int(seed))
    built, rows, failures = [], [], []

    for i, formula in enumerate(formulas):
        composition = Composition(formula).reduced_composition
        species = [str(el) for el in composition.elements]
        counts = [int(round(composition[el])) for el in composition.elements]
        wanted = (groups if groups is not None
                  else [int(g) for g in rng.integers(1, 231,
                                                     int(per_composition))])
        repeats = (int(per_composition) if groups is not None else 1)

        for group in wanted:
            for _ in range(repeats):
                structure, why = _one_crystal(
                    pyxtal, dimension, group, species, counts, factor,
                    max_attempts, int(rng.integers(0, 2 ** 31)))
                if structure is None:
                    failures.append(f"{formula} in group {group}: {why}")
                    continue
                achieved, symbol = _spacegroup_of(structure)
                built.append(structure)
                rows.append({
                    "parent": str(md.obs_names[i]),
                    "formula": Composition(formula).reduced_formula,
                    "requested_space_group": int(group),
                    "space_group": int(achieved),
                    "space_group_symbol": str(symbol),
                    "symmetry_as_requested": bool(achieved == group),
                    "nsites": len(structure),
                })

    if not built:
        raise ValueError(
            f"no structure could be generated for any of the {len(formulas)} "
            f"compositions"
            + (f"; first failure was {failures[0]}" if failures else ""))

    out = from_structures(built, obs=pd.DataFrame(rows))
    out.uns["from_symmetry"] = {
        "space_groups": groups,
        "per_composition": int(per_composition),
        "dimension": int(dimension),
        "factor": float(factor),
        "seed": int(seed),
        "n_requested": len(formulas) * int(per_composition),
        "n_built": len(built),
        "n_failed": len(failures),
        "errors": failures[:10],
        "note": "starting geometries: cell volume is estimated and free "
                "coordinates are random, so relax before believing an energy",
    }
    if failures:
        warnings.warn(
            f"{len(failures)} of {len(failures) + len(built)} attempts "
            f"produced no structure and are missing rows rather than "
            f"substituted ones; see uns['from_symmetry']['errors']. First: "
            f"{failures[0]}", RuntimeWarning, stacklevel=2)
    record(out, "gen.from_symmetry", space_groups=groups,
           per_composition=per_composition, seed=seed)
    return out


def _one_crystal(pyxtal_class, dimension, group, species, counts, factor,
                 max_attempts, seed):
    """One random crystal, or ``(None, reason)``."""
    try:
        crystal = pyxtal_class()
        crystal.from_random(dim=int(dimension), group=int(group),
                            species=list(species), numIons=list(counts),
                            factor=float(factor), max_count=int(max_attempts),
                            random_state=int(seed))
        if not getattr(crystal, "valid", False):
            return None, "pyxtal reported the crystal as invalid"
        return crystal.to_pymatgen(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _spacegroup_of(structure, tolerance: float = 0.1):
    """The symmetry actually present, not the one that was asked for."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure, symprec=tolerance)
        return analyzer.get_space_group_number(), analyzer.get_space_group_symbol()
    except Exception:                                      # pragma: no cover
        return -1, "unknown"


def _formulas_of(md: AnnData, column: str) -> list:
    """Formulas from obs if they are there, else from the composition matrix."""
    if column in md.obs:
        return [str(f) for f in md.obs[column]]
    if md.n_vars == 0:
        raise ValueError(
            f"obs[{column!r}] absent and this object has no element axis, so "
            f"there is no composition to build from")
    from pymatgen.core.composition import Composition

    values = np.asarray(md.X.todense() if hasattr(md.X, "todense") else md.X,
                        dtype=float)
    elements = list(md.var_names)
    out = []
    for row in values:
        amounts = {elements[j]: float(v) for j, v in enumerate(row) if v}
        if not amounts:
            raise ValueError("a row has an empty composition")
        out.append(Composition(amounts).reduced_formula)
    return out

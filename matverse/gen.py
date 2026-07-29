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


__all__ = ["validate", "substitute", "predict_substitutions",
           "predict_dopants", "DEFAULTS"]


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

"""``mv.disorder`` — partial occupancy, ordered approximants and doping.

Most of matverse assumes a structure is ordered: this atom is here, that one is
there. A large fraction of real materials are not. A solid solution, a doped
semiconductor, a high-entropy alloy and a cathode part-way through charging all
have sites that are *fractionally* occupied, and a first-principles code cannot
take one as input — it needs a specific arrangement of specific atoms.

Bridging that gap is what this namespace does, and it is a bigger gap than it
looks: a disordered cell stands for an ensemble, and any single ordered cell you
pick is a sample from it. Which sample you pick changes the answer.

```python
mv.disorder.describe(md)                     # how disordered, and where
ordered = mv.disorder.orderings(md, n=8)     # ordered approximants
mv.disorder.sqs(md, scaling=2)               # a special quasirandom structure
```

``describe`` also reports the **configurational entropy**, which is the term
that stabilises high-entropy alloys and that a zero-temperature hull leaves out
entirely — a material can sit above the hull at 0 K and be the phase that forms
at 1500 K for no other reason.
"""

from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function

#: Boltzmann's constant in eV/K, for configurational entropy.
KB_EV = 8.617333262e-5


@register_function(
    aliases=["describe disorder", "partial occupancy", "is it disordered",
             "site occupancy", "configurational entropy", "solid solution"],
    category="disorder",
    description="Report which sites are fractionally occupied, how many, and "
                "the ideal configurational entropy of the mixing.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["is_ordered", "n_disordered_sites",
                      "max_site_disorder", "configurational_entropy",
                      "entropy_term_300K"]},
    examples=["mv.disorder.describe(md)",
              "mv.disorder.describe(md, temperature=1500.0)"],
    related=["mv.disorder.orderings", "mv.disorder.sqs", "mv.thermo.hull"],
    notes="``configurational_entropy`` is the ideal mixing entropy per atom, "
          "``-k_B sum(x ln x)`` summed over sites — the number that makes "
          "high-entropy alloys stable. It is an upper bound: real mixing is "
          "never ideal, and short-range order reduces it.\n\n"
          "``entropy_term_300K`` is ``-T*S`` in eV/atom, which is what a hull "
          "would have to be shifted by. At room temperature it is tens of meV "
          "and usually ignorable; at a synthesis temperature of 1500 K a "
          "five-component equiatomic alloy gets about 0.2 eV/atom, which is "
          "larger than most of the distances a screen calls 'close to the "
          "hull'.",
)
def describe(md: AnnData, source: str = "input",
             temperature: float = 300.0) -> None:
    """Disorder metrics per material. Deposits; returns ``None``."""
    ordered = np.ones(md.n_obs, dtype=bool)
    counts = np.zeros(md.n_obs, dtype=int)
    worst = np.zeros(md.n_obs)
    entropy = np.zeros(md.n_obs)

    for i, structure in enumerate(structures(md, source)):
        ordered[i] = bool(structure.is_ordered)
        total = 0.0
        for site in structure:
            occupancies = np.array(
                [float(v) for v in site.species.as_dict().values()])
            if occupancies.size > 1 or occupancies.sum() < 1 - 1e-9:
                counts[i] += 1
                # How far from a single occupant this site is: 0 when one
                # species owns it, 1 when it is evenly split.
                worst[i] = max(worst[i], 1.0 - occupancies.max())
            positive = occupancies[occupancies > 0]
            if positive.size > 1:
                total += float(-(positive * np.log(positive)).sum())
        entropy[i] = KB_EV * total / max(len(structure), 1)

    md.obs["is_ordered"] = ordered
    md.obs["n_disordered_sites"] = counts
    md.obs["max_site_disorder"] = worst
    md.obs["configurational_entropy"] = entropy
    md.obs["entropy_term_300K"] = -float(temperature) * entropy
    md.uns["disorder"] = {
        "source": source, "temperature": float(temperature),
        "entropy_units": "eV/K/atom, ideal mixing (an upper bound)",
        "note": "-T*S is what a zero-temperature hull leaves out; at 1500 K a "
                "five-component equiatomic alloy gets roughly 0.2 eV/atom",
    }
    record(md, "disorder.describe", source=source, temperature=temperature)


@register_function(
    aliases=["ordered approximants", "order the structure", "enumerate "
             "orderings", "make it ordered", "supercell orderings"],
    category="disorder",
    description="Turn a fractionally occupied structure into specific ordered "
                "cells a DFT code can accept, ranked by electrostatic energy.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["parent", "ordering_index", "ewald_energy"],
              "structures": ["input"]},
    prerequisites=["mv.disorder.describe"],
    examples=["ordered = mv.disorder.orderings(md, n=8)",
              "ordered = mv.disorder.orderings(md, n=4, use_oxidation=True)"],
    related=["mv.disorder.sqs", "mv.disorder.describe", "mv.mag.orderings"],
    notes="Returns an ordinary materials object whose rows are arrangements, "
          "with ``obs['parent']`` pointing back — the same shape as "
          "``mv.mag.orderings`` and ``mv.pp.defects``, because it is the same "
          "move: enumerate the configurations, compute them all, let the "
          "object record which won.\n\n"
          "Ranking is by Ewald energy, which needs oxidation states. Without "
          "them every arrangement scores zero and the ordering is arbitrary — "
          "honest for an alloy, where electrostatics is not what decides, and "
          "wrong for an oxide. Pass ``use_oxidation=True`` there.\n\n"
          "The number of distinct arrangements grows combinatorially. This "
          "returns the best ``n`` by the ranking, not all of them, and says so "
          "in ``uns['orderings']``.",
)
def orderings(md: AnnData, source: str = "input", n: int = 4,
              use_oxidation: bool = False) -> AnnData:
    """Ordered approximants. Returns a materials object."""
    from pymatgen.transformations.advanced_transformations import (
        OrderDisorderedStructureTransformation)

    from .data import from_structures

    labels = [str(x) for x in md.obs.get("name", md.obs_names)]
    built, parents, indices, energies, failures = [], [], [], [], []

    transform = OrderDisorderedStructureTransformation(
        no_oxi_states=not use_oxidation)

    for i, structure in enumerate(structures(md, source)):
        if structure.is_ordered:
            built.append(structure)
            parents.append(labels[i])
            indices.append(0)
            energies.append(np.nan)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                found = transform.apply_transformation(
                    structure, return_ranked_list=n)
        except Exception as exc:
            failures.append(f"{labels[i]}: {type(exc).__name__}: {exc}")
            continue

        for k, entry in enumerate(found[:n]):
            built.append(entry["structure"])
            parents.append(labels[i])
            indices.append(k)
            energies.append(float(entry.get("energy", np.nan)))

    if not built:
        raise ValueError(
            "no ordered structure could be produced. "
            + (f"Errors: {failures[:3]}" if failures else
               "Every input was already ordered and none was returned, which "
               "should not happen — please report it."))

    out = from_structures(built)
    out.obs["parent"] = parents
    out.obs["ordering_index"] = indices
    out.obs["ewald_energy"] = energies
    out.uns["orderings"] = {
        "source": source, "n_requested": int(n),
        "use_oxidation": bool(use_oxidation),
        "n_failed": len(failures), "errors": failures[:10],
        "note": "the best n by Ewald energy, not every arrangement; the count "
                "of distinct arrangements grows combinatorially",
    }
    if not use_oxidation:
        out.uns["orderings"]["ranking"] = (
            "no oxidation states, so every Ewald energy is zero and the order "
            "is arbitrary. Honest for an alloy; pass use_oxidation=True for an "
            "oxide, where electrostatics is what decides.")
    record(out, "disorder.orderings", source=source, n=n,
           use_oxidation=use_oxidation)
    return out


@register_function(
    aliases=["sqs", "special quasirandom structure", "random alloy",
             "high entropy alloy", "quasirandom"],
    category="disorder",
    description="Build a special quasirandom structure — the small cell whose "
                "correlation functions best match a genuinely random alloy.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["sqs"], "obs": ["sqs_objective"]},
    prerequisites=["mv.disorder.describe"],
    examples=["mv.disorder.sqs(md, scaling=2)",
              "mv.disorder.sqs(md, scaling=[2, 2, 2], search_time=5.0)"],
    related=["mv.disorder.orderings", "mv.disorder.describe"],
    notes="An SQS is the right tool where ``orderings`` is the wrong one. "
          "Enumerating arrangements and taking the lowest gives you the "
          "*ordered ground state*, which is the opposite of a solid solution; "
          "an SQS is built to look random on short length scales, which is "
          "what a solid solution actually is.\n\n"
          "Needs ATAT's ``mcsqs``, which is a separate Fortran program and not "
          "pip-installable. When it is absent this raises with the download "
          "link rather than returning the ordered ground state under a "
          "different name.",
)
def sqs(md: AnnData, source: str = "input", scaling=2,
        search_time: float = 10.0, key_added: str = "sqs") -> None:
    """A quasirandom cell per material. Deposits; returns ``None``."""
    from pymatgen.transformations.advanced_transformations import (
        SQSTransformation)

    from ._core import deposit_structures

    transform = SQSTransformation(scaling=scaling, search_time=search_time)
    built, objectives, failures = [], [], []

    for i, structure in enumerate(structures(md, source)):
        if structure.is_ordered:
            built.append(structure)
            objectives.append(np.nan)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                built.append(transform.apply_transformation(structure))
                objectives.append(np.nan)
        except RuntimeError as exc:
            if "AT-AT" in str(exc) or "mcsqs" in str(exc):
                raise ImportError(
                    "mv.disorder.sqs needs ATAT's mcsqs, a separate Fortran "
                    "program: https://axelvandewalle.github.io/www-avdw/atat/ "
                    "\n\nDo not substitute mv.disorder.orderings for it — that "
                    "gives the ordered ground state, which is the opposite of "
                    "a solid solution."
                ) from exc
            failures.append(f"{i}: {exc}")
            built.append(structure)
            objectives.append(np.nan)
        except Exception as exc:
            failures.append(f"{i}: {type(exc).__name__}: {exc}")
            built.append(structure)
            objectives.append(np.nan)

    deposit_structures(md, key_added, built)
    md.obs["sqs_objective"] = objectives
    md.uns["sqs"] = {"source": source, "scaling": scaling,
                     "search_time": float(search_time),
                     "n_failed": len(failures), "errors": failures[:10]}
    record(md, "disorder.sqs", source=source, scaling=scaling,
           key_added=key_added)


@register_function(
    # 'doping' stays with mv.pp.defects, which enumerates substitutions as a
    # kind of point defect. This is the dilute, charge-compensated version.
    aliases=["dope", "add a dopant", "substitutional doping",
             "aliovalent doping", "dilute dopant"],
    category="disorder",
    description="Enumerate doped cells — substitute a dopant onto the sites "
                "its ionic radius and charge allow, with charge compensation.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["parent", "dopant", "doping_index"],
              "structures": ["input"]},
    examples=["doped = mv.disorder.dope(md, 'Nb5+')",
              "doped = mv.disorder.dope(md, 'La3+', min_length=12.0)"],
    related=["mv.gen.substitute", "mv.pp.defects", "mv.disorder.orderings"],
    notes="Unlike ``mv.gen.substitute``, which swaps every atom of an element, "
          "this puts a *dilute* dopant on a subset of sites and compensates "
          "the charge — which is what doping a semiconductor actually is.\n\n"
          "pymatgen's DopingTransformation enumerates internally with enumlib, "
          "a separate Fortran program. Without it the transformation returns "
          "**zero structures and no error**, so a doping study would silently "
          "produce nothing. matverse checks for that and raises.",
)
def dope(md: AnnData, dopant: str, source: str = "input",
         min_length: float = 10.0, n: int = 4,
         max_per_enum: int = 20) -> AnnData:
    """Doped cells. Returns a materials object."""
    from pymatgen.transformations.advanced_transformations import (
        DopingTransformation)

    from .data import from_structures

    labels = [str(x) for x in md.obs.get("name", md.obs_names)]
    transform = DopingTransformation(dopant, min_length=min_length,
                                     max_structures_per_enum=max_per_enum)

    built, parents, indices, failures = [], [], [], []
    for i, structure in enumerate(structures(md, source)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                found = transform.apply_transformation(
                    structure, return_ranked_list=n)
        except Exception as exc:
            failures.append(f"{labels[i]}: {type(exc).__name__}: {exc}")
            continue
        for k, entry in enumerate(found[:n]):
            built.append(entry["structure"])
            parents.append(labels[i])
            indices.append(k)

    if not built:
        raise ValueError(
            f"no doped structure was produced for {dopant!r}. "
            f"pymatgen's DopingTransformation enumerates with **enumlib**, a "
            f"separate Fortran program, and returns an empty list rather than "
            f"raising when it is missing — so this is usually a missing "
            f"install rather than a chemistry answer. Get it from "
            f"https://github.com/msg-byu/enumlib.\n\n"
            f"If enumlib is present, the dopant may genuinely fit nowhere: "
            f"check its charge and ionic radius against the host sites."
            + (f" Errors: {failures[:3]}" if failures else ""))

    out = from_structures(built)
    out.obs["parent"] = parents
    out.obs["dopant"] = str(dopant)
    out.obs["doping_index"] = indices
    out.uns["doping"] = {"dopant": str(dopant), "source": source,
                         "min_length": float(min_length),
                         "n_failed": len(failures), "errors": failures[:10]}
    record(out, "disorder.dope", dopant=dopant, source=source,
           min_length=min_length)
    return out


__all__ = ["KB_EV", "describe", "orderings", "sqs", "dope", "sro"]


@register_function(
    aliases=["sro", "short range order", "warren-cowley", "warren cowley",
             "chemical order", "sro parameter", "is my sqs random",
             "clustering tendency", "ordering tendency"],
    category="disorder",
    description="Measure the chemical short-range order of a structure — "
                "whether like atoms sit next to like, next to unlike, or at "
                "random — as Warren-Cowley parameters per element pair.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["sro_shell{shell}"], "uns": ["sro"],
              "obs": ["sro_max_shell{shell}", "sro_rms_shell{shell}"]},
    prerequisites=["mv.disorder.sqs"],
    examples=["mv.disorder.sro(md)",
              "mv.disorder.sro(md, source='sqs_2x2x2', shell=1)"],
    related=["mv.disorder.sqs", "mv.disorder.orderings", "mv.env.coordination"],
    notes="The Warren-Cowley parameter for an ordered pair (A, B) in shell k "
          "is\n\n"
          "    alpha_AB = 1 - P(B | A) / c_B\n\n"
          "where P(B|A) is the chance that a shell-k neighbour of an A atom is "
          "a B atom, and c_B is B's overall fraction. **Zero is random.** "
          "Negative means A prefers B as a neighbour — ordering. Positive "
          "means A avoids B, so A clusters with itself.\n\n"
          "The check that fixes the sign convention in your head: B2 brass, "
          "where every nearest neighbour of a copper is a zinc, gives -1 for "
          "the unlike pair and +1 for the like one. A random solid solution "
          "gives 0 for every pair, to within the noise of a finite cell.\n\n"
          "This is what an SQS is *for*, and the only way to know whether one "
          "worked. mv.disorder.sqs returns a cell that is supposed to imitate "
          "a random alloy; running this on it tells you how well it managed, "
          "and obs['sro_rms'] is the single number to screen or sort on.\n\n"
          "Computed here from the definition rather than through pymatgen's "
          "analysis.disorder, whose get_warren_cowley_parameters returns the "
          "same value for every pair — on B2 it gives -1 for the like pairs as "
          "well as the unlike ones, where the definition requires +1 and -1.\n\n"
          "The shell is found from the sorted neighbour distances of each "
          "site, so it follows the structure rather than a cutoff you have to "
          "guess. Pass cutoff to override it with a distance in angstroms.",
)
def sro(md: AnnData, source: str = "input", shell: int = 1,
        cutoff: float | None = None, tolerance: float = 0.1,
        key_added: str | None = None) -> None:
    """Warren-Cowley short-range order parameters. Deposits; returns ``None``."""
    elements = sorted({site.specie.symbol
                       for structure in structures(md, source)
                       for site in structure})
    pairs = [(a, b) for a in elements for b in elements]
    labels = [f"{a}-{b}" for a, b in pairs]

    name = key_added or f"shell{shell}"
    values = np.full((md.n_obs, len(pairs)), np.nan)
    failures: list[str] = []

    for row, structure in enumerate(structures(md, source)):
        symbols = np.array([site.specie.symbol for site in structure])
        fractions = {el: float((symbols == el).mean()) for el in elements}

        # Counts[a][b] = how many shell-k neighbours of species b an atom of
        # species a has, summed over every a atom in the cell.
        counts = {a: {b: 0.0 for b in elements} for a in elements}
        totals = {a: 0.0 for a in elements}

        for index, site in enumerate(structure):
            if cutoff is not None:
                neighbours = structure.get_neighbors(site, cutoff)
            else:
                low, high = _shell_band(structure, index, shell, tolerance)
                # The k-th shell, not the first k shells. Taking everything
                # inside the outer radius mixes them: on B2 the second shell is
                # six like neighbours and the first is eight unlike ones, and
                # pooling them turns an alpha of -1 into +0.14.
                neighbours = [n for n in structure.get_neighbors(site, high + 1e-9)
                              if n.nn_distance >= low - 1e-9]
            if not neighbours:
                continue
            a = symbols[index]
            for neighbour in neighbours:
                counts[a][neighbour.specie.symbol] += 1.0
                totals[a] += 1.0

        for column, (a, b) in enumerate(pairs):
            if totals[a] <= 0 or fractions.get(b, 0.0) <= 0:
                continue
            probability = counts[a][b] / totals[a]
            values[row, column] = 1.0 - probability / fractions[b]

        if not np.isfinite(values[row]).any():
            failures.append(str(md.obs_names[row]))

    # Not deposit_grid: that axis is numeric — a 2-theta, an energy — and this
    # one is a list of element pairs. The matrix goes to obsm all the same, so
    # the level suffix keeps working, and the pair names go beside it.
    md.obsm[f"sro_{name}"] = values
    md.uns.setdefault("sro", {})[name] = {
        "pairs": labels, "shell": int(shell), "cutoff": cutoff,
        "definition": "alpha_AB = 1 - P(B|A) / c_B; 0 is random, negative is "
                      "ordering, positive is clustering",
        "errors": failures,
    }

    with np.errstate(invalid="ignore"):
        md.obs[f"sro_max_{name}"] = np.nanmax(np.abs(values), axis=1)
        md.obs[f"sro_rms_{name}"] = np.sqrt(np.nanmean(values ** 2, axis=1))
    record(md, "disorder.sro", source=source, shell=shell, cutoff=cutoff,
           key_added=name)


def _shell_band(structure, index: int, shell: int,
                tolerance: float) -> tuple[float, float]:
    """The inner and outer radius of the nth distinct neighbour shell.

    Distances are grouped rather than taken one at a time: the twelve
    neighbours of an fcc site are one shell, not twelve, and a cell relaxed off
    its ideal geometry spreads them by a little. `tolerance` is that little, as
    a fraction of the distance.
    """
    site = structure[index]
    distances = sorted(n.nn_distance
                       for n in structure.get_neighbors(site, 12.0))
    if not distances:
        return 0.0, 0.0
    shells, current = [], [distances[0]]
    for distance in distances[1:]:
        if distance - current[0] > tolerance * current[0]:
            shells.append(current)
            current = [distance]
        else:
            current.append(distance)
    shells.append(current)
    index_ = min(max(int(shell), 1), len(shells)) - 1
    return float(min(shells[index_])), float(max(shells[index_]))

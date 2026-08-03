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


@register_function(
    aliases=["cluster expansion", "fit a cluster expansion", "effective "
             "cluster interactions", "ECI", "lattice model", "alloy "
             "hamiltonian", "CE"],
    category="disorder",
    description="Fit a cluster expansion to computed energies of ordered "
                "arrangements, giving an effective Hamiltonian that scores any "
                "configuration on the same lattice without another calculator "
                "call.",
    requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
    # No uns claim here: the model lands on parent, not on md, and a
    # produces entry names what appears on the object the function is called
    # with. The probe caught the difference.
    produces={"obs": ["ce_energy_{level}", "ce_residual_{level}"]},
    dispatch="level= names the energies being fitted, as for mv.calc.energy",
    prerequisites=["mv.disorder.orderings", "mv.calc.energy"],
    examples=["mv.disorder.cluster_expansion(ordered, parent=alloy, "
              "level='emt')",
              "mv.disorder.cluster_expansion(ordered, parent=alloy, "
              "level='pbe', cutoffs={2: 7.0, 3: 5.0})"],
    related=["mv.disorder.monte_carlo", "mv.disorder.orderings",
             "mv.disorder.sqs", "mv.disorder.sro"],
    notes="A cluster expansion is a fit, and the number that says whether it "
          "is any good is the **cross-validated** error, not the training "
          "error. Both are reported in uns['cluster_expansion'], and a fit "
          "whose training error is much the smaller of the two has too many "
          "clusters for the data — which is the normal failure, because the "
          "number of clusters grows fast with the cutoffs and the number of "
          "training structures does not.\n\n"
          "The fit is ordinary least squares. That is deliberate: a "
          "regularised fit hides an under-determined problem by shrinking "
          "coefficients rather than failing, and the condition number and the "
          "feature-to-structure ratio are both reported so an "
          "under-determined fit is visible instead. Widen the training set or "
          "shorten the cutoffs.\n\n"
          "The model lands on **parent**, not on the training object, because "
          "the parent is what owns the sublattice the expansion is defined "
          "on — and mv.disorder.monte_carlo samples the parent. The training "
          "object gets its predictions and residuals so the fit can be "
          "inspected row by row. The fit itself is per primitive cell, which "
          "is smol's convention and the only scale on which a cluster "
          "expansion is size-independent; ce_energy_{level} is multiplied "
          "back up to a total energy so it can be compared with "
          "energy_{level} directly, while the errors in uns are quoted per "
          "primitive cell.\n\n"
          "Structures the expansion cannot map onto the parent lattice are "
          "skipped and counted rather than silently dropped: a relaxation "
          "that moved atoms off their lattice sites is the usual cause, and "
          "it means the expansion is being asked about a different problem.",
)
def cluster_expansion(md: AnnData, parent: AnnData, level: str = "emt",
                      source: str = "input", parent_source: str = "input",
                      cutoffs: dict | None = None,
                      basis: str = "indicator") -> None:
    """Fit a cluster expansion. Deposits; returns ``None``."""
    try:
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        from smol.cofe import ClusterSubspace, StructureWrangler
    except ImportError as exc:
        raise ImportError(
            f"mv.disorder.cluster_expansion needs smol: `pip install "
            f"matverse[alloys]` or `pip install smol`. ({exc})") from exc

    energy_key = f"energy_{level}"
    if energy_key not in md.obs:
        raise ValueError(
            f"obs[{energy_key!r}] absent; a cluster expansion is a fit to "
            f"energies, so run mv.calc.energy(md, level={level!r}) on the "
            f"ordered structures first")
    if parent.n_obs != 1:
        raise ValueError(
            f"parent has {parent.n_obs} rows; pass the single disordered "
            f"structure the expansion is defined on, e.g. parent[[0]]")

    cutoffs = dict(cutoffs or {2: 7.0, 3: 5.0})
    prim = structures(parent, parent_source)[0]
    subspace = ClusterSubspace.from_cutoffs(
        prim, cutoffs={int(k): float(v) for k, v in cutoffs.items()},
        basis=basis)

    wrangler = StructureWrangler(subspace)
    energies = md.obs[energy_key].to_numpy(dtype=float)
    kept, skipped = [], []
    for i, cell in enumerate(structures(md, source)):
        if not np.isfinite(energies[i]):
            skipped.append(f"{md.obs_names[i]}: no {energy_key}")
            continue
        try:
            wrangler.add_entry(
                ComputedStructureEntry(cell, float(energies[i])), verbose=False)
        except Exception as exc:
            skipped.append(f"{md.obs_names[i]}: {type(exc).__name__}: {exc}")
            continue
        if wrangler.num_structures == len(kept):
            skipped.append(f"{md.obs_names[i]}: could not be matched onto the "
                           f"parent lattice")
            continue
        kept.append(i)

    if wrangler.num_structures < 2:
        raise ValueError(
            f"only {wrangler.num_structures} of {md.n_obs} structures could be "
            f"fitted; a cluster expansion needs several. First reason: "
            f"{skipped[0] if skipped else 'unknown'}")

    features = np.asarray(wrangler.feature_matrix, dtype=float)
    # smol normalises the target per primitive cell, and the correlation
    # functions are intensive, so the coefficients come out in eV per prim.
    # obs['energy_{level}'] is a total energy, so the prediction has to be
    # multiplied back by the number of prims before the two can be compared -
    # without it the residual column is a units error rather than a residual.
    values = np.asarray(wrangler.get_property_vector("energy", normalize=True),
                        dtype=float)
    sizes = np.asarray(wrangler.sizes, dtype=float)
    coefficients, *_ = np.linalg.lstsq(features, values, rcond=None)

    # Rank rather than shape. Having more structures than clusters does not
    # make a fit determined: on a training set of one supercell shape, whole
    # clusters can be indistinguishable, and the shape test says nothing about
    # it — 50 structures and 8 clusters looked fine while the rank was 6.
    rank = int(np.linalg.matrix_rank(features))
    residuals = features @ coefficients - values
    train_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    cv_rmse = _leave_one_out(features, values)

    predicted = np.full(md.n_obs, np.nan)
    residual = np.full(md.n_obs, np.nan)
    for position, row in enumerate(kept):
        total = float(features[position] @ coefficients) * float(sizes[position])
        predicted[row] = total
        residual[row] = total - float(energies[row])

    md.obs[f"ce_energy_{level}"] = predicted
    md.obs[f"ce_residual_{level}"] = residual
    parent.uns.setdefault("cluster_expansion", {})[level] = {
        "cutoffs": {int(k): float(v) for k, v in cutoffs.items()},
        "basis": str(basis),
        "coefficients": np.asarray(coefficients, dtype=float),
        "n_features": int(features.shape[1]),
        "n_structures": int(features.shape[0]),
        "n_skipped": int(len(skipped)),
        "skipped": skipped[:10],
        # Both in eV per primitive cell, which is the scale the fit is on.
        "train_rmse": train_rmse,
        "cv_rmse": cv_rmse,
        "units": "eV per primitive cell",
        "condition_number": float(np.linalg.cond(features)),
        "rank": int(rank),
        "rank_deficient": bool(rank < features.shape[1]),
        "parent_source": str(parent_source),
    }
    if rank < features.shape[1]:
        warnings.warn(
            f"the feature matrix has rank {rank} for {features.shape[1]} "
            f"clusters, so {features.shape[1] - rank} of them are not "
            f"identified by this training set and their coefficients are "
            f"whatever the least-norm solution happened to pick. More "
            f"structures will not help if they are all the same supercell — "
            f"the usual cause is that every training cell is too small to "
            f"tell two clusters apart. Vary the supercell shape, or shorten "
            f"cutoffs=.", RuntimeWarning, stacklevel=2)
    if skipped:
        warnings.warn(
            f"{len(skipped)} of {md.n_obs} structures were not fitted; see "
            f"parent.uns['cluster_expansion'][{level!r}]['skipped']. First: "
            f"{skipped[0]}", RuntimeWarning, stacklevel=2)

    record(md, "disorder.cluster_expansion", level=level, source=source,
           cutoffs=cutoffs, basis=basis)


def _leave_one_out(features: np.ndarray, values: np.ndarray) -> float:
    """Leave-one-out cross-validated RMSE of an ordinary least-squares fit.

    The training error of a cluster expansion says almost nothing — adding
    clusters drives it to zero whether or not they mean anything — so this is
    the number worth reporting.
    """
    if len(values) <= features.shape[1] + 1:
        return float("nan")
    errors = []
    for i in range(len(values)):
        mask = np.ones(len(values), dtype=bool)
        mask[i] = False
        try:
            fit, *_ = np.linalg.lstsq(features[mask], values[mask], rcond=None)
        except np.linalg.LinAlgError:                       # pragma: no cover
            return float("nan")
        errors.append(float(features[i] @ fit - values[i]))
    return float(np.sqrt(np.mean(np.square(errors))))


@register_function(
    aliases=["monte carlo", "order disorder transition", "finite temperature "
             "order", "transition temperature", "MC", "thermodynamic "
             "sampling", "critical temperature"],
    category="disorder",
    description="Sample a fitted cluster expansion at temperature, giving the "
                "energy, the heat capacity and the order-disorder transition "
                "temperature on the condition axis.",
    requires={"uns": ["cluster_expansion"], "structures": ["{source}"]},
    produces={"obsm": ["mc_energy_{level}", "mc_heat_capacity_{level}"],
              "obs": ["order_disorder_temperature_{level}"],
              "uns": ["grids", "monte_carlo"], "levels": ["{level}"]},
    prerequisites=["mv.disorder.cluster_expansion"],
    dispatch="level= selects which fitted expansion to sample",
    examples=["mv.disorder.monte_carlo(alloy, level='emt')",
              "mv.disorder.monte_carlo(alloy, level='emt', "
              "temperatures=(200, 400, 600, 800), supercell=(4, 4, 4))"],
    related=["mv.disorder.cluster_expansion", "mv.disorder.sro",
             "mv.disorder.describe", "mv.md.sweep"],
    notes="This is where a cluster expansion earns itself. The expansion "
          "scores a configuration for the cost of a dot product, so a "
          "temperature sweep over millions of configurations costs what one "
          "DFT calculation would — and the order-disorder transition is a "
          "collective effect that no single ordered cell shows.\\n\\n"
          "The transition temperature is read off the **heat-capacity peak**, "
          "which is a finite-size estimate and biased: a real transition is a "
          "divergence and a finite cell gives a rounded bump, shifted and "
          "broadened by an amount that only shrinks as the cell grows. Treat "
          "the number as an estimate whose error is bounded by the "
          "temperature spacing at best, and check it moves little between two "
          "supercell sizes before quoting it.\\n\\n"
          "A peak must also **stand out**, or it is noise. A flat heat "
          "capacity has a maximum like any other array, and on Cu-Au at a "
          "3x3x3 cell the curve was flat to within 20% and the reported peak "
          "moved from 380 K to 260 K purely by running longer chains, while a "
          "4x4x4 gave a peak six times the baseline that stayed put. A "
          "maximum below min_prominence= times the median capacity is "
          "therefore reported as NaN rather than as a temperature. The fix is "
          "a larger supercell=, not longer chains: a transition is a "
          "collective effect and a cell too small to hold the ordered domain "
          "cannot show one however long it is sampled.\\n\\n"
          "A peak at either **end** of the scanned range is reported as NaN "
          "rather than as a temperature, and the reason lands in "
          "uns['monte_carlo'][level]['unresolved_transitions']. An edge "
          "maximum means either the transition is outside the range or the "
          "coldest chains never equilibrated — a chain still relaxing has a "
          "large energy variance that is the approach to equilibrium and not "
          "a fluctuation, and the two are indistinguishable from the "
          "variance alone. Both are fixed by widening temperatures= or "
          "lengthening steps=, and neither is fixed by believing the number.\\n\\n"
          "The heat capacity is computed from the energy variance, and "
          "d<E>/dT computed by finite differences across the sweep is an "
          "independent route to the same quantity. Where the sampling is good "
          "the two agree to a couple of percent; where they disagree the "
          "sampling is not good. That is worth checking on a new system, and "
          "it is what the suite checks.\\n\\n"
          "Sampling is canonical, so the composition is whatever the starting "
          "configuration had and is conserved exactly. That is the right "
          "ensemble for an order-disorder transition at fixed composition and "
          "the wrong one for anything that exchanges atoms with a reservoir.\\n\\n"
          "The first ``discard`` fraction of each chain is dropped as burn-in "
          "before any average is taken, because an average that includes the "
          "approach to equilibrium is not an equilibrium average. The "
          "acceptance rate is recorded per temperature: near zero means the "
          "chain never moved and the numbers are the starting configuration, "
          "not a sample.\\n\\n"
          "threads= defaults to 1 on purpose. smol defaults it to the CPU "
          "count, and for the cell sizes this is used at that is a "
          "catastrophic default rather than a fast one - a single swap is a "
          "few dozen multiply-adds, and spawning a thread team per evaluation "
          "costs orders of magnitude more than the evaluation. Measured on a "
          "17-core node: 2000 steps on 27 sites took 62 s at the default and "
          "0.10 s at one thread, a factor of 620. threadpoolctl does not "
          "reach it. Raise threads= only for cells of many thousands of "
          "sites, and measure rather than assume.",
)
def monte_carlo(md: AnnData, level: str = "emt", source: str = "input",
                temperatures=(200.0, 400.0, 600.0, 800.0, 1000.0),
                supercell=(3, 3, 3), steps: int = 20000,
                discard: float = 0.3, seed: int = 0,
                threads: int = 1, min_prominence: float = 2.0) -> None:
    """Canonical Monte Carlo on a fitted expansion. Deposits; returns ``None``."""
    try:
        from smol.cofe import ClusterExpansion, ClusterSubspace
        from smol.moca import Ensemble, Sampler
    except ImportError as exc:
        raise ImportError(
            f"mv.disorder.monte_carlo needs smol: `pip install "
            f"matverse[alloys]` or `pip install smol`. ({exc})") from exc

    from ._core import deposit_grid, set_level

    fits = md.uns.get("cluster_expansion", {})
    if level not in fits:
        raise ValueError(
            f"no cluster expansion for level={level!r}; run "
            f"mv.disorder.cluster_expansion(ordered, parent=md, "
            f"level={level!r}) first. Present: {sorted(fits) or 'none'}")
    fit = fits[level]

    grid = np.asarray(list(temperatures), dtype=float)
    if not len(grid):
        raise ValueError("temperatures= is empty")

    energy = np.full((md.n_obs, len(grid)), np.nan)
    capacity = np.full((md.n_obs, len(grid)), np.nan)
    acceptance = np.full((md.n_obs, len(grid)), np.nan)
    transition = np.full(md.n_obs, np.nan)
    failures, edge_peaks = [], []

    for i, prim in enumerate(structures(md, source)):
        try:
            subspace = ClusterSubspace.from_cutoffs(
                prim, cutoffs={int(k): float(v)
                               for k, v in fit["cutoffs"].items()},
                basis=str(fit["basis"]))
            expansion = ClusterExpansion(
                subspace, coefficients=np.asarray(fit["coefficients"],
                                                  dtype=float))
            ensemble = Ensemble.from_cluster_expansion(
                expansion, supercell_matrix=np.diag(list(supercell)))
            # smol defaults this to the CPU count, which for a cell of a few
            # hundred sites is a catastrophic default rather than a fast one:
            # one swap is a few dozen multiply-adds, and spawning a thread
            # team per evaluation costs orders of magnitude more than the
            # evaluation. On a 17-core node, 2000 steps on 27 sites took 62 s
            # at the default and 0.10 s at one thread. threadpoolctl does not
            # reach it; this attribute is the knob.
            ensemble.processor.num_threads = int(threads)
            try:
                ensemble.processor.num_threads_full = int(threads)
            except Exception:                              # pragma: no cover
                pass
            start = _random_occupancy(ensemble, prim, seed)
            n_sites = int(ensemble.num_sites)
        except Exception as exc:
            failures.append(f"{md.obs_names[i]}: {type(exc).__name__}: {exc}")
            continue

        for j, temperature in enumerate(grid):
            try:
                sampler = Sampler.from_ensemble(
                    ensemble, temperature=float(temperature),
                    seeds=[int(seed) + j])
                sampler.run(int(steps), initial_occupancies=start[None, :],
                            thin_by=10, progress=False)
                trace = np.asarray(sampler.samples.get_energies(), dtype=float)
                keep = trace[int(len(trace) * float(discard)):]
                if not len(keep):                          # pragma: no cover
                    continue
                energy[i, j] = float(keep.mean()) / n_sites
                # C = (<E^2> - <E>^2) / (kB T^2), per site.
                capacity[i, j] = float(keep.var()) / (
                    KB_EV * temperature ** 2 * n_sites)
                # Thinning makes this an estimate rather than the exact
                # accepted fraction, which is why it is reported as a
                # diagnostic and not used for anything.
                acceptance[i, j] = float(np.atleast_1d(
                    sampler.samples.sampling_efficiency(
                        discard=int(len(trace) * float(discard))))[0])
            except Exception as exc:
                failures.append(
                    f"{md.obs_names[i]} at {temperature:g} K: "
                    f"{type(exc).__name__}: {exc}")

        finite = np.isfinite(capacity[i])
        if finite.sum() >= 3:
            available = grid[finite]
            values = capacity[i][finite]
            peak = int(np.argmax(values))
            baseline = float(np.median(values))
            prominence = (float(values[peak]) / baseline
                          if baseline > 0 else np.inf)
            if prominence < float(min_prominence):
                # A flat heat capacity has a maximum too, and it is wherever
                # the noise happened to be largest. Measured on Cu-Au: at a
                # 3x3x3 cell the curve was flat to within 20% and the reported
                # peak moved from 380 K to 260 K purely by lengthening the
                # chains, while a 4x4x4 gave a peak six times the baseline
                # that did not move. Reporting the first as a transition
                # temperature is reporting noise with a unit attached.
                edge_peaks.append(
                    f"{md.obs_names[i]}: the heat capacity is flat - its "
                    f"maximum is only {prominence:.2f}x the baseline, below "
                    f"min_prominence={float(min_prominence):g} - so no "
                    f"transition is resolved. A larger supercell= is what "
                    f"sharpens this; longer chains alone will not")
            elif peak in (0, len(available) - 1):
                # The maximum sits on the edge of the scanned range, so there
                # is no peak inside it: either the transition is outside the
                # range, or the coldest chains never equilibrated and their
                # variance is the approach to equilibrium rather than a
                # fluctuation. Both look identical here, and reporting the
                # edge temperature as a transition would be reporting an
                # artefact. Widen temperatures= or lengthen steps=.
                edge_peaks.append(
                    f"{md.obs_names[i]}: the heat capacity peaks at "
                    f"{available[peak]:g} K, an end of the scanned range, so "
                    f"no transition is resolved inside it")
            else:
                transition[i] = float(available[peak])

    deposit_grid(md, "mc_energy", level, energy, grid, unit="K",
                 value_unit="eV/site", supercell=list(supercell), steps=steps)
    deposit_grid(md, "mc_heat_capacity", level, capacity, grid, unit="K",
                 value_unit="eV/K/site")
    md.obs[f"order_disorder_temperature_{level}"] = transition
    md.uns.setdefault("monte_carlo", {})[level] = {
        "threads": int(threads),
        "temperatures": grid.tolist(),
        "supercell": list(supercell),
        "n_sites": int(np.prod(supercell) * len(structures(md, source)[0])),
        "steps": int(steps),
        "discard": float(discard),
        "seed": int(seed),
        "acceptance": acceptance.tolist(),
        "ensemble": "canonical (composition conserved)",
        "min_prominence": float(min_prominence),
        "transition_from": "heat-capacity peak, required to be interior to "
                           "the scanned range and at least min_prominence "
                           "times the median capacity; a finite-size estimate "
                           "resolved no better than the temperature spacing",
        "n_failed": len(failures),
        "errors": failures[:10],
        "unresolved_transitions": edge_peaks[:10],
    }
    set_level(md, level, kind="model", method="cluster expansion Monte Carlo",
              surrogate=True, source=source, supercell=list(supercell),
              steps=steps)
    if failures:
        warnings.warn(
            f"{len(failures)} Monte Carlo runs failed; see "
            f"uns['monte_carlo'][{level!r}]['errors']. First: {failures[0]}",
            RuntimeWarning, stacklevel=2)
    if edge_peaks:
        warnings.warn(
            f"{len(edge_peaks)} of {md.n_obs} materials have no transition "
            f"resolved inside the scanned temperatures, so the column is NaN "
            f"there rather than the edge value. First: {edge_peaks[0]}",
            RuntimeWarning, stacklevel=2)
    record(md, "disorder.monte_carlo", level=level, source=source,
           temperatures=grid.tolist(), supercell=list(supercell), steps=steps)


def _random_occupancy(ensemble, prim, seed: int) -> np.ndarray:
    """A starting configuration at the parent's own composition.

    Canonical sampling conserves whatever composition it starts from, so this
    is what sets it: the disordered parent's site occupancies, rounded to
    whole atoms on the supercell.
    """
    rng = np.random.default_rng(int(seed))
    occupancy = []
    for sublattice in ensemble.sublattices:
        species = list(sublattice.species)
        sites = list(sublattice.sites)
        fractions = np.array(
            [float(prim[0].species.get(s, 0.0)) if hasattr(prim[0], "species")
             else 0.0 for s in species], dtype=float)
        if fractions.sum() <= 0:
            fractions = np.ones(len(species))
        counts = np.floor(fractions / fractions.sum() * len(sites)).astype(int)
        counts[int(np.argmax(fractions))] += len(sites) - counts.sum()
        codes = np.repeat(np.arange(len(species)), counts)
        rng.shuffle(codes)
        occupancy.append((sites, codes))

    out = np.zeros(ensemble.num_sites, dtype=int)
    for sites, codes in occupancy:
        out[np.asarray(sites, dtype=int)] = codes
    return out

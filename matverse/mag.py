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
    requires={"orderings_.obs": ["energy_per_atom_{level}", "parent",
                                 "ordering"]},
    produces={"md.obs": ["magnetic_ordering_{level}",
                         "magnetic_spread_{level}",
                         "energy_per_atom_{level}", "total_magmom_{level}"],
              "orderings_.obs": ["is_ground_state_{level}"]},
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


__all__ = ["orderings", "ground_state", "describe", "jahn_teller",
           "MAGNETIC_ELEMENTS"]


@register_function(
    aliases=["jahn teller", "jahn-teller", "jt distortion",
             "orbital degeneracy", "is it jahn teller active",
             "octahedral distortion"],
    category="mag",
    description="Whether each material contains an ion whose partly filled d "
                "shell makes its coordination octahedron unstable against "
                "distortion, and how strongly.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["jahn_teller_active", "jahn_teller_strength",
                      "jahn_teller_species"],
              "uns": ["jahn_teller"]},
    examples=["mv.mag.jahn_teller(md)",
              "mv.mag.jahn_teller(md, guess_spin=True)"],
    related=["mv.env.chemenv", "mv.mag.describe", "mv.transform.oxidation_states"],
    notes="A degenerate electronic ground state in an octahedral site lowers "
          "its energy by distorting, and the distortion is not a detail: it is "
          "why LaMnO3 is orthorhombic rather than cubic, why manganese "
          "spinels fade on cycling, and why a structure prediction that "
          "assumed the ideal polyhedron can be qualitatively wrong.\n\n"
          "The answer depends on the oxidation state and the spin state. "
          "Valences are assigned from the structure by bond valence sums; "
          "guess_spin=True additionally guesses high or low spin from the "
          "species, which is a further assumption and off by default. A "
          "structure that already carries oxidation states — from "
          "mv.transform.oxidation_states — is used as given.\n\n"
          "'strong' means an e_g degeneracy, which distorts markedly; 'weak' "
          "means a t_2g one, which usually does not survive room temperature. "
          "Reading 'weak' as 'distorted' is the common mistake.",
)
def jahn_teller(md: AnnData, source: str = "input", guess_spin: bool = False,
                tolerance: float = 0.1) -> None:
    """Jahn-Teller activity per material. Deposits; returns ``None``."""
    from pymatgen.analysis.magnetism.jahnteller import JahnTellerAnalyzer

    analyzer = JahnTellerAnalyzer()
    active = np.zeros(md.n_obs, dtype=bool)
    strength = np.empty(md.n_obs, dtype=object)
    species = np.empty(md.n_obs, dtype=object)
    detail: dict = {}
    failed = 0

    for i, (row, structure) in enumerate(
            zip(md.obs_names, structures(md, source))):
        strength[i] = "none"
        species[i] = ""
        try:
            result = analyzer.get_analysis(
                structure, calculate_valences=True,
                guesstimate_spin=guess_spin, op_threshold=tolerance)
        except Exception:
            strength[i] = "unknown"
            failed += 1
            continue
        active[i] = bool(result.get("active", False))
        if active[i]:
            strength[i] = str(result.get("strength", "unknown"))
            # The list holds the active sites only, each with the ion, the
            # ligand it is coordinated by and the spread of those bond
            # lengths — which is the distortion itself.
            sites = result.get("sites") or []
            names = sorted({str(site.get("species", "")) for site in sites})
            species[i] = ", ".join(n for n in names if n)
        detail[str(row)] = {"active": bool(active[i]),
                            "strength": str(strength[i]),
                            "sites": result.get("sites", [])}

    md.obs["jahn_teller_active"] = active
    md.obs["jahn_teller_strength"] = strength.astype(str)
    md.obs["jahn_teller_species"] = species.astype(str)
    md.uns["jahn_teller"] = {"source": source, "guess_spin": bool(guess_spin),
                             "n_failed": int(failed), "per_material": detail}
    record(md, "mag.jahn_teller", source=source, guess_spin=guess_spin)


@register_function(
    aliases=["exchange", "exchange coupling", "heisenberg", "curie "
             "temperature", "neel temperature", "ordering temperature",
             "magnetic coupling", "J"],
    category="mag",
    description="Fit Heisenberg exchange couplings to the energies of "
                "different magnetic orderings, and estimate the ordering "
                "temperature from them.",
    requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
    produces={"obs": ["exchange_{level}", "ordering_temperature_{level}"],
              "uns": ["exchange"]},
    prerequisites=["mv.mag.orderings", "mv.calc.energy"],
    examples=["mv.mag.exchange(orderings, level='pbe')",
              "mv.mag.exchange(orderings, level='pbe', cutoff=5.0)"],
    related=["mv.mag.orderings", "mv.mag.ground_state", "mv.mag.describe"],
    notes="Which ordering is lowest tells you whether a material is a "
          "ferromagnet or an antiferromagnet. It does not tell you how far "
          "above room temperature it stays one, and that is usually the "
          "question. Mapping the energies onto a Heisenberg Hamiltonian gives "
          "the exchange couplings, and a mean-field estimate of the ordering "
          "temperature follows from them.\\n\\n"
          "**Read the temperature as an upper bound.** Mean-field theory "
          "ignores the fluctuations that destroy order, so it overestimates "
          "Curie and Neel temperatures systematically, often by a third to a "
          "half. It is useful for ranking candidates and for saying 'not "
          "anywhere near room temperature'; it is not a prediction of a "
          "measurement.\\n\\n"
          "Needs several orderings — one energy cannot determine a coupling — "
          "and they must be genuinely different arrangements of the same "
          "cell, which is what mv.mag.orderings produces. The couplings come "
          "back in meV under pymatgen's own convention, which counts per site "
          "rather than per bond; the ratio between two materials is "
          "convention-free and the absolute value is not.\\n\\n"
          "This is only worth running on energies from a spin-polarised "
          "calculation. A potential that does not distinguish spin gives the "
          "same energy for every ordering, the fit is then degenerate, and "
          "the failure is recorded rather than dressed up as a small "
          "coupling.",
)
def exchange(md: AnnData, level: str = "pbe", source: str = "input",
             cutoff: float = 0.0, tol: float = 0.02,
             key_added: str | None = None) -> None:
    """Heisenberg couplings and an ordering temperature. Deposits ``None``."""
    from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper

    energy_key = f"energy_{level}"
    if energy_key not in md.obs:
        raise ValueError(f"obs[{energy_key!r}] absent; run "
                         f"mv.calc.energy(orderings, level={level!r}) first")
    if md.n_obs < 2:
        raise ValueError(f"exchange needs at least two orderings to compare, "
                         f"got {md.n_obs}; one energy cannot determine a "
                         f"coupling")

    energies = md.obs[energy_key].to_numpy(dtype=float)
    cells = structures(md, source)
    finite = np.isfinite(energies)
    if finite.sum() < 2:
        raise ValueError(f"only {int(finite.sum())} of {md.n_obs} orderings "
                         f"have a finite {energy_key}")

    name = key_added or level
    spread = float(np.nanmax(energies) - np.nanmin(energies))
    couplings, temperature, error = {}, np.nan, ""

    if spread < 1e-9:
        error = (f"every ordering has the same energy to within {spread:.2e} "
                 f"eV, so the Heisenberg fit is degenerate — this is what a "
                 f"calculator that does not distinguish spin produces")
    else:
        try:
            mapper = HeisenbergMapper(
                [cells[i] for i in np.flatnonzero(finite)],
                [float(e) for e in energies[finite]],
                cutoff=float(cutoff), tol=float(tol))
            couplings = {str(k): float(v)
                         for k, v in mapper.get_exchange().items()}
            if couplings:
                temperature = float(
                    mapper.get_mft_temperature(list(couplings.values())[0]))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    primary = (float(list(couplings.values())[0]) if couplings else np.nan)
    md.obs[f"exchange_{name}"] = np.full(md.n_obs, primary)
    md.obs[f"ordering_temperature_{name}"] = np.full(md.n_obs, temperature)
    md.uns.setdefault("exchange", {})[name] = {
        "couplings": couplings, "unit": "meV",
        "convention": "pymatgen's, counted per site rather than per bond",
        "ordering_temperature": temperature,
        "temperature_note": "mean-field, an upper bound — fluctuations are "
                            "ignored and it overestimates systematically",
        "energy_spread": spread, "n_orderings": int(finite.sum()),
        "error": error or None,
    }
    if error:
        import warnings as _warnings
        _warnings.warn(f"mv.mag.exchange could not fit couplings: {error}",
                       stacklevel=2)
    record(md, "mag.exchange", level=level, cutoff=float(cutoff),
           key_added=name)

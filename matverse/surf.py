"""``mv.surf`` — surfaces, shapes and adsorption.

Everything a material does that is useful — catalysis, corrosion, wetting,
sintering, the shape a nanoparticle grows into — happens at a surface, and none
of it is visible from the bulk cell.

This used to sit outside a screening library because a surface energy meant a
DFT slab calculation per facet per material, and an adsorption energy meant a
campaign. Universal potentials changed the arithmetic: an adsorption energy is
now one model call, and enumerating twenty facets across a thousand candidates is
an afternoon rather than a grant.

Slabs make more rows than they consume
--------------------------------------
One bulk structure yields many facets, and one facet yields many adsorption
sites, so :func:`slabs` and :func:`adsorption_sites` **return new datasets**
rather than depositing. Each carries ``obs['parent']`` back to the material it
came from, the same foreign key the sites axis uses — so a screen over facets can
always be rolled back up to a screen over materials.

Surface energy needs a bulk reference
-------------------------------------
The energy of a slab is not a surface energy. Subtracting the bulk energy of the
same number of atoms is what leaves the cost of the two surfaces created, and
dividing by twice the cross-sectional area gives an intensive quantity. Get the
reference wrong and every number is wrong by a constant, which is invisible in a
ranking and fatal in a Wulff construction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import record, set_level, structures
from ._registry import register_function

#: eV/angstrom^2 -> J/m^2, the unit surface energies are quoted in.
_EV_PER_A2_TO_J_PER_M2 = 16.021766208


@register_function(
    aliases=["slabs", "surfaces", "generate slabs", "cleave", "facets",
             "miller indices", "make surface"],
    category="surf",
    description="Cut every distinct low-index surface from each bulk structure "
                "and return them as a new dataset, one row per facet and "
                "termination.",
    requires={"structures": ["{source}"]},
    examples=["facets = mv.surf.slabs(md, max_index=1)",
              "facets = mv.surf.slabs(md, miller=(1, 1, 1), min_slab=12.0)"],
    related=["mv.surf.surface_energy", "mv.surf.wulff",
             "mv.surf.adsorption_sites"],
    notes="Returns rather than deposits, because one bulk gives many facets. "
          "obs['parent'] points back at the material, obs['miller'] names the "
          "facet, and obs['termination'] distinguishes the inequivalent ways of "
          "cutting the same plane.\n\n"
          "Slab and vacuum thickness are the parameters people get wrong. Too "
          "thin a slab and the two surfaces interact through the material; too "
          "thin a vacuum and they interact through the gap. The defaults here "
          "are reasonable and are not a substitute for a convergence test — "
          "surfaxe exists for that.",
)
def slabs(md: AnnData, source: str = "input", max_index: int = 1,
          miller=None, min_slab: float = 10.0, min_vacuum: float = 12.0,
          symmetrize: bool = False) -> AnnData:
    """Cut surfaces from every bulk structure. Returns a new dataset."""
    from pymatgen.core.surface import SlabGenerator, generate_all_slabs

    from .data import from_structures

    built, rows, failed = [], [], []
    for name, structure in zip(md.obs_names, structures(md, source)):
        try:
            if miller is not None:
                generator = SlabGenerator(structure, tuple(miller), min_slab,
                                          min_vacuum, center_slab=True)
                cut = generator.get_slabs(symmetrize=symmetrize)
            else:
                cut = generate_all_slabs(structure, max_index, min_slab,
                                         min_vacuum, center_slab=True,
                                         symmetrize=symmetrize)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        seen: dict[tuple, int] = {}
        for slab in cut:
            index = tuple(int(v) for v in slab.miller_index)
            termination = seen.get(index, 0)
            seen[index] = termination + 1
            built.append(slab)
            rows.append({
                "parent": str(name),
                "miller": _miller_label(index),
                "termination": termination,
                "slab_area": float(_area(slab)),
                "is_polar": bool(getattr(slab, "is_polar", lambda: False)()),
                "is_symmetric": bool(
                    getattr(slab, "is_symmetric", lambda: True)()),
            })

    if not built:
        raise ValueError(f"no slab was generated; {len(failed)} structures "
                         f"failed: {failed[:3]}")

    facets = from_structures(built, pd.DataFrame(rows))
    facets.uns["slabs"] = {"source": source, "max_index": max_index,
                           "miller": list(miller) if miller else None,
                           "min_slab": min_slab, "min_vacuum": min_vacuum,
                           "n_parents": int(md.n_obs), "errors": failed}
    record(facets, "surf.slabs", source=source, max_index=max_index,
           n_parents=int(md.n_obs))
    return facets


def _miller_label(index) -> str:
    """One spelling of a Miller index, used everywhere in this module.

    Underscores separate the components rather than nothing at all, because a
    bare join makes (1, -1, 1) into '1-11', which reads as three different
    facets depending on where the eye stops.
    """
    return "_".join(str(int(v)) for v in index)


def _area(slab) -> float:
    """Cross-sectional area of the slab, perpendicular to the surface normal."""
    matrix = np.asarray(slab.lattice.matrix, dtype=float)
    return float(np.linalg.norm(np.cross(matrix[0], matrix[1])))


@register_function(
    aliases=["surface energy", "cleavage energy", "gamma", "surface tension"],
    category="surf",
    description="Compute the surface energy of every slab against the bulk "
                "energy per atom of the material it was cut from.",
    requires={"obs": ["energy_{level}", "parent", "slab_area"]},
    produces={"obs": ["surface_energy_{level}",
                      "surface_energy_{level}_off_stoichiometry"]},
    prerequisites=["mv.calc.relax", "mv.surf.slabs"],
    examples=["mv.surf.surface_energy(facets, bulk=md, level='emt')"],
    related=["mv.surf.slabs", "mv.surf.wulff",
             "mv.surf.surface_energy_chempot"],
    notes="Only defined for a slab whose composition is the bulk formula times "
          "an integer. A slab that is off-stoichiometry has leftover atoms "
          "whose energy depends on the reservoir they came from, so this "
          "function marks those NaN rather than returning a number that looks "
          "like a surface energy and is not one — "
          "mv.surf.surface_energy_chempot handles them. mv.surf.slabs keeps "
          "stoichiometry by default; symmetrize=True is what breaks it.\n\n"
          "Takes the bulk dataset as an argument rather than guessing. The "
          "reference must be the same material at the same level of theory, and "
          "getting it wrong shifts every surface energy by a constant — "
          "invisible in a ranking, fatal in a Wulff construction, which is why "
          "the bulk object is required rather than optional.",
)
def surface_energy(facets: AnnData, bulk: AnnData, level: str = "emt",
                   key_added: str | None = None) -> None:
    """Surface energy in J/m², from slab and bulk energies at one level."""
    slab_key = f"energy_{level}"
    bulk_key = f"energy_per_atom_{level}"
    if slab_key not in facets.obs:
        raise ValueError(f"obs[{slab_key!r}] absent on the slabs; run "
                         f"mv.calc.relax(facets, level={level!r}) first")
    if bulk_key not in bulk.obs:
        raise ValueError(f"obs[{bulk_key!r}] absent on the bulk object; run "
                         f"mv.calc.energy(bulk, level={level!r}) first")
    for column in ("parent", "slab_area"):
        if column not in facets.obs:
            raise ValueError(f"obs[{column!r}] absent; these slabs did not "
                             f"come from mv.surf.slabs")

    reference = dict(zip(map(str, bulk.obs_names),
                         bulk.obs[bulk_key].to_numpy(dtype=float)))
    slab_energy = facets.obs[slab_key].to_numpy(dtype=float)
    area = facets.obs["slab_area"].to_numpy(dtype=float)
    parents = facets.obs["parent"].astype(str).to_numpy()
    slabs_ = list(structures(facets, "input"))
    counts = np.array([len(s) for s in slabs_], dtype=float)

    # E_slab - N * e_bulk only removes the bulk cost if the slab is the bulk
    # formula times an integer. A slab that is off-stoichiometry has left atoms
    # over, and their energy has no business being called a surface energy - it
    # depends on where those atoms came from, which is a chemical potential.
    # SlabGenerator keeps stoichiometry by default, but symmetrize=True deletes
    # sites to make the two faces equivalent and routinely breaks it: on rutile
    # (100) it returns Ti3O8 and Ti4O6 beside the stoichiometric Ti4O8.
    bulk_formula = {str(name_): st.composition.reduced_composition
                    for name_, st in zip(map(str, bulk.obs_names),
                                         structures(bulk, "input"))}
    off = np.zeros(facets.n_obs, dtype=bool)
    for i, slab in enumerate(slabs_):
        formula = bulk_formula.get(parents[i])
        if formula is None:
            continue
        composition = slab.composition
        ratios = [composition[el] / formula[el] for el in formula]
        off[i] = (max(ratios) - min(ratios)) > 1e-6

    per_atom = np.array([reference.get(p, np.nan) for p in parents])
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = ((slab_energy - counts * per_atom) / (2.0 * area)
                 * _EV_PER_A2_TO_J_PER_M2)
    gamma[off] = np.nan

    name = key_added or f"surface_energy_{level}"
    facets.obs[name] = gamma
    facets.obs[f"{name}_off_stoichiometry"] = off
    facets.uns.setdefault("surface_energy", {})[level] = {
        "unit": "J/m^2", "reference": bulk_key,
        "n_unmatched": int(np.isnan(per_atom).sum()),
        "n_off_stoichiometry": int(off.sum()),
    }
    if off.any():
        import warnings as _warnings
        _warnings.warn(
            f"{int(off.sum())} of {facets.n_obs} slabs are not the bulk "
            f"formula times an integer, so a surface energy against the bulk "
            f"energy per atom is not defined for them; "
            f"obs[{name!r}] is NaN there. Use "
            f"mv.surf.surface_energy_chempot, which references the leftover "
            f"atoms to an elemental reservoir.",
            stacklevel=2)
    record(facets, "surf.surface_energy", level=level, key_added=name)


@register_function(
    aliases=["surface energy chempot", "non-stoichiometric surface energy",
             "off-stoichiometry surface energy", "surface excess",
             "surface phase diagram", "gamma vs chemical potential"],
    category="surf",
    description="Compute the surface energy of slabs that are off the bulk "
                "stoichiometry, referencing the leftover atoms to elemental "
                "reservoirs at a chosen chemical potential.",
    requires={"facets.obs": ["energy_{level}", "parent", "slab_area"],
              "bulk.obs": ["energy_{level}"],
              "refs.obs": ["energy_{level}"]},
    produces={"facets.obs": ["surface_energy_{level}"]},
    prerequisites=["mv.surf.slabs", "mv.calc.energy"],
    examples=["mv.surf.surface_energy_chempot(facets, bulk=md, refs=elemental,"
              " level='emt')",
              "mv.surf.surface_energy_chempot(facets, bulk=md, refs=elemental,"
              " chempot={'Au': -0.3})"],
    related=["mv.surf.surface_energy", "mv.surf.slabs", "mv.surf.wulff"],
    notes="A slab off the bulk stoichiometry has no single surface energy. It "
          "has a line: gamma(dmu) = gamma_0 - Gamma * dmu, where Gamma is the "
          "surface excess of the element in surplus and dmu is how far that "
          "element's reservoir sits below its elemental reference. This "
          "deposits gamma at the chemical potential you name (zero, meaning "
          "the element-rich limit, if you name none) and the excess Gamma "
          "beside it, so the line can be redrawn anywhere without recomputing "
          "an energy.\n\n"
          "The excess lands in obs['surface_excess_<element>_<level>'], one "
          "column per element that any slab is in surplus or deficit of, in "
          "J/m^2 per eV. It is not in `produces` because it is not always "
          "produced: which elements appear is a property of the slabs handed "
          "in, and a fully stoichiometric set makes none of these columns at "
          "all. A claim that holds only sometimes is worse than no claim.\n\n"
          "Which facet wins is a function of dmu, not a fact about the "
          "material: the ordering of two terminations can invert between the "
          "metal-rich and the oxygen-rich end of the same phase diagram. A "
          "Wulff shape built from one column is a shape at one chemical "
          "potential.\n\n"
          "Stoichiometric slabs are handled too and agree with "
          "mv.surf.surface_energy, so a mixed set can go through this in one "
          "call.",
)
def surface_energy_chempot(facets: AnnData, bulk: AnnData, refs: AnnData,
                           level: str = "emt", chempot: dict | None = None,
                           key_added: str | None = None) -> None:
    """Surface energy of off-stoichiometric slabs, at a chemical potential."""
    from pymatgen.analysis.surface_analysis import SlabEntry
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    energy_key = f"energy_{level}"
    for obj, label in ((facets, "slabs"), (bulk, "bulk"), (refs, "refs")):
        if energy_key not in obj.obs:
            raise ValueError(f"obs[{energy_key!r}] absent on the {label} "
                             f"object; run mv.calc.energy({label}, "
                             f"level={level!r}) first")
    for column in ("parent", "slab_area"):
        if column not in facets.obs:
            raise ValueError(f"obs[{column!r}] absent; these slabs did not "
                             f"come from mv.surf.slabs")

    # One elemental reservoir per element, cheapest per atom if several are
    # given: a reservoir is the phase the atoms would leave to, and that is the
    # stable elemental form, not whichever polymorph happened to be listed
    # first.
    reservoir: dict[str, ComputedStructureEntry] = {}
    for name, structure, energy in zip(map(str, refs.obs_names),
                                       structures(refs, "input"),
                                       refs.obs[energy_key].to_numpy(float)):
        composition = structure.composition
        if not composition.is_element:
            raise ValueError(f"refs row {name!r} is {composition.reduced_formula}, "
                             f"not an element; the reservoirs must be "
                             f"elemental structures")
        element = str(next(iter(composition.elements)))
        entry = ComputedStructureEntry(structure, float(energy))
        current = reservoir.get(element)
        if current is None or (float(energy) / len(structure)
                               < current.energy / len(current.structure)):
            reservoir[element] = entry

    bulk_entry = {
        str(name): ComputedStructureEntry(structure, float(energy))
        for name, structure, energy in zip(map(str, bulk.obs_names),
                                           structures(bulk, "input"),
                                           bulk.obs[energy_key].to_numpy(float))
    }

    shifts = {str(k): float(v) for k, v in (chempot or {}).items()}
    parents = facets.obs["parent"].astype(str).to_numpy()
    energies = facets.obs[energy_key].to_numpy(dtype=float)
    gamma = np.full(facets.n_obs, np.nan)
    excess: dict[str, np.ndarray] = {}
    unresolved: list[str] = []

    for i, slab in enumerate(structures(facets, "input")):
        ucell = bulk_entry.get(parents[i])
        if ucell is None:
            unresolved.append(f"{parents[i]}: no bulk row")
            continue
        needed = {str(el) for el in slab.composition.elements}
        missing = sorted(needed - set(reservoir))
        if missing:
            unresolved.append(f"{parents[i]}: no reservoir for "
                              f"{', '.join(missing)}")
            continue
        miller = tuple(int(v) for v in str(facets.obs["miller"].iloc[i]).split("_"))
        entry = SlabEntry(slab, float(energies[i]), miller)
        try:
            value = entry.surface_energy(
                ucell, ref_entries=[reservoir[el] for el in sorted(needed)])
        except Exception as exc:
            unresolved.append(f"{parents[i]} {miller}: "
                              f"{type(exc).__name__}: {exc}")
            continue

        # Stoichiometric slabs come back as a plain float; off-stoichiometric
        # ones as gamma_0 + coefficient * delu_<element>, one symbol per
        # element in surplus or deficit.
        symbols = getattr(value, "free_symbols", set())
        if symbols:
            substitution = {}
            for symbol in symbols:
                element = str(symbol).replace("delu_", "")
                substitution[symbol] = shifts.get(element, 0.0)
                column = excess.setdefault(
                    element, np.full(facets.n_obs, np.nan))
                column[i] = -float(value.coeff(symbol)) * _EV_PER_A2_TO_J_PER_M2
            value = value.subs(substitution)
        gamma[i] = float(value) * _EV_PER_A2_TO_J_PER_M2

    name = key_added or f"surface_energy_{level}"
    facets.obs[name] = gamma
    for element, column in excess.items():
        facets.obs[f"surface_excess_{element}_{level}"] = column
    facets.uns.setdefault("surface_energy_chempot", {})[level] = {
        "unit": "J/m^2", "excess_unit": "J/m^2 per eV",
        "chempot": shifts,
        "reservoirs": {el: e.composition.reduced_formula
                       for el, e in reservoir.items()},
        "n_off_stoichiometry": int(
            np.any([~np.isnan(c) for c in excess.values()], axis=0).sum()
            if excess else 0),
        "errors": unresolved,
    }
    record(facets, "surf.surface_energy_chempot", level=level,
           chempot=shifts, key_added=name)


@register_function(
    aliases=["wulff", "wulff construction", "equilibrium shape",
             "crystal shape", "nanoparticle shape"],
    category="surf",
    description="Build the equilibrium crystal shape from the surface energies "
                "of a material's facets, and record which facets survive on it "
                "and in what proportion.",
    requires={"obs": ["surface_energy_{level}", "parent", "miller"]},
    produces={"facets.obs": ["wulff_area_fraction_{level}"],
              "bulk.obs": ["wulff_effective_radius_{level}",
                           "wulff_shape_factor_{level}"]},
    prerequisites=["mv.surf.surface_energy"],
    examples=["mv.surf.wulff(facets, bulk=md, level='emt')"],
    related=["mv.surf.surface_energy", "mv.surf.slabs"],
    notes="A facet with a high surface energy does not appear on the "
          "equilibrium shape at all, which is why an area fraction of zero is "
          "an answer rather than a failure — that plane is not expressed.\n\n"
          "Deposits the per-facet area fractions onto the facet rows and the "
          "shape summary onto the bulk object, because the shape belongs to the "
          "material and the fractions belong to its facets.",
)
def wulff(facets: AnnData, bulk: AnnData, level: str = "emt",
          symprec: float = 0.1) -> None:
    """Equilibrium crystal shape from per-facet surface energies."""
    from pymatgen.analysis.wulff import WulffShape
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    key = f"surface_energy_{level}"
    if key not in facets.obs:
        raise ValueError(f"obs[{key!r}] absent; run mv.surf.surface_energy "
                         f"first")

    parents = facets.obs["parent"].astype(str).to_numpy()
    millers = facets.obs["miller"].astype(str).to_numpy()
    gamma = facets.obs[key].to_numpy(dtype=float)

    fractions = np.full(facets.n_obs, np.nan)
    radius = np.full(bulk.n_obs, np.nan)
    shape_factor = np.full(bulk.n_obs, np.nan)
    summary: dict = {}

    bulk_structures = structures(bulk, "input")
    for i, name in enumerate(map(str, bulk.obs_names)):
        rows = np.where((parents == name) & np.isfinite(gamma))[0]
        if not len(rows):
            continue
        # Lowest energy termination wins for each facet.
        best: dict[tuple, tuple[float, int]] = {}
        for row in rows:
            index = tuple(int(c) for c in millers[row].split("_"))
            if index not in best or gamma[row] < best[index][0]:
                best[index] = (float(gamma[row]), int(row))
        try:
            lattice = SpacegroupAnalyzer(
                bulk_structures[i], symprec=symprec
            ).get_conventional_standard_structure().lattice
            shape = WulffShape(lattice, list(best), [v[0] for v in best.values()])
            areas = shape.area_fraction_dict
        except Exception:
            continue

        for index, (_, row) in best.items():
            fractions[row] = float(areas.get(index, 0.0))
        radius[i] = float(shape.effective_radius)
        shape_factor[i] = float(shape.shape_factor)
        summary[name] = {
            "n_facets": len(best),
            "expressed": [_miller_label(k) for k, v in areas.items()
                          if v > 1e-6],
            "anisotropy": float(shape.anisotropy),
        }

    facets.obs[f"wulff_area_fraction_{level}"] = fractions
    bulk.obs[f"wulff_effective_radius_{level}"] = radius
    bulk.obs[f"wulff_shape_factor_{level}"] = shape_factor
    bulk.uns.setdefault("wulff", {})[level] = summary
    record(bulk, "surf.wulff", level=level, n_shapes=len(summary))
    record(facets, "surf.wulff", level=level)


@register_function(
    aliases=["adsorption sites", "adsorbate", "adsorption", "binding sites",
             "place adsorbate", "surface sites"],
    category="surf",
    description="Enumerate distinct adsorption sites on every slab and return "
                "the adsorbate-decorated structures as a new dataset, one row "
                "per site.",
    requires={"structures": ["{source}"]},
    examples=["configs = mv.surf.adsorption_sites(facets, 'H')",
              "configs = mv.surf.adsorption_sites(facets, 'O', height=1.8)"],
    related=["mv.surf.adsorption_energy", "mv.surf.slabs"],
    notes="Enumerates the symmetry-distinct atop, bridge and hollow sites "
          "rather than one guess, because which site binds most strongly is "
          "usually not the one that looks obvious and is the entire question in "
          "a catalysis screen. That is the AdsorbML finding: place several, "
          "relax all, take the minimum.",
)
def adsorption_sites(facets: AnnData, adsorbate: str, source: str = "input",
                     height: float = 2.0, min_distance: float = 1.5,
                     max_sites: int | None = None) -> AnnData:
    """Adsorbate-decorated slabs, one row per distinct site."""
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.core import Molecule

    from .data import from_structures

    molecule = Molecule([adsorbate], [[0.0, 0.0, 0.0]])
    built, rows, failed = [], [], []

    for name, slab in zip(facets.obs_names, structures(facets, source)):
        try:
            finder = AdsorbateSiteFinder(slab)
            sites = finder.find_adsorption_sites(distance=height)
            configurations = finder.generate_adsorption_structures(
                molecule, find_args={"distance": height})
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        kinds = [k for k in ("ontop", "bridge", "hollow") for _ in
                 sites.get(k, [])]
        for j, configuration in enumerate(configurations):
            if max_sites is not None and j >= max_sites:
                break
            built.append(configuration)
            rows.append({
                "parent": str(name),
                "adsorbate": adsorbate,
                "site_index": j,
                "site_kind": kinds[j] if j < len(kinds) else "",
                "facet": str(facets.obs["miller"].iloc[
                    list(facets.obs_names).index(str(name))])
                if "miller" in facets.obs else "",
            })

    if not built:
        raise ValueError(f"no adsorption configuration was generated; "
                         f"{len(failed)} slabs failed: {failed[:3]}")

    configs = from_structures(built, pd.DataFrame(rows))
    configs.uns["adsorption"] = {"adsorbate": adsorbate, "height": height,
                                 "n_slabs": int(facets.n_obs),
                                 "errors": failed}
    record(configs, "surf.adsorption_sites", adsorbate=adsorbate,
           source=source, n_slabs=int(facets.n_obs))
    return configs


@register_function(
    aliases=["adsorption energy", "binding energy", "chemisorption",
             "adsorbate binding"],
    category="surf",
    description="Compute the adsorption energy of every configuration against "
                "its clean slab and a reference for the adsorbate, and record "
                "the strongest binding site per slab.",
    requires={"obs": ["energy_{level}", "parent"]},
    produces={"obs": ["adsorption_energy_{level}", "is_best_site_{level}"]},
    prerequisites=["mv.calc.relax", "mv.surf.adsorption_sites"],
    examples=["mv.surf.adsorption_energy(configs, clean=facets, "
              "reference=-3.2, level='emt')"],
    related=["mv.surf.adsorption_sites"],
    notes="The adsorbate reference is an argument with no default, because "
          "there is no neutral choice: a hydrogen adsorption energy against "
          "half an H2 molecule and one against an isolated H atom differ by "
          "about 2.3 eV, and both conventions are in use. Stating it is the "
          "only way the number means anything.\n\n"
          "is_best_site marks the strongest binding configuration per slab, "
          "which is the AdsorbML protocol: enumerate many, relax all, keep the "
          "minimum.",
)
def adsorption_energy(configs: AnnData, clean: AnnData, reference: float,
                      level: str = "emt") -> None:
    """Adsorption energy in eV against the clean slab and a stated reference."""
    key = f"energy_{level}"
    for obj, label in ((configs, "configurations"), (clean, "clean slabs")):
        if key not in obj.obs:
            raise ValueError(f"obs[{key!r}] absent on the {label}; run "
                             f"mv.calc.relax(..., level={level!r}) on both")
    if "parent" not in configs.obs:
        raise ValueError("obs['parent'] absent; these configurations did not "
                         "come from mv.surf.adsorption_sites")

    slab_energy = dict(zip(map(str, clean.obs_names),
                           clean.obs[key].to_numpy(dtype=float)))
    total = configs.obs[key].to_numpy(dtype=float)
    parents = configs.obs["parent"].astype(str).to_numpy()
    bare = np.array([slab_energy.get(p, np.nan) for p in parents])

    with np.errstate(invalid="ignore"):
        binding = total - bare - reference

    configs.obs[f"adsorption_energy_{level}"] = binding

    best = np.zeros(configs.n_obs, dtype=bool)
    for parent in set(parents.tolist()):
        rows = np.where((parents == parent) & np.isfinite(binding))[0]
        if len(rows):
            best[rows[int(np.argmin(binding[rows]))]] = True
    configs.obs[f"is_best_site_{level}"] = best

    configs.uns.setdefault("adsorption_energy", {})[level] = {
        "reference": float(reference),
        "n_unmatched": int(np.isnan(bare).sum()),
        "convention": "E_ads = E(slab+adsorbate) - E(slab) - reference",
    }
    record(configs, "surf.adsorption_energy", level=level, reference=reference)


__all__ = ["slabs", "surface_energy", "wulff", "adsorption_sites",
           "adsorption_energy"]


@register_function(
    aliases=["scaling relation", "brønsted-evans-polanyi", "BEP", "linear "
             "scaling", "adsorbate scaling", "descriptor for catalysis"],
    category="surf",
    description="Fit the linear relation between two adsorbates' binding "
                "energies across a set of surfaces — the reason one descriptor "
                "can stand for a whole reaction.",
    requires={"obs": ["{x}", "{y}"]},
    produces={"obs": ["scaling_residual"], "uns": ["scaling"]},
    prerequisites=["mv.surf.adsorption_energy"],
    examples=["mv.surf.scaling(md, x='adsorption_energy_O_emt', "
              "y='adsorption_energy_OH_emt')",
              "mv.surf.scaling(md, x='E_O', y='E_OOH', group='facet')"],
    related=["mv.surf.adsorption_energy", "mv.surf.volcano",
             "mv.screen.rank", "mv.pl.parity"],
    notes="Adsorbates that bind through the same atom bind in proportion: on "
          "a metal that holds oxygen tightly, hydroxyl is held tightly too, "
          "and the ratio is set by how many bonds each makes rather than by "
          "which metal it is. That is why a screen over a whole reaction can "
          "be run on one number, and it is also why the reaction has a "
          "ceiling — the intermediate you want to bind weakly is tied to the "
          "one you want to bind strongly.\\n\\n"
          "The slope is the physical claim. A species bonding through one "
          "atom with n remaining valences scales against a reference species "
          "with m as roughly n/m: OH against O is about 1/2, OOH against O "
          "about 1/2 as well, and a fitted slope far from a small rational "
          "number usually means the two species are not binding the way the "
          "argument assumes.\\n\\n"
          "obs['scaling_residual'] is where a surface departs from the line, "
          "and it is the interesting column rather than the fit. A catalyst "
          "that beats the scaling ceiling has to break the relation, so a "
          "large residual is a candidate and a small one is a confirmation "
          "that this surface offers nothing new.\\n\\n"
          "Fitted by least squares on whatever rows carry both columns; rows "
          "missing either are excluded and counted rather than imputed.",
)
def scaling(md: AnnData, x: str, y: str, group: str | None = None) -> None:
    """Linear scaling between two adsorbates. Deposits; returns ``None``."""
    for column in (x, y):
        if column not in md.obs:
            raise ValueError(
                f"obs[{column!r}] absent; run mv.surf.adsorption_energy for "
                f"each adsorbate first, or point x=/y= at the columns that "
                f"hold them — this object has "
                f"{[c for c in md.obs.columns if 'adsorption' in c][:6]}")
    if group is not None and group not in md.obs:
        raise ValueError(f"obs[{group!r}] absent")

    left = pd.to_numeric(md.obs[x], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(md.obs[y], errors="coerce").to_numpy(dtype=float)
    labels = (md.obs[group].astype(str).to_numpy() if group is not None
              else np.full(md.n_obs, "all", dtype=object))

    residual = np.full(md.n_obs, np.nan)
    fits, skipped = {}, 0

    for label in dict.fromkeys(labels):
        mask = (labels == label) & np.isfinite(left) & np.isfinite(right)
        if mask.sum() < 3:
            skipped += int(mask.sum())
            continue
        slope, intercept = np.polyfit(left[mask], right[mask], 1)
        predicted = slope * left[mask] + intercept
        residual[mask] = right[mask] - predicted
        spread = right[mask] - right[mask].mean()
        total = float((spread ** 2).sum())
        fits[str(label)] = {
            "slope": float(slope),
            "intercept_eV": float(intercept),
            "rmse_eV": float(np.sqrt(np.mean((right[mask] - predicted) ** 2))),
            "r_squared": float(1.0 - ((right[mask] - predicted) ** 2).sum()
                               / total) if total > 0 else float("nan"),
            "n_points": int(mask.sum()),
        }

    md.obs["scaling_residual"] = residual
    md.uns["scaling"] = {
        "x": str(x), "y": str(y), "group": group,
        "fits": fits,
        "n_excluded": int(np.isnan(residual).sum()),
        "note": "least squares on rows carrying both columns; the residual is "
                "the departure from the line and is the column worth reading",
    }
    if not fits:
        raise ValueError(
            f"no group had three points with both {x!r} and {y!r}; a line "
            f"through two points is not a scaling relation")
    if skipped:
        warnings.warn(
            f"{skipped} rows are in groups too small to fit and have no "
            f"residual; a scaling relation needs at least three surfaces",
            RuntimeWarning, stacklevel=2)
    record(md, "surf.scaling", x=x, y=y, group=group)


@register_function(
    aliases=["volcano", "volcano plot", "sabatier", "activity descriptor",
             "catalytic activity", "optimal binding"],
    category="surf",
    description="Sabatier activity against a binding-energy descriptor — the "
                "volcano — with the optimum and each surface's distance from "
                "it.",
    requires={"obs": ["{descriptor}"]},
    produces={"obs": ["volcano_activity", "distance_from_optimum"],
              "uns": ["volcano"]},
    prerequisites=["mv.surf.adsorption_energy"],
    examples=["mv.surf.volcano(md, descriptor='adsorption_energy_O_emt', "
              "optimum=-1.6)",
              "mv.surf.volcano(md, descriptor='E_O', optimum=-1.6, "
              "slopes=(1.0, -1.0))"],
    related=["mv.surf.scaling", "mv.surf.adsorption_energy",
             "mv.screen.rank", "mv.pl.scatter"],
    notes="Sabatier's argument in one column: bind the intermediate too "
          "weakly and it never forms, too strongly and it never leaves, so "
          "activity peaks somewhere in between. The two limits are lines with "
          "opposite slopes in the binding energy, and the activity is "
          "whichever is smaller — a minimum of two straight lines, which is "
          "why the plot is a peak with straight flanks rather than a "
          "curve.\\n\\n"
          "**The optimum is an input, not a result.** Where the peak sits "
          "depends on the reaction, the potential and the conditions, and it "
          "comes from a microkinetic model or from experiment. Passing one "
          "and reading the ranking is legitimate; treating the ranking as a "
          "prediction of turnover is not, and the difference is that "
          "everything here is a *relative* ordering of surfaces against an "
          "optimum somebody else established.\\n\\n"
          "The activity is in arbitrary units and only its ordering means "
          "anything. obs['distance_from_optimum'] is the honest column: it is "
          "in electronvolts, it says which side the surface falls on through "
          "its sign, and it does not pretend to be a rate.",
)
def volcano(md: AnnData, descriptor: str, optimum: float,
            slopes: tuple = (1.0, -1.0)) -> None:
    """Sabatier activity against a descriptor. Deposits; returns ``None``."""
    if descriptor not in md.obs:
        raise ValueError(
            f"obs[{descriptor!r}] absent; this object has "
            f"{[c for c in md.obs.columns if 'adsorption' in c][:6]}")
    if len(slopes) != 2 or slopes[0] <= 0 or slopes[1] >= 0:
        raise ValueError(
            f"slopes={slopes} must be one positive and one negative — the "
            f"two limbs of a volcano have opposite signs, and giving them the "
            f"same sign produces a straight line dressed as a peak")

    values = pd.to_numeric(md.obs[descriptor], errors="coerce").to_numpy(
        dtype=float)
    offset = values - float(optimum)
    weak, strong = float(slopes[0]), float(slopes[1])
    activity = np.minimum(weak * offset, strong * offset)

    md.obs["volcano_activity"] = activity
    md.obs["distance_from_optimum"] = offset
    md.uns["volcano"] = {
        "descriptor": str(descriptor),
        "optimum": float(optimum),
        "slopes": [weak, strong],
        "optimum_source": "supplied by the caller; not derived here",
        "n_missing": int(np.isnan(values).sum()),
        "note": "activity is in arbitrary units and only its ordering is "
                "meaningful; distance_from_optimum is in eV and signed",
    }
    record(md, "surf.volcano", descriptor=descriptor, optimum=optimum)

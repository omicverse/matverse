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
    produces={"obs": ["surface_energy_{level}"]},
    prerequisites=["mv.calc.relax", "mv.surf.slabs"],
    examples=["mv.surf.surface_energy(facets, bulk=md, level='emt')"],
    related=["mv.surf.slabs", "mv.surf.wulff"],
    notes="Takes the bulk dataset as an argument rather than guessing. The "
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
    counts = np.array([len(s) for s in structures(facets, "input")],
                      dtype=float)

    per_atom = np.array([reference.get(p, np.nan) for p in parents])
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = ((slab_energy - counts * per_atom) / (2.0 * area)
                 * _EV_PER_A2_TO_J_PER_M2)

    name = key_added or f"surface_energy_{level}"
    facets.obs[name] = gamma
    facets.uns.setdefault("surface_energy", {})[level] = {
        "unit": "J/m^2", "reference": bulk_key,
        "n_unmatched": int(np.isnan(per_atom).sum()),
    }
    record(facets, "surf.surface_energy", level=level, key_added=name)


@register_function(
    aliases=["wulff", "wulff construction", "equilibrium shape",
             "crystal shape", "nanoparticle shape"],
    category="surf",
    description="Build the equilibrium crystal shape from the surface energies "
                "of a material's facets, and record which facets survive on it "
                "and in what proportion.",
    requires={"obs": ["surface_energy_{level}", "parent", "miller"]},
    produces={"obs": ["wulff_area_fraction_{level}",
                      "wulff_effective_radius_{level}",
                      "wulff_shape_factor_{level}"],
              "uns": ["wulff"]},
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

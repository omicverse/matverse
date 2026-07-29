"""``mv.env`` — local and coordination environments.

Composition says what a material is made of and a space group says how the cell
repeats. Neither says what an atom's *neighbourhood* looks like, and a great
deal of solid-state chemistry is exactly that: octahedral or tetrahedral, corner-
or edge-sharing, four-fold or six-fold. Two polymorphs with the same composition
and the same space group can differ entirely in coordination.

This namespace wraps the pymatgen machinery for that question — ``local_env``'s
near-neighbour algorithms, ``chemenv``'s coordination-environment classifier, and
``StructureGraph``'s bond network — and puts each result where its shape belongs:

===============================  =====================================
per atom                          the **sites** object (``mv.multi``)
per material                      ``obs`` on the materials object
the bond network                  ``obsp`` on the sites object
===============================  =====================================

```python
sites = mv.multi.sites(md)
mv.env.coordination(md, sites)              # -> sites.obs['coordination_number']
mv.env.chemenv(md, sites)                   # -> sites.obs['environment']
mv.env.bonds(md, sites)                     # -> sites.obsp['bonds']
mv.env.summarise(sites, md)                 # -> obs['mean_coordination'], ...
```

Coordination number is the one descriptor in matverse that a composition vector
provably cannot produce, which is the whole reason the namespace exists: `X` is
the same for two polymorphs and their coordination is not.
"""

from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function

#: Near-neighbour strategies, and what each is good for.
STRATEGIES = {
    "crystalnn": "CrystalNN — Voronoi with a solid-angle cutoff tuned on the "
                 "Materials Project; the general-purpose default",
    "voronoi": "VoronoiNN — raw Voronoi solid angles, no chemistry-aware "
               "weighting",
    "minimum_distance": "MinimumDistanceNN — everything within a tolerance of "
                        "the shortest bond; fast and crude",
    "econ": "EconNN — the effective coordination number of Hoppe (1979), "
            "which weights neighbours continuously rather than counting them",
    "brunner": "BrunnerNN_real — the largest gap in the distance histogram",
}


def _strategy(name: str):
    """Build a pymatgen NearNeighbors object from a short name."""
    from pymatgen.analysis import local_env

    key = str(name).strip().lower()
    builders = {
        "crystalnn": lambda: local_env.CrystalNN(),
        "voronoi": lambda: local_env.VoronoiNN(),
        "minimum_distance": lambda: local_env.MinimumDistanceNN(),
        "econ": lambda: local_env.EconNN(),
        "brunner": lambda: local_env.BrunnerNN_real(),
    }
    if key not in builders:
        raise ValueError(f"unknown strategy {name!r}; known: "
                         f"{sorted(builders)}. Each is described in "
                         f"mv.env.STRATEGIES.")
    return builders[key]()


def _require_sites(sites: AnnData, md: AnnData) -> None:
    """A sites object built from this parent, or a clear refusal."""
    from .multi import AXIS_KEY

    if sites.uns.get(AXIS_KEY) != "sites":
        raise ValueError(
            "this is not a sites object; build one with mv.multi.sites(md). "
            "Coordination is one number per atom, so it needs the axis whose "
            "rows are atoms.")
    counts = sites.obs["material"].value_counts()
    expected = {str(name) for name in md.obs_names}
    if not set(counts.index).issubset(expected):
        raise ValueError(
            "the sites object was built from a different dataset; rebuild it "
            "with mv.multi.sites(md) so the two axes line up.")


@register_function(
    aliases=["coordination number", "CN", "near neighbours", "local "
             "environment", "how many neighbours", "coordination"],
    category="env",
    description="Coordination number and near-neighbour distances for every "
                "atom, by any of pymatgen's near-neighbour strategies.",
    requires={"structures": ["{source}"]},
    produces={"sites.obs": ["coordination_number", "mean_neighbour_distance",
                            "coordination_strategy"]},
    prerequisites=["mv.multi.sites"],
    examples=["mv.env.coordination(md, sites)",
              "mv.env.coordination(md, sites, strategy='voronoi')"],
    related=["mv.env.chemenv", "mv.env.bonds", "mv.multi.sites"],
    notes="Deposits on the **sites** object, because a coordination number is "
          "one value per atom and the material axis has no room for a ragged "
          "column.\n\n"
          "Which strategy produced the number is recorded next to it. "
          "Near-neighbour algorithms disagree — CrystalNN and MinimumDistanceNN "
          "routinely differ by one or two on the same site — so a coordination "
          "number without its strategy is not reproducible.",
)
def coordination(md: AnnData, sites: AnnData, strategy: str = "crystalnn",
                 source: str = "input") -> None:
    """Per-atom coordination number. Deposits on ``sites``; returns ``None``."""
    _require_sites(sites, md)
    finder = _strategy(strategy)

    numbers = np.full(sites.n_obs, np.nan)
    distances = np.full(sites.n_obs, np.nan)
    failures = 0

    row = 0
    for structure in structures(md, source):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in range(len(structure)):
                try:
                    neighbours = finder.get_nn_info(structure, index)
                    numbers[row] = len(neighbours)
                    if neighbours:
                        distances[row] = float(np.mean([
                            structure[index].distance(n["site"])
                            for n in neighbours]))
                except Exception:
                    failures += 1
                row += 1

    sites.obs["coordination_number"] = numbers
    sites.obs["mean_neighbour_distance"] = distances
    sites.obs["coordination_strategy"] = str(strategy)
    sites.uns["coordination"] = {"strategy": str(strategy),
                                 "description": STRATEGIES[strategy.lower()],
                                 "source": source, "n_failed": failures}
    record(sites, "env.coordination", strategy=strategy, source=source)


@register_function(
    aliases=["chemenv", "coordination environment", "octahedral or "
             "tetrahedral", "environment symbol", "site geometry"],
    category="env",
    description="Classify each atom's coordination polyhedron — octahedral, "
                "tetrahedral, square planar and the rest — using pymatgen's "
                "ChemEnv, with the continuous symmetry measure that says how "
                "good the match is.",
    requires={"structures": ["{source}"]},
    produces={"sites.obs": ["environment", "environment_csm",
                            "environment_coordination"]},
    prerequisites=["mv.multi.sites"],
    examples=["mv.env.chemenv(md, sites)",
              "mv.env.chemenv(md, sites, max_csm=6.0)"],
    related=["mv.env.coordination", "mv.multi.sites"],
    notes="A coordination number of 6 does not mean octahedral — it can be "
          "trigonal prismatic, or a badly distorted octahedron, and those are "
          "different materials. ChemEnv fits model polyhedra and reports the "
          "**continuous symmetry measure**: 0 is perfect, and anything above "
          "roughly 2.5 is a poor match that should be read as 'distorted' "
          "rather than as the name it was given.\n\n"
          "This is the slowest function in matverse per atom. It fits every "
          "model polyhedron to every site, so budget seconds per structure, "
          "not milliseconds.",
)
def chemenv(md: AnnData, sites: AnnData, source: str = "input",
            max_csm: float = 8.0, distance_cutoff: float = 1.4,
            angle_cutoff: float = 0.3) -> None:
    """Coordination environment per atom. Deposits on ``sites``."""
    _require_sites(sites, md)

    from pymatgen.analysis.chemenv.coordination_environments.\
        chemenv_strategies import SimplestChemenvStrategy
    from pymatgen.analysis.chemenv.coordination_environments.\
        coordination_geometry_finder import LocalGeometryFinder
    from pymatgen.analysis.chemenv.coordination_environments.\
        structure_environments import LightStructureEnvironments

    finder = LocalGeometryFinder()
    strategy = SimplestChemenvStrategy(distance_cutoff=distance_cutoff,
                                       angle_cutoff=angle_cutoff)

    symbols = np.full(sites.n_obs, "", dtype=object)
    measures = np.full(sites.n_obs, np.nan)
    counts = np.full(sites.n_obs, np.nan)
    failures = []

    row = 0
    for i, structure in enumerate(structures(md, source)):
        n = len(structure)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                finder.setup_structure(structure=structure)
                environments = finder.compute_structure_environments(
                    maximum_distance_factor=distance_cutoff + 0.2,
                    only_cations=False)
                light = LightStructureEnvironments.\
                    from_structure_environments(
                        strategy=strategy, structure_environments=environments)
            for j in range(n):
                found = light.coordination_environments[j]
                if found:
                    best = found[0]
                    csm = float(best.get("csm", np.nan))
                    if np.isnan(csm) or csm <= max_csm:
                        symbols[row + j] = str(best.get("ce_symbol", ""))
                        measures[row + j] = csm
                        counts[row + j] = float(
                            str(best.get("ce_symbol", ":0")).split(":")[-1])
        except Exception as exc:
            failures.append(f"{i}: {type(exc).__name__}: {exc}")
        row += n

    sites.obs["environment"] = symbols
    sites.obs["environment_csm"] = measures
    sites.obs["environment_coordination"] = counts
    sites.uns["chemenv"] = {
        "max_csm": float(max_csm), "distance_cutoff": float(distance_cutoff),
        "angle_cutoff": float(angle_cutoff), "source": source,
        "n_failed": len(failures), "errors": failures[:10],
        "note": "ce_symbol is pymatgen's IUPAC-style code: O:6 octahedral, "
                "T:4 tetrahedral, S:4 square planar, CU:8 cubic. csm is the "
                "continuous symmetry measure; 0 is perfect.",
    }
    record(sites, "env.chemenv", source=source, max_csm=max_csm)


@register_function(
    aliases=["bond network", "structure graph", "connectivity", "bonds",
             "adjacency", "who is bonded to whom"],
    category="env",
    description="The bond network as a sparse adjacency matrix on the sites "
                "axis, so connectivity is a matrix rather than a nested list.",
    requires={"structures": ["{source}"]},
    produces={"sites.obsp": ["bonds", "bond_distances"]},
    prerequisites=["mv.multi.sites"],
    examples=["mv.env.bonds(md, sites)",
              "mv.env.bonds(md, sites, strategy='minimum_distance')"],
    related=["mv.env.coordination", "mv.tl.neighbors"],
    notes="Lands in ``obsp`` — the atoms x atoms slot — which is exactly what "
          "AnnData has for a graph over its own rows. The same slot holds a "
          "kNN graph in single-cell analysis, and the machinery written for "
          "that works here without modification.\n\n"
          "Bonds are within a material only: two atoms in different structures "
          "are never connected, so the matrix is block-diagonal by "
          "construction.",
)
def bonds(md: AnnData, sites: AnnData, strategy: str = "crystalnn",
          source: str = "input") -> None:
    """Bond network into ``sites.obsp``. Returns ``None``."""
    from scipy import sparse

    _require_sites(sites, md)
    finder = _strategy(strategy)

    rows, cols, lengths = [], [], []
    offset = 0
    failures = 0
    for structure in structures(md, source):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in range(len(structure)):
                try:
                    for neighbour in finder.get_nn_info(structure, index):
                        j = int(neighbour["site_index"])
                        rows.append(offset + index)
                        cols.append(offset + j)
                        lengths.append(float(
                            structure[index].distance(neighbour["site"])))
                except Exception:
                    failures += 1
        offset += len(structure)

    shape = (sites.n_obs, sites.n_obs)
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=shape).tocsr()
    distances = sparse.coo_matrix(
        (np.asarray(lengths, dtype=float), (rows, cols)), shape=shape).tocsr()

    sites.obsp["bonds"] = adjacency
    sites.obsp["bond_distances"] = distances
    sites.uns["bonds"] = {"strategy": str(strategy), "source": source,
                          "n_edges": int(adjacency.nnz), "n_failed": failures}
    record(sites, "env.bonds", strategy=strategy, source=source)


@register_function(
    aliases=["summarise environments", "coordination summary", "aggregate "
             "coordination", "per material coordination"],
    category="env",
    description="Summarise per-atom environments back onto the material axis, "
                "so a screen can filter on coordination.",
    requires={"sites.obs": ["coordination_number"]},
    produces={"md.obs": ["mean_coordination", "min_coordination",
                         "max_coordination", "coordination_spread"]},
    prerequisites=["mv.env.coordination"],
    examples=["mv.env.summarise(sites, md)"],
    related=["mv.env.coordination", "mv.multi.aggregate", "mv.screen.filter"],
    notes="The same move as mv.multi.aggregate, with the statistics a "
          "coordination question actually asks. ``coordination_spread`` is the "
          "one worth screening on: a spread of zero means every atom sits in "
          "the same environment, which is what a high-symmetry structure looks "
          "like from the inside.",
)
def summarise(sites: AnnData, md: AnnData) -> None:
    """Per-material coordination statistics. Deposits on ``md``."""
    if "coordination_number" not in sites.obs:
        raise ValueError("sites.obs['coordination_number'] absent; run "
                         "mv.env.coordination(md, sites) first")

    values = sites.obs["coordination_number"].to_numpy(dtype=float)
    material = sites.obs["material"].astype(str).to_numpy()

    means = np.full(md.n_obs, np.nan)
    lows = np.full(md.n_obs, np.nan)
    highs = np.full(md.n_obs, np.nan)
    spreads = np.full(md.n_obs, np.nan)

    for i, name in enumerate(map(str, md.obs_names)):
        block = values[material == name]
        block = block[np.isfinite(block)]
        if block.size:
            means[i] = block.mean()
            lows[i] = block.min()
            highs[i] = block.max()
            spreads[i] = block.max() - block.min()

    md.obs["mean_coordination"] = means
    md.obs["min_coordination"] = lows
    md.obs["max_coordination"] = highs
    md.obs["coordination_spread"] = spreads
    record(md, "env.summarise")


__all__ = ["STRATEGIES", "coordination", "chemenv", "bonds", "summarise"]

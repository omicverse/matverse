"""matverse — a materials analysis ecosystem on the AnnData substrate.

    import matverse as mv

    md = mv.data.from_cif('candidates/')   # or from_mp / from_ase / from_matminer
    mv.pp.standardize(md)                  # primitive + conventional + symmetry
    mv.pp.qc(md)                           # -> obs['is_valid'], obs['min_distance']
    md = mv.pp.filter_materials(md)        # drop the broken ones
    mv.feat.element_stats(md)              # -> obsm['X_element_stats']
    mv.calc.relax(md, level='emt')         # -> uns['structures']['relaxed_emt']
    mv.thermo.hull(md, level='emt')        # -> obs['e_above_hull_emt']
    mv.screen.filter(md, e_above_hull_emt__lt=0.05)
    mv.tl.rank_elements_groups(md, 'passes')   # what chemistry passed, and why

``md`` is an ordinary ``AnnData`` throughout: subsettable, writable to h5ad,
readable without matverse installed.

Namespaces
----------
``mv.data``    build a dataset — CIF, Materials Project, matminer, ASE, pymatgen
``mv.pp``      standardisation, symmetry, quality control, filtering, dedup
``mv.feat``    descriptors into ``obsm``
``mv.tl``      analysis on the composition matrix — ordination, clustering,
               and which elements distinguish one group of materials from another
``mv.calc``    energies and relaxation, tagged by level of theory
``mv.prop``    derived properties — elastic moduli, phonons, and curves
``mv.thermo``  stability: convex hull, reactions, chemical potentials
``mv.dft``     first-principles inputs out, results back in
``mv.multi``   the sites axis — one row per atom
``mv.exp``     measured data, on the same footing as computed data
``mv.gen``     scoring generated candidates
``mv.model``   property prediction with splits that do not leak
``mv.opt``     design campaigns
``mv.pl``      plotting, including the periodic-table heatmap
``mv.utils``   units, checkpointing, cluster submission
``mv.screen``  filtering, ranking and Pareto fronts that leave a record

Three conventions carry the design
----------------------------------
Operations **deposit** rather than return, so a pipeline is reproducible from
the object alone.

A computed quantity carries its **level of theory** in the slot name —
``obs['energy_emt']`` versus ``obs['energy_pbe']``, with what produced each in
``uns['levels'][level]`` — so mixing a surrogate potential with DFT is a visible
mistake rather than a silent one.

``X`` is the **composition matrix** and ``var`` is the **periodic table**.
Materials by elements is the same shape as cells by genes: sparse, non-negative,
and mostly zero. That correspondence is what lets ordination, clustering and
differential enrichment apply to chemical space without being rewritten.

Every public function carries a registry entry describing what it consumes and
creates::

    mv.describe('convex hull')          # find the function from the intent
    mv.registry.find('stability')       # rank candidates
"""

from __future__ import annotations

from . import (calc, data, datasets, dft, disorder, elec,  # noqa: F401
               elements, env, exp, feat, gen, iface, mag, md, mol, model, multi,
               neb, opt, pl, pp, prop, screen, struct, surf, thermo, tl,
               transform, utils)
from ._core import (check_commercial_use, compare_levels, grid_of,  # noqa: F401
                    level_info, levels_used, new, provenance, records,
                    set_level, structures, variants)
from ._registry import get_registry

__version__ = "0.1.20"

#: The process-global function registry.
registry = get_registry()


def describe(name: str) -> str:
    """What a function does, what it needs, and what it will write.

    >>> print(mv.describe('convex hull'))
    mv.thermo.hull(md, level='emt', source='input', references=None, ...)
    ...
    """
    return registry.describe(name)


def find(query: str, limit: int = 5) -> list:
    """Rank functions against a free-text intent, best first."""
    return [e["public_name"] for e in registry.find(query, limit=limit)]


__all__ = [
    "data", "pp", "feat", "tl", "calc", "prop", "thermo", "screen", "exp",
    "multi", "gen", "pl", "model", "opt", "utils", "dft",
    "md", "neb", "surf", "mag", "datasets",
    "disorder", "elec", "elements", "env", "iface", "struct",
    "mol", "transform",
    "new", "structures", "variants", "provenance", "compare_levels",
    "levels_used", "level_info", "set_level", "records",
    "check_commercial_use",
    "grid_of",
    "registry", "describe", "find", "__version__",
]

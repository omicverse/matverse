"""matverse — a materials analysis ecosystem on the AnnData substrate.

    import matverse as mv

    md = mv.data.from_matminer(df)        # or from_mp / from_ase / from_structures
    mv.struct.standardize(md)             # primitive + conventional + symmetry
    mv.feat.composition(md)               # -> obsm['X_composition']
    mv.calc.relax(md, level='emt')        # -> uns['structures']['relaxed_emt'],
                                          #    obs['energy_emt']
    mv.thermo.hull(md, level='emt')       # -> obs['e_above_hull_emt']
    mv.screen.filter(md, e_above_hull_emt__lt=0.05)

``md`` is an ordinary ``AnnData`` throughout: subsettable, writable to h5ad,
readable without matverse installed.

Namespaces
----------
``mv.data``    build a dataset from Materials Project, matminer, ASE, or
               bare pymatgen structures
``mv.struct``  structure operations — standardisation, symmetry, supercells
``mv.feat``    descriptors into ``obsm``
``mv.calc``    energies and relaxation, tagged by level of theory
``mv.thermo``  stability: convex hull, energy above hull
``mv.screen``  high-throughput filtering and ranking that leaves a record

Two conventions carry the design. Operations **deposit** rather than return, so
a pipeline is reproducible from the object alone. And a computed quantity
carries its **level of theory in the slot name** — ``obs['energy_emt']`` versus
``obs['energy_pbe']``, settings in ``uns['calc'][level]`` — so mixing a
surrogate potential with DFT is a visible mistake rather than a silent one.
"""

from . import calc, data, feat, screen, struct, thermo  # noqa: F401
from ._core import new, structures  # noqa: F401

__version__ = "0.1.0"
__all__ = ["data", "struct", "feat", "calc", "thermo", "screen", "new", "structures"]

"""``mv.struct`` — structure operations.

Every operation deposits a named structure variant. ``pymatgen`` returns new
objects and leaves the input untouched, which means the caller decides where
each variant lives and everything downstream must know which variable that
was. Naming them keeps the variants with the data.
"""

from __future__ import annotations

from anndata import AnnData

from ._core import deposit_structures, record, structures


def standardize(md: AnnData, source: str = "input", symprec: float = 0.01) -> None:
    """Primitive and conventional standard cells, plus symmetry.

    Named for what pymatgen calls it — ``get_primitive_standard_structure`` /
    ``get_conventional_standard_structure`` — rather than 'normalize', which in
    a dataset context reads as rescaling numbers.

    produces: uns['structures']['primitive'], uns['structures']['conventional'],
              obs['spacegroup'], obs['spacegroup_number'], obs['crystal_system'],
              obs['nsites_primitive']
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    S = structures(md, source)
    prim, conv, sg, num, sys_ = [], [], [], [], []
    for s in S:
        sga = SpacegroupAnalyzer(s, symprec=symprec)
        prim.append(sga.get_primitive_standard_structure())
        conv.append(sga.get_conventional_standard_structure())
        sg.append(sga.get_space_group_symbol())
        num.append(int(sga.get_space_group_number()))
        sys_.append(sga.get_crystal_system())
    deposit_structures(md, "primitive", prim)
    deposit_structures(md, "conventional", conv)
    md.obs["spacegroup"] = sg
    md.obs["spacegroup_number"] = num
    md.obs["crystal_system"] = sys_
    md.obs["nsites_primitive"] = [len(s) for s in prim]
    record(md, f"struct.standardize({source}, symprec={symprec})")


def supercell(md: AnnData, scaling, source: str = "input", name: str | None = None) -> None:
    """produces: uns['structures'][name or 'supercell_<scaling>']"""
    S = structures(md, source)
    out = []
    for s in S:
        c = s.copy(); c.make_supercell(scaling); out.append(c)
    key = name or f"supercell_{''.join(map(str, scaling)) if hasattr(scaling,'__iter__') else scaling}"
    deposit_structures(md, key, out)
    record(md, f"struct.supercell({source}, {scaling}) -> {key}")


def describe(md: AnnData, source: str = "input") -> None:
    """produces: obs['formula'], obs['nsites'], obs['volume'], obs['density'],
                 obs['n_elements']"""
    S = structures(md, source)
    md.obs["formula"] = [s.composition.reduced_formula for s in S]
    md.obs["nsites"] = [len(s) for s in S]
    md.obs["volume"] = [float(s.volume) for s in S]
    md.obs["density"] = [float(s.density) for s in S]
    md.obs["n_elements"] = [len(s.composition.elements) for s in S]
    record(md, f"struct.describe({source})")


__all__ = ["standardize", "supercell", "describe"]

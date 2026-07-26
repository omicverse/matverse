"""``mv.data`` — get materials in.

Entry points for the shapes the ecosystem already emits. Each returns an
``AnnData`` following the convention in ``_core``; none of them invents a
format.
"""

from __future__ import annotations

import pandas as pd
from anndata import AnnData

from ._core import new, record


def from_structures(structures: list, obs: pd.DataFrame | None = None) -> AnnData:
    """From a bare list of pymatgen ``Structure`` objects."""
    return new(structures, obs, source="data.from_structures")


def from_matminer(df: pd.DataFrame, structure_col: str = "structure") -> AnnData:
    """Adopt a matminer-style DataFrame.

    matminer's convention is a DataFrame with a ``structure`` column of
    pymatgen objects and featuriser-appended numeric columns — already "rows
    are materials". Numeric columns become a feature block; everything else
    becomes ``obs``. ``to_matminer`` takes it back out, because a one-way
    adapter is an adoption dead end.
    """
    if structure_col not in df.columns:
        raise KeyError(f"no {structure_col!r} column; matminer's convention puts "
                       f"pymatgen Structures there")
    meta = [c for c in df.columns if c != structure_col]
    num = [c for c in meta if pd.api.types.is_numeric_dtype(df[c])]
    obs_cols = [c for c in meta if c not in num]

    md = new(list(df[structure_col]),
             df[obs_cols] if obs_cols else None,
             source="data.from_matminer")
    if num:
        md.obsm["X_matminer"] = df[num].to_numpy(dtype=float)
        md.uns["features"]["X_matminer"] = {"names": num, "featurizer": "matminer"}
    return md


def to_matminer(md: AnnData, variant: str = "input") -> pd.DataFrame:
    """Back to the DataFrame the ecosystem reads."""
    from ._core import structures as _s
    df = md.obs.copy().reset_index(drop=True)
    df.insert(0, "structure", list(_s(md, variant)))
    for block, meta in md.uns.get("features", {}).items():
        arr = md.obsm[block]
        for i, name in enumerate(meta["names"]):
            df[name] = arr[:, i]
    return df


def from_ase(atoms_list: list, obs: pd.DataFrame | None = None) -> AnnData:
    """From ASE ``Atoms``. Per-atom arrays are kept, not silently dropped."""
    from pymatgen.io.ase import AseAtomsAdaptor
    ad = AseAtomsAdaptor()
    md = new([ad.get_structure(a) for a in atoms_list], obs, source="data.from_ase")
    md.uns["sites"] = [{k: v.tolist() for k, v in a.arrays.items()
                        if k not in ("positions", "numbers")} for a in atoms_list]
    return md


def from_mp(criteria: dict, api_key: str | None = None, max_n: int = 200) -> AnnData:
    """Query the Materials Project.

    Kept thin on purpose: ``mp_api`` owns the query language, and duplicating
    it here would mean tracking their schema changes forever.
    """
    import os
    from mp_api.client import MPRester            # optional dependency
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("set MP_API_KEY or pass api_key=")
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(**criteria)[:max_n]
    obs = pd.DataFrame({
        "material_id": [str(d.material_id) for d in docs],
        "formula": [d.formula_pretty for d in docs],
        "energy_above_hull_mp": [d.energy_above_hull for d in docs],
    })
    md = new([d.structure for d in docs], obs, source="data.from_mp")
    md.uns["calc"]["mp"] = {"provider": "Materials Project", "criteria": criteria}
    return md


__all__ = ["from_structures", "from_matminer", "to_matminer", "from_ase", "from_mp"]

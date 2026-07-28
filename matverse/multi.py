"""``mv.multi`` — the sites axis.

Some results are not one number per material. Forces, Bader charges, magnetic
moments and local-environment descriptors are one number *per atom*, and the
number of atoms differs from material to material. v0.1 stored them as a list of
records in ``uns`` and called it the open design problem, which was accurate: a
list of ragged records is honest and unvectorised, and nothing can be computed
from it without unpacking it first.

The answer is a second axis. A **sites object** is an ordinary ``AnnData`` whose
rows are atoms rather than materials, carrying ``obs['material']`` as a foreign
key back to the parent. Per-atom results are then columns on a matrix again, and
a training set for a fitted potential is a subset rather than a script.

```python
sites = mv.multi.sites(md)                    # one row per atom
mv.calc.forces(md, sites, level='emt')        # -> sites.obsm['forces_emt']
mv.multi.aggregate(sites, md, 'force_magnitude_emt', how='max')
```

``X`` on the sites object is the one-hot element indicator, so ``var`` is the
same periodic table the parent carries and element-wise questions work on both
axes — "which elements carry the largest forces" is
``mv.tl.rank_elements_groups`` on the sites object.

Two objects rather than one is deliberate. ``mv.multi.to_mudata`` assembles them
into a single ``MuData`` when you want one file, but nothing in matverse requires
it, and the sites object is useful on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function

#: Set on a sites object so functions can tell the two axes apart.
AXIS_KEY = "matverse_axis"


@register_function(
    aliases=["sites", "sites axis", "per atom", "atom table", "expand to atoms",
             "site level"],
    category="multi",
    description="Build a companion object whose rows are atoms rather than "
                "materials, so per-site results — forces, charges, moments, "
                "local descriptors — become columns of a matrix instead of "
                "ragged records.",
    requires={"structures": ["{source}"]},
    examples=["sites = mv.multi.sites(md)",
              "sites = mv.multi.sites(md, source='relaxed_emt')"],
    related=["mv.calc.forces", "mv.multi.aggregate", "mv.multi.to_mudata"],
    notes="Returns rather than deposits, because the result has a different "
          "number of rows than the object it came from — it cannot be a slot on "
          "it. obs['material'] is the foreign key back.",
)
def sites(md: AnnData, source: str = "input") -> AnnData:
    """One row per atom, with a foreign key back to its material."""
    from .elements import element_frame

    S = structures(md, source)
    material, material_index, site_index = [], [], []
    symbols, frac, cart = [], [], []
    properties: dict[str, list] = {}

    for i, (name, structure) in enumerate(zip(md.obs_names, S)):
        for j, site in enumerate(structure):
            material.append(str(name))
            material_index.append(i)
            site_index.append(j)
            symbols.append(str(site.specie.symbol))
            frac.append(np.asarray(site.frac_coords, dtype=float))
            cart.append(np.asarray(site.coords, dtype=float))
            for key, value in (site.properties or {}).items():
                properties.setdefault(key, []).append(value)

    n = len(material)
    elements = sorted(set(symbols), key=_by_atomic_number)
    index = {el: k for k, el in enumerate(elements)}

    from scipy.sparse import csr_matrix
    X = csr_matrix((np.ones(n, dtype=np.float32),
                    np.asarray([index[s] for s in symbols], dtype=np.int32),
                    np.arange(n + 1, dtype=np.int32)),
                   shape=(n, len(elements)))

    # Name the rows before handing the frame to AnnData rather than after, so
    # the construction does not warn about coercing an integer index. See the
    # same note in _core.new.
    names = pd.Index([f"{m}:{j}" for m, j in zip(material, site_index)],
                     dtype=object)
    obs = pd.DataFrame({
        "material": pd.Categorical(material),
        "material_index": material_index,
        "site_index": site_index,
        "element": pd.Categorical(symbols),
    }, index=names)
    for key, values in properties.items():
        if len(values) == n:
            obs[f"site_{key}"] = values

    out = AnnData(X=X, obs=obs, var=element_frame(elements))
    out.obsm["X_frac"] = np.vstack(frac) if n else np.zeros((0, 3))
    out.obsm["X_cart"] = np.vstack(cart) if n else np.zeros((0, 3))
    out.uns[AXIS_KEY] = "sites"
    out.uns["parent"] = {"source": source, "n_materials": int(md.n_obs)}
    out.uns["provenance"] = []
    record(out, "multi.sites", source=source, n_materials=int(md.n_obs))
    return out


def _by_atomic_number(symbol: str) -> tuple:
    try:
        from pymatgen.core.periodic_table import Element
        return (0, int(Element(symbol).Z))
    except Exception:
        return (1, symbol)


def _check_sites(sites_obj: AnnData) -> None:
    if sites_obj.uns.get(AXIS_KEY) != "sites":
        raise ValueError(
            "expected a sites object from mv.multi.sites; got an object whose "
            f"axis is {sites_obj.uns.get(AXIS_KEY, 'materials')!r}")


@register_function(
    aliases=["aggregate sites", "per material summary", "roll up sites",
             "summarise per atom", "sites to materials"],
    category="multi",
    description="Summarise a per-atom column back onto the material axis, "
                "which is how a per-site result becomes something a screen can "
                "filter on.",
    requires={"obs": ["{column}"]},
    produces={"obs": ["{key_added}"]},
    prerequisites=["mv.multi.sites"],
    examples=["mv.multi.aggregate(sites, md, 'force_magnitude_emt', how='max')"],
    related=["mv.multi.sites", "mv.screen.filter"],
    notes="The bridge between the two axes. Deposits onto the materials object, "
          "so the summary is screenable while the per-atom detail stays where "
          "it can still be inspected.",
)
def aggregate(sites_obj: AnnData, md: AnnData, column: str, how: str = "max",
              key_added: str | None = None) -> None:
    """Reduce a per-atom column onto the material axis."""
    _check_sites(sites_obj)
    if column not in sites_obj.obs:
        raise ValueError(f"sites obs[{column!r}] absent; available: "
                         f"{list(sites_obj.obs.columns)}")
    reducers = {"max": np.nanmax, "min": np.nanmin, "mean": np.nanmean,
                "sum": np.nansum, "std": np.nanstd, "count": len}
    if how not in reducers:
        raise ValueError(f"unknown how={how!r}; use {sorted(reducers)}")

    values = sites_obj.obs[column].to_numpy(dtype=float)
    parent = sites_obj.obs["material_index"].to_numpy(dtype=int)

    out = np.full(md.n_obs, np.nan)
    with np.errstate(invalid="ignore"):
        for i in range(md.n_obs):
            selected = values[parent == i]
            if len(selected):
                out[i] = float(reducers[how](selected))

    name = key_added or f"{column}_{how}"
    md.obs[name] = out
    record(md, "multi.aggregate", column=column, how=how, key_added=name)


@register_function(
    aliases=["to mudata", "multi modal", "combine axes", "mudata export",
             "one object"],
    category="multi",
    description="Assemble the materials object and its sites companion into a "
                "single MuData, for when one file is wanted instead of two "
                "aligned objects.",
    examples=["mdata = mv.multi.to_mudata(md, sites)"],
    related=["mv.multi.sites"],
    notes="Optional throughout. matverse's operations take AnnData, and the "
          "sites object is useful without ever being assembled — this exists "
          "for storage and for handing one thing to somebody else.",
)
def to_mudata(md: AnnData, sites_obj: AnnData | None = None, **modalities):
    """A MuData holding the material axis and any companion axes."""
    try:
        from mudata import MuData
    except ImportError as exc:
        raise ImportError(
            "mv.multi.to_mudata needs mudata (`pip install matverse[multi]`). "
            "The sites object works without it.") from exc

    mods = {"materials": md}
    if sites_obj is not None:
        _check_sites(sites_obj)
        mods["sites"] = sites_obj
    mods.update(modalities)
    return MuData(mods)


__all__ = ["sites", "aggregate", "to_mudata", "AXIS_KEY"]

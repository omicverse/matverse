"""``mv.feat`` — descriptors into ``obsm``.

``X`` already holds the composition and ``var`` already holds the periodic
table, so the classic composition descriptor — the weighted mean, spread and
range of element properties across a formula — is a matrix product of two things
the object is carrying anyway. :func:`element_stats` is that product. It is the
clearest dividend of putting elements on the ``var`` axis rather than flattening
them into anonymous feature columns.

Everything else here is a block deposited into ``obsm``, with its column names
recorded in ``uns['features'][block]``. Not into ``X``: AnnData ties ``X``'s
width to ``var``, and ``var`` is the element axis.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function

#: Statistics taken over the elements of each composition, weighted by amount.
STATISTICS = ("mean", "std", "min", "max", "range")


@register_function(
    aliases=["element statistics", "magpie", "composition features",
             "composition descriptor", "elemental properties", "composition"],
    category="feat",
    description="Composition-weighted mean, spread, minimum, maximum and range "
                "of every element property in var, giving the classic "
                "composition descriptor without an external featuriser.",
    requires={"X": ["composition"]},
    produces={"obsm": ["X_element_stats"], "features": ["X_element_stats"]},
    examples=["mv.feat.element_stats(md)",
              "mv.feat.element_stats(md, properties=['electronegativity', "
              "'atomic_radius'])"],
    related=["mv.feat.soap", "mv.tl.pca"],
    notes="Computed from X and var directly, so it costs one matrix product and "
          "stays consistent with whatever element axis the object currently has "
          "after filtering. It requires no *particular* var column — a claim on "
          "var['Z'] was probed and deleted, because the default takes whatever "
          "numeric columns var happens to carry.",
)
def element_stats(md: AnnData, properties=None, statistics=STATISTICS,
                  key_added: str = "X_element_stats") -> None:
    """Weighted statistics of element properties across each composition."""
    if md.n_vars == 0:
        raise ValueError(
            "this object has no element axis; it was built with build_X=False")

    numeric = [c for c in md.var.columns
               if np.issubdtype(md.var[c].dtype, np.number)]
    props = list(properties) if properties is not None else numeric
    missing = [p for p in props if p not in md.var.columns]
    if missing:
        raise KeyError(f"var has no column(s) {missing}; available: {numeric}")
    bad = [s for s in statistics if s not in STATISTICS]
    if bad:
        raise ValueError(f"unknown statistic(s) {bad}; use {list(STATISTICS)}")

    counts = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    weights = np.divide(counts, totals, out=np.zeros_like(counts),
                        where=totals > 0)
    present = counts > 0

    P = md.var[props].to_numpy(dtype=float)               # elements x properties
    blocks, names = [], []
    for stat in statistics:
        blocks.append(_statistic(stat, weights, present, P))
        names += [f"{p}_{stat}" for p in props]

    md.obsm[key_added] = np.column_stack(blocks)
    md.uns["features"][key_added] = {
        "names": names, "featurizer": "matverse.feat.element_stats",
        "properties": props, "statistics": list(statistics)}
    record(md, "feat.element_stats", n_properties=len(props),
           statistics=list(statistics))


def _statistic(stat: str, weights: np.ndarray, present: np.ndarray,
               P: np.ndarray) -> np.ndarray:
    """One statistic over the elements present in each composition.

    ``mean`` and ``std`` are weighted by amount; ``min``, ``max`` and ``range``
    are taken over the elements that are present at all, which is what makes
    'the most electronegative element in this compound' the quantity it sounds
    like rather than an amount-weighted blend.
    """
    with np.errstate(invalid="ignore"):
        if stat == "mean":
            return _nan_safe(weights @ np.nan_to_num(P), weights, P)
        if stat == "std":
            mean = _nan_safe(weights @ np.nan_to_num(P), weights, P)
            second = _nan_safe(weights @ np.nan_to_num(P ** 2), weights, P)
            return np.sqrt(np.maximum(second - mean ** 2, 0.0))
        masked = np.where(present[:, :, None], P[None, :, :], np.nan)
        if stat == "min":
            return np.nanmin(masked, axis=1)
        if stat == "max":
            return np.nanmax(masked, axis=1)
        if stat == "range":
            return np.nanmax(masked, axis=1) - np.nanmin(masked, axis=1)
    raise ValueError(stat)                                # pragma: no cover


def _nan_safe(result: np.ndarray, weights: np.ndarray,
              P: np.ndarray) -> np.ndarray:
    """Blank out entries whose weighted sum touched a missing property.

    ``nan_to_num`` above lets the matrix product run; this puts the NaN back
    wherever an element actually lacked a published value, so a missing
    electronegativity does not silently become a zero.
    """
    touched = (weights > 0) @ np.isnan(P)
    return np.where(touched > 0, np.nan, result)


#: The v0.1 name. ``element_stats`` says what it computes.
composition = element_stats


@register_function(
    aliases=["soap", "local environment descriptor", "smooth overlap",
             "dscribe", "structure descriptor"],
    category="feat",
    description="Smooth Overlap of Atomic Positions descriptor, averaged over "
                "sites, giving a structure-aware fingerprint that distinguishes "
                "polymorphs of one composition.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["X_soap"], "features": ["X_soap"]},
    examples=["mv.feat.soap(md)", "mv.feat.soap(md, r_cut=5.0, n_max=6)"],
    related=["mv.feat.element_stats", "mv.tl.neighbors"],
    notes="Composition descriptors cannot tell two polymorphs apart, because "
          "they have the same composition. This can. Needs dscribe.",
)
def soap(md: AnnData, source: str = "input", r_cut: float = 5.0,
         n_max: int = 6, l_max: int = 4, sigma: float = 0.5,
         key_added: str = "X_soap") -> None:
    """Site-averaged SOAP descriptor for every structure."""
    try:
        from dscribe.descriptors import SOAP
    except ImportError as exc:
        raise ImportError(
            "mv.feat.soap needs dscribe (`pip install matverse[descriptors]`). "
            "mv.feat.element_stats is the dependency-free alternative, but it "
            "cannot separate polymorphs.") from exc
    from pymatgen.io.ase import AseAtomsAdaptor

    S = structures(md, source)
    species = sorted({str(el) for s in S for el in s.composition.elements})
    descriptor = SOAP(species=species, r_cut=r_cut, n_max=n_max, l_max=l_max,
                      sigma=sigma, periodic=True, sparse=False)

    adaptor = AseAtomsAdaptor()
    rows = [descriptor.create(adaptor.get_atoms(s)).mean(axis=0) for s in S]
    md.obsm[key_added] = np.asarray(rows, dtype=float)
    md.uns["features"][key_added] = {
        "names": [f"soap_{i}" for i in range(md.obsm[key_added].shape[1])],
        "featurizer": "dscribe.SOAP",
        "params": {"r_cut": r_cut, "n_max": n_max, "l_max": l_max,
                   "sigma": sigma, "species": species}}
    record(md, "feat.soap", source=source, r_cut=r_cut, n_max=n_max,
           l_max=l_max)


@register_function(
    aliases=["matminer features", "matminer featurizer", "featurize with "
             "matminer"],
    category="feat",
    description="Delegate featurisation to matminer when it is installed and "
                "deposit the resulting block.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["X_matminer"], "features": ["X_matminer"]},
    examples=["mv.feat.matminer(md)"],
    related=["mv.feat.element_stats"],
    notes="matminer's last release was April 2024. It is an optional delegate, "
          "never a dependency of anything matverse needs to do.",
)
def matminer(md: AnnData, featurizers=None, source: str = "input",
             key_added: str = "X_matminer") -> None:
    """Delegate to matminer, which owns this problem and has hundreds of these."""
    try:
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.structure import DensityFeatures
    except ImportError as exc:
        raise ImportError(
            "mv.feat.matminer needs `pip install matverse[matminer]`") from exc
    import pandas as pd

    S = structures(md, source)
    featurizers = featurizers or MultipleFeaturizer([DensityFeatures()])
    df = featurizers.featurize_dataframe(pd.DataFrame({"structure": S}),
                                         "structure", ignore_errors=True)
    names = [c for c in df.columns if c != "structure"]
    md.obsm[key_added] = df[names].to_numpy(dtype=float)
    md.uns["features"][key_added] = {"names": names, "featurizer": "matminer"}
    record(md, "feat.matminer", source=source)


@register_function(
    aliases=["similarity", "pairwise similarity", "cosine similarity",
             "structure similarity"],
    category="feat",
    description="Cosine similarity between every pair of materials in a "
                "descriptor block, deposited as a pairwise matrix.",
    requires={"obsm": ["{block}"]},
    produces={"obsp": ["similarity_{block}"]},
    prerequisites=["mv.feat.element_stats"],
    examples=["mv.feat.similarity(md)",
              "mv.feat.similarity(md, block='X_soap')"],
    related=["mv.tl.neighbors"],
    notes="Dense and quadratic. Past a few thousand materials use "
          "mv.tl.neighbors, which keeps only the k nearest.",
)
def similarity(md: AnnData, block: str = "X_element_stats") -> None:
    """Cosine similarity over a descriptor block."""
    if block not in md.obsm:
        raise ValueError(
            f"obsm[{block!r}] absent; run mv.feat.element_stats first, or name "
            f"a block that exists ({sorted(md.obsm)})")
    Z = np.nan_to_num(np.asarray(md.obsm[block], dtype=float), nan=0.0)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    md.obsp[f"similarity_{block}"] = Z @ Z.T
    record(md, "feat.similarity", block=block)


__all__ = ["element_stats", "composition", "soap", "matminer", "similarity",
           "STATISTICS"]

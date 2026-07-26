"""``mv.feat`` — descriptors into ``obsm``.

Feature blocks live in ``obsm`` with their column names in
``uns['features'][block]``. Not in ``X``: AnnData ties ``X``'s width to ``var``
so it cannot be widened in place, and these operations write in place.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import record, structures

_COMPOSITION = ["mean_atomic_mass", "mean_electronegativity", "mean_atomic_radius",
                "n_elements", "volume_per_atom", "density"]


def composition(md: AnnData, source: str = "input") -> None:
    """Cheap composition + cell descriptors, no external featuriser needed.

    produces: obsm['X_composition'], uns['features']['X_composition']
    """
    rows = []
    for s in structures(md, source):
        comp = s.composition
        els = list(comp.elements)
        en = [e.X for e in els if e.X is not None]
        rad = [e.atomic_radius for e in els if e.atomic_radius is not None]
        rows.append([
            float(comp.weight / comp.num_atoms),
            float(np.mean(en)) if en else np.nan,
            float(np.mean(rad)) if rad else np.nan,
            float(len(els)),
            float(s.volume / len(s)),
            float(s.density),
        ])
    md.obsm["X_composition"] = np.asarray(rows, dtype=float)
    md.uns["features"]["X_composition"] = {"names": list(_COMPOSITION),
                                           "featurizer": "matverse.feat.composition"}
    record(md, f"feat.composition({source})")


def matminer(md: AnnData, featurizers=None, source: str = "input") -> None:
    """Delegate to matminer when it is installed.

    matminer owns this problem and has hundreds of featurisers; reimplementing
    them would be a maintenance liability with no upside. This only adapts the
    call and deposits the result.

    produces: obsm['X_matminer'], uns['features']['X_matminer']
    """
    try:
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.composition import ElementProperty
        from matminer.featurizers.structure import DensityFeatures
    except ImportError as exc:                     # pragma: no cover
        raise ImportError("mv.feat.matminer needs `pip install matminer`") from exc
    import pandas as pd

    S = structures(md, source)
    if featurizers is None:
        featurizers = MultipleFeaturizer([DensityFeatures()])
        df = featurizers.featurize_dataframe(pd.DataFrame({"structure": S}), "structure")
        names = [c for c in df.columns if c != "structure"]
    else:
        df = featurizers.featurize_dataframe(pd.DataFrame({"structure": S}), "structure")
        names = [c for c in df.columns if c != "structure"]
    md.obsm["X_matminer"] = df[names].to_numpy(dtype=float)
    md.uns["features"]["X_matminer"] = {"names": names, "featurizer": "matminer"}
    record(md, f"feat.matminer({source})")


def similarity(md: AnnData, block: str = "X_composition") -> None:
    """produces: obsp['similarity_<block>']"""
    if block not in md.obsm:
        raise ValueError(f"obsm[{block!r}] absent; run mv.feat.composition first")
    Z = np.nan_to_num(md.obsm[block], nan=0.0)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    md.obsp[f"similarity_{block}"] = Z @ Z.T
    record(md, f"feat.similarity({block})")


__all__ = ["composition", "matminer", "similarity"]

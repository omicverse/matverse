"""``mv.screen`` — filtering that leaves a record.

A screen that returns a shorter list loses why. These deposit a boolean column
and the criteria that produced it, so a dataset carries its own selection
history and ``md[md.obs['passes']]`` stays an ordinary AnnData subset.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import record


def filter(md: AnnData, name: str = "passes", **criteria) -> None:
    """Deposit a boolean mask from ``column__op=value`` criteria.

    ``mv.screen.filter(md, e_above_hull_emt__lt=0.05, n_elements__le=3)``

    produces: obs[name], uns['screens'][name]
    """
    ops = {"lt": np.less, "le": np.less_equal, "gt": np.greater,
           "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}
    mask = np.ones(md.n_obs, dtype=bool)
    applied = {}
    for spec, value in criteria.items():
        col, _, op = spec.rpartition("__")
        if not col or op not in ops:
            raise ValueError(f"criterion {spec!r} must be '<column>__<op>' with op "
                             f"in {sorted(ops)}")
        if col not in md.obs:
            raise ValueError(f"obs[{col!r}] absent; available: {list(md.obs.columns)}")
        v = md.obs[col].to_numpy()
        with np.errstate(invalid="ignore"):
            m = ops[op](v, value)
        m = np.asarray(m, dtype=bool) & (v == v if v.dtype.kind == "f" else True)
        mask &= m
        applied[spec] = value
    md.obs[name] = mask
    md.uns.setdefault("screens", {})[name] = {"criteria": applied,
                                              "n_pass": int(mask.sum()),
                                              "n_total": int(md.n_obs)}
    record(md, f"screen.filter({name}, {applied})")


def rank(md: AnnData, by: str, name: str = "rank", ascending: bool = True) -> None:
    """produces: obs[name]"""
    if by not in md.obs:
        raise ValueError(f"obs[{by!r}] absent")
    v = md.obs[by].to_numpy(dtype=float)
    order = np.argsort(v if ascending else -v, kind="stable")
    r = np.empty(len(v), dtype=float); r[order] = np.arange(1, len(v) + 1)
    r[np.isnan(v)] = np.nan
    md.obs[name] = r
    record(md, f"screen.rank(by={by}, ascending={ascending})")


__all__ = ["filter", "rank"]

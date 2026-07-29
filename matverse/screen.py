"""``mv.screen`` — selection that leaves a record.

A screen that returns a shorter list loses why. These deposit a boolean column
and the criteria that produced it, so a dataset carries its own selection
history and ``md[md.obs['passes']]`` stays an ordinary AnnData subset.

The distinction from ``mv.pp.filter_materials`` is deliberate: that one drops
rows because they are broken, and there is nothing to learn from them. This one
keeps rows because they failed a scientific criterion, and which criterion they
failed is a result.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import record
from ._registry import register_function

_OPS = {"lt": np.less, "le": np.less_equal, "gt": np.greater,
        "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}


@register_function(
    aliases=["screen", "filter candidates", "apply criteria", "select materials",
             "high throughput screening", "shortlist"],
    category="screen",
    description="Deposit a boolean pass/fail column from threshold criteria "
                "written as column__op=value, together with the criteria "
                "themselves and how many candidates passed.",
    produces={"obs": ["{name}"], "uns": ["screens"]},
    examples=["mv.screen.filter(md, e_above_hull_emt__lt=0.05)",
              "mv.screen.filter(md, e_above_hull_emt__lt=0.05, n_elements__le=3, "
              "name='shortlist')"],
    related=["mv.screen.rank", "mv.screen.pareto", "mv.pp.filter_materials"],
    notes="Deposits rather than subsets, so the criteria and the rejected "
          "candidates both survive in the object. Subset afterwards with "
          "md[md.obs[name]] when you want the shorter list.\n\n"
          "Claims no requires, and this is the one place in matverse where the "
          "contract genuinely does not bind. The columns this call consumes are "
          "named by the *keys* of **criteria, each of which is a column and an "
          "operator joined by __ — there is no parameter holding a column name "
          "for a slot template to interpolate. The claim it used to make, "
          "obs['{column}'], was unresolvable rather than merely wrong. An API "
          "whose consumed state is encoded in keyword names puts that state "
          "beyond what a slot template can say.",
)
def filter(md: AnnData, name: str = "passes", **criteria) -> None:
    """Deposit a boolean mask from ``column__op=value`` criteria.

    NaN never passes: a candidate whose energy failed to converge has not met
    the criterion, and silently admitting it is how a broken calculation reaches
    a shortlist.
    """
    mask = np.ones(md.n_obs, dtype=bool)
    applied = {}
    for spec, value in criteria.items():
        column, _, op = spec.rpartition("__")
        if not column or op not in _OPS:
            raise ValueError(
                f"criterion {spec!r} must be '<column>__<op>' with op in "
                f"{sorted(_OPS)}")
        if column not in md.obs:
            raise ValueError(f"obs[{column!r}] absent; available: "
                             f"{list(md.obs.columns)}")
        values = md.obs[column].to_numpy()
        with np.errstate(invalid="ignore"):
            m = np.asarray(_OPS[op](values, value), dtype=bool)
        if values.dtype.kind == "f":
            m &= ~np.isnan(values)
        mask &= m
        applied[spec] = value

    md.obs[name] = mask
    md.uns.setdefault("screens", {})[name] = {
        "criteria": applied,
        "n_pass": int(mask.sum()),
        "n_total": int(md.n_obs),
    }
    record(md, "screen.filter", name=name, **applied)


@register_function(
    aliases=["rank", "sort candidates", "order by", "best candidates"],
    category="screen",
    description="Rank materials by one column, leaving the ranking in obs and "
                "the rows in place.",
    requires={"obs": ["{by}"]},
    produces={"obs": ["{name}"]},
    examples=["mv.screen.rank(md, by='e_above_hull_emt')"],
    related=["mv.screen.filter", "mv.screen.pareto"],
)
def rank(md: AnnData, by: str, name: str = "rank",
         ascending: bool = True) -> None:
    """Rank 1 is best. NaN ranks as NaN rather than last."""
    if by not in md.obs:
        raise ValueError(f"obs[{by!r}] absent; available: "
                         f"{list(md.obs.columns)}")
    values = md.obs[by].to_numpy(dtype=float)
    order = np.argsort(values if ascending else -values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    ranks[np.isnan(values)] = np.nan
    md.obs[name] = ranks
    record(md, "screen.rank", by=by, name=name, ascending=ascending)


@register_function(
    aliases=["pareto", "pareto front", "multi objective", "trade off",
             "non dominated"],
    category="screen",
    description="Find the non-dominated set across several objectives at once, "
                "for screens where no single column decides — stability against "
                "band gap, or performance against cost.",
    requires={"obs": ["{objectives}"]},
    produces={"obs": ["{name}", "{name}_rank"], "uns": ["pareto"]},
    examples=["mv.screen.pareto(md, {'e_above_hull_emt': 'min', "
              "'band_gap_pbe': 'max'})"],
    related=["mv.screen.filter", "mv.screen.rank"],
    notes="Deposits both membership of the first front and the front index, so "
          "the second-best trade-offs remain reachable instead of being "
          "discarded with everything that is not optimal.",
)
def pareto(md: AnnData, objectives: dict, name: str = "pareto") -> None:
    """Non-dominated sorting over ``{column: 'min' | 'max'}``."""
    if not objectives:
        raise ValueError("pareto needs at least one objective")
    cols = list(objectives)
    missing = [c for c in cols if c not in md.obs]
    if missing:
        raise ValueError(f"obs column(s) {missing} absent; available: "
                         f"{list(md.obs.columns)}")
    for col, sense in objectives.items():
        if sense not in ("min", "max"):
            raise ValueError(f"objective {col!r} must be 'min' or 'max', "
                             f"got {sense!r}")

    # Flip maximisation to minimisation so one comparison covers both.
    M = np.column_stack([
        md.obs[c].to_numpy(dtype=float) * (1.0 if objectives[c] == "min" else -1.0)
        for c in cols])
    valid = ~np.isnan(M).any(axis=1)

    fronts = np.full(len(M), np.nan)
    remaining = np.where(valid)[0]
    front_index = 0
    while len(remaining):
        block = M[remaining]
        # dominates[i, j] is "j beats i": j is no worse on every objective and
        # strictly better on at least one. Point i is dominated if any j does,
        # so the reduction is over j — axis 1, not axis 0.
        le = (block[None, :, :] <= block[:, None, :]).all(axis=2)
        lt = (block[None, :, :] < block[:, None, :]).any(axis=2)
        dominated = (le & lt).any(axis=1)
        current = remaining[~dominated]
        if not len(current):
            break
        fronts[current] = front_index
        remaining = remaining[dominated]
        front_index += 1

    md.obs[name] = fronts == 0
    md.obs[f"{name}_rank"] = fronts
    md.uns.setdefault("pareto", {})[name] = {
        "objectives": dict(objectives),
        "n_fronts": int(front_index),
        "n_optimal": int((fronts == 0).sum()),
        "n_incomparable": int((~valid).sum()),
    }
    record(md, "screen.pareto", name=name, **objectives)


__all__ = ["filter", "rank", "pareto"]

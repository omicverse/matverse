"""``mv.exp`` — measured data, on the same footing as computed data.

Experiment is a level of theory. That sounds like a slogan and is actually the
whole implementation: ``uns['levels']['experiment']`` records an instrument
instead of a functional, a measured pattern lands in ``obsm['xrd_experiment']``
next to the computed ``obsm['xrd_calc']``, and every comparison written for two
computed levels works unchanged on a computed one against a measured one.

Nothing new was needed to make that true, which is the argument for having typed
the level of theory in the first place.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import deposit_grid, grid_of, record, set_level
from ._registry import register_function


@register_function(
    aliases=["attach measurement", "add measured spectrum", "measured data",
             "experimental pattern", "attach experiment"],
    category="exp",
    description="Attach a measured curve for every material onto the same grid "
                "as its computed counterpart, recording the instrument as a "
                "level of theory.",
    produces={"obsm": ["{quantity}_{level}"], "levels": ["{level}"]},
    prerequisites=["mv.prop.xrd"],
    examples=["mv.exp.attach(md, 'xrd', patterns, two_theta, "
              "instrument='Bruker D8')"],
    related=["mv.prop.compare_grids", "mv.exp.match_xrd"],
    notes="Resampled onto the existing grid rather than stored on its own, "
          "because two curves on different grids cannot be subtracted and a "
          "resample done once is better than one done in every comparison. "
          "Attach a measurement at its own resolution or better: diffraction "
          "peaks are narrow, and resampling onto a grid coarser than the peak "
          "width discards them permanently. Points outside the measured range "
          "become NaN rather than zero, and comparisons use the overlap.\n\n"
          "Claims no requires. It used to claim uns['grids'], and probing "
          "deleted it: when the quantity has no grid yet the measured one "
          "becomes it, so a measurement can be attached before anything was "
          "computed.",
)
def attach(md: AnnData, quantity: str, values, grid, level: str = "experiment",
           instrument: str | None = None, unit: str = "",
           **meta) -> None:
    """Attach measured curves, resampled onto the quantity's existing grid."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    grid = np.asarray(grid, dtype=float)
    if values.shape[0] != md.n_obs:
        raise ValueError(
            f"got {values.shape[0]} measured curves for {md.n_obs} materials; "
            f"pass one row per material, or use mv.exp.match_xrd for a single "
            f"pattern you want ranked against every candidate")

    target = _target_grid(md, quantity, grid, unit)
    resampled = np.vstack([np.interp(target, grid, row, left=np.nan,
                                     right=np.nan) for row in values])

    deposit_grid(md, quantity, level, resampled, target, unit=unit)
    set_level(md, level, kind="experiment", method=instrument or "measurement",
              reference=None, surrogate=False, license=None,
              uncertainty="measured", instrument=instrument, **meta)
    record(md, "exp.attach", quantity=quantity, level=level,
           instrument=instrument)


def _target_grid(md: AnnData, quantity: str, grid: np.ndarray, unit: str):
    """The grid to resample onto — the existing one, or this one if it is new."""
    try:
        return grid_of(md, quantity)
    except KeyError:
        return grid


@register_function(
    aliases=["match xrd", "phase identification", "identify phase",
             "search match", "which phase is this", "index a pattern"],
    category="exp",
    description="Rank every candidate in the object by how well its simulated "
                "diffraction pattern matches one measured pattern — phase "
                "identification against the candidate set you already have.",
    requires={"obsm": ["xrd_{level}"]},
    produces={"obs": ["xrd_match", "xrd_match_rank"], "uns": ["xrd_match"]},
    prerequisites=["mv.prop.xrd"],
    examples=["mv.exp.match_xrd(md, intensity, two_theta)"],
    related=["mv.prop.xrd", "mv.screen.rank"],
    notes="Scores against the candidates in this object and nothing else. A "
          "high score means 'the best of what you gave it', not 'identified' — "
          "the true phase can be absent from the library entirely.",
)
def match_xrd(md: AnnData, intensity, two_theta, level: str = "calc",
              key_added: str = "xrd_match") -> None:
    """Rank candidates against one measured powder pattern.

    Both patterns are baseline-shifted to zero and unit-normalised before the
    dot product, so the score reflects peak positions and relative heights
    rather than absolute counts or exposure time.
    """
    key = f"xrd_{level}"
    if key not in md.obsm:
        raise ValueError(f"obsm[{key!r}] absent; run mv.prop.xrd(md, "
                         f"level={level!r}) first")

    grid = grid_of(md, "xrd")
    measured = np.interp(grid, np.asarray(two_theta, dtype=float),
                         np.asarray(intensity, dtype=float),
                         left=0.0, right=0.0)
    computed = np.asarray(md.obsm[key], dtype=float)

    m = _unit(measured - np.nanmin(measured))
    C = np.vstack([_unit(row - np.nanmin(row)) if np.isfinite(row).all()
                   else np.full(len(grid), np.nan) for row in computed])
    with np.errstate(invalid="ignore"):
        score = C @ m

    md.obs[key_added] = score
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    ranks[np.isnan(score)] = np.nan
    md.obs[f"{key_added}_rank"] = ranks

    best = int(order[0]) if len(order) else None
    md.uns[key_added] = {
        "level": level,
        "n_candidates": int(md.n_obs),
        "best": str(md.obs_names[best]) if best is not None else None,
        "best_score": float(score[best]) if best is not None else None,
        "scored_against": "this object's candidates only",
    }
    record(md, "exp.match_xrd", level=level, key_added=key_added)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


@register_function(
    aliases=["measured property", "attach measurement column",
             "experimental value", "measured scalar"],
    category="exp",
    description="Attach a measured scalar property as its own level of theory, "
                "so it sits beside the computed values of the same quantity and "
                "can be compared with mv.compare_levels.",
    produces={"obs": ["{quantity}_{level}"], "levels": ["{level}"]},
    examples=["mv.exp.measure(md, 'band_gap', values, instrument='UV-Vis')"],
    related=["mv.compare_levels", "mv.exp.attach"],
    notes="The point of the naming convention. Once a measurement is a level, "
          "mv.compare_levels(md, 'band_gap') puts PBE, HSE06 and the "
          "spectrometer in one table without anyone deciding which is the "
          "band gap.",
)
def measure(md: AnnData, quantity: str, values, level: str = "experiment",
            instrument: str | None = None, uncertainty=None, **meta) -> None:
    """Attach a measured scalar, tagged as an experimental level."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] != md.n_obs:
        raise ValueError(f"got {values.shape[0]} values for {md.n_obs} "
                         f"materials")
    md.obs[f"{quantity}_{level}"] = values
    if uncertainty is not None:
        md.obs[f"{quantity}_{level}_std"] = np.asarray(uncertainty, dtype=float)
    set_level(md, level, kind="experiment", method=instrument or "measurement",
              reference=None, surrogate=False, license=None,
              uncertainty="measured" if uncertainty is not None else None,
              instrument=instrument, **meta)
    record(md, "exp.measure", quantity=quantity, level=level,
           instrument=instrument)


__all__ = ["attach", "match_xrd", "measure"]

"""``mv.prop`` — derived physical properties.

Properties that are a single number per material go to ``obs``. Properties that
are a curve — a diffraction pattern, a density of states, a phonon spectrum — go
to ``obsm`` as a ``materials x grid`` block, with the shared grid axis recorded
once in ``uns['grids'][quantity]``.

Both carry the level of theory as a name suffix, so ``obs['band_gap_pbe']`` and
``obsm['xrd_pbe']`` read the same way and a measured pattern is
``obsm['xrd_experiment']`` rather than a different kind of object. That is what
makes comparing computation against measurement a subtraction rather than an
export.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import deposit_grid, record, set_level, structures
from ._registry import register_function


@register_function(
    aliases=["xrd", "x-ray diffraction", "diffraction pattern", "powder pattern",
             "simulate xrd", "xrd pattern"],
    category="prop",
    description="Simulate a powder X-ray diffraction pattern for every "
                "structure and store the patterns on a shared two-theta grid, "
                "so they can be compared with each other and with a measured "
                "pattern.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["xrd_{level}"], "uns": ["grids"], "levels": ["{level}"]},
    examples=["mv.prop.xrd(md)",
              "mv.prop.xrd(md, two_theta=(10, 90), step=0.02, fwhm=0.15)"],
    related=["mv.exp.match_xrd", "mv.pp.standardize"],
    notes="pymatgen returns a peak list; a peak list cannot be compared across "
          "materials because no two share the same peak positions. Broadening "
          "onto a common grid is what makes the patterns a matrix.",
)
def xrd(md: AnnData, source: str = "input", level: str = "calc",
        wavelength: str = "CuKa", two_theta: tuple = (5.0, 90.0),
        step: float = 0.02, fwhm: float = 0.1,
        normalize: bool = True) -> None:
    """Powder XRD patterns on a shared two-theta grid.

    ``fwhm`` is the full width at half maximum of the Gaussian each reflection
    is broadened by, in degrees. It stands in for instrumental resolution and
    finite crystallite size together, which is enough to compare candidates and
    is not enough to refine anything.
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    grid = np.arange(two_theta[0], two_theta[1] + step / 2, step)
    sigma = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    calculator = XRDCalculator(wavelength=wavelength)

    rows, failed = [], 0
    for structure in structures(md, source):
        try:
            pattern = calculator.get_pattern(structure, two_theta_range=two_theta)
            rows.append(_broaden(np.asarray(pattern.x, dtype=float),
                                 np.asarray(pattern.y, dtype=float),
                                 grid, sigma, normalize))
        except Exception:
            rows.append(np.full(len(grid), np.nan))
            failed += 1

    deposit_grid(md, "xrd", level, np.vstack(rows), grid, unit="degrees 2theta",
                 wavelength=wavelength, fwhm=fwhm, normalized=bool(normalize))
    set_level(md, level, kind="model", method=f"XRD ({wavelength})",
              reference=None, surrogate=False, license=None, uncertainty=None,
              source=source, n_failed=failed)
    record(md, "prop.xrd", source=source, level=level, wavelength=wavelength,
           fwhm=fwhm)


def _broaden(positions: np.ndarray, intensities: np.ndarray, grid: np.ndarray,
             sigma: float, normalize: bool) -> np.ndarray:
    """Sum of Gaussians at each reflection, evaluated on the grid."""
    if not len(positions):
        return np.zeros(len(grid))
    delta = grid[:, None] - positions[None, :]
    profile = np.exp(-0.5 * (delta / sigma) ** 2) @ intensities
    peak = profile.max()
    return profile / peak * 100.0 if normalize and peak > 0 else profile


@register_function(
    aliases=["radial distribution", "rdf", "pair distribution",
             "structure fingerprint curve"],
    category="prop",
    description="Radial distribution function for every structure on a shared "
                "distance grid, a cheap structural fingerprint that separates "
                "polymorphs without an external descriptor package.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["rdf_{level}"], "uns": ["grids"], "levels": ["{level}"]},
    examples=["mv.prop.rdf(md)", "mv.prop.rdf(md, r_max=12.0, sigma=0.1)"],
    related=["mv.prop.xrd", "mv.feat.soap"],
    notes="Composition descriptors cannot separate polymorphs and SOAP needs "
          "dscribe. This needs neither, at the cost of being a much coarser "
          "fingerprint than either.",
)
def rdf(md: AnnData, source: str = "input", level: str = "calc",
        r_max: float = 10.0, step: float = 0.05, sigma: float = 0.1) -> None:
    """Radial distribution function on a shared distance grid, in angstrom."""
    grid = np.arange(step, r_max + step / 2, step)

    rows, failed = [], 0
    for structure in structures(md, source):
        try:
            neighbours = structure.get_all_neighbors(r_max)
            distances = np.array([n.nn_distance for site in neighbours
                                  for n in site], dtype=float)
            if not len(distances):
                rows.append(np.zeros(len(grid)))
                continue
            delta = grid[:, None] - distances[None, :]
            counts = np.exp(-0.5 * (delta / sigma) ** 2).sum(axis=1)
            # Normalise by the shell volume and the ideal-gas density, so the
            # curve tends to 1 at large r instead of growing with r squared.
            shell = 4.0 * np.pi * grid ** 2 * step
            density = len(structure) / structure.volume
            rows.append(counts / (shell * density * len(structure)))
        except Exception:
            rows.append(np.full(len(grid), np.nan))
            failed += 1

    deposit_grid(md, "rdf", level, np.vstack(rows), grid, unit="angstrom",
                 sigma=sigma)
    set_level(md, level, kind="model", method="radial distribution function",
              reference=None, surrogate=False, license=None, uncertainty=None,
              source=source, n_failed=failed)
    record(md, "prop.rdf", source=source, level=level, r_max=r_max, sigma=sigma)


@register_function(
    aliases=["compare grids", "compare spectra", "spectrum difference",
             "computed versus measured"],
    category="prop",
    description="Compare one grid-shaped quantity across two levels of theory "
                "and record the per-material agreement, which is how a computed "
                "pattern is checked against a measured one.",
    requires={"obsm": ["{quantity}_{a}", "{quantity}_{b}"]},
    produces={"obs": ["{quantity}_cosine_{a}_vs_{b}",
                      "{quantity}_rmse_{a}_vs_{b}",
                      "{quantity}_overlap_{a}_vs_{b}"]},
    prerequisites=["mv.prop.xrd"],
    examples=["mv.prop.compare_grids(md, 'xrd', 'calc', 'experiment')"],
    related=["mv.prop.xrd", "mv.exp.match_xrd", "mv.compare_levels"],
    notes="Compared over the points where both curves are defined, and the "
          "number of points used is recorded. A measurement covering a narrower "
          "range than the calculation is the normal case, not an error, so one "
          "undefined point must not take the whole comparison with it.",
)
def compare_grids(md: AnnData, quantity: str, a: str, b: str) -> None:
    """Cosine similarity and RMSE between two levels of one grid quantity."""
    key_a, key_b = f"{quantity}_{a}", f"{quantity}_{b}"
    for key in (key_a, key_b):
        if key not in md.obsm:
            raise ValueError(
                f"obsm[{key!r}] absent; compute {quantity!r} at that level "
                f"first. Present: {sorted(k for k in md.obsm)}")
    A = np.asarray(md.obsm[key_a], dtype=float)
    B = np.asarray(md.obsm[key_b], dtype=float)

    both = np.isfinite(A) & np.isfinite(B)
    A0 = np.where(both, A, 0.0)
    B0 = np.where(both, B, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        norms = np.sqrt((A0 ** 2).sum(axis=1)) * np.sqrt((B0 ** 2).sum(axis=1))
        cosine = np.where(norms > 0, (A0 * B0).sum(axis=1) / norms, np.nan)
        counts = both.sum(axis=1)
        rmse = np.where(counts > 0,
                        np.sqrt(((A0 - B0) ** 2).sum(axis=1)
                                / np.maximum(counts, 1)),
                        np.nan)

    md.obs[f"{quantity}_cosine_{a}_vs_{b}"] = cosine
    md.obs[f"{quantity}_rmse_{a}_vs_{b}"] = rmse
    md.obs[f"{quantity}_overlap_{a}_vs_{b}"] = counts.astype(float)
    record(md, "prop.compare_grids", quantity=quantity, a=a, b=b)


__all__ = ["xrd", "rdf", "compare_grids"]

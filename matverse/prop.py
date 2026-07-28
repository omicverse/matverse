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
    aliases=["elastic", "elastic constants", "bulk modulus", "shear modulus",
             "stiffness tensor", "elastic moduli", "mechanical properties"],
    category="prop",
    description="Compute the elastic stiffness tensor of every structure by "
                "finite strains at one level of theory, and derive the Voigt-"
                "Reuss-Hill bulk and shear moduli from it.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["bulk_modulus_{level}", "shear_modulus_{level}",
                      "youngs_modulus_{level}", "poisson_ratio_{level}",
                      "elastic_stable_{level}"],
              "obsm": ["elastic_tensor_{level}"], "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    dispatch="level= selects the calculator, as for mv.calc.energy",
    examples=["mv.prop.elastic(md, level='emt', source='relaxed_emt')"],
    related=["mv.calc.relax", "mv.screen.filter"],
    notes="Run this on a relaxed structure. An elastic constant is the second "
          "derivative of energy about a minimum, and taking it about a "
          "geometry that is not one gives a number with residual stress folded "
          "in — which is why elastic_stable is reported: a negative eigenvalue "
          "of the stiffness tensor usually means the input was not relaxed "
          "rather than that the material is unstable.",
)
def elastic(md: AnnData, level: str = "emt", source: str = "input",
            strain: float = 0.01, key_added: str | None = None) -> None:
    """Elastic stiffness tensor and the moduli derived from it, in GPa."""
    from pymatgen.io.ase import AseAtomsAdaptor

    from .calc import _get

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    calculator = factory()
    block = key_added or f"elastic_tensor_{level}"

    tensors, failed = [], 0
    for structure in structures(md, source):
        try:
            tensors.append(_stiffness(structure, adaptor, calculator, strain))
        except Exception:
            tensors.append(np.full((6, 6), np.nan))
            failed += 1

    stacked = np.stack(tensors)
    md.obsm[block] = stacked.reshape(len(stacked), 36)

    bulk, shear, young, poisson, stable = [], [], [], [], []
    for C in stacked:
        k, g, e, nu, ok = _moduli(C)
        bulk.append(k); shear.append(g); young.append(e)
        poisson.append(nu); stable.append(ok)

    md.obs[f"bulk_modulus_{level}"] = bulk
    md.obs[f"shear_modulus_{level}"] = shear
    md.obs[f"youngs_modulus_{level}"] = young
    md.obs[f"poisson_ratio_{level}"] = poisson
    md.obs[f"elastic_stable_{level}"] = stable
    set_level(md, level, **meta, source=source, strain=strain, n_failed=failed)
    record(md, "prop.elastic", level=level, source=source, strain=strain)


#: eV/angstrom^3 -> GPa
_EV_PER_A3_TO_GPA = 160.21766208

#: Voigt order: xx, yy, zz, yz, xz, xy.
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _stiffness(structure, adaptor, calculator, amount: float) -> np.ndarray:
    """Stiffness by central differences of stress with respect to strain."""
    C = np.zeros((6, 6))
    for j, (a, b) in enumerate(_VOIGT):
        plus = _stress(structure, adaptor, calculator,
                       _deformation(a, b, amount))
        minus = _stress(structure, adaptor, calculator,
                        _deformation(a, b, -amount))
        C[:, j] = (plus - minus) / (2.0 * amount)
    return 0.5 * (C + C.T) * _EV_PER_A3_TO_GPA


def _deformation(a: int, b: int, amount: float) -> np.ndarray:
    """Deformation gradient for one Voigt strain component."""
    epsilon = np.zeros((3, 3))
    if a == b:
        epsilon[a, a] = amount
    else:
        epsilon[a, b] = epsilon[b, a] = amount / 2.0
    return np.eye(3) + epsilon


def _stress(structure, adaptor, calculator, F: np.ndarray) -> np.ndarray:
    """Stress in Voigt order, in eV/angstrom^3, under a deformation."""
    deformed = structure.copy()
    deformed.lattice = deformed.lattice.__class__(
        np.asarray(deformed.lattice.matrix) @ F.T)
    atoms = adaptor.get_atoms(deformed)
    atoms.calc = calculator
    return np.asarray(atoms.get_stress(voigt=True), dtype=float)


def _moduli(C: np.ndarray):
    """Voigt-Reuss-Hill averages from a stiffness tensor."""
    if not np.isfinite(C).all():
        return (np.nan,) * 4 + (False,)
    try:
        eigenvalues = np.linalg.eigvalsh(C)
        stable = bool((eigenvalues > 0).all())          # Born stability
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return (np.nan,) * 4 + (False,)

    kv = (C[:3, :3].sum()) / 9.0
    gv = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2])
          + 3.0 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    kr_denominator = S[:3, :3].sum()
    gr_denominator = (4.0 * (S[0, 0] + S[1, 1] + S[2, 2])
                      - 4.0 * (S[0, 1] + S[0, 2] + S[1, 2])
                      + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5]))
    kr = 1.0 / kr_denominator if abs(kr_denominator) > 1e-12 else np.nan
    gr = 15.0 / gr_denominator if abs(gr_denominator) > 1e-12 else np.nan

    k = float(np.nanmean([kv, kr]))
    g = float(np.nanmean([gv, gr]))
    if not np.isfinite(k) or not np.isfinite(g) or (3.0 * k + g) == 0:
        return k, g, np.nan, np.nan, stable
    young = 9.0 * k * g / (3.0 * k + g)
    poisson = (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))
    return k, g, float(young), float(poisson), stable


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


__all__ = ["xrd", "rdf", "elastic", "compare_grids"]

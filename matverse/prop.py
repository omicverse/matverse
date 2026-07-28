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
    aliases=["phonon", "phonons", "vibrational spectrum", "phonon dos",
             "frozen phonon", "lattice dynamics", "imaginary modes"],
    category="prop",
    description="Compute the vibrational spectrum of every structure by frozen "
                "displacements, store the phonon density of states on a shared "
                "frequency grid, and flag structures with imaginary modes.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["phonon_dos_{level}"],
              "obs": ["n_imaginary_modes_{level}", "dynamically_stable_{level}",
                      "zero_point_energy_{level}"],
              "uns": ["grids"], "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    dispatch="level= selects the calculator, as for mv.calc.energy",
    examples=["mv.prop.phonon(md, level='emt', source='relaxed_emt')",
              "mv.prop.phonon(md, level='emt', supercell=(2, 2, 2))"],
    related=["mv.calc.relax", "mv.prop.free_energy", "mv.thermo.hull"],
    notes="Gamma-point frozen phonons on a supercell: displace each atom, read "
          "the forces, diagonalise. Cheap, and coarser than phonopy's full "
          "q-mesh — a supercell samples only the q-points commensurate with it. "
          "Run it on a relaxed structure; imaginary modes on an unrelaxed one "
          "mean the geometry, not the material.\n\n"
          "An imaginary mode is the check a hull cannot make. A composition can "
          "sit on the convex hull and still be dynamically unstable, and "
          "generated structures fail this far more often than they fail the "
          "hull.",
)
def phonon(md: AnnData, level: str = "emt", source: str = "input",
           supercell=(2, 2, 2), displacement: float = 0.01,
           f_max: float = 15.0, n_bins: int = 200,
           sigma: float = 0.3) -> None:
    """Phonon density of states by frozen displacements, in THz."""
    from pymatgen.io.ase import AseAtomsAdaptor

    from .calc import _get

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    calculator = factory()
    grid = np.linspace(0.0, f_max, n_bins)

    rows, imaginary, stable, zpe, failed = [], [], [], [], 0
    for structure in structures(md, source):
        try:
            frequencies = _frequencies(structure, supercell, adaptor,
                                       calculator, displacement)
            negative = int((frequencies < -_IMAGINARY_TOLERANCE).sum())
            real = frequencies[frequencies > _IMAGINARY_TOLERANCE]
            rows.append(_smear(real, grid, sigma))
            imaginary.append(negative)
            stable.append(negative == 0)
            # ZPE is the sum of hbar*omega/2 over modes, reported per atom so
            # it is comparable between cells of different size.
            n_atoms = len(structure) * int(np.prod(supercell))
            zpe.append(float(0.5 * _THZ_TO_EV * real.sum() / n_atoms))
        except Exception:
            rows.append(np.full(len(grid), np.nan))
            imaginary.append(-1)
            stable.append(False)
            zpe.append(np.nan)
            failed += 1

    deposit_grid(md, "phonon_dos", level, np.vstack(rows), grid, unit="THz",
                 supercell=list(supercell), displacement=displacement,
                 smearing=sigma)
    md.obs[f"n_imaginary_modes_{level}"] = imaginary
    md.obs[f"dynamically_stable_{level}"] = stable
    md.obs[f"zero_point_energy_{level}"] = zpe
    set_level(md, level, **meta, source=source, supercell=list(supercell),
              displacement=displacement, n_failed=failed)
    record(md, "prop.phonon", level=level, source=source,
           supercell=list(supercell))


#: Frequencies below this magnitude are the three acoustic modes at gamma, which
#: are zero by translational invariance and numerically are not quite.
_IMAGINARY_TOLERANCE = 0.05        # THz

#: sqrt(eV / (angstrom^2 * amu)) -> THz, and h * THz -> eV.
_OMEGA_TO_THZ = 15.633302
_THZ_TO_EV = 4.135667696e-3


def _frequencies(structure, supercell, adaptor, calculator,
                 displacement: float) -> np.ndarray:
    """Gamma-point frequencies of a supercell, in THz.

    Central differences of the forces with respect to each atomic displacement
    give the force constants; mass-weighting and diagonalising gives the modes.
    Negative eigenvalues come back as negative frequencies rather than complex
    ones, which is the convention every phonon code prints.
    """
    cell = structure.copy()
    cell.make_supercell(list(supercell))
    n = len(cell)
    if n > 64:
        raise ValueError(
            f"supercell has {n} atoms; frozen phonons cost 6N force "
            f"evaluations, so this would need {6 * n}. Use a smaller supercell, "
            f"or phonopy, which exploits symmetry.")

    masses = np.array([site.specie.atomic_mass for site in cell], dtype=float)
    force_constants = np.zeros((3 * n, 3 * n))

    for atom in range(n):
        for axis in range(3):
            plus = _forces_displaced(cell, atom, axis, displacement,
                                     adaptor, calculator)
            minus = _forces_displaced(cell, atom, axis, -displacement,
                                      adaptor, calculator)
            force_constants[3 * atom + axis] = -(plus - minus).ravel() / \
                (2.0 * displacement)

    force_constants = 0.5 * (force_constants + force_constants.T)
    weights = np.repeat(masses, 3)
    dynamical = force_constants / np.sqrt(np.outer(weights, weights))

    eigenvalues = np.linalg.eigvalsh(dynamical)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _OMEGA_TO_THZ


def _forces_displaced(cell, atom: int, axis: int, amount: float,
                      adaptor, calculator) -> np.ndarray:
    moved = cell.copy()
    shift = np.zeros(3)
    shift[axis] = amount
    moved.translate_sites([atom], shift, frac_coords=False, to_unit_cell=False)
    atoms = adaptor.get_atoms(moved)
    atoms.calc = calculator
    return np.asarray(atoms.get_forces(), dtype=float)


def _smear(frequencies: np.ndarray, grid: np.ndarray,
           sigma: float) -> np.ndarray:
    """Gaussian-broadened density of states, normalised to unit area."""
    if not len(frequencies):
        return np.zeros(len(grid))
    delta = grid[:, None] - frequencies[None, :]
    dos = np.exp(-0.5 * (delta / sigma) ** 2).sum(axis=1)
    area = np.trapezoid(dos, grid) if hasattr(np, "trapezoid") \
        else np.trapz(dos, grid)
    return dos / area if area > 0 else dos


@register_function(
    aliases=["free energy", "vibrational free energy", "heat capacity",
             "finite temperature", "entropy", "thermal properties"],
    category="prop",
    description="Derive the harmonic vibrational free energy, entropy and heat "
                "capacity at one temperature from a phonon density of states.",
    requires={"obsm": ["phonon_dos_{level}"], "uns": ["grids"]},
    produces={"obs": ["vibrational_free_energy_{level}",
                      "vibrational_entropy_{level}", "heat_capacity_{level}"]},
    prerequisites=["mv.prop.phonon"],
    examples=["mv.prop.free_energy(md, level='emt', temperature=300.0)"],
    related=["mv.prop.phonon", "mv.thermo.hull"],
    notes="The harmonic approximation, which is where a hull built at 0 K "
          "starts to become a hull at temperature. It is also where the "
          "approximation shows: near melting, and for anything with a soft "
          "mode, harmonic free energies are wrong in a way this cannot detect.",
)
def free_energy(md: AnnData, level: str = "emt",
                temperature: float = 300.0) -> None:
    """Harmonic vibrational thermodynamics from the phonon DOS.

    Free energy in eV/atom, entropy and heat capacity in eV/K/atom. Well above
    the Debye temperature the heat capacity approaches ``3 k_B`` per atom, which
    is the check worth running on a new calculator.
    """
    key = f"phonon_dos_{level}"
    if key not in md.obsm:
        raise ValueError(f"obsm[{key!r}] absent; run mv.prop.phonon(md, "
                         f"level={level!r}) first")
    from ._core import grid_of

    grid = grid_of(md, "phonon_dos")
    dos = np.asarray(md.obsm[key], dtype=float)

    kT = _BOLTZMANN_EV_PER_K * float(temperature)
    energy = _THZ_TO_EV * grid                       # hbar*omega, in eV
    positive = energy > 1e-6

    free, entropy, capacity = [], [], []
    for row in dos:
        if not np.isfinite(row).all():
            free.append(np.nan); entropy.append(np.nan); capacity.append(np.nan)
            continue
        weight = row[positive]
        e = energy[positive]
        x = e / kT if kT > 0 else np.full_like(e, np.inf)

        # F = integral g(w) [ hw/2 + kT ln(1 - exp(-hw/kT)) ] dw
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            f_density = 0.5 * e + kT * np.log1p(-np.exp(-x))
            n = 1.0 / np.expm1(x)
            s_density = _BOLTZMANN_EV_PER_K * ((1.0 + n) * np.log1p(n)
                                               - n * np.log(np.maximum(n, 1e-300)))
            c_density = _BOLTZMANN_EV_PER_K * x ** 2 * np.exp(x) / \
                np.maximum(np.expm1(x) ** 2, 1e-300)

        # The stored DOS is normalised to unit area, so each integral is an
        # average per vibrational mode. Three modes per atom turns that into a
        # per-atom quantity, which is what a hull and a heat capacity are
        # normally quoted in — and lets the classical limit be checked: the
        # heat capacity should approach 3k_B per atom well above the Debye
        # temperature.
        free.append(float(_MODES_PER_ATOM * _integrate(
            np.nan_to_num(f_density), weight, grid[positive])))
        entropy.append(float(_MODES_PER_ATOM * _integrate(
            np.nan_to_num(s_density), weight, grid[positive])))
        capacity.append(float(_MODES_PER_ATOM * _integrate(
            np.nan_to_num(c_density), weight, grid[positive])))

    md.obs[f"vibrational_free_energy_{level}"] = free
    md.obs[f"vibrational_entropy_{level}"] = entropy
    md.obs[f"heat_capacity_{level}"] = capacity
    md.uns.setdefault("thermal", {})[level] = {"temperature": float(temperature)}
    record(md, "prop.free_energy", level=level, temperature=temperature)


#: eV/K
_BOLTZMANN_EV_PER_K = 8.617333262e-5

#: Three vibrational modes per atom, which converts a per-mode average taken
#: over a unit-area density of states into a per-atom quantity.
_MODES_PER_ATOM = 3.0


def _integrate(density: np.ndarray, weight: np.ndarray,
               grid: np.ndarray) -> float:
    product = density * weight
    return (np.trapezoid(product, grid) if hasattr(np, "trapezoid")
            else np.trapz(product, grid))


@register_function(
    aliases=["thermal conductivity", "lattice thermal conductivity", "kappa",
             "heat transport", "thermoelectric", "slack model"],
    category="prop",
    description="Estimate the lattice thermal conductivity from the phonon and "
                "elastic data already computed, using the Slack model, together "
                "with the Debye temperature and Gruneisen parameter it needs.",
    requires={"obsm": ["phonon_dos_{level}"], "uns": ["grids"]},
    produces={"obs": ["debye_temperature_{level}", "gruneisen_{level}",
                      "sound_velocity_{level}",
                      "thermal_conductivity_{level}"]},
    prerequisites=["mv.prop.phonon"],
    examples=["mv.prop.thermal_conductivity(md, level='emt')",
              "mv.prop.thermal_conductivity(md, level='emt', "
              "temperature=300.0)"],
    related=["mv.prop.phonon", "mv.prop.elastic", "mv.screen.rank"],
    notes="An order-of-magnitude model, not a solution of the Boltzmann "
          "transport equation. Slack's expression captures the right scaling — "
          "kappa falls with mass, with anharmonicity and with temperature — and "
          "is routinely a factor of two off in absolute terms. It is the right "
          "tool for ranking a thousand candidates and the wrong one for quoting "
          "a number.\n\n"
          "The honest alternative is phono3py: third-order force constants and "
          "a real phonon-phonon scattering calculation, at perhaps a thousand "
          "times the cost. Matbench Discovery weights thermal conductivity at "
          "40% of its combined score precisely because it is the property "
          "cheap methods get wrong, so treat a screen ranked on this as a "
          "shortlist for phono3py rather than an answer.\n\n"
          "The Gruneisen parameter is taken from the Poisson ratio when "
          "mv.prop.elastic has run, and defaults to 1.5 — a typical value for "
          "a simple solid — when it has not. Which was used is recorded.",
)
def thermal_conductivity(md: AnnData, level: str = "emt",
                         temperature: float = 300.0,
                         gruneisen: float | None = None) -> None:
    """Lattice thermal conductivity in W/m/K, by the Slack model."""
    key = f"phonon_dos_{level}"
    if key not in md.obsm:
        raise ValueError(f"obsm[{key!r}] absent; run mv.prop.phonon(md, "
                         f"level={level!r}) first")
    from ._core import grid_of

    grid = grid_of(md, "phonon_dos")
    dos = np.asarray(md.obsm[key], dtype=float)
    structures_ = structures(md, "input")

    debye = np.full(md.n_obs, np.nan)
    gamma = np.full(md.n_obs, np.nan)
    velocity = np.full(md.n_obs, np.nan)
    kappa = np.full(md.n_obs, np.nan)

    poisson = (md.obs[f"poisson_ratio_{level}"].to_numpy(dtype=float)
               if f"poisson_ratio_{level}" in md.obs else None)
    source = "explicit" if gruneisen is not None else (
        "Poisson ratio" if poisson is not None else "default 1.5")

    for i, structure in enumerate(structures_):
        row = dos[i]
        if not np.isfinite(row).all() or not row.any():
            continue
        theta = _debye_temperature(row, grid)
        debye[i] = theta

        if gruneisen is not None:
            g = float(gruneisen)
        elif poisson is not None and np.isfinite(poisson[i]):
            g = _gruneisen_from_poisson(float(poisson[i]))
        else:
            g = 1.5
        gamma[i] = g

        n = len(structure)
        volume_per_atom = float(structure.volume) / n
        mean_mass = float(structure.composition.weight) / n     # amu

        # Debye velocity from the Debye temperature and the atom density.
        velocity[i] = _sound_velocity(theta, volume_per_atom)
        kappa[i] = _slack(theta, mean_mass, volume_per_atom, g, n, temperature)

    md.obs[f"debye_temperature_{level}"] = debye
    md.obs[f"gruneisen_{level}"] = gamma
    md.obs[f"sound_velocity_{level}"] = velocity
    md.obs[f"thermal_conductivity_{level}"] = kappa
    md.uns.setdefault("thermal_conductivity", {})[level] = {
        "temperature": float(temperature),
        "model": "Slack",
        "gruneisen_source": source,
        "unit": "W/m/K",
        "note": "order-of-magnitude; phono3py is the honest calculation",
    }
    record(md, "prop.thermal_conductivity", level=level,
           temperature=temperature)


#: THz -> K, for a phonon frequency expressed as a temperature.
_THZ_TO_K = 47.9924341590788

#: Slack's prefactor, in SI, for kappa in W/m/K with mass in amu, volume per
#: atom in angstrom^3 and temperatures in kelvin.
_SLACK_A = 3.1e-6


def _debye_temperature(dos: np.ndarray, grid: np.ndarray) -> float:
    """Debye temperature from the second moment of the phonon spectrum.

    The moment-based definition rather than the cutoff frequency: a real
    spectrum has a tail, and reading the highest frequency off it makes the
    answer depend on where the smearing was truncated.
    """
    weight = np.maximum(dos, 0.0)
    total = np.trapezoid(weight, grid) if hasattr(np, "trapezoid") \
        else np.trapz(weight, grid)
    if total <= 0:
        return float("nan")
    second = (np.trapezoid(weight * grid ** 2, grid) if hasattr(np, "trapezoid")
              else np.trapz(weight * grid ** 2, grid)) / total
    return float(np.sqrt(5.0 / 3.0 * second) * _THZ_TO_K)


def _gruneisen_from_poisson(nu: float) -> float:
    """Gruneisen parameter from the Poisson ratio, after Belomestnykh-Tesleva.

    An elastic proxy for anharmonicity: a material that resists shear relative
    to compression is the one whose phonons scatter least.
    """
    nu = float(np.clip(nu, -0.4, 0.49))
    return float(1.5 * (1.0 + nu) / (2.0 - 3.0 * nu))


def _sound_velocity(theta: float, volume_per_atom: float) -> float:
    """Debye sound velocity in m/s from the Debye temperature."""
    if not np.isfinite(theta) or volume_per_atom <= 0:
        return float("nan")
    # v = (k_B theta / hbar) * (V_atom / (6 pi^2))^(1/3), in SI.
    k_b, hbar = 1.380649e-23, 1.054571817e-34
    volume = volume_per_atom * 1e-30                      # m^3
    return float(k_b * theta / hbar * (volume / (6.0 * np.pi ** 2)) ** (1 / 3))


def _slack(theta: float, mean_mass: float, volume_per_atom: float,
           gamma: float, n_atoms: int, temperature: float) -> float:
    """Slack's expression for the lattice thermal conductivity.

        kappa = A * M_avg * theta^3 * delta / (gamma^2 * n^(2/3) * T)

    with ``delta`` the cube root of the volume per atom. Every dependence in it
    is the physically expected one, which is why it ranks well even where the
    magnitude is off.
    """
    if not np.isfinite(theta) or theta <= 0 or temperature <= 0 or gamma <= 0:
        return float("nan")
    delta = volume_per_atom ** (1 / 3)                    # angstrom
    return float(_SLACK_A * mean_mass * theta ** 3 * delta
                 / (gamma ** 2 * n_atoms ** (2 / 3) * temperature))


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


__all__ = ["xrd", "rdf", "elastic", "phonon", "free_energy",
           "thermal_conductivity", "compare_grids"]

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

import warnings

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


@register_function(
    aliases=["equation of state", "eos", "energy volume curve", "e-v curve",
             "birch murnaghan", "compressibility", "fit eos"],
    category="prop",
    description="Fit an equation of state to the energy-volume curve of every "
                "structure at one level of theory, giving the bulk modulus, "
                "its pressure derivative and the equilibrium volume.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["bulk_modulus_eos_{level}",
                      "bulk_modulus_derivative_{level}",
                      "equilibrium_volume_{level}",
                      "equilibrium_energy_{level}",
                      "eos_residual_{level}"],
              "obsm": ["eos_{level}"], "uns": ["grids"], "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    dispatch="level= selects the calculator, as for mv.calc.energy",
    examples=["mv.prop.eos(md, level='emt')",
              "mv.prop.eos(md, level='emt', source='relaxed_emt', "
              "model='vinet')"],
    related=["mv.prop.elastic", "mv.calc.relax", "mv.screen.filter"],
    notes="The bulk modulus this returns and the one mv.prop.elastic derives "
          "from the stiffness tensor are the same quantity by two routes — a "
          "curvature in volume against a curvature in strain — and they agree "
          "when the input is relaxed and the calculator is well behaved. When "
          "they disagree, the usual cause is that the input was not at a "
          "minimum, and the disagreement is the cheapest available warning. "
          "Both live on the object, so comparing them is a subtraction.\n\n"
          "The curve is stored against the **volume scale factor**, not against "
          "absolute volume: materials of different size have no common volume "
          "axis, and the strain series they were computed on is common by "
          "construction. obs['eos_residual'] is the RMS misfit in eV/atom; a "
          "value far above a meV is a fit that should not be read.",
)
def eos(md: AnnData, level: str = "emt", source: str = "input",
        scales=None, model: str = "birch_murnaghan",
        key_added: str | None = None) -> None:
    """Equation of state by isotropic compression. Deposits; returns ``None``."""
    from pymatgen.analysis.eos import EOS
    from pymatgen.io.ase import AseAtomsAdaptor

    from .calc import _get

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    calculator = factory()
    quantity = key_added or "eos"

    grid = (np.linspace(0.94, 1.06, 7) if scales is None
            else np.asarray(scales, dtype=float))
    if grid.ndim != 1 or grid.size < 4:
        raise ValueError(
            f"an equation of state needs at least four volume scale factors to "
            f"fit four parameters; got {grid.size}. Pass scales=np.linspace("
            f"0.94, 1.06, 7) or leave it at the default.")

    curves = np.full((md.n_obs, grid.size), np.nan)
    moduli = np.full(md.n_obs, np.nan)
    derivatives = np.full(md.n_obs, np.nan)
    volumes = np.full(md.n_obs, np.nan)
    energies0 = np.full(md.n_obs, np.nan)
    residuals = np.full(md.n_obs, np.nan)
    failed = 0

    for i, structure in enumerate(structures(md, source)):
        n = len(structure)
        try:
            sampled_v, sampled_e = [], []
            for j, scale in enumerate(grid):
                strained = structure.copy()
                strained.scale_lattice(structure.volume * float(scale))
                atoms = adaptor.get_atoms(strained)
                atoms.calc = calculator
                energy = float(atoms.get_potential_energy())
                curves[i, j] = energy / n
                sampled_v.append(strained.volume)
                sampled_e.append(energy)

            fit = EOS(eos_name=model).fit(sampled_v, sampled_e)
            moduli[i] = float(fit.b0_GPa)
            derivatives[i] = float(fit.b1)
            volumes[i] = float(fit.v0)
            energies0[i] = float(fit.e0) / n
            predicted = np.asarray(
                [fit.func(v) for v in sampled_v], dtype=float)
            residuals[i] = float(np.sqrt(np.mean(
                ((predicted - np.asarray(sampled_e)) / n) ** 2)))
        except Exception:
            failed += 1

    deposit_grid(md, quantity, level, curves, grid, unit="eV/atom",
                 grid_unit="V/V_input", model=model, source=source)
    md.obs[f"bulk_modulus_eos_{level}"] = moduli
    md.obs[f"bulk_modulus_derivative_{level}"] = derivatives
    md.obs[f"equilibrium_volume_{level}"] = volumes
    md.obs[f"equilibrium_energy_{level}"] = energies0
    md.obs[f"eos_residual_{level}"] = residuals
    set_level(md, level, **meta, source=source, eos_model=model,
              n_scales=int(grid.size), n_failed=failed)
    record(md, "prop.eos", level=level, source=source, model=model,
           n_scales=int(grid.size))


@register_function(
    aliases=["dimensionality", "is it layered", "2d material", "layered",
             "bonding dimensionality", "van der waals layered",
             "molecular crystal"],
    category="prop",
    description="Classify every structure by the dimensionality of its bonded "
                "network — 0D molecular, 1D chains, 2D layers, 3D framework — "
                "so a screen can ask for exfoliable candidates directly.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["dimensionality", "n_components", "is_layered",
                      "dimensionality_strategy"]},
    examples=["mv.prop.dimensionality(md)",
              "mv.prop.dimensionality(md, strategy='minimum_distance')",
              "mv.prop.dimensionality(md); "
              "mv.screen.filter(md, dimensionality__eq=2, name='exfoliable')"],
    related=["mv.env.bonds", "mv.env.coordination", "mv.screen.filter"],
    notes="A layered material and a framework can have identical compositions, "
          "identical space groups and nearly identical densities, so this is "
          "not derivable from X or from obs — it is a property of the bond "
          "graph, which is why it needs a near-neighbour algorithm to exist "
          "at all.\n\n"
          "Which algorithm is recorded next to the answer, for the same reason "
          "mv.env.coordination records it: the algorithms disagree, and a "
          "dimensionality without its strategy is not reproducible. The "
          "disagreement is worse here than for a coordination number, because "
          "the classification turns on whether a long contact counts as a bond "
          "at all — which is exactly the question a van der Waals gap poses.\n\n"
          "n_components counts the disconnected pieces in the unit cell, so a "
          "2D material with two layers per cell reports 2. It is one and the "
          "same number for a 3D framework, which is 1 by construction.",
)
def dimensionality(md: AnnData, source: str = "input",
                   strategy: str = "crystalnn") -> None:
    """Bonded-network dimensionality per material. Deposits; returns ``None``."""
    from pymatgen.analysis.dimensionality import (get_dimensionality_larsen,
                                                  get_structure_components)
    from pymatgen.analysis.graphs import StructureGraph

    from .env import _strategy

    finder = _strategy(strategy)
    dims = np.full(md.n_obs, -1, dtype=int)
    counts = np.full(md.n_obs, -1, dtype=int)
    failed = 0

    for i, structure in enumerate(structures(md, source)):
        try:
            graph = StructureGraph.from_local_env_strategy(structure, finder)
            dims[i] = int(get_dimensionality_larsen(graph))
            counts[i] = len(get_structure_components(graph))
        except Exception:
            failed += 1

    md.obs["dimensionality"] = dims
    md.obs["n_components"] = counts
    md.obs["is_layered"] = dims == 2
    md.obs["dimensionality_strategy"] = str(strategy)
    md.uns["dimensionality"] = {"strategy": str(strategy), "source": source,
                                "n_failed": int(failed)}
    record(md, "prop.dimensionality", source=source, strategy=str(strategy))


@register_function(
    aliases=["nmr", "chemical shielding", "chemical shift", "shielding tensor",
             "solid state nmr", "magnetic shielding", "csa"],
    category="prop",
    description="Reduce a per-atom chemical shielding tensor to the parameters "
                "a solid-state NMR spectrum is described by — isotropic shift, "
                "anisotropy, asymmetry, span and skew — on the sites axis.",
    requires={"sites.obs": ["material"]},
    produces={"sites.obs": ["shielding_iso_{level}",
                            "shielding_anisotropy_{level}",
                            "shielding_asymmetry_{level}",
                            "shielding_span_{level}",
                            "shielding_skew_{level}"],
              "sites.obsm": ["shielding_tensor_{level}"],
              "levels": ["{level}"]},
    prerequisites=["mv.multi.sites"],
    examples=["sites = mv.multi.sites(md); mv.prop.nmr(md, sites, shieldings)",
              "mv.prop.nmr(md, sites, shieldings, level='pbe')"],
    related=["mv.prop.efg", "mv.multi.sites", "mv.dft.read_outputs"],
    notes="Takes the tensors as an argument rather than reading them off the "
          "object, for the same reason mv.elec.bands takes band structures: "
          "nothing in matverse computes a shielding tensor, and the honest "
          "place for a result someone else computed is an argument.\n\n"
          "Both conventions are reported because both are in use and they "
          "disagree on what 'anisotropy' names. Haeberlen's zeta is "
          "sigma_33 - sigma_iso; the reduced anisotropy quoted by most "
          "spectrometer software is 3/2 of it. span and skew are the "
          "Herzfeld-Berger pair, which is what a sideband analysis returns.\n\n"
          "These are **shieldings**, not shifts. A chemical shift is a "
          "shielding referenced to a standard compound and runs the other way "
          "in sign; converting needs a reference this function is not given.",
)
def nmr(md: AnnData, sites: AnnData, shieldings, level: str = "dft") -> None:
    """NMR shielding parameters per atom. Deposits on ``sites``."""
    from pymatgen.analysis.nmr import ChemicalShielding

    from .env import _require_sites

    _require_sites(sites, md)
    tensors = np.asarray(shieldings, dtype=float)
    if tensors.shape != (sites.n_obs, 3, 3):
        raise ValueError(
            f"got shieldings of shape {tensors.shape} for {sites.n_obs} atoms; "
            f"expected ({sites.n_obs}, 3, 3) — one 3x3 tensor per row of the "
            f"sites object, in the order mv.multi.sites produced them")

    iso = np.full(sites.n_obs, np.nan)
    zeta = np.full(sites.n_obs, np.nan)
    eta = np.full(sites.n_obs, np.nan)
    span = np.full(sites.n_obs, np.nan)
    skew = np.full(sites.n_obs, np.nan)

    for i, tensor in enumerate(tensors):
        shielding = ChemicalShielding(tensor)
        haeberlen = shielding.haeberlen_values
        mehring = shielding.mehring_values
        s11, s22, s33 = (float(np.real(mehring.sigma_11)),
                         float(np.real(mehring.sigma_22)),
                         float(np.real(mehring.sigma_33)))
        iso[i] = float(np.real(haeberlen.sigma_iso))
        zeta[i] = float(np.real(haeberlen.zeta))
        eta[i] = float(np.real(haeberlen.eta))
        # Herzfeld-Berger: the span is the full width, the skew says which
        # side of the isotropic value the third principal component sits on.
        span[i] = s33 - s11
        skew[i] = (3.0 * (s22 - iso[i]) / span[i]
                   if abs(span[i]) > 1e-12 else 0.0)

    sites.obsm[f"shielding_tensor_{level}"] = tensors.reshape(sites.n_obs, 9)
    sites.obs[f"shielding_iso_{level}"] = iso
    sites.obs[f"shielding_anisotropy_{level}"] = zeta
    sites.obs[f"shielding_asymmetry_{level}"] = eta
    sites.obs[f"shielding_span_{level}"] = span
    sites.obs[f"shielding_skew_{level}"] = skew
    set_level(md, level, kind="dft", method="chemical shielding tensor",
              reference=None, surrogate=False, license=None, uncertainty=None,
              quantity="nmr shielding")
    record(md, "prop.nmr", level=level, n_sites=int(sites.n_obs))


@register_function(
    aliases=["efg", "electric field gradient", "quadrupolar coupling",
             "quadrupole coupling constant", "nuclear quadrupole"],
    category="prop",
    description="Reduce a per-atom electric field gradient tensor to Vzz, the "
                "asymmetry parameter and the quadrupolar coupling constant, "
                "which is what a quadrupolar NMR lineshape is set by.",
    requires={"sites.obs": ["element"]},
    produces={"sites.obs": ["efg_vzz_{level}", "efg_asymmetry_{level}",
                            "efg_coupling_{level}"],
              "sites.obsm": ["efg_tensor_{level}"],
              "levels": ["{level}"]},
    prerequisites=["mv.multi.sites"],
    examples=["sites = mv.multi.sites(md); mv.prop.efg(md, sites, gradients)",
              "mv.prop.efg(md, sites, gradients, level='pbe')"],
    related=["mv.prop.nmr", "mv.multi.sites"],
    notes="The coupling constant needs the nuclear quadrupole moment of a "
          "specific isotope, which is a property of the nucleus rather than of "
          "the calculation. It is looked up from the element on each site, so "
          "it is the most abundant NMR-active isotope; for an element with no "
          "tabulated moment the coupling is NaN while Vzz and the asymmetry "
          "are still reported, because those are properties of the gradient "
          "alone.\n\n"
          "The convention is |Vzz| >= |Vyy| >= |Vxx|, so eta lies in [0, 1] "
          "by construction and a value outside it means the tensor was not "
          "traceless.",
)
def efg(md: AnnData, sites: AnnData, gradients, level: str = "dft") -> None:
    """Electric field gradient parameters per atom. Deposits on ``sites``."""
    from pymatgen.analysis.nmr import ElectricFieldGradient

    from .env import _require_sites

    _require_sites(sites, md)
    tensors = np.asarray(gradients, dtype=float)
    if tensors.shape != (sites.n_obs, 3, 3):
        raise ValueError(
            f"got gradients of shape {tensors.shape} for {sites.n_obs} atoms; "
            f"expected ({sites.n_obs}, 3, 3) — one 3x3 tensor per row of the "
            f"sites object, in the order mv.multi.sites produced them")

    elements = sites.obs["element"].astype(str).to_numpy()
    vzz = np.full(sites.n_obs, np.nan)
    asymmetry = np.full(sites.n_obs, np.nan)
    coupling = np.full(sites.n_obs, np.nan)

    for i, tensor in enumerate(tensors):
        gradient = ElectricFieldGradient(tensor)
        vzz[i] = float(np.real(gradient.V_zz))
        asymmetry[i] = float(np.real(gradient.asymmetry))
        try:
            coupling[i] = float(np.real(
                gradient.coupling_constant(elements[i])))
        except Exception:
            coupling[i] = np.nan       # no tabulated quadrupole moment

    sites.obsm[f"efg_tensor_{level}"] = tensors.reshape(sites.n_obs, 9)
    sites.obs[f"efg_vzz_{level}"] = vzz
    sites.obs[f"efg_asymmetry_{level}"] = asymmetry
    sites.obs[f"efg_coupling_{level}"] = coupling
    set_level(md, level, kind="dft", method="electric field gradient",
              reference=None, surrogate=False, license=None, uncertainty=None,
              quantity="efg")
    record(md, "prop.efg", level=level, n_sites=int(sites.n_obs))


@register_function(
    aliases=["piezoelectric", "piezo", "piezoelectric tensor", "d33",
             "piezoelectric coefficient", "electromechanical"],
    category="prop",
    description="Check a piezoelectric tensor against the crystal symmetry, "
                "put it in the IEEE frame, and derive the largest longitudinal "
                "response over all directions for screening.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["piezo_max_longitudinal_{level}",
                      "piezo_max_direction_{level}",
                      "piezo_symmetry_valid_{level}"],
              "obsm": ["piezo_tensor_{level}"], "levels": ["{level}"]},
    examples=["mv.prop.piezoelectric(md, tensors)",
              "mv.prop.piezoelectric(md, tensors, level='pbe')"],
    related=["mv.prop.elastic", "mv.screen.rank", "mv.dft.read_outputs"],
    notes="A piezoelectric tensor is forbidden by symmetry in any "
          "centrosymmetric crystal, so a non-zero one there is an error in the "
          "calculation or in the structure it was paired with rather than a "
          "discovery. piezo_symmetry_valid is pymatgen's check of the tensor "
          "against the structure's point group, and it is worth reading before "
          "the magnitude.\n\n"
          "The screening number is the largest **longitudinal** response — "
          "the maximum of d_ijk n_i n_j n_k over unit vectors n — because that "
          "is the quantity a stack actuator or a single-crystal transducer is "
          "built around, and it is invariant to how the tensor was oriented. "
          "It is found by sampling directions rather than solved in closed "
          "form, so it is a lower bound that tightens with n_directions.",
)
def piezoelectric(md: AnnData, tensors, level: str = "dft",
                  source: str = "input", n_directions: int = 2000,
                  tolerance: float = 1e-3) -> None:
    """Piezoelectric response per material. Deposits; returns ``None``."""
    from pymatgen.analysis.piezo import PiezoTensor

    array = np.asarray(tensors, dtype=float)
    if array.shape == (md.n_obs, 3, 6):
        array = np.stack([_from_voigt(v) for v in array])
    if array.shape != (md.n_obs, 3, 3, 3):
        raise ValueError(
            f"got tensors of shape {array.shape} for {md.n_obs} materials; "
            f"expected ({md.n_obs}, 3, 3, 3) in full notation or "
            f"({md.n_obs}, 3, 6) in Voigt notation")

    directions = _fibonacci_sphere(int(n_directions))
    longitudinal = np.full(md.n_obs, np.nan)
    best = np.empty(md.n_obs, dtype=object)
    valid = np.zeros(md.n_obs, dtype=bool)
    ieee = np.full((md.n_obs, 3, 3, 3), np.nan)

    for i, (structure, tensor) in enumerate(zip(structures(md, source), array)):
        try:
            piezo = PiezoTensor(tensor)
            valid[i] = bool(piezo.is_fit_to_structure(structure,
                                                      tol=tolerance))
            try:
                ieee[i] = np.asarray(piezo.convert_to_ieee(structure),
                                     dtype=float)
            except Exception:
                ieee[i] = tensor
        except Exception:
            ieee[i] = tensor
        # d(n) = d_ijk n_i n_j n_k, maximised over the sampled directions.
        response = np.einsum("ijk,ni,nj,nk->n", ieee[i], directions,
                             directions, directions)
        winner = int(np.argmax(np.abs(response)))
        longitudinal[i] = float(np.abs(response[winner]))
        best[i] = ",".join(f"{c:.3f}" for c in directions[winner])

    md.obsm[f"piezo_tensor_{level}"] = ieee.reshape(md.n_obs, 27)
    md.obs[f"piezo_max_longitudinal_{level}"] = longitudinal
    md.obs[f"piezo_max_direction_{level}"] = best
    md.obs[f"piezo_symmetry_valid_{level}"] = valid
    set_level(md, level, kind="dft", method="piezoelectric tensor",
              reference=None, surrogate=False, license=None, uncertainty=None,
              source=source, n_directions=int(n_directions))
    record(md, "prop.piezoelectric", level=level, source=source,
           n_directions=int(n_directions))


#: Voigt index -> the pair of Cartesian indices it stands for.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _from_voigt(voigt: np.ndarray) -> np.ndarray:
    """A 3x6 piezoelectric matrix as the full 3x3x3 tensor."""
    full = np.zeros((3, 3, 3))
    for i in range(3):
        for v, (j, k) in enumerate(_VOIGT_PAIRS):
            # The shear columns carry a factor of two in the Voigt convention
            # for strain-like second indices, split between the two symmetric
            # positions so the full tensor stays symmetric in j and k.
            value = voigt[i, v] / (1.0 if v < 3 else 2.0)
            full[i, j, k] = value
            full[i, k, j] = value
    return full


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniformly spaced unit vectors, deterministically."""
    n = max(int(n), 1)
    index = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


@register_function(
    aliases=["dielectric function", "optical absorption", "absorption "
             "coefficient", "refractive index", "optical properties",
             "epsilon"],
    category="prop",
    description="Deposit a frequency-dependent dielectric function and derive "
                "the refractive index and absorption coefficient from it, on "
                "the shared energy grid the rest of the object uses for curves.",
    produces={"obsm": ["dielectric_real_{level}", "dielectric_imag_{level}",
                       "absorption_{level}", "extinction_{level}"],
              "obs": ["static_dielectric_{level}",
                      "refractive_index_{level}"],
              "uns": ["grids"], "levels": ["{level}"]},
    examples=["mv.prop.dielectric(md, energies, eps_real, eps_imag)",
              "mv.prop.dielectric(md, energies, eps1, eps2, level='pbe')"],
    related=["mv.prop.slme", "mv.exp.attach", "mv.dft.read_outputs"],
    notes="The absorption coefficient is derived rather than taken, because "
          "alpha = 2 E k / (hbar c) is a definition and a dielectric function "
          "that has been through it twice is a dielectric function nobody can "
          "reconstruct. Storing epsilon and deriving alpha keeps the input "
          "recoverable.\n\n"
          "alpha is in m^-1, which is the unit mv.prop.slme expects and a "
          "factor of 100 away from the cm^-1 that optics papers quote. The "
          "static dielectric constant is epsilon_1 at the lowest energy on the "
          "grid, so it is only the true zero-frequency limit if the grid "
          "starts near zero.",
)
def dielectric(md: AnnData, energies, real, imag, level: str = "dft") -> None:
    """Dielectric function and what follows from it. Deposits; returns ``None``."""
    grid = np.asarray(energies, dtype=float)
    eps1 = np.atleast_2d(np.asarray(real, dtype=float))
    eps2 = np.atleast_2d(np.asarray(imag, dtype=float))
    for name, block in (("real", eps1), ("imag", eps2)):
        if block.shape != (md.n_obs, grid.size):
            raise ValueError(
                f"got {name} part of shape {block.shape}; expected "
                f"({md.n_obs}, {grid.size}) — one row per material on the "
                f"energy grid given")

    modulus = np.sqrt(eps1 ** 2 + eps2 ** 2)
    n = np.sqrt(np.maximum(modulus + eps1, 0.0) / 2.0)
    k = np.sqrt(np.maximum(modulus - eps1, 0.0) / 2.0)

    # alpha = 2 omega k / c, with omega = E / hbar and E in joules.
    joules = grid * _ELEMENTARY_CHARGE
    alpha = 2.0 * (joules / _HBAR) * k / _SPEED_OF_LIGHT

    deposit_grid(md, "dielectric_real", level, eps1, grid, unit="eV")
    deposit_grid(md, "dielectric_imag", level, eps2, grid, unit="eV")
    deposit_grid(md, "extinction", level, k, grid, unit="eV")
    deposit_grid(md, "absorption", level, alpha, grid, unit="eV",
                 value_unit="m^-1")
    md.obs[f"static_dielectric_{level}"] = eps1[:, 0]
    md.obs[f"refractive_index_{level}"] = n[:, 0]
    set_level(md, level, kind="dft", method="dielectric function",
              reference=None, surrogate=False, license=None, uncertainty=None,
              quantity="optics")
    record(md, "prop.dielectric", level=level, n_points=int(grid.size))


#: Physical constants, SI, CODATA 2018.
_ELEMENTARY_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_SPEED_OF_LIGHT = 299792458.0


@register_function(
    aliases=["slme", "solar efficiency", "photovoltaic efficiency",
             "spectroscopic limited maximum efficiency", "solar absorber",
             "shockley queisser"],
    category="prop",
    description="Spectroscopic limited maximum efficiency: the ceiling on a "
                "single-junction solar cell made of each material, from its "
                "absorption spectrum and its gap under the AM1.5G spectrum.",
    requires={"obsm": ["absorption_{level}"], "uns": ["grids"],
              "obs": ["band_gap_{level}"]},
    produces={"obs": ["slme_{level}", "sq_limit_{level}"]},
    prerequisites=["mv.prop.dielectric"],
    examples=["mv.prop.slme(md, level='pbe')",
              "mv.prop.slme(md, level='pbe', thickness=1e-6)"],
    related=["mv.prop.dielectric", "mv.screen.rank", "mv.elec.band_features"],
    notes="Returned as a **percentage**, matching how cell efficiencies are "
          "quoted, and the unit is recorded so mv.utils.check_units can say so. "
          "The Shockley-Queisser limit for the same gap is reported next to it: "
          "SLME is always the smaller of the two, and the gap between them is "
          "what the material's own absorption costs it. A material with a "
          "perfect step absorption edge and no indirect gap reproduces "
          "Shockley-Queisser exactly, peaking near 33% at 1.34 eV.\n\n"
          "indirect_key names the column holding the indirect gap. Left unset, "
          "the direct gap is used for both, which sets the radiative fraction "
          "to one — that is the Shockley-Queisser assumption and it is "
          "optimistic for any real indirect semiconductor, silicon most of all.",
)
def slme(md: AnnData, level: str = "dft", thickness: float = 5e-7,
         temperature: float = 293.15, indirect_key: str | None = None) -> None:
    """Spectroscopic limited maximum efficiency, in percent. Deposits."""
    from pymatgen.analysis.solar.slme import slme as _slme

    from ._core import grid_of

    block = f"absorption_{level}"
    if block not in md.obsm:
        raise ValueError(
            f"obsm[{block!r}] absent; run mv.prop.dielectric(md, energies, "
            f"eps1, eps2, level={level!r}) first, which derives the absorption "
            f"coefficient from the dielectric function")
    gap_key = f"band_gap_{level}"
    if gap_key not in md.obs:
        raise ValueError(
            f"obs[{gap_key!r}] absent; a solar efficiency needs a gap. "
            f"mv.elec.band_features or mv.dft.read_dos deposits one.")

    grid = grid_of(md, "absorption")
    alpha = np.asarray(md.obsm[block], dtype=float)
    direct = md.obs[gap_key].to_numpy(dtype=float)
    indirect = (md.obs[indirect_key].to_numpy(dtype=float)
                if indirect_key else direct)

    efficiency = np.full(md.n_obs, np.nan)
    ceiling = np.full(md.n_obs, np.nan)
    for i in range(md.n_obs):
        if not np.isfinite(direct[i]) or direct[i] <= 0:
            continue
        try:
            efficiency[i] = float(_slme(grid, alpha[i], direct[i], indirect[i],
                                        thickness=thickness,
                                        temperature=temperature))
            perfect = np.where(grid >= direct[i], 1e8, 0.0)
            ceiling[i] = float(_slme(grid, perfect, direct[i], direct[i],
                                     thickness=thickness,
                                     temperature=temperature))
        except Exception:
            continue

    md.obs[f"slme_{level}"] = efficiency
    md.obs[f"sq_limit_{level}"] = ceiling
    md.uns.setdefault("units", {})[f"slme_{level}"] = "percent"
    md.uns["units"][f"sq_limit_{level}"] = "percent"
    record(md, "prop.slme", level=level, thickness=thickness,
           temperature=temperature)


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


__all__ = ["xrd", "rdf", "neutron", "tem", "elastic", "eos", "dimensionality",
           "phonon", "free_energy", "quasiharmonic",
           "nmr", "efg", "piezoelectric", "dielectric", "slme",
           "cost", "supply_risk",
           "thermal_conductivity", "compare_grids"]


@register_function(
    aliases=["cost", "material cost", "price", "how much does it cost",
             "raw material price", "economic screening"],
    category="prop",
    description="Raw-material cost per kilogram and per mole from elemental "
                "prices, so a screen can rank on what a candidate would cost "
                "to make rather than only on what it would do.",
    requires={"X": ["composition"]},
    produces={"obs": ["cost_per_kg", "cost_per_mol"]},
    examples=["mv.prop.cost(md)",
              "mv.prop.cost(md); mv.screen.filter(md, cost_per_kg__lt=50)"],
    related=["mv.prop.supply_risk", "mv.screen.filter", "mv.screen.pareto"],
    notes="Elemental prices only — the cost of the elements in the formula, "
          "not of the synthesis, the processing or the yield. A material made "
          "of cheap elements by an expensive route will look cheap here. What "
          "it does catch is the case that ends a project: platinum oxide "
          "comes out around 42,000 $/kg against iron oxide's 0.92, and five "
          "orders of magnitude is not a number any amount of process "
          "optimisation closes.\n\n"
          "The prices are pymatgen's table, which is a snapshot rather than a "
          "feed. Use it to separate the affordable from the impossible, not to "
          "quote a budget.",
)
def cost(md: AnnData, source: str = "input") -> None:
    """Raw-material cost per material. Deposits; returns ``None``."""
    from pymatgen.analysis.cost import CostAnalyzer, CostDBElements

    try:
        analyzer = CostAnalyzer(CostDBElements())
        analyzer.get_cost_per_kg("Fe")            # forces the table to load
    except ImportError as exc:
        raise ImportError(
            f"mv.prop.cost needs bibtexparser, which pymatgen uses to read the "
            f"citations in its elemental price table and does not require "
            f"itself. Install it with `pip install bibtexparser`. ({exc})"
        ) from exc
    per_kg = np.full(md.n_obs, np.nan)
    per_mol = np.full(md.n_obs, np.nan)
    failed = 0

    for i, structure in enumerate(structures(md, source)):
        formula = structure.composition.reduced_formula
        try:
            per_kg[i] = float(analyzer.get_cost_per_kg(formula))
            per_mol[i] = float(analyzer.get_cost_per_mol(formula))
        except Exception:
            failed += 1

    md.obs["cost_per_kg"] = per_kg
    md.obs["cost_per_mol"] = per_mol
    md.uns.setdefault("units", {})["cost_per_kg"] = "USD/kg"
    md.uns["units"]["cost_per_mol"] = "USD/mol"
    md.uns["cost"] = {"source": source, "database": "pymatgen elemental",
                      "n_failed": int(failed)}
    record(md, "prop.cost", source=source)


@register_function(
    aliases=["supply risk", "hhi", "criticality", "herfindahl",
             "element criticality", "supply chain"],
    category="prop",
    description="Herfindahl-Hirschman indices for the elements in each "
                "material, measuring how concentrated their production and "
                "their reserves are in a few countries.",
    requires={"X": ["composition"]},
    produces={"obs": ["hhi_production", "hhi_reserve", "supply_risk"]},
    examples=["mv.prop.supply_risk(md)",
              "mv.prop.supply_risk(md); "
              "mv.screen.filter(md, hhi_reserve__lt=4000)"],
    related=["mv.prop.cost", "mv.screen.filter"],
    notes="A high index means a few countries hold most of the world's "
          "production or reserves, which is a different risk from being "
          "expensive: cobalt and the rare earths are affordable and "
          "concentrated, and that is what makes them awkward. The US "
          "Department of Justice reads above 2500 as concentrated and above "
          "1500 as moderately so; supply_risk carries pymatgen's own "
          "designation rather than a threshold chosen here.\n\n"
          "Production and reserve indices differ and both are reported, "
          "because they answer different questions — production is who makes "
          "it now, reserves are who could.",
)
def supply_risk(md: AnnData, source: str = "input") -> None:
    """Supply-concentration indices per material. Deposits; returns ``None``."""
    from pymatgen.analysis.hhi import HHIModel

    model = HHIModel()
    production = np.full(md.n_obs, np.nan)
    reserve = np.full(md.n_obs, np.nan)
    designation = np.empty(md.n_obs, dtype=object)
    failed = 0

    for i, structure in enumerate(structures(md, source)):
        formula = structure.composition.reduced_formula
        designation[i] = ""
        try:
            production[i] = float(model.get_hhi_production(formula))
            reserve[i] = float(model.get_hhi_reserve(formula))
            designation[i] = str(model.get_hhi_designation(production[i]))
        except Exception:
            failed += 1

    md.obs["hhi_production"] = production
    md.obs["hhi_reserve"] = reserve
    md.obs["supply_risk"] = designation.astype(str)
    md.uns["supply_risk"] = {"source": source, "n_failed": int(failed)}
    record(md, "prop.supply_risk", source=source)


@register_function(
    aliases=["neutron diffraction", "neutron pattern", "nd pattern",
             "simulate neutron", "powder neutron"],
    category="prop",
    description="Simulate a powder neutron diffraction pattern on the same "
                "shared grid convention as the X-ray one, so the two can be "
                "compared or fitted together.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["neutron_{level}"], "uns": ["grids"],
              "levels": ["{level}"]},
    examples=["mv.prop.neutron(md)",
              "mv.prop.neutron(md, two_theta=(10, 120), wavelength=1.54)"],
    related=["mv.prop.xrd", "mv.exp.attach", "mv.prop.compare_grids"],
    notes="Neutrons scatter off nuclei rather than off electrons, so the two "
          "patterns are not redundant. X-rays barely see hydrogen or lithium "
          "next to a transition metal and cannot tell neighbouring elements "
          "apart at all; neutron scattering lengths do not follow atomic "
          "number, so light atoms and Mn/Fe ordering show up here and nowhere "
          "else. For a lithium battery cathode this is the pattern that "
          "locates the lithium.",
)
def neutron(md: AnnData, source: str = "input", level: str = "calc",
            wavelength: float = 1.54184, two_theta: tuple = (5.0, 90.0),
            step: float = 0.02, fwhm: float = 0.1,
            normalize: bool = True) -> None:
    """Powder neutron diffraction patterns on a shared two-theta grid."""
    from pymatgen.analysis.diffraction.neutron import NDCalculator

    grid = np.arange(two_theta[0], two_theta[1] + step / 2, step)
    sigma = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    calculator = NDCalculator(wavelength=wavelength)

    rows, failed = [], 0
    for structure in structures(md, source):
        try:
            pattern = calculator.get_pattern(structure,
                                             two_theta_range=two_theta)
            rows.append(_broaden(np.asarray(pattern.x, dtype=float),
                                 np.asarray(pattern.y, dtype=float),
                                 grid, sigma, normalize))
        except Exception:
            rows.append(np.full(len(grid), np.nan))
            failed += 1

    deposit_grid(md, "neutron", level, np.vstack(rows), grid,
                 unit="degrees 2theta", wavelength=wavelength, fwhm=fwhm,
                 normalized=bool(normalize))
    set_level(md, level, kind="model", method=f"neutron ({wavelength} A)",
              reference=None, surrogate=False, license=None, uncertainty=None,
              source=source, n_failed=failed)
    record(md, "prop.neutron", source=source, level=level,
           wavelength=wavelength, fwhm=fwhm)


@register_function(
    aliases=["quasiharmonic", "qha", "quasiharmonic expansion",
             "thermal expansion from an equation of state",
             "gibbs free energy", "debye model", "gruneisen",
             "finite temperature volume"],
    category="prop",
    description="Quasi-harmonic Debye model over an energy-volume curve, "
                "giving the Gibbs free energy, thermal expansion and Gruneisen "
                "parameter as functions of temperature.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["thermal_expansion_qha_{level}",
                      "gruneisen_{level}", "debye_temperature_qha_{level}",
                      "heat_capacity_300K_{level}"],
              "obsm": ["gibbs_{level}", "thermal_expansion_{level}"],
              "uns": ["grids"], "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    dispatch="level= selects the calculator, as for mv.calc.energy",
    examples=["mv.prop.quasiharmonic(md, level='emt', source='relaxed_emt')",
              "mv.prop.quasiharmonic(md, level='emt', t_max=1200)"],
    related=["mv.prop.eos", "mv.prop.phonon", "mv.prop.free_energy"],
    notes="A 0 K hull is a hull at 0 K. The quasi-harmonic approximation is "
          "the cheapest way to move off it: compute the energy at several "
          "volumes, let the Debye model supply the vibrational free energy at "
          "each, and minimise the Gibbs free energy over volume at every "
          "temperature. The cell expands because the minimum moves, which is "
          "what thermal expansion is.\n\n"
          "The Debye model is a coarse phonon spectrum — one sound velocity "
          "standing in for the whole dispersion — so this is the right tool "
          "for a trend across candidates and the wrong one for a number to "
          "quote. mv.prop.phonon and mv.prop.free_energy do the harmonic "
          "calculation properly at one volume; this does a crude one at many, "
          "and the volume dependence is the part harmonic theory cannot give "
          "you at all.\n\n"
          "**The expansion is computed from the thermodynamic identity** "
          "alpha = gamma C_V / (B V), not from where pymatgen's model puts the "
          "volume minimum. The two disagree by a factor of twelve, and the "
          "identity is the one that is right: with the model's own Gruneisen "
          "parameter and the bulk modulus fitted from the same E(V) points, "
          "copper comes out at 4.5e-5 /K against a measured 5.0e-5, silver at "
          "5.1e-5 against 5.7e-5, while the model's own optimum_volumes give "
          "4.3e-6 for copper. gamma is right and B is right, so a volume "
          "minimisation inconsistent with both is the part to discard. C_V is "
          "the Debye heat capacity rather than the Dulong-Petit constant, so "
          "the expansion falls off correctly below the Debye temperature.\n\n"
          "That comparison was only available because the bulk modulus from "
          "mv.prop.eos and the Gruneisen parameter from this model sit on the "
          "same object under names that say what they are.\n\n"
          "The bare alias 'thermal expansion' belongs to mv.md.sweep, which "
          "measures it by running the dynamics at several temperatures and so "
          "carries the anharmonicity this model approximates away. Both "
          "compute the same quantity by different routes and the registry "
          "cannot give one name to two functions, so the direct measurement "
          "keeps the plain name and this one is reached as 'quasiharmonic'.",
)
def quasiharmonic(md: AnnData, level: str = "emt", source: str = "input",
                  scales=None, t_min: float = 300.0, t_max: float = 1000.0,
                  t_step: float = 100.0, eos_model: str = "vinet",
                  poisson: float = 0.25) -> None:
    """Quasi-harmonic thermodynamics from an E(V) curve. Deposits."""
    from pymatgen.analysis.eos import EOS
    from pymatgen.analysis.quasiharmonic import QuasiharmonicDebyeApprox
    from pymatgen.io.ase import AseAtomsAdaptor

    from .calc import _get

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    calculator = factory()

    fractions = (np.linspace(0.94, 1.06, 7) if scales is None
                 else np.asarray(scales, dtype=float))
    if fractions.ndim != 1 or fractions.size < 5:
        raise ValueError(
            f"the Debye model is fitted through an equation of state, which "
            f"needs at least five volume points; got {fractions.size}")
    grid = np.arange(t_min, t_max + t_step / 2, t_step)

    gibbs = np.full((md.n_obs, grid.size), np.nan)
    alpha_of_t = np.full((md.n_obs, grid.size), np.nan)
    expansion = np.full(md.n_obs, np.nan)
    gruneisen = np.full(md.n_obs, np.nan)
    debye = np.full(md.n_obs, np.nan)
    heat_capacity = np.full(md.n_obs, np.nan)
    failed = 0

    for i, structure in enumerate(structures(md, source)):
        try:
            energies, volumes = [], []
            for scale in fractions:
                strained = structure.copy()
                strained.scale_lattice(structure.volume * float(scale))
                atoms = adaptor.get_atoms(strained)
                atoms.calc = calculator
                energies.append(float(atoms.get_potential_energy()))
                volumes.append(float(strained.volume))

            model = QuasiharmonicDebyeApprox(
                energies, volumes, structure, t_min=float(grid[0]),
                t_step=float(t_step), t_max=float(grid[-1]),
                eos=eos_model, poisson=poisson)
            summary = model.get_summary_dict()
            temperatures = np.asarray(summary["temperatures"], dtype=float)

            gibbs[i] = np.interp(
                grid, temperatures,
                np.asarray(summary["gibbs_free_energy"], dtype=float))
            gamma = float(np.mean(summary["gruneisen_parameter"]))
            theta = float(np.mean(summary["debye_temperature"]))
            gruneisen[i] = gamma
            debye[i] = theta

            # Bulk modulus from the same E(V) points, so every term in the
            # expansion comes from one curve.
            fit = EOS(eos_name=eos_model).fit(volumes, energies)
            bulk_pa = float(fit.b0_GPa) * 1e9
            volume_m3 = float(fit.v0) * 1e-30

            capacity = _debye_heat_capacity(grid, theta, len(structure))
            heat_capacity[i] = float(np.interp(300.0, grid, capacity))
            alpha_of_t[i] = gamma * capacity / (bulk_pa * volume_m3)
            expansion[i] = float(np.interp(300.0, grid, alpha_of_t[i]))
        except Exception:
            failed += 1

    deposit_grid(md, "gibbs", level, gibbs, grid, unit="K", value_unit="eV")
    deposit_grid(md, "thermal_expansion", level, alpha_of_t, grid, unit="K",
                 value_unit="1/K")
    md.obs[f"thermal_expansion_qha_{level}"] = expansion
    md.obs[f"gruneisen_{level}"] = gruneisen
    md.obs[f"debye_temperature_qha_{level}"] = debye
    md.obs[f"heat_capacity_300K_{level}"] = heat_capacity
    set_level(md, level, **meta, source=source, eos_model=eos_model,
              poisson=poisson, n_failed=failed)
    record(md, "prop.quasiharmonic", level=level, source=source,
           t_min=t_min, t_max=t_max, eos_model=eos_model)


#: Boltzmann's constant, J/K.
_K_B = 1.380649e-23


def _debye_heat_capacity(temperatures: np.ndarray, theta: float,
                         n_atoms: int) -> np.ndarray:
    """Debye heat capacity at constant volume, J/K per cell.

    Nine n k_B (T/theta)^3 times the Debye integral, evaluated by quadrature.
    Tends to the Dulong-Petit value 3 n k_B well above the Debye temperature
    and falls as T^3 well below it, which is the part a constant would miss.
    """
    out = np.zeros_like(np.asarray(temperatures, dtype=float))
    if not np.isfinite(theta) or theta <= 0:
        return out
    for i, temperature in enumerate(np.atleast_1d(temperatures)):
        if temperature <= 0:
            continue
        upper = theta / float(temperature)
        x = np.linspace(1e-8, upper, 512)
        integrand = x ** 4 * np.exp(x) / np.expm1(x) ** 2
        integral = float(np.trapezoid(integrand, x))
        out[i] = 9.0 * n_atoms * _K_B * (float(temperature) / theta) ** 3 \
            * integral
    return out


@register_function(
    aliases=["tem", "electron diffraction", "saed", "selected area diffraction",
             "tem pattern", "zone axis"],
    category="prop",
    description="Simulate selected-area electron diffraction: the reflections "
                "excited along a zone axis, reduced to a ring profile on the "
                "shared grid convention.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["tem_{level}"], "uns": ["grids"],
              "obs": ["tem_n_reflections_{level}", "tem_strongest_{level}",
                      "tem_zone_axis"],
              "levels": ["{level}"]},
    examples=["mv.prop.tem(md)",
              "mv.prop.tem(md, beam_direction=(1, 1, 1), voltage=300)"],
    related=["mv.prop.xrd", "mv.prop.neutron", "mv.prop.compare_grids"],
    notes="A TEM pattern is spots on a plane, not a curve, and a plane of "
          "spots is not a shape that compares across materials — two crystals "
          "on the same zone axis have different spots in different places. "
          "What is stored is the **ring profile**: intensity against the "
          "magnitude of the scattering vector, which is what a "
          "polycrystalline selected-area pattern actually looks like and what "
          "can be put on one axis with an X-ray or neutron pattern.\\n\\n"
          "obs keeps the two facts the reduction loses: how many reflections "
          "were excited, and the strongest one's Miller indices. The zone axis "
          "is recorded because the pattern is meaningless without it — the "
          "same crystal down [001] and [111] gives different patterns, and "
          "that is the point of choosing an axis.\\n\\n"
          "Electron scattering factors are much larger than X-ray ones and "
          "multiple scattering is the rule rather than the exception, so a "
          "kinematic intensity like this one orders reflections correctly and "
          "should not be fitted against.",
)
def tem(md: AnnData, source: str = "input", level: str = "calc",
        beam_direction: tuple = (0, 0, 1), voltage: float = 200.0,
        r_max: float = 2.0, step: float = 0.01, sigma: float = 0.02) -> None:
    """Electron diffraction ring profiles on a shared grid. Deposits."""
    from pymatgen.analysis.diffraction.tem import TEMCalculator

    grid = np.arange(step, r_max + step / 2, step)
    calculator = TEMCalculator(voltage=voltage,
                               beam_direction=tuple(beam_direction))
    axis = ",".join(str(int(x)) for x in beam_direction)

    rows = []
    counts = np.full(md.n_obs, np.nan)
    strongest = np.empty(md.n_obs, dtype=object)
    reasons: list[str] = []
    failed = 0

    for i, structure in enumerate(structures(md, source)):
        strongest[i] = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pattern = calculator.get_pattern(structure)
            intensity = np.asarray(
                pattern["Intensity (norm)"], dtype=float)
            spacing = np.asarray(
                pattern["Interplanar Spacing"], dtype=float)
            keep = np.isfinite(intensity) & np.isfinite(spacing) & \
                (spacing > 0) & (intensity > 1e-6 * max(intensity.max(), 1e-30))
            g = 1.0 / spacing[keep]
            weight = intensity[keep]
            counts[i] = int(keep.sum())
            if keep.any():
                best = np.argmax(weight)
                strongest[i] = str(pattern["(hkl)"].to_numpy()[keep][best])
            rows.append(_broaden(g, weight, grid, sigma, True))
        except Exception as exc:
            reasons.append(f"{structure.composition.reduced_formula}: "
                           f"{type(exc).__name__}: {exc}".split("\n")[0])
            rows.append(np.full(len(grid), np.nan))
            failed += 1

    deposit_grid(md, "tem", level, np.vstack(rows), grid,
                 unit="1/angstrom", beam_direction=axis, voltage=voltage,
                 normalized=True)
    md.obs[f"tem_n_reflections_{level}"] = counts
    md.obs[f"tem_strongest_{level}"] = strongest.astype(str)
    md.obs["tem_zone_axis"] = axis
    set_level(md, level, kind="model", method=f"electron diffraction ({voltage} kV)",
              reference=None, surrogate=False, license=None, uncertainty=None,
              source=source, beam_direction=axis, n_failed=failed,
              failures=reasons[:5])
    record(md, "prop.tem", source=source, level=level,
           beam_direction=axis, voltage=voltage)

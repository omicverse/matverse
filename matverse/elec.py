"""``mv.elec`` — electronic structure.

A band structure is the answer to "what can the electrons do", and almost every
functional property — whether a material conducts, absorbs light, carries a
thermoelectric current, or catalyses anything — is downstream of it. It was also
the largest hole in matverse: ``mv.dft`` could read a density of states and
nothing else.

The shape question is the interesting one. A band structure is *bands x
k-points* per material, ragged in the number of bands. That is not one number
per material and it is not one number per atom, so it needs its own axis — and
once you say that, the AnnData shape is obvious:

===========================  =========================================
one row                       one band of one material, at one spin
``X``                         its energy along the k-path
``var``                       the k-points, with labels and distances
``obs['material']``           the foreign key back to the parent
===========================  =========================================

which is *bands x k-points*, structurally a cells x genes matrix again. Two
band structures become comparable because they share a normalised path
coordinate, exactly as two diffraction patterns become comparable by sharing a
2-theta grid.

```python
kpath  = mv.elec.kpath(md)                       # high-symmetry path per material
bands  = mv.elec.read_bands(md, 'runs/', level='pbe')
mv.elec.band_features(bands, md, level='pbe')    # -> obs['band_gap_pbe'], ...
mv.elec.transport(md, level='pbe')               # needs BoltzTraP2
```

Scalars derived from the bands — the gap, the valence-band maximum, whether the
gap is direct — land back on the material axis where a screen can reach them,
which is the same move ``mv.multi.aggregate`` makes for per-atom data.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import AXIS_KEY, deposit_grid, record, set_level, structures
from ._registry import register_function

__all_axis__ = AXIS_KEY   # re-exported; the definition lives in _core

#: Conventions for choosing the high-symmetry path.
PATH_TYPES = {
    "setyawan_curtarolo": "Setyawan and Curtarolo (2010) — the paths most "
                          "published band structures use",
    "hinuma": "Hinuma et al. (2017), via seekpath — handles more Bravais "
              "lattice edge cases correctly",
    "latimer_munro": "Latimer and Munro — derived from the point group rather "
                     "than tabulated",
}


@register_function(
    aliases=["k path", "high symmetry path", "band path", "brillouin zone "
             "path", "kpoints for bands"],
    category="elec",
    description="The high-symmetry k-path through the Brillouin zone for each "
                "material, which is what a band structure is plotted along.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["kpath_labels", "n_kpoints", "kpath_type"],
              "obsm": ["kpath"]},
    examples=["mv.elec.kpath(md)",
              "mv.elec.kpath(md, path_type='hinuma', line_density=40)"],
    related=["mv.elec.read_bands", "mv.elec.bands"],
    notes="The path depends on the Bravais lattice, so two materials in a "
          "dataset generally get different paths and different numbers of "
          "k-points. That raggedness is why the path goes to obsm as JSON "
          "rather than to a shared grid — and why the bands object resamples "
          "onto a normalised path coordinate before it can be a matrix.\n\n"
          "Which convention produced the path is recorded. Setyawan-Curtarolo "
          "and Hinuma disagree on several Bravais lattices, so a band "
          "structure whose path convention is unstated is not reproducible.",
)
def kpath(md: AnnData, source: str = "input",
          path_type: str = "setyawan_curtarolo",
          line_density: int = 20) -> None:
    """High-symmetry path per material. Deposits; returns ``None``."""
    from pymatgen.symmetry.bandstructure import HighSymmKpath

    if path_type not in PATH_TYPES:
        raise ValueError(f"unknown path_type {path_type!r}; known: "
                         f"{sorted(PATH_TYPES)}")

    encoded, labels, counts, failures = [], [], [], []
    for i, structure in enumerate(structures(md, source)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                finder = HighSymmKpath(structure, path_type=path_type)
                points, point_labels = finder.get_kpoints(
                    line_density=line_density, coords_are_cartesian=False)
            encoded.append(json.dumps({
                "kpoints": np.asarray(points, dtype=float).tolist(),
                "labels": list(point_labels),
                "path": finder.kpath["path"],
            }))
            named = [x for x in point_labels if x]
            labels.append(" - ".join(dict.fromkeys(named)))
            counts.append(len(points))
        except Exception as exc:
            failures.append(f"{i}: {type(exc).__name__}: {exc}")
            encoded.append("")
            labels.append("")
            counts.append(0)

    md.obsm["kpath"] = pd.DataFrame({"path": encoded}, index=md.obs_names)
    md.obs["kpath_labels"] = labels
    md.obs["n_kpoints"] = counts
    md.obs["kpath_type"] = str(path_type)
    md.uns["kpath"] = {"path_type": str(path_type),
                       "convention": PATH_TYPES[path_type],
                       "line_density": int(line_density),
                       "source": source, "n_failed": len(failures),
                       "errors": failures[:10]}
    record(md, "elec.kpath", source=source, path_type=path_type,
           line_density=line_density)


def _resample(energies: np.ndarray, n_points: int) -> np.ndarray:
    """Put a band onto a normalised path coordinate of fixed length."""
    original = np.linspace(0.0, 1.0, len(energies))
    target = np.linspace(0.0, 1.0, n_points)
    return np.interp(target, original, energies)


@register_function(
    aliases=["band structure", "attach bands", "bands axis", "electronic "
             "bands", "band energies"],
    category="elec",
    description="Build the bands axis from pymatgen BandStructure objects — "
                "one row per band per spin per material, energies along a "
                "shared normalised k-path coordinate.",
    produces={"obs": ["material", "spin", "band_index", "band_minimum",
                      "band_maximum", "band_width", "crosses_fermi"]},
    examples=["bands = mv.elec.bands(md, structures_by_row)",
              "bands = mv.elec.bands(md, bs_list, level='pbe', n_points=300)"],
    related=["mv.elec.read_bands", "mv.elec.band_features", "mv.elec.kpath"],
    notes="Energies are stored **relative to the Fermi level**, because an "
          "absolute eigenvalue is meaningless across codes and even across "
          "runs of one code. Zero is the Fermi level everywhere on this axis.\n\n"
          "Bands are resampled onto a fixed number of points along the path, "
          "so two materials with different paths still share an abscissa and "
          "the block is a matrix. The resampling is linear interpolation in "
          "path fraction: it preserves band extrema and the gap, and it does "
          "**not** preserve the physical k-spacing, so read the axis as "
          "'fraction along this material's own path', never as a wavevector.",
)
def bands(md: AnnData, bandstructures, level: str = "dft",
          n_points: int = 200, max_bands: int | None = None) -> AnnData:
    """A bands object: rows are bands, columns are path points."""
    from pymatgen.electronic_structure.core import Spin

    if len(bandstructures) != md.n_obs:
        raise ValueError(
            f"got {len(bandstructures)} band structures for {md.n_obs} "
            f"materials; pass one per row, using None where a run is missing")

    rows, material, material_index, spins, indices = [], [], [], [], []
    fermi_levels = np.full(md.n_obs, np.nan)

    for i, (name, bs) in enumerate(zip(md.obs_names, bandstructures)):
        if bs is None:
            continue
        fermi = float(getattr(bs, "efermi", 0.0) or 0.0)
        fermi_levels[i] = fermi
        for spin, block in bs.bands.items():
            block = np.asarray(block, dtype=float)
            keep = block.shape[0] if max_bands is None else min(
                block.shape[0], max_bands)
            for b in range(keep):
                rows.append(_resample(block[b] - fermi, n_points))
                material.append(str(name))
                material_index.append(i)
                spins.append("up" if spin == Spin.up else "down")
                indices.append(b)

    if not rows:
        raise ValueError("no band structures were supplied; every entry was "
                         "None")

    X = np.vstack(rows)
    names = pd.Index([f"{m}:{s}:{b}" for m, s, b in
                      zip(material, spins, indices)], dtype=object)
    obs = pd.DataFrame({
        "material": pd.Categorical(material),
        "material_index": material_index,
        "spin": pd.Categorical(spins),
        "band_index": indices,
        "band_minimum": X.min(axis=1),
        "band_maximum": X.max(axis=1),
        "band_width": X.max(axis=1) - X.min(axis=1),
        "crosses_fermi": (X.min(axis=1) < 0.0) & (X.max(axis=1) > 0.0),
    }, index=names)

    distance = np.linspace(0.0, 1.0, n_points)
    var = pd.DataFrame({"path_fraction": distance},
                       index=pd.Index([f"k{i}" for i in range(n_points)],
                                      dtype=object))

    out = AnnData(X=X, obs=obs, var=var)
    out.uns[AXIS_KEY] = "bands"
    out.uns["provenance"] = []
    out.uns["bands"] = {
        "level": str(level), "n_points": int(n_points),
        "n_materials": int(np.unique(material_index).size),
        "energy_reference": "Fermi level; zero is E_F",
        "abscissa": "fraction along each material's own high-symmetry path",
    }
    md.obs[f"efermi_{level}"] = fermi_levels
    record(out, "elec.bands", level=level, n_points=n_points)
    record(md, "elec.bands", level=level, n_points=n_points)
    return out


@register_function(
    aliases=["read bands", "parse band structure", "harvest bands",
             "vasprun bands", "load band structure"],
    category="elec",
    description="Parse band structures out of finished DFT runs and build the "
                "bands axis from them.",
    prerequisites=["mv.dft.write_inputs"],
    examples=["bands = mv.elec.read_bands(md, 'runs/', level='pbe')"],
    related=["mv.elec.bands", "mv.dft.read_outputs", "mv.elec.band_features"],
    notes="A run that produced no parseable band structure becomes a missing "
          "row rather than a dropped material, and the reason lands in "
          "obs['band_error_{level}']. The bias from silently dropping failed "
          "runs points the wrong way: the calculations that fail are the "
          "difficult, interesting ones.",
)
def read_bands(md: AnnData, root, level: str = "dft",
               filename: str = "vasprun.xml", n_points: int = 200) -> AnnData:
    """Parse band structures from a directory of runs."""
    from pathlib import Path

    from pymatgen.io.vasp.outputs import Vasprun

    root = Path(root)
    manifest = root / "matverse_runs.json"
    mapping = {}
    if manifest.exists():
        mapping = json.loads(manifest.read_text()).get("directories", {})

    collected, errors = [], []
    for name in map(str, md.obs_names):
        directory = Path(mapping.get(name, root / f"run-{name}"))
        target = directory / filename
        if not target.exists():
            collected.append(None)
            errors.append("no output found")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run = Vasprun(str(target), parse_projected_eigen=False)
                collected.append(run.get_band_structure(line_mode=True))
            errors.append("")
        except Exception as exc:
            collected.append(None)
            errors.append(f"{type(exc).__name__}: {exc}")

    md.obs[f"band_error_{level}"] = errors
    if all(bs is None for bs in collected):
        raise ValueError(
            f"no band structure could be read from {root}; "
            f"obs['band_error_{level}'] says why for each row")
    return bands(md, collected, level=level, n_points=n_points)


@register_function(
    # 'band gap' stays with mv.dft.read_dos, which derives one from a density
    # of states. This derives it from the bands themselves, and an exact alias
    # can only point at one of them.
    aliases=["band features", "band gap from bands", "vbm cbm",
             "direct or indirect", "is it a metal", "gap from bands"],
    category="elec",
    description="Derive the band gap, valence and conduction band edges and "
                "whether the gap is direct, and put them back on the material "
                "axis where a screen can reach them.",
    requires={"bands_obj.obs": ["material"]},
    produces={"md.obs": ["band_gap_{level}", "vbm_{level}", "cbm_{level}",
                         "is_direct_{level}", "is_metal_{level}",
                         "n_bands_{level}"]},
    prerequisites=["mv.elec.bands"],
    examples=["mv.elec.band_features(bands, md, level='pbe')"],
    related=["mv.elec.bands", "mv.screen.filter", "mv.dft.read_dos"],
    notes="The gap is read off the bands rather than taken from the DFT code's "
          "own report, so it is derived from data you can see. A material with "
          "any band crossing the Fermi level is reported as a metal with gap "
          "zero, not as a semiconductor with a negative gap.\n\n"
          "Semilocal DFT underestimates gaps by roughly half. This function "
          "reports what the bands say; it does not correct them, and the level "
          "name is what tells a later reader whether a correction is needed.",
)
def band_features(bands_obj: AnnData, md: AnnData, level: str = "dft") -> None:
    """Gap, edges and directness per material. Deposits on ``md``."""
    if bands_obj.uns.get(AXIS_KEY) != "bands":
        raise ValueError("this is not a bands object; build one with "
                         "mv.elec.bands or mv.elec.read_bands")
    # mv.prop.dispersion builds the same axis for phonons, and a band gap is
    # not a thing a phonon spectrum has. Without this it would return a
    # plausible-looking number in THz labelled as an electronic gap.
    if bands_obj.uns.get("quantity") == "phonon_frequency":
        raise ValueError(
            "this is a phonon dispersion, not an electronic band structure; "
            "a band gap is not defined for it. For the phonon equivalents see "
            "obs['is_imaginary'] on the object mv.prop.dispersion returned")

    X = np.asarray(bands_obj.X, dtype=float)
    material = bands_obj.obs["material"].astype(str).to_numpy()

    gaps = np.full(md.n_obs, np.nan)
    vbms = np.full(md.n_obs, np.nan)
    cbms = np.full(md.n_obs, np.nan)
    direct = np.full(md.n_obs, False)
    metal = np.full(md.n_obs, False)
    counts = np.zeros(md.n_obs, dtype=int)

    for i, name in enumerate(map(str, md.obs_names)):
        block = X[material == name]
        if not block.size:
            continue
        counts[i] = block.shape[0]

        below = block[block.max(axis=1) <= 0.0]
        above = block[block.min(axis=1) >= 0.0]
        crossing = block.shape[0] - below.shape[0] - above.shape[0]

        if crossing or not below.size or not above.size:
            metal[i] = True
            gaps[i] = 0.0
            if below.size:
                vbms[i] = below.max()
            if above.size:
                cbms[i] = above.min()
            continue

        valence = below.max(axis=0)
        conduction = above.min(axis=0)
        vbms[i] = valence.max()
        cbms[i] = conduction.min()
        gaps[i] = cbms[i] - vbms[i]
        # Direct means both extrema sit at the same k-point.
        direct[i] = bool(np.argmax(valence) == np.argmin(conduction))

    md.obs[f"band_gap_{level}"] = gaps
    md.obs[f"vbm_{level}"] = vbms
    md.obs[f"cbm_{level}"] = cbms
    md.obs[f"is_direct_{level}"] = direct
    md.obs[f"is_metal_{level}"] = metal
    md.obs[f"n_bands_{level}"] = counts
    record(md, "elec.band_features", level=level)


@register_function(
    aliases=["dos fingerprint", "compare density of states", "dos similarity",
             "electronic fingerprint"],
    category="elec",
    description="Reduce a density of states to a fixed-length fingerprint, so "
                "two materials' electronic structures can be compared with a "
                "distance rather than by eye.",
    requires={"obsm": ["dos_{level}"], "uns": ["grids"]},
    produces={"obsm": ["dos_fingerprint_{level}"]},
    prerequisites=["mv.dft.read_dos"],
    examples=["mv.elec.dos_fingerprint(md, level='pbe')",
              "mv.elec.dos_fingerprint(md, level='pbe', n_bins=64)"],
    related=["mv.dft.read_dos", "mv.feat.similarity", "mv.tl.pca"],
    notes="Binned over a window around the Fermi level and normalised, "
          "following pymatgen's DosFingerprint. Everything more than a few eV "
          "from E_F is deep core and valence structure that no property "
          "depends on, so including it would swamp the part that matters.",
)
def dos_fingerprint(md: AnnData, level: str = "dft", window: float = 5.0,
                    n_bins: int = 32) -> None:
    """Fixed-length DOS descriptor into ``obsm``. Returns ``None``."""
    from ._core import grid_of

    key = f"dos_{level}"
    if key not in md.obsm:
        raise ValueError(f"obsm[{key!r}] absent; run mv.dft.read_dos(md, "
                         f"level={level!r}) first")

    energy = np.asarray(grid_of(md, "dos"), dtype=float)
    block = np.asarray(md.obsm[key], dtype=float)
    inside = np.abs(energy) <= window
    if not inside.any():
        raise ValueError(f"no DOS grid points within {window} eV of the Fermi "
                         f"level; widen window=")

    edges = np.linspace(-window, window, n_bins + 1)
    which = np.clip(np.digitize(energy[inside], edges) - 1, 0, n_bins - 1)

    out = np.zeros((md.n_obs, n_bins))
    for b in range(n_bins):
        mask = which == b
        if mask.any():
            out[:, b] = block[:, inside][:, mask].mean(axis=1)
    totals = out.sum(axis=1, keepdims=True)
    out = np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)

    md.obsm[f"dos_fingerprint_{level}"] = out
    md.uns.setdefault("features", {})[f"dos_fingerprint_{level}"] = {
        "names": [f"dos_bin_{i}" for i in range(n_bins)],
        "featurizer": "matverse.elec.dos_fingerprint",
        "window": float(window), "n_bins": int(n_bins), "level": str(level),
    }
    record(md, "elec.dos_fingerprint", level=level, window=window,
           n_bins=n_bins)


@register_function(
    aliases=["cohp", "lobster", "bonding analysis", "icohp", "bond strength",
             "is this bond bonding or antibonding"],
    category="elec",
    description="Read LOBSTER's crystal orbital Hamilton populations, which "
                "say whether each bond is bonding or antibonding and how "
                "strongly.",
    produces={"obs": ["icohp_mean_{level}", "icohp_min_{level}",
                      "n_bonds_{level}"]},
    examples=["mv.elec.cohp(md, 'lobster_runs/', level='pbe')"],
    related=["mv.env.bonds", "mv.elec.band_features"],
    notes="ICOHP is the energy-integrated COHP up to the Fermi level: negative "
          "is bonding, positive antibonding, and the magnitude is a bond "
          "strength in eV. It is the closest thing electronic structure theory "
          "offers to a bond order you can screen on.\n\n"
          "Needs LOBSTER to have run — it is a separate program that "
          "post-processes a VASP calculation, not something matverse can "
          "compute.",
)
def cohp(md: AnnData, root, level: str = "dft",
         filename: str = "ICOHPLIST.lobster") -> None:
    """Per-material ICOHP summaries. Deposits on ``md``; returns ``None``."""
    from pathlib import Path

    from pymatgen.io.lobster import Icohplist

    root = Path(root)
    manifest = root / "matverse_runs.json"
    mapping = {}
    if manifest.exists():
        mapping = json.loads(manifest.read_text()).get("directories", {})

    means = np.full(md.n_obs, np.nan)
    lowest = np.full(md.n_obs, np.nan)
    counts = np.zeros(md.n_obs, dtype=int)
    errors = []

    for i, name in enumerate(map(str, md.obs_names)):
        directory = Path(mapping.get(name, root / f"run-{name}"))
        target = directory / filename
        if not target.exists():
            errors.append("no ICOHPLIST found")
            continue
        try:
            listing = Icohplist(filename=str(target))
            values = np.array([v.summed_icohp
                               for v in listing.icohpcollection], dtype=float)
            if values.size:
                means[i] = float(values.mean())
                lowest[i] = float(values.min())
                counts[i] = int(values.size)
            errors.append("")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    md.obs[f"icohp_mean_{level}"] = means
    md.obs[f"icohp_min_{level}"] = lowest
    md.obs[f"n_bonds_{level}"] = counts
    md.obs[f"cohp_error_{level}"] = errors
    md.uns["cohp"] = {"level": str(level), "root": str(root),
                      "note": "ICOHP: negative is bonding, positive "
                              "antibonding; magnitude is a bond strength in eV"}
    record(md, "elec.cohp", level=level)


@register_function(
    # 'thermoelectric' stays with mv.prop.thermal_conductivity, which owns the
    # lattice half of zT. This is the electronic half.
    aliases=["transport", "seebeck", "boltztrap", "electronic transport",
             "electrical conductivity", "power factor"],
    category="elec",
    description="Semiclassical transport from a band structure — Seebeck "
                "coefficient, conductivity over relaxation time and power "
                "factor — through BoltzTraP2.",
    produces={"obs": ["seebeck_{level}", "sigma_over_tau_{level}",
                      "power_factor_{level}"]},
    prerequisites=["mv.elec.read_bands"],
    examples=["mv.elec.transport(md, bandstructures, level='pbe')",
              "mv.elec.transport(md, bandstructures, mu=0.2, "
              "temperature=600.0)"],
    related=["mv.elec.band_features", "mv.prop.thermal_conductivity"],
    notes="The conductivity is reported as **sigma/tau**, because the "
          "constant relaxation time approximation cannot supply tau. "
          "Multiplying by a guessed tau to reach S/m is how thermoelectric "
          "screens produce figures of merit that do not survive measurement. "
          "The Seebeck coefficient does not depend on tau and is the number "
          "to trust.\n\n"
          "**mu is where you dope to**, in eV relative to the Fermi level of "
          "the band structure you passed. At mu = 0 an intrinsic "
          "semiconductor gives almost nothing — the conductivity in the "
          "middle of a gap is exponentially small and underflows to zero, "
          "which is a true answer and a useless one. The interesting numbers "
          "are at finite doping, and choosing how much is a decision this "
          "function will not make for you.\n\n"
          "**Needs BoltzTraP2**, which links against netCDF, is not "
          "pip-installable everywhere, and on conda-forge exists only up to a "
          "Python 3.10 build: `conda install -c conda-forge boltztrap2`.\n\n"
          "Two upstream traps are worked around here so they need not be "
          "rediscovered. BandstructureLoader accepts nelect and stores it as "
          "nelect_all while BztTransportProperties reads .nelect. And the "
          "interpolation must run with curvature=True, or the Hall term is "
          "None and the constructor fails on a TypeError that names neither "
          "the term nor the cause.",
)
def transport(md: AnnData, bandstructures, level: str = "dft",
              temperature: float = 300.0, mu: float = 0.0,
              lpfac: int = 10) -> None:
    """Semiclassical transport coefficients. Deposits on ``md``."""
    try:
        from pymatgen.electronic_structure.boltztrap2 import (
            BandstructureLoader, BztInterpolator, BztTransportProperties)
        # BoltzTraP2's top level imports without touching netCDF; the module
        # that does the work does not. Guarding on the package alone let an
        # installation that cannot run through, and the failure then surfaced
        # much later as something unrelated. ValueError is caught alongside
        # ImportError because a netCDF4 built against a different numpy ABI
        # raises "numpy.dtype size changed" rather than failing to import.
        import BoltzTraP2.dft                              # noqa: F401
        import BoltzTraP2.fite                             # noqa: F401
    except (ImportError, ValueError) as exc:
        raise ImportError(
            f"mv.elec.transport needs a working BoltzTraP2, which links "
            f"against netCDF and often will not build from a wheel. Try "
            f"`conda install -c conda-forge boltztrap2` — note conda-forge "
            f"carries it only up to a Python 3.10 build. A "
            f"'numpy.dtype size changed' here means netCDF4 was built for a "
            f"different numpy than the one installed, which is a rebuild "
            f"rather than a missing package. Or read the band edges off "
            f"mv.elec.band_features and estimate the Seebeck coefficient "
            f"yourself. ({type(exc).__name__}: {exc})"
        ) from exc

    if len(bandstructures) != md.n_obs:
        raise ValueError(f"got {len(bandstructures)} band structures for "
                         f"{md.n_obs} rows; one per row is needed")

    seebeck = np.full(md.n_obs, np.nan)
    sigma = np.full(md.n_obs, np.nan)
    power = np.full(md.n_obs, np.nan)
    failed: list[str] = []

    for row, band_structure in enumerate(bandstructures):
        if band_structure is None:
            failed.append(f"row {row}: no band structure")
            continue
        try:
            loader = BandstructureLoader(band_structure)
            # Stored as nelect_all by the loader, read as .nelect by the
            # transport class. The AttributeError names neither.
            if not hasattr(loader, "nelect"):
                loader.nelect = loader.nelect_all
            # curvature=True or Hall_mu comes back None and the constructor
            # dies subscripting it.
            interpolated = BztInterpolator(loader, lpfac=int(lpfac),
                                           curvature=True)
            properties = BztTransportProperties(
                interpolated, temp_r=np.array([float(temperature)]))
            grid = np.asarray(properties.mu_r_eV, dtype=float)
            index = int(np.argmin(np.abs(grid - float(mu))))

            def average(tensor):
                block = np.asarray(tensor)[0][index]
                return float(np.trace(block) / 3.0)

            seebeck[row] = average(properties.Seebeck_mu)
            sigma[row] = average(properties.Conductivity_mu)
            # S^2 sigma, with S in uV/K, so 1e-12 puts it in W/(m K^2 s).
            power[row] = seebeck[row] ** 2 * sigma[row] * 1e-12
        except Exception as exc:
            failed.append(f"row {row}: {type(exc).__name__}: {exc}")

    md.obs[f"seebeck_{level}"] = seebeck
    md.obs[f"sigma_over_tau_{level}"] = sigma
    md.obs[f"power_factor_{level}"] = power
    md.uns.setdefault("transport", {})[level] = {
        "temperature": float(temperature), "mu": float(mu),
        "seebeck_unit": "uV/K", "sigma_unit": "1/(ohm m s)",
        "power_factor_unit": "W/(m K^2 s)",
        "errors": failed,
        "caveat": "sigma is per unit relaxation time; only the Seebeck "
                  "coefficient is independent of tau",
    }
    record(md, "elec.transport", level=level, mu=float(mu),
           temperature=float(temperature))


__all__ = ["AXIS_KEY", "PATH_TYPES", "kpath", "bands", "read_bands", "xps",
           "band_features", "dos_fingerprint", "cohp", "transport"]


@register_function(
    aliases=["xps", "photoemission", "photoelectron spectrum", "esca",
             "binding energy spectrum", "core level", "valence band spectrum"],
    category="elec",
    description="X-ray photoelectron spectrum from a projected density of "
                "states, weighted by each orbital's photoionisation cross-"
                "section.",
    produces={"obsm": ["xps_{level}"], "uns": ["grids"],
              "obs": ["xps_peak_{level}"], "levels": ["{level}"]},
    prerequisites=["mv.dft.read_dos"],
    examples=["mv.elec.xps(md, doses, level='pbe')",
              "mv.elec.xps(md, doses, level='pbe', n_points=400)"],
    related=["mv.dft.read_dos", "mv.elec.dos_fingerprint",
             "mv.prop.frontier_orbitals"],
    notes="An XPS spectrum is not a density of states with the axis flipped. "
          "Photoemission sees each orbital through its photoionisation cross-"
          "section, and those differ by more than an order of magnitude "
          "between elements and between shells of the same element: copper's "
          "3d is 0.0012 and oxygen's 2p is 0.00006, a factor of twenty. Two "
          "states contributing equally to the DOS of a copper oxide therefore "
          "contribute twenty-to-one to its photoemission, which is why a "
          "measured spectrum looks nothing like a plotted DOS and why "
          "comparing them directly is a mistake people make constantly.\\n\\n"
          "The cross-sections are Yeh and Lindau's tabulation, shipped with "
          "pymatgen, per element and orbital type. Elements past uranium are "
          "not in it and their contribution is dropped.\\n\\n"
          "**The DOS objects are an argument**, one per row, on the same "
          "reasoning as mv.elec.bands taking band structures: they come from "
          "a real calculation. mv.dft.read_dos parses them out of a directory "
          "of vasprun files, and the projections have to be there — a total "
          "DOS carries no orbital character and there is nothing to weight.\\n\\n"
          "The grid is binding energy, so it runs the opposite way to a DOS: "
          "a state at -3 eV relative to the Fermi level appears at +3 eV.",
)
def xps(md: AnnData, doses, level: str = "dft", n_points: int = 301,
        e_min: float | None = None, e_max: float | None = None) -> None:
    """XPS from projected densities of states. Deposits; returns ``None``."""
    from pymatgen.analysis.xps import XPS

    if len(doses) != md.n_obs:
        raise ValueError(f"got {len(doses)} densities of states for "
                         f"{md.n_obs} rows; one per row is needed")

    spectra, peaks, failed = [], [], []
    computed = []
    for row, dos in enumerate(doses):
        if dos is None:
            computed.append(None)
            failed.append(f"row {row}: no DOS")
            continue
        try:
            computed.append(XPS.from_dos(dos))
        except Exception as exc:
            computed.append(None)
            failed.append(f"row {row}: {type(exc).__name__}: {exc}")

    live = [s for s in computed if s is not None]
    if not live:
        raise ValueError(f"no XPS could be built from these {md.n_obs} "
                         f"densities of states: {failed[:3]}")

    low = e_min if e_min is not None else min(float(np.min(s.x)) for s in live)
    high = e_max if e_max is not None else max(float(np.max(s.x)) for s in live)
    grid = np.linspace(float(low), float(high), int(n_points))

    for spectrum in computed:
        if spectrum is None:
            spectra.append(np.full(len(grid), np.nan))
            peaks.append(np.nan)
            continue
        # Each XPS comes back on its own energy axis, so they are put onto one
        # shared grid before deposit - the grid convention exists so two rows
        # can be compared point by point.
        order = np.argsort(np.asarray(spectrum.x, dtype=float))
        x = np.asarray(spectrum.x, dtype=float)[order]
        y = np.asarray(spectrum.y, dtype=float)[order]
        resampled = np.interp(grid, x, y, left=np.nan, right=np.nan)
        spectra.append(resampled)
        finite = np.isfinite(resampled)
        peaks.append(float(grid[finite][np.argmax(resampled[finite])])
                     if finite.any() else np.nan)

    deposit_grid(md, "xps", level, np.vstack(spectra), grid,
                 unit="binding energy (eV)", n_failed=len(failed))
    set_level(md, level, kind="derived",
              method="XPS from a projected DOS, weighted by Yeh-Lindau "
                     "photoionisation cross-sections",
              note="the level of theory is whatever produced the DOS; this "
                   "step only reweights it")
    md.obs[f"xps_peak_{level}"] = np.array(peaks, dtype=float)
    md.uns.setdefault("xps", {})[level] = {
        "cross_sections": "Yeh and Lindau, as shipped with pymatgen",
        "axis": "binding energy, positive below the Fermi level",
        "errors": failed,
    }
    record(md, "elec.xps", level=level, n_points=int(n_points))


@register_function(
    aliases=["fermi surface", "fermi surfaces", "ifermi", "fermi surface "
             "area", "fermi sheets", "fermi pockets", "band topology at the "
             "fermi level"],
    category="elec",
    description="Fermi surfaces from uniform-mesh band structures — area, "
                "how many disconnected sheets, and whether there is one at "
                "all.",
    produces={"obs": ["fermi_surface_area_{level}",
                      "fermi_sheets_{level}", "has_fermi_surface_{level}"],
              "uns": ["fermi_surface"], "levels": ["{level}"]},
    prerequisites=["mv.elec.read_bands"],
    dispatch="level= names the theory the bands came from",
    examples=["mv.elec.fermi_surface(md, band_structures, level='pbe')",
              "mv.elec.fermi_surface(md, bs_list, level='pbe', mu=0.1)"],
    related=["mv.elec.bands", "mv.elec.read_bands", "mv.elec.band_features",
             "mv.elec.dos_fingerprint"],
    notes="The band structure has to be on a **uniform k-mesh**, not a "
          "high-symmetry line. Fourier interpolation needs a grid to "
          "interpolate from, and a line-mode calculation has no interior to "
          "sample — mv.elec.bands takes line-mode structures for plotting and "
          "this takes the other kind. A BandStructureSymmLine is refused "
          "rather than interpolated into nonsense.\\n\\n"
          "The area is the total over all sheets, in inverse square "
          "angstrom, and it is clipped to the first Brillouin zone. That "
          "clipping is physics, not an approximation: a free-electron sphere "
          "wider than the zone is genuinely truncated where it crosses the "
          "boundary, and the area that remains is the one that carries "
          "current.\\n\\n"
          "Verified against the analytic free-electron result. On a simple "
          "cubic cell with a = 4 A, where the zone boundary is at 0.785 "
          "1/A, the computed area matches 4*pi*kF^2 to within 1% at kF of "
          "0.324, 0.458 and 0.628 — and falls to 0.66 of it at kF = 0.887, "
          "which is where the sphere has grown past the boundary.\\n\\n"
          "It is not cheap. Fourier interpolation at the default factor of "
          "five takes minutes for a single 12x12x12 mesh, and the cost rises "
          "with the cube of the factor — this belongs on a shortlist, not on "
          "a library, and lowering interpolation_factor is the first thing to "
          "try when it is too slow.\n\n"
          "keep_mesh stores each sheet's vertices and faces in "
          "uns['fermi_surface'][level]['meshes'], which is what "
          "mv.pl.fermi_surface draws. A few hundred kilobytes buys not having "
          "to repeat the interpolation, and the alternative — a plotting "
          "function that recomputes for minutes every time it is called — is "
          "not an interface worth having. Pass keep_mesh=False for a screen "
          "that only wants the numbers.\n\n"
          "obs['fermi_sheets_{level}'] counts disconnected pieces, which is "
          "what distinguishes a simple metal from one with pockets. Zero "
          "sheets means no band crosses the level, which is the definition of "
          "an insulator and is reported rather than raised.",
)
def fermi_surface(md: AnnData, bandstructures, level: str = "dft",
                  mu: float = 0.0, interpolation_factor: float = 5.0,
                  wigner_seitz: bool = True, keep_mesh: bool = True) -> None:
    """Fermi surface area and sheet count. Deposits; returns ``None``."""
    try:
        from ifermi.interpolate import FourierInterpolator
        from ifermi.surface import FermiSurface
    except ImportError as exc:
        raise ImportError(
            f"mv.elec.fermi_surface needs IFermi: `pip install "
            f"matverse[fermi]` or `pip install ifermi`. ({exc})") from exc

    from pymatgen.electronic_structure.bandstructure import (
        BandStructureSymmLine)

    if len(bandstructures) != md.n_obs:
        raise ValueError(
            f"got {len(bandstructures)} band structures for {md.n_obs} "
            f"materials; pass one per row, using None where a run is missing")

    areas = np.full(md.n_obs, np.nan)
    sheets = np.full(md.n_obs, np.nan)
    present = np.zeros(md.n_obs, dtype=bool)
    meshes: dict = {}
    failures = []

    for i, bs in enumerate(bandstructures):
        if bs is None:
            continue
        if isinstance(bs, BandStructureSymmLine):
            # A line through the zone has no interior to interpolate, and
            # feeding one in produces a surface rather than an error.
            failures.append(
                f"{md.obs_names[i]}: this is a line-mode band structure. A "
                f"Fermi surface needs a uniform k-mesh; mv.elec.bands is what "
                f"takes the line-mode kind")
            continue
        try:
            dense = FourierInterpolator(bs).interpolate_bands(
                interpolation_factor=float(interpolation_factor))
            surface = FermiSurface.from_band_structure(
                dense, mu=float(mu), wigner_seitz=bool(wigner_seitz))
        except Exception as exc:
            failures.append(f"{md.obs_names[i]}: {type(exc).__name__}: {exc}")
            continue

        collected = []
        for spin, isosurfaces in getattr(surface, "isosurfaces", {}).items():
            for sheet in isosurfaces:
                # Vertices and faces are the surface. Keeping them costs a few
                # hundred kilobytes and means mv.pl.fermi_surface can draw it
                # without repeating an interpolation that takes minutes.
                collected.append({
                    "spin": str(spin),
                    "band": int(getattr(sheet, "band_idx", -1)),
                    "area": float(sheet.area),
                    "vertices": np.asarray(sheet.vertices, dtype=float),
                    "faces": np.asarray(sheet.faces, dtype=int),
                })
        pieces = len(collected)
        areas[i] = float(surface.area)
        sheets[i] = int(pieces)
        present[i] = pieces > 0
        if keep_mesh:
            meshes[str(md.obs_names[i])] = collected

    md.obs[f"fermi_surface_area_{level}"] = areas
    md.obs[f"fermi_sheets_{level}"] = sheets
    md.obs[f"has_fermi_surface_{level}"] = present
    md.uns.setdefault("fermi_surface", {})[level] = {
        "mu": float(mu),
        "interpolation_factor": float(interpolation_factor),
        "wigner_seitz": bool(wigner_seitz),
        "area_unit": "angstrom^-2",
        "meshes": meshes,
        "n_failed": len(failures),
        "errors": failures[:10],
        "note": "area is the total over all sheets, clipped to the first "
                "Brillouin zone; the clipping is physics, not an "
                "approximation",
    }
    set_level(md, level, kind="dft", method="Fermi surface by Fourier "
              "interpolation", reference=None, surrogate=False,
              mu=float(mu), interpolation_factor=float(interpolation_factor))
    if failures:
        warnings.warn(
            f"{len(failures)} of {md.n_obs} band structures produced no Fermi "
            f"surface; see uns['fermi_surface'][{level!r}]['errors']. First: "
            f"{failures[0]}", RuntimeWarning, stacklevel=2)
    record(md, "elec.fermi_surface", level=level, mu=mu,
           interpolation_factor=interpolation_factor)

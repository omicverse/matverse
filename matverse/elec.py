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

from ._core import deposit_grid, record, set_level, structures
from ._registry import register_function

#: Set on a bands object so functions can tell the axes apart.
AXIS_KEY = "matverse_axis"

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
    description="Semiclassical transport from the band structure — Seebeck "
                "coefficient, electrical conductivity and power factor — via "
                "BoltzTraP2.",
    produces={"obs": ["seebeck_{level}", "sigma_over_tau_{level}",
                      "power_factor_{level}"]},
    prerequisites=["mv.elec.read_bands"],
    examples=["mv.elec.transport(md, bands, level='pbe', temperature=300.0)"],
    related=["mv.elec.band_features", "mv.prop.thermal_conductivity"],
    notes="BoltzTraP2 links against netCDF and is not pip-installable "
          "everywhere; when it is absent this raises with the install command "
          "rather than returning zeros.\n\n"
          "The conductivity is reported as **sigma/tau** because the constant "
          "relaxation time approximation cannot supply tau. Multiplying by a "
          "guessed tau to get a number in S/m is how thermoelectric screens "
          "produce figures of merit that do not survive measurement — the "
          "Seebeck coefficient, which is independent of tau, is the number to "
          "trust here.",
)
def transport(md: AnnData, bands_obj: AnnData, level: str = "dft",
              temperature: float = 300.0, doping=None) -> None:
    """Semiclassical transport coefficients. Deposits on ``md``."""
    try:
        import BoltzTraP2                                # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mv.elec.transport needs BoltzTraP2, which links against netCDF "
            "and often will not build from a wheel. Try `conda install -c "
            "conda-forge boltztrap2`, or compute the Seebeck coefficient from "
            "the bands yourself — mv.elec.band_features gives you the edges."
        ) from exc

    raise NotImplementedError(
        "BoltzTraP2 is importable but the matverse binding is not wired yet; "
        "open an issue with the code you would like to run.")


__all__ = ["AXIS_KEY", "PATH_TYPES", "kpath", "bands", "read_bands",
           "band_features", "dos_fingerprint", "cohp", "transport"]

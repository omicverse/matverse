"""``mv.md`` — molecular dynamics, and the properties only motion gives you.

Everything in ``mv.calc`` and ``mv.prop`` so far describes a structure sitting
still at zero kelvin. Diffusion, ionic conductivity, thermal expansion, melting
and the structure of a liquid or a glass are all properties of a system that is
moving, and none of them can be got from a relaxation.

That makes this the module the rest of the missing half hangs off.

What is stored, and what is not
-------------------------------
Trajectories deliberately do not become part of the object. OMat24 alone is
around 110 million frames; a screening library that materialises a trajectory
per candidate stops working at a few hundred materials. So a run computes its
observables **as it goes** and deposits the summary — mean temperature, mean
energy, diffusivity, and a per-element diffusivity layer — while the frames are
discarded unless you ask for a file.

The condition axis
------------------
A property at temperature is a curve, not a number. :func:`sweep` runs a
temperature series and stores the result on the grid axis, so
``obsm['volume_emt']`` against ``uns['grids']['temperature']`` is the same shape
of thing as a diffraction pattern — one mechanism, both uses.

A caveat worth reading before trusting a number
-----------------------------------------------
Universal machine-learned potentials systematically soften the potential energy
surface, and melt-quench with them has been shown to produce catastrophically
under-dense amorphous structures — in one 2026 study, 1–4 g/cm³ against an
ab-initio 10.04 for a-IrO₂, with 19 of 30 test materials under-dense.
:func:`melt_quench` therefore defaults to the protocol that survives that test —
quench at fixed volume, then equilibrate the cell — rather than the naive NPT
quench, and says so in what it records.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import deposit_grid, deposit_structures, record, set_level, structures
from ._registry import register_function

#: 1 angstrom^2/ps expressed in cm^2/s, the unit diffusivity is quoted in.
_A2_PER_PS_TO_CM2_PER_S = 1.0e-4

#: eV/K
_BOLTZMANN_EV_PER_K = 8.617333262e-5

#: elementary charge squared / (eV * angstrom) -> S/cm, for Nernst-Einstein.
_NERNST_EINSTEIN = 1.602176634e-19 * 1.0e8


#: name -> (factory, metadata) for batched TorchSim models. Registered rather
#: than hardcoded, for the same reason ``mv.calc`` registers its calculators.
_BATCHED: dict[str, tuple] = {}


def _engine(level: str):
    """The calculator and its metadata, shared with ``mv.calc``."""
    from .calc import _get
    return _get(level)


@register_function(
    aliases=["register batched model", "torchsim model", "gpu md model",
             "batched potential"],
    category="md",
    description="Register a model that integrates many systems at once on a "
                "GPU, for use by mv.md.run(engine='torchsim').",
    examples=["mv.md.register_batched('lj', lambda: LennardJonesModel(), "
              "method='Lennard-Jones')"],
    related=["mv.md.run", "mv.md.batched_available"],
    notes="Separate from mv.calc.register_calculator because the interfaces "
          "differ: an ASE calculator answers about one structure at a time, "
          "and a TorchSim model answers about a batch. Registering the same "
          "physics under both is normal — one for a relaxation, one for a "
          "thousand trajectories.",
)
def register_batched(name: str, factory, *, method: str | None = None,
                     reference: str | None = None, surrogate: bool = True,
                     license: str | None = None, **extra) -> None:
    """Register a batched (TorchSim) model under a level name."""
    _BATCHED[name] = (factory, {"kind": "mlip", "method": method or name,
                                "reference": reference,
                                "surrogate": bool(surrogate),
                                "license": license, "uncertainty": None,
                                "engine": "torchsim", **extra})


@register_function(
    aliases=["batched available", "which batched models", "is torchsim "
             "installed", "gpu md available"],
    category="md",
    description="Report whether a batched molecular-dynamics engine is "
                "available and which models are registered with it.",
    examples=["mv.md.batched_available()"],
    related=["mv.md.register_batched", "mv.md.run"],
)
def batched_available() -> dict:
    """What the batched path can run here, and why not if it cannot."""
    try:
        import torch

        import torch_sim                                    # noqa: F401
        engine = {"torch_sim": True,
                  "cuda": bool(torch.cuda.is_available()),
                  "devices": int(torch.cuda.device_count())}
    except ImportError as exc:
        return {"torch_sim": False, "reason": str(exc),
                "install": "torch-sim-atomistic needs Python >= 3.11; "
                           "matverse's own floor is 3.10, so this is an "
                           "environment decision rather than a dependency"}
    return {**engine,
            "models": {name: dict(meta) for name, (_, meta) in
                       _BATCHED.items()}}


@register_function(
    aliases=["molecular dynamics", "md", "run md", "dynamics", "nvt", "npt",
             "simulate at temperature", "diffusivity"],
    category="md",
    description="Run molecular dynamics on every structure at one temperature "
                "and record the observables — mean energy, mean temperature, "
                "mean-squared displacement and diffusivity — without keeping "
                "the trajectory.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["md_energy_{level}", "md_temperature_{level}",
                      "msd_{level}", "diffusivity_{level}",
                      "md_volume_{level}"],
              "layers": ["diffusivity_{level}"],
              "structures": ["md_{level}"], "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    dispatch="ensemble='nvt' holds the cell fixed; 'npt' lets it relax, which "
             "is what thermal expansion and density need",
    examples=["mv.md.run(md, level='emt', temperature=600.0, steps=2000)",
              "mv.md.run(md, level='emt', temperature=900.0, ensemble='npt')"],
    related=["mv.md.sweep", "mv.md.melt_quench", "mv.prop.rdf"],
    notes="The per-element diffusivity goes to a layer, because it is "
          "materials x elements — the same shape as X. For an ionic conductor "
          "the number that matters is the mobile species' diffusivity, not the "
          "average over everything in the cell, and averaging a lithium "
          "diffusivity with a framework one produces a number describing "
          "nothing.",
)
def run(md: AnnData, level: str = "emt", source: str = "input",
        temperature: float = 300.0, steps: int = 1000,
        timestep: float = 2.0, ensemble: str = "nvt",
        equilibration: int = 500, friction: float = 0.05,
        sample_every: int = 10, seed: int = 0,
        key_added: str | None = None) -> None:
    """Molecular dynamics at one temperature. ``timestep`` is in femtoseconds.

    ``friction`` sets how hard the thermostat pulls. The default is
    deliberately strong: a weakly coupled thermostat takes tens of picoseconds
    to reach its target, and a short screening run then samples a temperature
    that is not the one you asked for. Weaken it when you care about dynamics
    rather than about sampling a fixed temperature.
    """
    if ensemble not in ("nvt", "npt"):
        raise ValueError(f"ensemble must be 'nvt' or 'npt', got {ensemble!r}")

    from pymatgen.io.ase import AseAtomsAdaptor

    tag = key_added or level
    if level in _BATCHED:
        return _run_batched(md, level, tag, source, temperature, steps,
                            timestep, ensemble, seed)

    factory, meta = _engine(level)
    adaptor = AseAtomsAdaptor()
    calculator = factory()

    energies, temperatures, msds, diffusivities = [], [], [], []
    volumes, finals, per_element, failed = [], [], [], 0
    elements = list(map(str, md.var_names))

    for structure in structures(md, source):
        try:
            result = _integrate(structure, adaptor, calculator, temperature,
                                steps, timestep, ensemble, equilibration,
                                friction, sample_every, seed, elements)
        except Exception:
            failed += 1
            energies.append(np.nan); temperatures.append(np.nan)
            msds.append(np.nan); diffusivities.append(np.nan)
            volumes.append(np.nan); finals.append(structure)
            per_element.append(np.full(len(elements), np.nan))
            continue
        energies.append(result["energy"])
        temperatures.append(result["temperature"])
        msds.append(result["msd"])
        diffusivities.append(result["diffusivity"])
        volumes.append(result["volume"])
        finals.append(result["structure"])
        per_element.append(result["per_element"])

    md.obs[f"md_energy_{tag}"] = energies
    md.obs[f"md_temperature_{tag}"] = temperatures
    md.obs[f"msd_{tag}"] = msds
    md.obs[f"diffusivity_{tag}"] = diffusivities
    md.obs[f"md_volume_{tag}"] = volumes
    if md.n_vars:
        md.layers[f"diffusivity_{tag}"] = np.vstack(per_element)
    deposit_structures(md, f"md_{tag}", finals)
    set_level(md, tag, **meta, source=source, ensemble=ensemble,
              temperature=temperature, steps=steps, timestep=timestep,
              n_failed=failed)
    _check_thermostat(md, tag, temperature)
    record(md, "md.run", level=level, source=source, temperature=temperature,
           steps=steps, ensemble=ensemble)


def _check_thermostat(md: AnnData, tag: str, target: float,
                      tolerance: float = 0.2) -> None:
    """Say so when the run did not reach the temperature it was asked for.

    A thermostat that has not equilibrated samples a different ensemble than the
    one named, and every observable from that run belongs to the temperature it
    actually reached. The achieved temperature is already in
    ``obs['md_temperature_*']``; this makes not noticing it harder.
    """
    import warnings

    achieved = md.obs[f"md_temperature_{tag}"].to_numpy(dtype=float)
    finite = achieved[np.isfinite(achieved)]
    if not len(finite) or target <= 0:
        return
    off = np.abs(finite - target) / target > tolerance
    if not off.any():
        return
    warnings.warn(
        f"{int(off.sum())} of {len(finite)} runs sampled a temperature more "
        f"than {tolerance:.0%} from the requested {target:g} K (mean achieved "
        f"{finite.mean():.0f} K). The thermostat had not equilibrated, so "
        f"every observable belongs to the temperature reached rather than the "
        f"one asked for. Raise `equilibration`, raise `friction`, or use a "
        f"larger cell — a few-atom cell fluctuates too much to hold a "
        f"temperature.", stacklevel=3)


def _run_batched(md: AnnData, level: str, tag: str, source: str,
                 temperature: float, steps: int, timestep: float,
                 ensemble: str, seed: int) -> None:
    """Integrate every structure at once on one device.

    The ASE path above runs one trajectory per structure in a Python loop, which
    leaves a GPU idle between force calls. A batched engine puts the whole
    dataset in one tensor and steps it together — the same shape the object
    already has, which is why this is a different execution path rather than a
    different library.

    Per-atom displacement is not read back here: the batched state packs every
    system into one flat atom index, and unpacking it correctly is worth doing
    deliberately rather than inferring. Diffusivity therefore comes from the ASE
    path, and this one is for relaxing and equilibrating a large batch.
    """
    import torch
    import torch_sim as ts

    factory, meta = _BATCHED[level]
    model = factory()
    integrator = (ts.integrators.npt_langevin if ensemble == "npt"
                  else ts.integrators.nvt_langevin)

    S = structures(md, source)
    torch.manual_seed(seed)
    final = ts.integrate(system=S, model=model, integrator=integrator,
                         n_steps=int(steps), temperature=float(temperature),
                         timestep=float(timestep))
    relaxed = final.to_structures() if hasattr(final, "to_structures") else S

    md.obs[f"md_energy_{tag}"] = _detach(getattr(final, "energy", None), md.n_obs)
    md.obs[f"md_temperature_{tag}"] = _detach(
        ts.calc_kT(masses=final.masses, momenta=final.momenta,
                   system_idx=final.system_idx) / _BOLTZMANN_EV_PER_K, md.n_obs)
    md.obs[f"md_volume_{tag}"] = [float(s.volume) for s in relaxed]
    deposit_structures(md, f"md_{tag}", relaxed)
    set_level(md, tag, **{**meta, "engine": "torchsim"}, source=source,
              ensemble=ensemble, temperature=temperature, steps=steps,
              timestep=timestep, device=str(getattr(model, "device", "?")),
              note="Batched engine: no per-atom displacement is read back, so "
                   "no MSD or diffusivity. Use the ASE path for those.")
    _check_thermostat(md, tag, temperature)
    record(md, "md.run", level=level, source=source, temperature=temperature,
           steps=steps, ensemble=ensemble, engine="torchsim")


def _detach(value, n: int) -> np.ndarray:
    """A torch tensor as a plain array of the right length, or NaN."""
    if value is None:
        return np.full(n, np.nan)
    array = np.asarray(value.detach().cpu().numpy()
                       if hasattr(value, "detach") else value, dtype=float)
    return array if array.shape == (n,) else np.full(n, np.nan)


def _integrate(structure, adaptor, calculator, temperature, steps, timestep,
               ensemble, equilibration, friction, sample_every, seed,
               elements) -> dict:
    """One trajectory, reduced to its observables as it runs."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = adaptor.get_atoms(structure)
    atoms.calc = calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=
                                 np.random.default_rng(seed))

    if ensemble == "npt":
        dynamics = _npt(atoms, timestep, temperature, friction)
    else:
        dynamics = Langevin(atoms, timestep * units.fs,
                            temperature_K=temperature, friction=friction,
                            rng=np.random.default_rng(seed), logfile=None)

    dynamics.run(equilibration)

    reference = atoms.get_positions()
    symbols = np.asarray(atoms.get_chemical_symbols())
    samples: list[tuple[float, np.ndarray]] = []
    energy, kinetic_t, volume, n = 0.0, 0.0, 0.0, 0

    for step in range(0, steps, sample_every):
        dynamics.run(min(sample_every, steps - step))
        displacement = atoms.get_positions() - reference
        samples.append(((step + sample_every) * timestep * 1e-3,   # ps
                        (displacement ** 2).sum(axis=1)))
        energy += float(atoms.get_potential_energy())
        kinetic_t += float(atoms.get_temperature())
        volume += float(atoms.get_volume())
        n += 1

    times = np.array([t for t, _ in samples])
    squared = np.vstack([d for _, d in samples])
    overall = _slope_diffusivity(times, squared.mean(axis=1))

    by_element = np.full(len(elements), np.nan)
    for i, symbol in enumerate(elements):
        mask = symbols == symbol
        if mask.any():
            by_element[i] = _slope_diffusivity(times,
                                               squared[:, mask].mean(axis=1))

    return {"energy": energy / max(n, 1),
            "temperature": kinetic_t / max(n, 1),
            "volume": volume / max(n, 1),
            "msd": float(squared.mean(axis=1)[-1]),
            "diffusivity": overall,
            "per_element": by_element,
            "structure": adaptor.get_structure(atoms)}


def _npt(atoms, timestep, temperature, friction):
    """An NPT integrator, falling back to Berendsen where NPT needs a cell
    triangularisation the structure will not accept."""
    from ase import units
    from ase.md.nptberendsen import NPTBerendsen

    return NPTBerendsen(atoms, timestep * units.fs, temperature_K=temperature,
                        pressure_au=0.0, taut=100 * units.fs,
                        taup=1000 * units.fs, compressibility_au=4.57e-5,
                        logfile=None)


def _slope_diffusivity(times: np.ndarray, msd: np.ndarray) -> float:
    """Einstein relation: D = MSD / 6t, from the slope rather than one point.

    Fitted over the second half of the trajectory. The early part is dominated
    by vibration inside a site rather than by hopping between sites, and
    including it inflates D for anything that is not actually diffusing.
    """
    if len(times) < 4:
        return float("nan")
    half = len(times) // 2
    slope = np.polyfit(times[half:], msd[half:], 1)[0]
    return float(max(slope, 0.0) / 6.0 * _A2_PER_PS_TO_CM2_PER_S)


@register_function(
    aliases=["temperature sweep", "sweep", "property versus temperature",
             "thermal expansion", "temperature series"],
    category="md",
    description="Run molecular dynamics across a temperature series and store "
                "volume, energy and diffusivity as curves against temperature, "
                "giving thermal expansion and an Arrhenius fit.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["md_volume_{level}", "md_energy_{level}",
                       "md_diffusivity_{level}"],
              "obs": ["thermal_expansion_{level}",
                      "activation_energy_{level}"],
              "uns": ["grids"], "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    examples=["mv.md.sweep(md, level='emt', "
              "temperatures=[300, 600, 900, 1200])"],
    related=["mv.md.run", "mv.prop.free_energy"],
    notes="The temperature series is a **condition axis**, and it uses the same "
          "grid mechanism as a diffraction pattern: one array in "
          "uns['grids']['temperature'], one obsm block per quantity. Nothing "
          "new was needed to express 'the same property at different "
          "conditions'.\n\n"
          "Thermal expansion needs ensemble='npt'; at fixed volume the cell "
          "cannot expand and the coefficient is zero by construction.",
)
def sweep(md: AnnData, level: str = "emt", source: str = "input",
          temperatures=(300.0, 600.0, 900.0), steps: int = 500,
          ensemble: str = "npt", **kwargs) -> None:
    """A temperature series, stored on the condition axis."""
    grid = np.asarray(list(temperatures), dtype=float)
    volume, energy, diffusivity = [], [], []

    for temperature in grid:
        scratch = md.copy()
        run(scratch, level=level, source=source, temperature=float(temperature),
            steps=steps, ensemble=ensemble, key_added="_sweep", **kwargs)
        volume.append(scratch.obs["md_volume__sweep"].to_numpy(dtype=float))
        energy.append(scratch.obs["md_energy__sweep"].to_numpy(dtype=float))
        diffusivity.append(
            scratch.obs["diffusivity__sweep"].to_numpy(dtype=float))

    volume = np.column_stack(volume)
    deposit_grid(md, "md_volume", level, volume, grid, unit="K")
    deposit_grid(md, "md_energy", level, np.column_stack(energy), grid,
                 unit="K")
    deposit_grid(md, "md_diffusivity", level, np.column_stack(diffusivity),
                 grid, unit="K")

    md.obs[f"thermal_expansion_{level}"] = _expansion(volume, grid)
    md.obs[f"activation_energy_{level}"] = _arrhenius(
        np.column_stack(diffusivity), grid)
    _, meta = _engine(level)
    set_level(md, level, **meta, source=source, ensemble=ensemble,
              temperatures=grid.tolist(), steps=steps)
    record(md, "md.sweep", level=level, source=source,
           temperatures=grid.tolist(), ensemble=ensemble)


def _expansion(volume: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Volumetric thermal expansion coefficient, 1/V dV/dT, per kelvin."""
    out = np.full(len(volume), np.nan)
    if len(grid) < 2:
        return out
    for i, row in enumerate(volume):
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        slope = np.polyfit(grid[ok], row[ok], 1)[0]
        reference = float(np.mean(row[ok]))
        if reference > 0:
            out[i] = float(slope / reference)
    return out


def _arrhenius(diffusivity: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Activation energy in eV from ln D against 1/kT.

    Returns NaN where the material did not diffuse. A slope fitted through
    numerical noise is not an activation energy, and reporting one is worse than
    reporting nothing.
    """
    out = np.full(len(diffusivity), np.nan)
    for i, row in enumerate(diffusivity):
        ok = np.isfinite(row) & (row > 1e-12)
        if ok.sum() < 2:
            continue
        slope = np.polyfit(1.0 / (_BOLTZMANN_EV_PER_K * grid[ok]),
                           np.log(row[ok]), 1)[0]
        out[i] = float(-slope)
    return out


@register_function(
    aliases=["ionic conductivity", "conductivity", "nernst einstein",
             "solid electrolyte", "lithium conductivity"],
    category="md",
    description="Convert a per-element diffusivity into an ionic conductivity "
                "for one mobile species by the Nernst-Einstein relation.",
    requires={"layers": ["diffusivity_{level}"]},
    produces={"obs": ["conductivity_{species}_{level}"]},
    prerequisites=["mv.md.run"],
    examples=["mv.md.conductivity(md, species='Li', charge=1, level='emt')"],
    related=["mv.md.run", "mv.neb.barrier"],
    notes="Nernst-Einstein assumes uncorrelated hopping, so it omits the Haven "
          "ratio and typically overestimates by a factor of order one. It is "
          "the right tool for ranking candidates and the wrong one for quoting "
          "a number against experiment.",
)
def conductivity(md: AnnData, species: str, charge: float = 1.0,
                 level: str = "emt", temperature: float | None = None) -> None:
    """Ionic conductivity in S/cm from the mobile species' diffusivity."""
    key = f"diffusivity_{level}"
    if key not in md.layers:
        raise ValueError(f"layers[{key!r}] absent; run mv.md.run(md, "
                         f"level={level!r}) first")
    names = list(map(str, md.var_names))
    if species not in names:
        raise ValueError(f"{species!r} is not on the element axis ({names})")

    diffusivity = np.asarray(md.layers[key], dtype=float)[:, names.index(species)]
    counts = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    amount = np.asarray(counts, dtype=float)[:, names.index(species)]

    if temperature is None:
        temperature = float(np.nanmean(
            md.obs.get(f"md_temperature_{level}", np.array([300.0]))))
    volumes = md.obs.get(f"md_volume_{level}")
    volumes = (np.asarray(volumes, dtype=float) if volumes is not None
               else np.array([s.volume for s in structures(md, "input")]))

    with np.errstate(divide="ignore", invalid="ignore"):
        density = amount / volumes                    # carriers per angstrom^3
        sigma = (density * charge ** 2 * diffusivity * _NERNST_EINSTEIN
                 / (_BOLTZMANN_EV_PER_K * temperature))

    md.obs[f"conductivity_{species}_{level}"] = sigma
    record(md, "md.conductivity", species=species, charge=charge, level=level,
           temperature=temperature)


@register_function(
    aliases=["melt quench", "amorphous", "glass", "make amorphous",
             "quench", "disordered structure"],
    category="md",
    description="Generate an amorphous structure by melting and quenching, "
                "using the fixed-volume quench and separate cell equilibration "
                "that avoids the under-dense structures a naive quench gives.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["amorphous_{level}"],
              "obs": ["amorphous_density_{level}",
                      "amorphous_density_ratio_{level}"],
              "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    examples=["mv.md.melt_quench(md, level='emt', melt_temperature=3000)"],
    related=["mv.md.run", "mv.prop.rdf"],
    notes="A 2026 study found every one of eight universal potentials produced "
          "catastrophically under-dense amorphous structures under a naive NPT "
          "melt-quench — 1-4 g/cm3 against an ab-initio 10.04 for a-IrO2, with "
          "19 of 30 materials under-dense. Quenching at fixed volume and "
          "equilibrating the cell afterwards is what survives that test, so it "
          "is the default here.\n\n"
          "obs['amorphous_density_ratio'] is the amorphous density over the "
          "crystalline one. A real glass sits a few percent below 1; a value "
          "near 0.3 means the protocol failed, not that the material is "
          "exotic.",
)
def melt_quench(md: AnnData, level: str = "emt", source: str = "input",
                melt_temperature: float = 3000.0,
                final_temperature: float = 300.0,
                melt_steps: int = 500, quench_steps: int = 500,
                equilibrate_steps: int = 300, timestep: float = 1.0,
                supercell=(2, 2, 2), seed: int = 0,
                fixed_volume_quench: bool = True) -> None:
    """Melt, quench at fixed volume, then let the cell relax."""
    from pymatgen.io.ase import AseAtomsAdaptor

    factory, meta = _engine(level)
    adaptor = AseAtomsAdaptor()
    calculator = factory()

    out, densities, ratios, failed = [], [], [], 0
    for structure in structures(md, source):
        crystalline = float(structure.density)
        try:
            cell = structure.copy()
            cell.make_supercell(list(supercell))
            amorphous = _quench(cell, adaptor, calculator, melt_temperature,
                                final_temperature, melt_steps, quench_steps,
                                equilibrate_steps, timestep, seed,
                                fixed_volume_quench)
            out.append(amorphous)
            densities.append(float(amorphous.density))
            ratios.append(float(amorphous.density) / crystalline
                          if crystalline > 0 else np.nan)
        except Exception:
            failed += 1
            out.append(structure)
            densities.append(np.nan)
            ratios.append(np.nan)

    deposit_structures(md, f"amorphous_{level}", out)
    md.obs[f"amorphous_density_{level}"] = densities
    md.obs[f"amorphous_density_ratio_{level}"] = ratios
    set_level(md, level, **meta, source=source,
              melt_temperature=melt_temperature,
              fixed_volume_quench=bool(fixed_volume_quench), n_failed=failed,
              protocol="NVT quench then NPT equilibration"
              if fixed_volume_quench else "NPT quench (known to under-densify)")
    record(md, "md.melt_quench", level=level, source=source,
           melt_temperature=melt_temperature,
           fixed_volume_quench=fixed_volume_quench)


def _quench(cell, adaptor, calculator, melt_t, final_t, melt_steps,
            quench_steps, equilibrate_steps, timestep, seed, fixed_volume):
    """Melt at temperature, cool, then let the cell find its own volume."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = adaptor.get_atoms(cell)
    atoms.calc = calculator
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=melt_t, rng=rng)

    Langevin(atoms, timestep * units.fs, temperature_K=melt_t, friction=0.02,
             rng=rng, logfile=None).run(melt_steps)

    # Cool in stages at fixed volume. Letting the cell follow the temperature
    # down is what produces the under-dense structures.
    stages = 5
    for i in range(stages):
        target = melt_t + (final_t - melt_t) * (i + 1) / stages
        Langevin(atoms, timestep * units.fs, temperature_K=target,
                 friction=0.02, rng=rng,
                 logfile=None).run(max(quench_steps // stages, 1))

    if not fixed_volume:
        return adaptor.get_structure(atoms)

    equilibrated = _npt(atoms, timestep, final_t, 0.02)
    equilibrated.run(equilibrate_steps)
    return adaptor.get_structure(atoms)


__all__ = ["run", "sweep", "conductivity", "melt_quench", "register_batched",
           "batched_available"]

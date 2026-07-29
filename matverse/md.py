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

import warnings

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
    traces, times = [], None
    elements = list(map(str, md.var_names))
    n_samples = len(range(0, steps, sample_every))

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
            traces.append(np.full(n_samples, np.nan))
            continue
        energies.append(result["energy"])
        temperatures.append(result["temperature"])
        msds.append(result["msd"])
        diffusivities.append(result["diffusivity"])
        volumes.append(result["volume"])
        finals.append(result["structure"])
        per_element.append(result["per_element"])
        traces.append(result["trace"])
        times = result["times"]

    md.obs[f"md_energy_{tag}"] = energies
    md.obs[f"md_temperature_{tag}"] = temperatures
    md.obs[f"msd_{tag}"] = msds
    md.obs[f"diffusivity_{tag}"] = diffusivities
    md.obs[f"md_volume_{tag}"] = volumes
    if md.n_vars:
        md.layers[f"diffusivity_{tag}"] = np.vstack(per_element)
    if times is not None and traces:
        _replace_trace(md, tag, np.vstack(traces), times)
    deposit_structures(md, f"md_{tag}", finals)
    set_level(md, tag, **meta, source=source, ensemble=ensemble,
              temperature=temperature, steps=steps, timestep=timestep,
              n_failed=failed)
    _check_thermostat(md, tag, temperature)
    record(md, "md.run", level=level, source=source, temperature=temperature,
           steps=steps, ensemble=ensemble)


def _replace_trace(md: AnnData, tag: str, block: np.ndarray,
                   times: np.ndarray) -> None:
    """Deposit the temperature trace, discarding any earlier one of a
    different length.

    Grids exist so two levels of the same quantity can be compared, and
    ``deposit_grid`` refuses when a new grid disagrees with the stored one.
    That is right for a diffraction pattern, whose axis the caller chooses.
    A trace axis is not chosen: its length falls out of ``steps`` and
    ``sample_every``, so a second run of different length produces a curve that
    genuinely cannot share an axis with the first. Keeping both is impossible,
    so the stale one goes.
    """
    grids = md.uns.setdefault("grids", {})
    stored = grids.get("md_temperature_trace")
    if stored is not None and not np.array_equal(
            np.asarray(stored.get("values"), dtype=float), times):
        for key in [k for k in md.obsm
                    if k.startswith("md_temperature_trace_")]:
            del md.obsm[key]
        del grids["md_temperature_trace"]
    deposit_grid(md, "md_temperature_trace", tag, block, times, unit="K")


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

    trace: list[float] = []
    for step in range(0, steps, sample_every):
        dynamics.run(min(sample_every, steps - step))
        displacement = atoms.get_positions() - reference
        samples.append(((step + sample_every) * timestep * 1e-3,   # ps
                        (displacement ** 2).sum(axis=1)))
        energy += float(atoms.get_potential_energy())
        instantaneous = float(atoms.get_temperature())
        kinetic_t += instantaneous
        # Keep the trace, not just its mean. The mean cannot distinguish a run
        # that equilibrated from one still drifting at the last step, and that
        # distinction is the difference between a result and an artefact.
        trace.append(instantaneous)
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
            "trace": np.asarray(trace, dtype=float),
            "times": times,
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


__all__ = ["run", "sweep", "rdf", "sites", "van_hove", "occupancy",
           "conductivity",
           "melt_quench",
           "register_batched", "batched_available"]


@register_function(
    aliases=["trajectory rdf", "dynamic rdf", "time averaged rdf",
             "pair correlation from md", "liquid structure",
             "md radial distribution"],
    category="md",
    description="Radial distribution function averaged over a trajectory, "
                "with the running coordination number, on the shared grid "
                "convention.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["rdf_md_{level}", "coordination_md_{level}"],
              "uns": ["grids"], "obs": ["first_shell_{level}",
                                        "first_shell_coordination_{level}"],
              "levels": ["{level}"]},
    examples=["mv.md.rdf(md, trajectories, species='Li')",
              "mv.md.rdf(md, trajectories, species='Li', r_max=8.0)"],
    related=["mv.prop.rdf", "mv.md.run", "mv.prop.compare_grids"],
    notes="mv.prop.rdf takes one static structure. This takes a trajectory and "
          "averages over it, which is a different quantity: the static "
          "function reports where the atoms are in one snapshot, and this "
          "reports where they spend their time. For a crystal near 0 K they "
          "converge; for a liquid, a superionic conductor or anything above "
          "half its melting point they do not, and the difference is the "
          "thermal broadening that makes a real diffraction pattern wider "
          "than a simulated one.\\n\\n"
          "**The trajectory is an argument**, because mv.md.run deliberately "
          "does not keep one — a screening library that materialised every "
          "frame would spend its memory on positions nobody reads. Pass "
          "fractional coordinates shaped (frames, atoms, 3), from whatever "
          "produced them.\\n\\n"
          "obs['first_shell'] is the position of the first peak and "
          "first_shell_coordination the running coordination number at the "
          "following minimum, which is the coordination number a "
          "diffractionist means.\n\n"
          "That integral is computed here from its definition, "
          "n(r) = integral of 4 pi r^2 rho g(r), rather than taken from "
          "pymatgen's coordination_number, which reports the count **per "
          "reference index**. On a cell where the mobile ion has twelve "
          "neighbours spread over three reference sites, pymatgen returns 4.0 "
          "and the definition returns 11.4 — the shortfall from twelve being "
          "the Gaussian smearing spilling past the cutoff. Four is not a "
          "coordination number for that cell.",
)
def rdf(md: AnnData, trajectories, species: str, source: str = "input",
        level: str = "md", reference: str | None = None, r_max: float = 10.0,
        n_grid: int = 101, sigma: float = 0.1) -> None:
    """Trajectory-averaged RDF on a shared grid. Deposits; returns ``None``."""
    try:
        from pymatgen.analysis.diffusion.aimd.rdf import (
            RadialDistributionFunction)
    except ImportError as exc:
        raise ImportError(
            f"mv.md.rdf needs pymatgen-analysis-diffusion, one of pymatgen's "
            f"own add-on packages. Install it with `pip install "
            f"pymatgen-analysis-diffusion`. ({exc})") from exc

    frames = np.asarray(trajectories, dtype=float)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(
            f"trajectories must be (frames, atoms, 3) fractional coordinates; "
            f"got shape {frames.shape}")

    S = structures(md, source)
    if md.n_obs != 1:
        raise ValueError(
            f"one trajectory belongs to one structure, and this dataset has "
            f"{md.n_obs} rows. Subset it first — md[[i]].copy() — or call this "
            f"once per material.")
    structure = S[0]
    if frames.shape[1] != len(structure):
        raise ValueError(
            f"the trajectory has {frames.shape[1]} atoms and the structure has "
            f"{len(structure)}; they must be the same cell in the same order")

    mobile = [i for i, site in enumerate(structure)
              if site.specie.symbol == str(species)]
    if not mobile:
        raise ValueError(
            f"no {species!r} in this structure; it has "
            f"{sorted({s.specie.symbol for s in structure})}")
    other = str(reference) if reference else None
    partners = [i for i, site in enumerate(structure)
                if (site.specie.symbol == other if other
                    else i not in mobile)]
    if not partners:
        raise ValueError(
            f"no reference atoms to measure against; reference={reference!r}")

    snapshots = []
    for frame in frames:
        snapshot = structure.copy()
        for index, coords in enumerate(frame):
            snapshot[index] = snapshot[index].specie, coords
        snapshots.append(snapshot)

    analysis = RadialDistributionFunction(
        snapshots, indices=mobile, reference_indices=partners,
        ngrid=int(n_grid), rmax=float(r_max), sigma=float(sigma))
    grid = np.asarray(analysis.interval, dtype=float)
    curve = np.asarray(analysis.rdf, dtype=float)
    # The running coordination number from its definition rather than from
    # pymatgen's coordination_number, which reports the count *per reference
    # index*: on a cell where the mobile ion has twelve neighbours across three
    # reference sites it returns 4.0, and 4 is not a coordination number.
    density = len(partners) / float(structure.volume)
    step = float(grid[1] - grid[0]) if grid.size > 1 else 0.0
    coordination = np.cumsum(
        4.0 * np.pi * grid ** 2 * density * curve) * step

    deposit_grid(md, "rdf_md", level, curve[None, :], grid, unit="angstrom",
                 species=str(species), reference=other or "all others",
                 n_frames=int(frames.shape[0]))
    deposit_grid(md, "coordination_md", level, coordination[None, :], grid,
                 unit="angstrom", species=str(species))

    peak, shell = _first_shell(grid, curve, coordination)
    md.obs[f"first_shell_{level}"] = [peak]
    md.obs[f"first_shell_coordination_{level}"] = [shell]
    set_level(md, level, kind="md", method=f"trajectory RDF ({species})",
              reference=None, surrogate=False, license=None, uncertainty=None,
              source=source, n_frames=int(frames.shape[0]))
    record(md, "md.rdf", species=str(species), level=level,
           n_frames=int(frames.shape[0]))


def _first_shell(grid, curve, coordination):
    """The first peak, and the coordination number at the minimum after it."""
    if not np.isfinite(curve).any() or curve.max() <= 0:
        return float("nan"), float("nan")
    peak = int(np.argmax(curve))
    tail = curve[peak:]
    if tail.size < 3:
        return float(grid[peak]), float(coordination[peak])
    # First point after the peak where the curve stops falling.
    falling = np.diff(tail)
    turning = np.argmax(falling > 0) if (falling > 0).any() else tail.size - 1
    minimum = peak + int(turning)
    return float(grid[peak]), float(coordination[minimum])


@register_function(
    aliases=["md sites", "occupied sites", "cluster trajectory",
             "where do the ions sit", "site occupation", "hopping sites",
             "kmeans pbc", "trajectory clustering"],
    category="md",
    description="Cluster where one species actually spent its time during a "
                "run, recovering the sites it occupied and how far it rattled "
                "about them.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["md_sites_{species}_{level}",
                      "md_site_spread_{species}_{level}",
                      "md_site_visits_{species}_{level}"]},
    prerequisites=["mv.md.run"],
    examples=["mv.md.sites(md, trajectories, species='Li')",
              "mv.md.sites(md, trajectories, species='Li', n_sites=8)"],
    related=["mv.md.run", "mv.md.rdf", "mv.neb.percolation",
             "mv.md.conductivity"],
    notes="A mean-squared displacement says how far the ions went. It cannot "
          "say whether they went there by rattling harder in the same well or "
          "by hopping between wells, and those are different materials. "
          "Clustering the sampled positions separates them: a vibrating ion "
          "gives tight clusters and each ion keeps to one of them, while a "
          "hopping ion visits several.\n\n"
          "obs['md_site_spread'] is the RMS distance from a position to its "
          "own site centre, in angstroms — a thermal vibration amplitude, of "
          "order 0.1 A for a solid well below melting. "
          "obs['md_site_visits'] is the mean number of distinct sites one "
          "atom of the species was found at over the run: 1.0 means nothing "
          "hopped, and anything above it counts hops the MSD alone would have "
          "reported as a larger number without saying why.\n\n"
          "**The trajectory is an argument**, on the same reasoning as "
          "mv.md.rdf: mv.md.run does not keep one. Pass fractional "
          "coordinates shaped (frames, atoms, 3).\n\n"
          "n_sites defaults to the number of atoms of that species in the "
          "cell, which is the right guess when each ion has its own site and "
          "the wrong one for an interstitial mechanism where there are more "
          "wells than ions. k-means needs the count in advance and cannot "
          "discover it, so this is a parameter rather than a result.",
)
def sites(md: AnnData, trajectories, species: str, source: str = "input",
          level: str = "md", n_sites: int | None = None,
          max_iterations: int = 200, key_added: str | None = None) -> None:
    """Cluster a trajectory into occupied sites. Deposits; returns ``None``."""
    try:
        from pymatgen.analysis.diffusion.aimd.clustering import KmeansPBC
    except ImportError as exc:                             # pragma: no cover
        raise ImportError(
            f"mv.md.sites needs pymatgen-analysis-diffusion, one of "
            f"pymatgen's own add-on packages. Install it with `pip install "
            f"pymatgen-analysis-diffusion`. ({exc})") from exc

    frames = np.asarray(trajectories, dtype=float)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(f"trajectories must be (frames, atoms, 3) fractional "
                         f"coordinates, got shape {frames.shape}")

    name = key_added or f"{species}_{level}"
    counts, spreads, visits = [], [], []

    for index, structure in enumerate(structures(md, source)):
        mobile = [i for i, site in enumerate(structure)
                  if site.specie.symbol == species]
        if not mobile or index >= len(frames) and frames.shape[0] == md.n_obs:
            counts.append(0)
            spreads.append(np.nan)
            visits.append(np.nan)
            continue
        if frames.shape[1] != len(structure):
            raise ValueError(
                f"row {index}: the trajectory has {frames.shape[1]} atoms and "
                f"the structure has {len(structure)}; they must be the same "
                f"cell")

        # (frames, mobile, 3) -> one point per atom per frame, but remember
        # which atom each point came from: the number of distinct sites *one
        # atom* visited is the hop count, and pooling the atoms would lose it.
        points = frames[:, mobile, :] % 1.0
        flat = points.reshape(-1, 3)
        k = int(n_sites) if n_sites else len(mobile)
        k = max(1, min(k, len(flat)))

        # Seed the search at the crystallographic sites of the species, not at
        # random points. KmeansPBC's default picks k of the input points with
        # an unseeded random.sample, so the same trajectory gives a different
        # answer on every call - unacceptable for a number anyone reports.
        # Starting from the sites is also the right guess: the question is
        # whether the ions stayed near them.
        seed = np.asarray([structure[i].frac_coords for i in mobile],
                          dtype=float) % 1.0
        if k <= len(seed):
            start = seed[:k]
        else:
            extra = flat[np.linspace(0, len(flat) - 1, k - len(seed),
                                     dtype=int)]
            start = np.vstack([seed, extra])

        try:
            centroids, labels, _ = KmeansPBC(
                structure.lattice, max_iterations=max_iterations
            ).cluster(flat, k, initial_centroids=start)
        except Exception:
            counts.append(0)
            spreads.append(np.nan)
            visits.append(np.nan)
            continue

        labels = np.asarray(labels).reshape(points.shape[0], len(mobile))
        centroids = np.asarray(centroids, dtype=float) % 1.0

        # Distance from each sampled position to the centre of its own site,
        # through the lattice rather than in fractional coordinates, so the
        # number is an angstrom a person can compare to a thermal amplitude.
        assigned = centroids[labels.reshape(-1)]
        deltas = flat - assigned
        deltas -= np.round(deltas)
        cartesian = deltas @ np.asarray(structure.lattice.matrix, dtype=float)
        spreads.append(float(np.sqrt((cartesian ** 2).sum(axis=1).mean())))
        counts.append(int(len(np.unique(labels))))
        visits.append(float(np.mean([len(np.unique(labels[:, atom]))
                                     for atom in range(len(mobile))])))

    md.obs[f"md_sites_{name}"] = np.array(counts, dtype=int)
    md.obs[f"md_site_spread_{name}"] = np.array(spreads, dtype=float)
    md.obs[f"md_site_visits_{name}"] = np.array(visits, dtype=float)
    md.uns.setdefault("md_sites", {})[name] = {
        "species": species, "n_sites": n_sites, "n_frames": int(frames.shape[0]),
        "spread_unit": "angstrom",
        "visits_meaning": "mean distinct sites visited by one atom; 1.0 means "
                          "no hopping",
    }
    record(md, "md.sites", species=species, source=source, level=level,
           n_sites=n_sites, key_added=name)


@register_function(
    aliases=["van hove", "van hove correlation", "self correlation",
             "distinct correlation", "displacement distribution",
             "how far did they move", "Gs", "Gd"],
    category="md",
    description="Van Hove correlation function from a trajectory: how the "
                "distribution of where an atom is, relative to where it or "
                "its neighbours were, spreads out with time.",
    requires={"structures": ["{source}"]},
    produces={"obsm": ["van_hove_self_{level}", "van_hove_distinct_{level}"],
              "uns": ["grids"],
              "obs": ["van_hove_rms_{level}", "van_hove_peak_{level}",
                      "van_hove_jump_{level}"]},
    prerequisites=["mv.md.run"],
    examples=["mv.md.van_hove(md, trajectories, species='Li')",
              "mv.md.van_hove(md, trajectories, species='Li', dt=20)"],
    related=["mv.md.rdf", "mv.md.sites", "mv.md.run", "mv.prop.rdf"],
    notes="Two functions, both on the shared r grid. The **self** part is the "
          "distribution of how far one atom moved in time dt — it starts as a "
          "spike at r=0 and spreads. The **distinct** part is where the other "
          "atoms were relative to it, and at dt=0 it is exactly the radial "
          "distribution function, which is the check worth running: "
          "obsm['van_hove_distinct'] at dt=0 must reproduce mv.prop.rdf on "
          "the same cell.\n\n"
          "What it tells you that a diffusivity cannot: the *shape*. In a "
          "liquid the self part stays a single spreading Gaussian. In a "
          "hopping solid it grows a second bump at the jump distance while "
          "the peak at zero survives, because most ions did not move and the "
          "few that did moved a lattice spacing. A diffusivity averages those "
          "two populations into one number that describes neither.\n\n"
          "Three scalars come off it. obs['van_hove_rms'] is the "
          "root-mean-square displacement over dt in angstroms. "
          "obs['van_hove_peak'] is where the self part is largest — the *most "
          "probable* displacement, which for a vibrating solid sits near "
          "sqrt(2) times the one-dimensional amplitude rather than at zero, "
          "because the volume of a shell grows as r^2 and beats the falling "
          "Gaussian. obs['van_hove_jump'] is the outermost local maximum "
          "worth more than a hundredth of the largest one, which is the "
          "jump distance when ions hop and NaN when they only rattle.\n\n"
          "Computed here from the definition. pymatgen's VanHoveAnalysis "
          "exposes get_1d_plot and get_3d_plot and no data accessor, so "
          "wrapping it would mean reading private attributes, and matverse "
          "deposits data rather than pictures.\n\n"
          "**Displacements use the minimum image convention**, so a "
          "displacement longer than half the shortest cell vector folds back "
          "and is reported short. That is a property of the cell, not of this "
          "function: run a bigger box or a shorter dt.",
)
def van_hove(md: AnnData, trajectories, species: str | None = None,
             source: str = "input", level: str = "md", dt: int = 1,
             r_max: float = 10.0, n_grid: int = 101, sigma: float = 0.1
             ) -> None:
    """Self and distinct van Hove functions. Deposits; returns ``None``."""
    frames = np.asarray(trajectories, dtype=float)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(f"trajectories must be (frames, atoms, 3) fractional "
                         f"coordinates, got shape {frames.shape}")
    if dt < 0 or dt >= frames.shape[0]:
        raise ValueError(f"dt must be between 0 and {frames.shape[0] - 1} "
                         f"frames, got {dt}")

    grid = np.linspace(0.0, float(r_max), int(n_grid))
    self_part = np.zeros((md.n_obs, len(grid)))
    distinct = np.zeros((md.n_obs, len(grid)))
    rms = np.full(md.n_obs, np.nan)
    peak = np.full(md.n_obs, np.nan)
    jump = np.full(md.n_obs, np.nan)

    for row, structure in enumerate(structures(md, source)):
        if frames.shape[1] != len(structure):
            raise ValueError(
                f"row {row}: the trajectory has {frames.shape[1]} atoms and "
                f"the structure has {len(structure)}; they must be the same "
                f"cell")
        chosen = [i for i, site in enumerate(structure)
                  if species is None or site.specie.symbol == species]
        if not chosen:
            continue

        matrix = np.asarray(structure.lattice.matrix, dtype=float)
        volume = float(structure.lattice.volume)
        positions = frames[:, chosen, :]
        n_pairs_self, n_pairs_distinct = 0, 0
        self_distances, distinct_distances = [], []

        # Average over every start frame that admits an interval of dt, which
        # is what makes this a correlation function rather than one snapshot
        # of one displacement.
        for start in range(frames.shape[0] - dt):
            delta = positions[start + dt] - positions[start]
            delta -= np.round(delta)                     # minimum image
            self_distances.append(
                np.linalg.norm(delta @ matrix, axis=1))
            n_pairs_self += len(chosen)

            cross = positions[start + dt][:, None, :] - positions[start][None]
            cross -= np.round(cross)
            lengths = np.linalg.norm(cross @ matrix, axis=2)
            off = ~np.eye(len(chosen), dtype=bool)
            distinct_distances.append(lengths[off])
            n_pairs_distinct += int(off.sum())

        self_distances = np.concatenate(self_distances)
        distinct_distances = np.concatenate(distinct_distances)

        self_part[row] = _smear(self_distances, grid, sigma) / max(
            n_pairs_self, 1)
        # The distinct part is normalised to the ideal-gas count, so that a
        # structureless system gives 1 and the dt=0 curve is the RDF itself
        # rather than an unnormalised histogram of pair separations.
        density = len(chosen) / volume
        shell = 4.0 * np.pi * np.maximum(grid, 1e-8) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            distinct[row] = (_smear(distinct_distances, grid, sigma)
                             / max(n_pairs_distinct, 1)
                             / (shell * density) * len(chosen))

        rms[row] = float(np.sqrt((self_distances ** 2).mean()))
        finite = np.isfinite(self_part[row])
        if finite.any():
            peak[row] = float(grid[np.argmax(np.where(finite, self_part[row],
                                                      -np.inf))])
        jump[row] = _outer_feature(grid, self_part[row])

    deposit_grid(md, "van_hove_self", level, self_part, grid, unit="1/A",
                 dt=int(dt), species=species, sigma=float(sigma))
    deposit_grid(md, "van_hove_distinct", level, distinct, grid, unit="",
                 dt=int(dt), species=species, sigma=float(sigma))
    md.obs[f"van_hove_rms_{level}"] = rms
    md.obs[f"van_hove_peak_{level}"] = peak
    md.obs[f"van_hove_jump_{level}"] = jump
    record(md, "md.van_hove", species=species, source=source, level=level,
           dt=int(dt), r_max=float(r_max))


def _smear(distances, grid, sigma: float):
    """Distances onto a grid as normalised Gaussians rather than a histogram.

    A histogram of a few thousand distances is mostly empty bins and its shape
    depends on where the bin edges fell. Each distance contributes a unit
    Gaussian instead, so the curve is smooth and integrates to the number of
    distances however the grid was chosen.
    """
    distances = np.asarray(distances, dtype=float)
    if not len(distances):
        return np.zeros(len(grid))
    inside = distances <= grid[-1] + 5.0 * sigma
    distances = distances[inside]
    if not len(distances):
        return np.zeros(len(grid))
    # Reflected at r = 0. A distance is a magnitude, so a Gaussian placed at
    # d leaks into r < 0, where nothing can live; the mirror image folds that
    # weight back. Without it a displacement of zero - every atom, at dt = 0 -
    # keeps only the half of its Gaussian above the origin and the self part
    # integrates to 0.5 rather than 1.
    delta = grid[None, :] - distances[:, None]
    mirror = grid[None, :] + distances[:, None]
    weight = (np.exp(-0.5 * (delta / sigma) ** 2)
              + np.exp(-0.5 * (mirror / sigma) ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return weight.sum(axis=0)


def _outer_feature(grid, curve, threshold: float = 0.01) -> float:
    """The outermost local maximum of the self part, or NaN if there is none.

    Not the largest one. At any useful dt most atoms have not hopped, and the
    self part's tallest feature is the vibrational peak near sqrt(2) times the
    one-dimensional amplitude — the r^2 shell volume puts it there rather than
    at the origin. A jump shows up as a *further out* bump carrying only the
    fraction of atoms that moved, so it is found by position, not by height,
    with a threshold to keep numerical ripple from qualifying.
    """
    curve = np.asarray(curve, dtype=float)
    if len(curve) < 5 or not np.isfinite(curve).any():
        return float("nan")
    tallest = float(np.nanmax(curve))
    if not np.isfinite(tallest) or tallest <= 0:
        return float("nan")
    interior = np.arange(1, len(curve) - 1)
    maxima = interior[(curve[1:-1] > curve[:-2]) & (curve[1:-1] >= curve[2:])
                      & (curve[1:-1] > threshold * tallest)]
    if len(maxima) < 2:
        # One maximum is the vibrational peak itself; there is no jump.
        return float("nan")
    return float(grid[maxima[-1]])


@register_function(
    aliases=["occupancy", "probability density", "where the ions go",
             "ion density", "delocalisation", "explored volume",
             "superionic", "smeared or localised"],
    category="md",
    description="How much of the cell the mobile ions actually explore, from "
                "the probability density of finding one anywhere in it.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["occupied_fraction_{level}", "occupancy_entropy_{level}",
                      "occupancy_peak_{level}"]},
    prerequisites=["mv.md.run"],
    examples=["mv.md.occupancy(md, trajectories, species='Li')",
              "mv.md.occupancy(md, trajectories, species='Li', bins=32)"],
    related=["mv.md.sites", "mv.md.van_hove", "mv.neb.percolation",
             "mv.md.conductivity"],
    notes="Histogram the mobile ions' fractional positions over the run and "
          "you have the probability of finding one in each corner of the "
          "cell. A normal solid puts nearly all of it in small blobs at the "
          "lattice sites; a superionic conductor smears it along the channels "
          "between them. That difference is visible in the density long "
          "before it is significant in a diffusivity, because it does not "
          "need the ion to complete a journey — only to spend time on the "
          "way.\n\n"
          "obs['occupied_fraction'] is the fraction of the cell holding 90% "
          "of the probability, which is the number to read: near zero for "
          "ions sitting still, rising as they delocalise, and about 0.87 for "
          "a well-sampled uniform distribution — not 1.0, because covering "
          "90% of a uniform probability still leaves out the thin tail of "
          "voxels that happened to be visited least. "
          "obs['occupancy_entropy'] is the Shannon entropy of the same "
          "histogram divided by its maximum, so it runs 0 to 1 on any grid "
          "and can be compared between cells of different size. "
          "obs['occupancy_peak'] is the largest single-voxel probability, "
          "which falls as the density spreads.\n\n"
          "Computed here from the definition. pymatgen's "
          "ProbabilityDensityAnalysis sits beside generate_stable_sites, "
          "which raises on a structure with one stable site — the condensed "
          "distance matrix is empty and scipy's linkage refuses it — and one "
          "well-localised site is the commonest case there is.\n\n"
          "**The grid is the parameter that matters, and it interacts with "
          "how long you ran.** Too coarse and everything looks delocalised "
          "because one voxel spans several sites. Too fine and everything "
          "looks localised for a reason that has nothing to do with the "
          "material: with fewer samples than voxels, most voxels are empty "
          "however the ions moved. A uniform distribution reads 0.87 at "
          "sixteen samples per voxel and 0.05 at a tenth of a sample per "
          "voxel — same physics, same function, different sampling — so this "
          "warns below five per voxel rather than returning the number "
          "quietly. bins is per lattice vector; the voxel edge lengths and "
          "the sampling density are both recorded in uns.\n\n"
          "Compare these numbers between runs of the same length on the same "
          "grid. Across different ones they are not comparable, and no "
          "normalisation makes them so.",
)
def occupancy(md: AnnData, trajectories, species: str | None = None,
              source: str = "input", level: str = "md", bins: int = 24,
              coverage: float = 0.9, key_added: str | None = None) -> None:
    """Probability density of the mobile ions. Deposits; returns ``None``."""
    frames = np.asarray(trajectories, dtype=float)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(f"trajectories must be (frames, atoms, 3) fractional "
                         f"coordinates, got shape {frames.shape}")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage is a fraction of the probability and must "
                         f"be in (0, 1], got {coverage}")

    name = key_added or level
    fraction = np.full(md.n_obs, np.nan)
    entropy = np.full(md.n_obs, np.nan)
    peak = np.full(md.n_obs, np.nan)
    voxels: list = []
    sampling: list = []

    for row, structure in enumerate(structures(md, source)):
        if frames.shape[1] != len(structure):
            raise ValueError(
                f"row {row}: the trajectory has {frames.shape[1]} atoms and "
                f"the structure has {len(structure)}; they must be the same "
                f"cell")
        mobile = [i for i, site in enumerate(structure)
                  if species is None or site.specie.symbol == species]
        if not mobile:
            voxels.append(None)
            sampling.append(float("nan"))
            continue

        points = (frames[:, mobile, :] % 1.0).reshape(-1, 3)
        per_voxel = len(points) / float(bins ** 3)
        if per_voxel < 5.0:
            warnings.warn(
                f"row {row}: {len(points)} sampled positions over "
                f"{bins ** 3} voxels is {per_voxel:.2f} per voxel. Below "
                f"about five, most voxels are empty because the run was "
                f"short rather than because the ions were localised, and "
                f"occupied_fraction is driven by the sampling. Use fewer "
                f"bins or more frames.", stacklevel=2)
        counts, _ = np.histogramdd(
            points, bins=(bins, bins, bins),
            range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
        total = counts.sum()
        if total <= 0:
            voxels.append(None)
            sampling.append(float("nan"))
            continue
        probability = (counts / total).ravel()

        # The smallest set of voxels that between them hold `coverage` of the
        # probability, as a fraction of the cell. Taking every visited voxel
        # instead would count the single frame an ion spent in flight the same
        # as the thousand it spent at rest.
        ordered = np.sort(probability)[::-1]
        needed = int(np.searchsorted(np.cumsum(ordered), coverage) + 1)
        fraction[row] = needed / probability.size

        positive = probability[probability > 0]
        entropy[row] = float(-(positive * np.log(positive)).sum()
                             / np.log(probability.size))
        peak[row] = float(probability.max())
        voxels.append([float(length / bins)
                       for length in structure.lattice.abc])
        sampling.append(float(per_voxel))

    md.obs[f"occupied_fraction_{name}"] = fraction
    md.obs[f"occupancy_entropy_{name}"] = entropy
    md.obs[f"occupancy_peak_{name}"] = peak
    md.uns.setdefault("occupancy", {})[name] = {
        "species": species, "bins": int(bins), "coverage": float(coverage),
        "n_frames": int(frames.shape[0]),
        "voxel_edges_angstrom": voxels,
        "samples_per_voxel": sampling,
        "meaning": "occupied_fraction is the fraction of the cell holding "
                   f"{coverage:.0%} of the probability of finding a mobile ion",
    }
    record(md, "md.occupancy", species=species, source=source, level=level,
           bins=int(bins), coverage=float(coverage), key_added=name)

# User API

Import matverse as:

```python
import matverse as mv
```

This page is generated from the `@register_function` entries in the matverse
registry, so it lists exactly what the library exposes to a caller — and to an
agent. Every entry names the state it reads and the state it writes, and each of
those claims is verified by execution in `tests/test_contracts.py` rather than
asserted.

Public registry entries listed here: 88

Look a function up by intent rather than by name:

```python
mv.find('thermodynamic stability')      # ['mv.thermo.hull', ...]
print(mv.describe('convex hull'))       # signature, contract, examples
```

```{eval-rst}
.. currentmodule:: matverse
```


## Data IO

Build a dataset, and get it back out again.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   data.from_ase
   data.from_cif
   data.from_matminer
   data.from_mp
   data.from_optimade
   data.from_optimade_response
   data.from_structures
   data.optimade_providers
   data.to_ase
   data.to_cif
   data.to_matminer
   data.to_pymatgen
```


## Preprocessing

Structure standardisation, quality control, filtering, deduplication and cross-database harmonisation.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pp.dedup
   pp.describe
   pp.filter_elements
   pp.filter_materials
   pp.harmonize
   pp.normalize_composition
   pp.qc
   pp.rattle
   pp.standardize
   pp.strain
   pp.supercell
```


## Featurisation

Descriptors into `obsm`.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   feat.element_stats
   feat.matminer
   feat.similarity
   feat.soap
```


## Tools

Analysis on the composition matrix — ordination, clustering, element enrichment and novelty.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.cluster
   tl.neighbors
   tl.novelty
   tl.pca
   tl.rank_elements_groups
```


## Calculators

Energies, forces and relaxation, tagged by level of theory.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   calc.available
   calc.check_licenses
   calc.committee
   calc.energy
   calc.forces
   calc.register_calculator
   calc.relax
```


## Properties

Derived properties, including curves stored on a shared grid.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   prop.compare_grids
   prop.elastic
   prop.free_energy
   prop.phonon
   prop.rdf
   prop.xrd
```


## Thermodynamics

Convex hull, reactions, chemical potentials.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   thermo.chempot_limits
   thermo.hull
   thermo.reaction
   thermo.references_from_mp
```


## First principles

Input generation and result harvesting. Submission stays delegated.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   dft.presets
   dft.read_outputs
   dft.status
   dft.write_inputs
```


## Sites axis

Per-atom results, on a companion object whose rows are atoms.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   multi.aggregate
   multi.sites
   multi.to_mudata
```


## Experiment

Measured data, carried as a level of theory like any other.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   exp.attach
   exp.match_xrd
   exp.measure
```


## Screening

Filtering, ranking and Pareto fronts that leave a record.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   screen.filter
   screen.pareto
   screen.rank
```


## Generated candidates

Scoring generated structures, and enumerating substitutions.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   gen.substitute
   gen.validate
```


## Machine learning

Property prediction, with splits that do not leak.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   model.available
   model.cross_validate
   model.fit
   model.register_model
   model.split
```


## Design campaigns

Choosing what to compute next, and recording each round.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   opt.history
   opt.observe
   opt.start
   opt.suggest
```


## Plotting

Publication defaults; every function draws onto an axis and returns it.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.embedding
   pl.hull
   pl.pareto
   pl.parity
   pl.periodic_table
   pl.provenance
   pl.rank_elements_groups
   pl.spectra
```


## Infrastructure

Units, checkpointing, cluster submission and object summaries.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   utils.check_units
   utils.checkpoint
   utils.convert
   utils.resume
   utils.set_units
   utils.slurm_script
   utils.summary
```


## What each function writes

The table below is the `produces` half of the registry contract: the slots a
call deposits into the object. Names in braces are templated on the call's own
arguments — `obs['energy_{level}']` becomes `obs['energy_emt']` when you pass
`level='emt'`.

| Function | Writes |
|---|---|
| `mv.data.from_ase` | `obsm['structures']['input']`, `uns['sites']`, `X` |
| `mv.data.from_cif` | `obsm['structures']['input']`, `obs['source_file']`, `X` |
| `mv.data.from_matminer` | `obsm['structures']['input']`, `obsm['X_matminer']`, `X` |
| `mv.data.from_mp` | `obsm['structures']['input']`, `obs['material_id']`, `obs['formula']`, `uns['levels']['mp']`, `X` |
| `mv.data.from_optimade` | `obsm['structures']['input']`, `obs['optimade_id']`, `obs['provider']`, `X`, `uns['levels']['{provider}']` |
| `mv.data.from_optimade_response` | `obsm['structures']['input']`, `obs['optimade_id']`, `obs['provider']`, `X`, `uns['levels']['{provider}']` |
| `mv.data.from_structures` | `obsm['structures']['input']`, `X`, `var['Z']` |
| `mv.data.to_cif` | `written to disk` |
| `mv.pp.dedup` | `obs['duplicate_of']`, `obs['is_duplicate']`, `uns['dedup']` |
| `mv.pp.describe` | `obs['formula']`, `obs['nsites']`, `obs['volume']`, `obs['density']`, `obs['n_elements']`, `obs['volume_per_atom']` |
| `mv.pp.harmonize` | `obs['{energy_key}_harmonized']`, `uns['harmonize']` |
| `mv.pp.normalize_composition` | `layers['fraction']` |
| `mv.pp.qc` | `obs['min_distance']`, `obs['is_ordered']`, `obs['is_charge_balanced']`, `obs['is_valid']`, `obs['qc_reason']` |
| `mv.pp.rattle` | `obsm['structures']['{name}']` |
| `mv.pp.standardize` | `obsm['structures']['primitive']`, `obsm['structures']['conventional']`, `obs['spacegroup']`, `obs['spacegroup_number']`, `obs['crystal_system']`, `obs['nsites_primitive']` |
| `mv.pp.strain` | `obsm['structures']['{name}']` |
| `mv.pp.supercell` | `obsm['structures']['{name}']` |
| `mv.feat.element_stats` | `obsm['X_element_stats']`, `uns['features']['X_element_stats']` |
| `mv.feat.matminer` | `obsm['X_matminer']`, `uns['features']['X_matminer']` |
| `mv.feat.similarity` | `obsp['similarity_{block}']` |
| `mv.feat.soap` | `obsm['X_soap']`, `uns['features']['X_soap']` |
| `mv.tl.cluster` | `obs['{key_added}']`, `uns['cluster']` |
| `mv.tl.neighbors` | `obsp['connectivities']`, `obsp['distances']`, `uns['neighbors']` |
| `mv.tl.novelty` | `obs['novelty_distance']`, `obs['nearest_reference']` |
| `mv.tl.pca` | `obsm['X_pca']`, `varm['PCs']`, `uns['pca']` |
| `mv.tl.rank_elements_groups` | `uns['rank_elements_groups']` |
| `mv.calc.committee` | `obs['energy_per_atom_{key}']`, `obs['energy_per_atom_{key}_std']`, `uns['levels']['{key}']` |
| `mv.calc.energy` | `obs['energy_{level}']`, `obs['energy_per_atom_{level}']`, `uns['levels']['{level}']` |
| `mv.calc.forces` | `uns['levels']['{level}']` |
| `mv.calc.relax` | `obsm['structures']['relaxed_{level}']`, `obs['energy_{level}']`, `obs['energy_per_atom_{level}']`, `obs['relax_converged_{level}']`, `obs['max_force_{level}']`, `uns['levels']['{level}']` |
| `mv.prop.compare_grids` | `obs['{quantity}_cosine_{a}_vs_{b}']`, `obs['{quantity}_rmse_{a}_vs_{b}']`, `obs['{quantity}_overlap_{a}_vs_{b}']` |
| `mv.prop.elastic` | `obs['bulk_modulus_{level}']`, `obs['shear_modulus_{level}']`, `obs['youngs_modulus_{level}']`, `obs['poisson_ratio_{level}']`, `obs['elastic_stable_{level}']`, `obsm['elastic_tensor_{level}']`, `uns['levels']['{level}']` |
| `mv.prop.free_energy` | `obs['vibrational_free_energy_{level}']`, `obs['vibrational_entropy_{level}']`, `obs['heat_capacity_{level}']` |
| `mv.prop.phonon` | `obsm['phonon_dos_{level}']`, `obs['n_imaginary_modes_{level}']`, `obs['dynamically_stable_{level}']`, `obs['zero_point_energy_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.rdf` | `obsm['rdf_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.xrd` | `obsm['xrd_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.thermo.chempot_limits` | `uns['chempot_limits']` |
| `mv.thermo.hull` | `obs['e_above_hull_{level}']`, `obs['is_stable_{level}']`, `obs['formation_energy_{level}']`, `obs['decomposes_to_{level}']`, `uns['phase_diagram']` |
| `mv.thermo.reaction` | `uns['reactions']` |
| `mv.dft.read_outputs` | `obs['energy_{level}']`, `obs['energy_per_atom_{level}']`, `obs['band_gap_{level}']`, `obs['converged_{level}']`, `obsm['structures']['relaxed_{level}']`, `uns['levels']['{level}']`, `uns['dft']` |
| `mv.dft.write_inputs` | `obs['dft_directory']`, `written to disk`, `uns['dft']` |
| `mv.multi.aggregate` | `obs['{key_added}']` |
| `mv.exp.attach` | `obsm['{quantity}_{level}']`, `uns['levels']['{level}']` |
| `mv.exp.match_xrd` | `obs['xrd_match']`, `obs['xrd_match_rank']`, `uns['xrd_match']` |
| `mv.exp.measure` | `obs['{quantity}_{level}']`, `uns['levels']['{level}']` |
| `mv.screen.filter` | `obs['{name}']`, `uns['screens']` |
| `mv.screen.pareto` | `obs['{name}']`, `obs['{name}_rank']`, `uns['pareto']` |
| `mv.screen.rank` | `obs['{name}']` |
| `mv.gen.validate` | `obs['gen_valid']`, `obs['gen_unique']`, `obs['gen_novel']`, `obs['gen_stable']`, `obs['gen_metastable']`, `obs['gen_sun']`, `obs['gen_msun']`, `uns['gen_validate']` |
| `mv.model.cross_validate` | `uns['cross_validate']` |
| `mv.model.fit` | `obs['{target}_{level}']`, `uns['levels']['{level}']`, `uns['model']` |
| `mv.model.split` | `obs['{key_added}']`, `uns['split']` |
| `mv.opt.observe` | `obs['campaign_round']`, `obs['selected']`, `uns['{name}']` |
| `mv.opt.start` | `obs['{observed_key}']`, `obs['campaign_round']`, `uns['campaign']` |
| `mv.opt.suggest` | `obs['acquisition']`, `obs['selected']`, `uns['{name}']` |
| `mv.utils.checkpoint` | `written to disk`, `uns['checkpoints']` |
| `mv.utils.convert` | `obs['{key_added}']`, `uns['units']` |
| `mv.utils.set_units` | `uns['units']` |
| `mv.utils.slurm_script` | `written to disk` |

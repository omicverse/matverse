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

Public registry entries listed here: 39

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
   data.from_structures
   data.to_ase
   data.to_cif
   data.to_matminer
   data.to_pymatgen
```


## Preprocessing

Structure standardisation, quality control, filtering and deduplication.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pp.dedup
   pp.describe
   pp.filter_elements
   pp.filter_materials
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

Energies and relaxation, tagged by level of theory.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   calc.available
   calc.check_licenses
   calc.committee
   calc.energy
   calc.register_calculator
   calc.relax
```


## Thermodynamics

Convex hull, energy above hull, decomposition products.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   thermo.hull
   thermo.references_from_mp
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
| `mv.data.from_structures` | `obsm['structures']['input']`, `X`, `var['Z']` |
| `mv.data.to_cif` | `written to disk` |
| `mv.pp.dedup` | `obs['duplicate_of']`, `obs['is_duplicate']`, `uns['dedup']` |
| `mv.pp.describe` | `obs['formula']`, `obs['nsites']`, `obs['volume']`, `obs['density']`, `obs['n_elements']`, `obs['volume_per_atom']` |
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
| `mv.calc.relax` | `obsm['structures']['relaxed_{level}']`, `obs['energy_{level}']`, `obs['energy_per_atom_{level}']`, `obs['relax_converged_{level}']`, `obs['max_force_{level}']`, `uns['levels']['{level}']` |
| `mv.thermo.hull` | `obs['e_above_hull_{level}']`, `obs['is_stable_{level}']`, `obs['formation_energy_{level}']`, `obs['decomposes_to_{level}']`, `uns['phase_diagram']` |
| `mv.screen.filter` | `obs['{name}']`, `uns['screens']` |
| `mv.screen.pareto` | `obs['{name}']`, `obs['{name}_rank']`, `uns['pareto']` |
| `mv.screen.rank` | `obs['{name}']` |

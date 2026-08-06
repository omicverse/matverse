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

Public registry entries listed here: 225

Look a function up by intent rather than by name:

```python
mv.find('thermodynamic stability')      # ['mv.thermo.hull', ...]
print(mv.describe('convex hull'))       # signature, contract, examples
```

```{eval-rst}
.. currentmodule:: matverse
```


## Datasets

Real published structures to work on, bundled or fetched.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   datasets.available
   datasets.cached
   datasets.fetch
   datasets.load
   datasets.metals
```


## Data IO

Build a dataset, and get it back out again.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   data.from_ase
   data.from_ase_file
   data.from_cif
   data.from_compositions
   data.from_iterable
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
   pp.defects
   pp.describe
   pp.filter_elements
   pp.filter_materials
   pp.harmonize
   pp.locate_defect
   pp.normalize_composition
   pp.predict_volume
   pp.prototype
   pp.qc
   pp.rattle
   pp.standardize
   pp.strain
   pp.supercell
   pp.symmetry
```


## Featurisation

Descriptors into `obsm`.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   feat.element_stats
   feat.embed
   feat.matminer
   feat.register_embedder
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

   prop.capture
   prop.compare_grids
   prop.configuration_coordinate
   prop.cost
   prop.dielectric
   prop.dimensionality
   prop.dispersion
   prop.efg
   prop.elastic
   prop.electrostatic
   prop.eos
   prop.free_energy
   prop.frontier_orbitals
   prop.neutron
   prop.nmr
   prop.phonon
   prop.phonon_at_temperature
   prop.piezo_from_dfpt
   prop.piezoelectric
   prop.polarization
   prop.quasiharmonic
   prop.rdf
   prop.scattering
   prop.slme
   prop.superconductivity
   prop.supply_risk
   prop.tem
   prop.thermal_conductivity
   prop.xrd
   prop.zt
```


## Molecular dynamics

Motion, and the properties only motion gives you.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   md.batched_available
   md.conductivity
   md.melt_quench
   md.occupancy
   md.rdf
   md.register_batched
   md.run
   md.sites
   md.sweep
   md.van_hove
```


## Magnetism

Magnetic orderings, and picking the ground state before the hull.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   mag.describe
   mag.exchange
   mag.ground_state
   mag.jahn_teller
   mag.orderings
   mag.symmetry
```


## Migration barriers

Nudged elastic band, and building the endpoints for it.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   neb.barrier
   neb.hop_endpoints
   neb.hops
   neb.percolation
```


## Surfaces

Slabs, surface energies, equilibrium shapes and adsorption.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   surf.adsorption_energy
   surf.adsorption_sites
   surf.scaling
   surf.slabs
   surf.surface_energy
   surf.surface_energy_chempot
   surf.volcano
   surf.wulff
```


## Thermodynamics

Convex hull, reactions, chemical potentials.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   thermo.calphad
   thermo.chempot_diagram
   thermo.chempot_limits
   thermo.corrections
   thermo.defect_formation
   thermo.fit_corrections
   thermo.hull
   thermo.pourbaix
   thermo.reaction
   thermo.references_from_mp
   thermo.theoretical_capacity
   thermo.voltage
```


## First principles

Input generation and result harvesting. Submission stays delegated.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   dft.presets
   dft.read_dos
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
   exp.formation_hull
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

   gen.alloy_pairs
   gen.compositions
   gen.from_symmetry
   gen.predict_dopants
   gen.predict_hosts
   gen.predict_substitutions
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

   pl.bands
   pl.distribution
   pl.elastic
   pl.embedding
   pl.fermi_surface
   pl.hull
   pl.pareto
   pl.parity
   pl.periodic_table
   pl.provenance
   pl.rank_elements_groups
   pl.scatter
   pl.set_style
   pl.spacegroups
   pl.spectra
   pl.structure
```


## Infrastructure

Units, checkpointing, cluster submission and object summaries.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   utils.check_units
   utils.checkpoint
   utils.chunks
   utils.convert
   utils.job_status
   utils.map_chunks
   utils.resume
   utils.set_units
   utils.slurm_script
   utils.submit
   utils.summary
```


## disorder

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   disorder.cluster_expansion
   disorder.describe
   disorder.dope
   disorder.monte_carlo
   disorder.orderings
   disorder.sqs
   disorder.sro
```


## elec

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   elec.band_features
   elec.bands
   elec.cohp
   elec.dos_fingerprint
   elec.fermi_surface
   elec.kpath
   elec.read_bands
   elec.transport
   elec.xps
```


## env

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   env.bonds
   env.chemenv
   env.connectivity
   env.coordination
   env.lobster
   env.summarise
   env.voronoi
```


## iface

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   iface.build
   iface.match
   iface.reactivity
```


## mol

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   mol.bond_lengths
   mol.bonds
   mol.descriptors
   mol.dissociation
   mol.fragments
   mol.from_molecules
   mol.functional_groups
   mol.match
   mol.point_group
   mol.quasirrho
```


## transform

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   transform.apply
   transform.available
   transform.chain
   transform.expand
   transform.oxidation_states
   transform.setting
```


## What each function writes

The table below is the `produces` half of the registry contract: the slots a
call deposits into the object. Names in braces are templated on the call's own
arguments — `obs['energy_{level}']` becomes `obs['energy_emt']` when you pass
`level='emt'`.

| Function | Writes |
|---|---|
| `mv.datasets.fetch` | `obsm['structures']['input']`, `X` |
| `mv.datasets.load` | `obsm['structures']['input']`, `X`, `obs['name']`, `obs['spacegroup']`, `obs['dataset']` |
| `mv.datasets.metals` | `obsm['structures']['input']`, `X`, `obs['name']`, `obs['lattice_parameter']` |
| `mv.data.from_ase` | `obsm['structures']['input']`, `X` |
| `mv.data.from_ase_file` | `obsm['structures']['input']`, `X` |
| `mv.data.from_cif` | `obsm['structures']['input']`, `obs['source_file']`, `X` |
| `mv.data.from_compositions` | `obs['formula']`, `X` |
| `mv.data.from_iterable` | `obsm['structures']['input']`, `X` |
| `mv.data.from_matminer` | `obsm['structures']['input']`, `obsm['X_matminer']`, `X` |
| `mv.data.from_mp` | `obsm['structures']['input']`, `obs['material_id']`, `obs['formula']`, `uns['levels']['mp']`, `X` |
| `mv.data.from_optimade` | `obsm['structures']['input']`, `obs['optimade_id']`, `obs['provider']`, `X`, `uns['levels']['{provider}']` |
| `mv.data.from_optimade_response` | `obsm['structures']['input']`, `obs['optimade_id']`, `obs['provider']`, `X`, `uns['levels']['{provider}']` |
| `mv.data.from_structures` | `obsm['structures']['input']`, `X`, `var['Z']` |
| `mv.data.to_cif` | `written to disk` |
| `mv.pp.dedup` | `obs['duplicate_of']`, `obs['is_duplicate']`, `uns['dedup']` |
| `mv.pp.describe` | `obs['formula']`, `obs['nsites']`, `obs['volume']`, `obs['density']`, `obs['n_elements']`, `obs['volume_per_atom']` |
| `mv.pp.harmonize` | `obs['{energy_key}_harmonized']`, `uns['harmonize']` |
| `mv.pp.locate_defect` | `obs['defect_a']`, `obs['defect_b']`, `obs['defect_c']`, `obs['defect_nearest_site']` |
| `mv.pp.normalize_composition` | `layers['fraction']` |
| `mv.pp.predict_volume` | `obs['predicted_volume']`, `obs['volume_scale']`, `obsm['structures']['{key_added}']` |
| `mv.pp.prototype` | `obs['prototype']`, `obs['prototype_mineral']`, `obs['strukturbericht']` |
| `mv.pp.qc` | `obs['min_distance']`, `obs['is_ordered']`, `obs['is_charge_balanced']`, `obs['is_valid']`, `obs['qc_reason']` |
| `mv.pp.rattle` | `obsm['structures']['{name}']` |
| `mv.pp.standardize` | `obsm['structures']['primitive']`, `obsm['structures']['conventional']`, `obs['spacegroup']`, `obs['spacegroup_number']`, `obs['crystal_system']`, `obs['nsites_primitive']` |
| `mv.pp.strain` | `obsm['structures']['{name}']` |
| `mv.pp.supercell` | `obsm['structures']['{name}']` |
| `mv.pp.symmetry` | `obs['crystal_system']`, `obs['point_group']`, `obs['spacegroup_number']`, `obs['spacegroup_symbol']`, `obs['n_symmetry_operations']`, `obs['n_wyckoff']`, `obs['wyckoff']`, `obs['min_site_symmetry']`, `obs['max_site_symmetry']` |
| `mv.feat.element_stats` | `obsm['X_element_stats']`, `uns['features']['X_element_stats']` |
| `mv.feat.embed` | `obsm['X_{model}']`, `uns['features']['X_{model}']` |
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
| `mv.calc.forces` | `uns['levels']['{level}']`, `sites.obsm['forces_{level}']`, `sites.obs['force_magnitude_{level}']` |
| `mv.calc.relax` | `obsm['structures']['{key_added}']`, `obs['energy_{level}']`, `obs['energy_per_atom_{level}']`, `obs['relax_converged_{level}']`, `obs['max_force_{level}']`, `uns['levels']['{level}']` |
| `mv.prop.capture` | `obs['capture_coefficient_{kind}']` |
| `mv.prop.compare_grids` | `obs['{quantity}_cosine_{a}_vs_{b}']`, `obs['{quantity}_rmse_{a}_vs_{b}']`, `obs['{quantity}_overlap_{a}_vs_{b}']` |
| `mv.prop.configuration_coordinate` | `obs['cc_frequency_{level}']`, `obs['cc_relaxation_{level}']`, `obs['cc_huang_rhys_{level}']` |
| `mv.prop.cost` | `obs['cost_per_kg']`, `obs['cost_per_mol']` |
| `mv.prop.dielectric` | `obsm['dielectric_real_{level}']`, `obsm['dielectric_imag_{level}']`, `obsm['absorption_{level}']`, `obsm['extinction_{level}']`, `obs['static_dielectric_{level}']`, `obs['refractive_index_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.dimensionality` | `obs['dimensionality']`, `obs['n_components']`, `obs['is_layered']`, `obs['dimensionality_strategy']` |
| `mv.prop.dispersion` | `obs['material']`, `obs['branch_index']`, `obs['branch_minimum']`, `obs['branch_maximum']`, `obs['is_imaginary']`, `obs['is_acoustic']` |
| `mv.prop.efg` | `sites.obs['efg_vzz_{level}']`, `sites.obs['efg_asymmetry_{level}']`, `sites.obs['efg_coupling_{level}']`, `sites.obsm['efg_tensor_{level}']`, `uns['levels']['{level}']` |
| `mv.prop.elastic` | `obs['bulk_modulus_{level}']`, `obs['shear_modulus_{level}']`, `obs['youngs_modulus_{level}']`, `obs['poisson_ratio_{level}']`, `obs['elastic_stable_{level}']`, `obsm['elastic_tensor_{level}']`, `uns['levels']['{level}']` |
| `mv.prop.electrostatic` | `obs['electrostatic_energy']`, `obs['electrostatic_per_atom']`, `obs['electrostatic_per_formula_unit']` |
| `mv.prop.eos` | `obs['bulk_modulus_eos_{level}']`, `obs['bulk_modulus_derivative_{level}']`, `obs['equilibrium_volume_{level}']`, `obs['equilibrium_energy_{level}']`, `obs['eos_residual_{level}']`, `obsm['eos_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.free_energy` | `obs['vibrational_free_energy_{level}']`, `obs['vibrational_entropy_{level}']`, `obs['heat_capacity_{level}']` |
| `mv.prop.frontier_orbitals` | `obs['homo_element']`, `obs['homo_orbital']`, `obs['homo_energy']`, `obs['lumo_element']`, `obs['lumo_orbital']`, `obs['lumo_energy']`, `obs['orbital_gap_estimate']`, `obs['likely_metal']` |
| `mv.prop.neutron` | `obsm['neutron_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.nmr` | `sites.obs['shielding_iso_{level}']`, `sites.obs['shielding_anisotropy_{level}']`, `sites.obs['shielding_asymmetry_{level}']`, `sites.obs['shielding_span_{level}']`, `sites.obs['shielding_skew_{level}']`, `sites.obsm['shielding_tensor_{level}']`, `uns['levels']['{level}']` |
| `mv.prop.phonon` | `obsm['phonon_dos_{level}']`, `obs['n_imaginary_modes_{level}']`, `obs['dynamically_stable_{level}']`, `obs['zero_point_energy_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.phonon_at_temperature` | `obsm['min_frequency_vs_temperature_{level}']`, `obsm['imaginary_modes_vs_temperature_{level}']`, `obs['stabilisation_temperature_{level}']`, `uns['grids']`, `uns['self_consistent_phonons']`, `uns['levels']['{level}']` |
| `mv.prop.piezo_from_dfpt` | `obs['piezo_max_{level}']`, `obs['piezo_norm_{level}']`, `uns['piezo_from_dfpt']`, `uns['levels']['{level}']` |
| `mv.prop.piezoelectric` | `obs['piezo_max_longitudinal_{level}']`, `obs['piezo_max_direction_{level}']`, `obs['piezo_symmetry_valid_{level}']`, `obsm['piezo_tensor_{level}']`, `uns['levels']['{level}']` |
| `mv.prop.polarization` | `obs['polarization_a']`, `obs['polarization_b']`, `obs['polarization_c']`, `uns['polarization']` |
| `mv.prop.quasiharmonic` | `obs['thermal_expansion_qha_{level}']`, `obs['gruneisen_{level}']`, `obs['debye_temperature_qha_{level}']`, `obs['heat_capacity_300K_{level}']`, `obsm['gibbs_{level}']`, `obsm['thermal_expansion_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.rdf` | `obsm['rdf_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.scattering` | `obsm['structure_factor_{level}']`, `obsm['pdf_{level}']`, `uns['grids']`, `uns['scattering']` |
| `mv.prop.slme` | `obs['slme_{level}']`, `obs['sq_limit_{level}']` |
| `mv.prop.superconductivity` | `obs['omega_log_{level}']`, `obs['omega_2_{level}']`, `obs['critical_temperature_{level}']`, `uns['superconductivity']` |
| `mv.prop.supply_risk` | `obs['hhi_production']`, `obs['hhi_reserve']`, `obs['supply_risk']` |
| `mv.prop.tem` | `obsm['tem_{level}']`, `uns['grids']`, `obs['tem_n_reflections_{level}']`, `obs['tem_strongest_{level}']`, `obs['tem_zone_axis']`, `uns['levels']['{level}']` |
| `mv.prop.thermal_conductivity` | `obs['debye_temperature_{level}']`, `obs['gruneisen_{level}']`, `obs['sound_velocity_{level}']`, `obs['thermal_conductivity_{level}']` |
| `mv.prop.xrd` | `obsm['xrd_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.prop.zt` | `obs['zt_{level}']`, `obs['zt_ceiling_{level}']`, `obs['kappa_electronic_{level}']`, `uns['figure_of_merit']` |
| `mv.md.conductivity` | `obs['conductivity_{species}_{level}']` |
| `mv.md.melt_quench` | `obsm['structures']['amorphous_{level}']`, `obs['amorphous_density_{level}']`, `obs['amorphous_density_ratio_{level}']`, `uns['levels']['{level}']` |
| `mv.md.occupancy` | `obs['occupied_fraction_{level}']`, `obs['occupancy_entropy_{level}']`, `obs['occupancy_peak_{level}']` |
| `mv.md.rdf` | `obsm['rdf_md_{level}']`, `obsm['coordination_md_{level}']`, `uns['grids']`, `obs['first_shell_{level}']`, `obs['first_shell_coordination_{level}']`, `uns['levels']['{level}']` |
| `mv.md.run` | `obs['md_energy_{level}']`, `obs['md_temperature_{level}']`, `obs['msd_{level}']`, `obs['diffusivity_{level}']`, `obs['md_volume_{level}']`, `layers['diffusivity_{level}']`, `obsm['structures']['md_{level}']`, `uns['levels']['{level}']` |
| `mv.md.sites` | `obs['md_sites_{species}_{level}']`, `obs['md_site_spread_{species}_{level}']`, `obs['md_site_visits_{species}_{level}']` |
| `mv.md.sweep` | `obsm['md_volume_{level}']`, `obsm['md_energy_{level}']`, `obsm['md_diffusivity_{level}']`, `obs['thermal_expansion_{level}']`, `obs['activation_energy_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.md.van_hove` | `obsm['van_hove_self_{level}']`, `obsm['van_hove_distinct_{level}']`, `uns['grids']`, `obs['van_hove_rms_{level}']`, `obs['van_hove_peak_{level}']`, `obs['van_hove_jump_{level}']` |
| `mv.mag.describe` | `obs['total_magmom']`, `obs['absolute_magmom']`, `obs['magnetic_order']`, `obs['n_magnetic_species']` |
| `mv.mag.exchange` | `obs['exchange_{level}']`, `obs['ordering_temperature_{level}']`, `uns['exchange']` |
| `mv.mag.ground_state` | `md.obs['magnetic_ordering_{level}']`, `md.obs['magnetic_spread_{level}']`, `md.obs['energy_per_atom_{level}']`, `md.obs['total_magmom_{level}']`, `orderings_.obs['is_ground_state_{level}']` |
| `mv.mag.jahn_teller` | `obs['jahn_teller_active']`, `obs['jahn_teller_strength']`, `obs['jahn_teller_species']`, `uns['jahn_teller']` |
| `mv.mag.symmetry` | `obs['magnetic_symmetry_order']`, `obs['parent_symmetry_order']`, `obs['magnetic_symmetry_fraction']` |
| `mv.neb.barrier` | `obs['barrier_{level}']`, `obs['barrier_reverse_{level}']`, `obs['reaction_energy_{level}']`, `obs['neb_converged_{level}']`, `obsm['neb_profile_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.neb.hop_endpoints` | `obsm['structures']['{key_added}_initial']`, `obsm['structures']['{key_added}_final']`, `obs['hop_distance']`, `obs['hop_species']` |
| `mv.neb.percolation` | `obs['percolation_dimensionality_{species}']`, `obs['percolation_threshold_{species}']`, `obs['percolation_sites_{species}']` |
| `mv.surf.adsorption_energy` | `obs['adsorption_energy_{level}']`, `obs['is_best_site_{level}']` |
| `mv.surf.scaling` | `obs['scaling_residual']`, `uns['scaling']` |
| `mv.surf.surface_energy` | `obs['surface_energy_{level}']`, `obs['surface_energy_{level}_off_stoichiometry']` |
| `mv.surf.surface_energy_chempot` | `facets.obs['surface_energy_{level}']` |
| `mv.surf.volcano` | `obs['volcano_activity']`, `obs['distance_from_optimum']`, `uns['volcano']` |
| `mv.surf.wulff` | `facets.obs['wulff_area_fraction_{level}']`, `bulk.obs['wulff_effective_radius_{level}']`, `bulk.obs['wulff_shape_factor_{level}']` |
| `mv.thermo.calphad` | `obs['calphad_phases']`, `obs['calphad_n_phases']`, `obs['calphad_major_phase']`, `obs['calphad_major_fraction']`, `uns['calphad']` |
| `mv.thermo.chempot_diagram` | `obs['chempot_stable_{level}']`, `obs['chempot_window_{level}']`, `uns['chempot_diagram']` |
| `mv.thermo.chempot_limits` | `uns['chempot_limits']` |
| `mv.thermo.corrections` | `obs['energy_{level}-{scheme}']`, `obs['energy_per_atom_{level}-{scheme}']`, `obs['correction_{level}-{scheme}']`, `obs['correction_per_atom_{level}-{scheme}']`, `obs['run_type_{level}-{scheme}']`, `uns['levels']['{level}-{scheme}']`, `uns['corrections']` |
| `mv.thermo.defect_formation` | `obs['defect_formation_energy_{level}']`, `obs['stable_charge_{level}']`, `obsm['formation_vs_fermi_{level}']`, `uns['grids']`, `uns['defect_thermodynamics']` |
| `mv.thermo.fit_corrections` | `uns['fitted_corrections']`, `obs['correction_{level}']`, `obs['energy_corrected_{level}']` |
| `mv.thermo.hull` | `obs['e_above_hull_{level}']`, `obs['is_stable_{level}']`, `obs['formation_energy_{level}']`, `obs['decomposes_to_{level}']`, `uns['phase_diagram']` |
| `mv.thermo.pourbaix` | `obs['pourbaix_decomposition']`, `uns['pourbaix']` |
| `mv.thermo.reaction` | `uns['reactions']` |
| `mv.thermo.theoretical_capacity` | `obs['max_ion_removal_{working_ion}']`, `obs['max_ion_insertion_{working_ion}']`, `obs['theoretical_capacity_{working_ion}']`, `obs['theoretical_capacity_volumetric_{working_ion}']`, `uns['theoretical_capacity']` |
| `mv.thermo.voltage` | `obs['voltage_{level}']`, `obs['capacity_gravimetric_{level}']`, `obs['capacity_volumetric_{level}']`, `obs['energy_density_{level}']`, `obs['volume_change_{level}']`, `uns['electrode']` |
| `mv.dft.read_dos` | `obsm['dos_{level}']`, `obs['band_gap_{level}']`, `obs['is_direct_gap_{level}']`, `obs['vbm_{level}']`, `obs['cbm_{level}']`, `obs['fermi_level_{level}']`, `obs['dos_at_fermi_{level}']`, `obs['is_metal_{level}']`, `uns['grids']`, `uns['levels']['{level}']` |
| `mv.dft.read_outputs` | `obs['energy_{level}']`, `obs['energy_per_atom_{level}']`, `obs['band_gap_{level}']`, `obs['converged_{level}']`, `obsm['structures']['relaxed_{level}']`, `uns['levels']['{level}']`, `uns['dft']` |
| `mv.dft.write_inputs` | `obs['dft_directory']`, `written to disk`, `uns['dft']` |
| `mv.multi.aggregate` | `md.obs['{key_added}']` |
| `mv.exp.attach` | `obsm['{quantity}_{level}']`, `uns['levels']['{level}']` |
| `mv.exp.formation_hull` | `obs['e_above_hull_{level}']`, `obs['is_stable_{level}']`, `obs['formation_energy_{level}']` |
| `mv.exp.match_xrd` | `obs['xrd_match']`, `obs['xrd_match_rank']`, `uns['xrd_match']` |
| `mv.exp.measure` | `obs['{quantity}_{level}']`, `uns['levels']['{level}']` |
| `mv.screen.filter` | `obs['{name}']`, `uns['screens']` |
| `mv.screen.pareto` | `obs['{name}']`, `obs['{name}_rank']`, `uns['pareto']` |
| `mv.screen.rank` | `obs['{name}']` |
| `mv.gen.compositions` | `obs['formula']`, `obs['n_elements']`, `obs['oxidation_states']`, `obs['n_oxidation_assignments']`, `obs['stoichiometry']`, `X` |
| `mv.gen.from_symmetry` | `obs['parent']`, `obs['formula']`, `obs['requested_space_group']`, `obs['space_group']`, `obs['space_group_symbol']`, `obs['symmetry_as_requested']`, `obs['nsites']`, `obsm['structures']['input']` |
| `mv.gen.predict_dopants` | `obs['n_type_dopant']`, `obs['n_type_probability']`, `obs['p_type_dopant']`, `obs['p_type_probability']`, `uns['dopants']` |
| `mv.gen.predict_hosts` | `obs['parent']`, `obs['target']`, `obs['host_probability']`, `obsm['structures']['input']` |
| `mv.gen.predict_substitutions` | `obs['parent']`, `obs['substitution']`, `obs['substitution_probability']`, `obsm['structures']['input']` |
| `mv.gen.validate` | `obs['gen_valid']`, `obs['gen_unique']`, `obs['gen_novel']`, `obs['gen_stable']`, `obs['gen_metastable']`, `obs['gen_sun']`, `obs['gen_msun']`, `uns['gen_validate']` |
| `mv.model.cross_validate` | `uns['cross_validate']` |
| `mv.model.fit` | `obs['{target}_{level}']`, `uns['levels']['{level}']`, `uns['model']` |
| `mv.model.split` | `obs['{key_added}']`, `uns['split']` |
| `mv.opt.observe` | `obs['campaign_round']`, `obs['selected']`, `uns['{name}']` |
| `mv.opt.start` | `obs['{observed_key}']`, `obs['campaign_round']`, `uns['campaign']` |
| `mv.opt.suggest` | `obs['acquisition']`, `obs['selected']`, `uns['{name}']` |
| `mv.utils.checkpoint` | `written to disk`, `uns['checkpoints']` |
| `mv.utils.convert` | `obs['{key_added}']`, `uns['units']` |
| `mv.utils.job_status` | `uns['submissions']` |
| `mv.utils.map_chunks` | `uns['chunked']` |
| `mv.utils.set_units` | `uns['units']` |
| `mv.utils.slurm_script` | `written to disk` |
| `mv.utils.submit` | `uns['submissions']` |

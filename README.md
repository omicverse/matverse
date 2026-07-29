# matverse

A materials analysis ecosystem on the [AnnData](https://anndata.readthedocs.io/)
substrate. One object carries a screening pipeline end to end — query,
standardise, featurise, relax, rank — keeping the structures, the annotations,
the descriptors and the results together, and writing to `h5ad`/`zarr` without
a new file format.

```python
import matverse as mv

md = mv.datasets.load('battery_cathodes')  # real structures, no network needed
# or: mv.data.from_cif('candidates/') / from_mp / from_optimade / from_ase
mv.pp.standardize(md)                    # primitive + conventional + symmetry
mv.pp.qc(md)                             # -> obs['is_valid'], obs['min_distance']
md = mv.pp.filter_materials(md)          # drop the broken ones
mv.feat.element_stats(md)                # -> obsm['X_element_stats']
mv.calc.relax(md, level='emt')           # -> obsm['structures']['relaxed_emt']
mv.thermo.hull(md, level='emt')          # -> obs['e_above_hull_emt']
mv.screen.filter(md, e_above_hull_emt__lt=0.05)
mv.tl.rank_elements_groups(md, 'passes') # which chemistry passed, and why

md[md.obs['passes']]                     # still an ordinary AnnData
```

## Three problems it addresses

**A DataFrame loses its structures.** The community convention — a pandas
DataFrame with a `structure` column and featuriser-appended columns — works
until you merge, filter or featurise twice, at which point the correspondence
between rows, structures and descriptor blocks is maintained by hand. AnnData's
axes maintain it by construction: subset the object and every aligned annotation
subsets with it.

**A number without its level of theory is not a result.** An energy from PBE,
from a model trained to reproduce PBE+U, and from one trained on r2SCAN are
three different quantities; in a flat table they are three columns whose names
happen to differ. Here the level of theory is the slot name:

```python
md.obs["energy_pbe"]         md.uns["levels"]["pbe"]         # functional, cutoff
md.obs["energy_mace-omat"]   md.uns["levels"]["mace-omat"]   # surrogate, reference, licence
```

**Chemical space has no toolchain.** "Which elements distinguish the candidates
that passed from the ones that failed" is the question that follows every
screen, and answering it from a table of averaged descriptors means writing the
test by hand each time. Here it is one call, because the composition matrix is
the same shape as a single-cell count matrix.

## Elements are the axis of `X`

`X` is materials × elements, sparse, non-negative, mostly zero — almost every
material draws on five or fewer of 118 columns. That is structurally a cells ×
genes matrix, and `var` is the periodic table.

| scanpy | materials |
|---|---|
| `normalize_total` | atom counts → atomic fractions |
| `filter_genes` | drop elements absent from this library |
| `pca` / `neighbors` / `leiden` | a chemical-space map |
| **`rank_genes_groups`** | **which elements characterise a group of materials** |

The correspondence also pays off inside the library: `mv.feat.element_stats`
computes the classic composition descriptor — weighted mean, spread and range of
element properties — as a matrix product of `X` and `var`, rather than as a
featuriser that re-derives the periodic table.

```python
mv.tl.rank_elements_groups(md, 'is_stable_emt')
md.uns['rank_elements_groups']['True']
#   element  n_in_group  frac_in_group  odds_ratio      pval      qval
# 0      Al           3            1.0        inf  0.047619  0.142857
```

`build_X=False` restores the width-zero `X` of v0.1 for datasets whose rows are
not single compositions.

## Namespaces

| | |
|---|---|
| `mv.datasets` | real published structures to work on, bundled or fetched |
| `mv.data` | build a dataset — OPTIMADE, Materials Project, CIF, matminer, ASE, pymatgen |
| `mv.pp` | standardisation, symmetry, quality control, filtering, deduplication, cross-database harmonisation |
| `mv.feat` | descriptors into `obsm` |
| `mv.tl` | ordination, clustering, element enrichment, novelty |
| `mv.calc` | energies, forces and relaxation, tagged by level of theory |
| `mv.prop` | derived properties — elastic moduli, equation of state, phonons, bonding dimensionality, NMR and piezoelectric tensors, optics and solar efficiency, and curves on a shared grid |
| `mv.md` | molecular dynamics — diffusivity, thermal expansion, amorphous structures |
| `mv.neb` | migration barriers by nudged elastic band |
| `mv.mag` | magnetic orderings, and which one the hull should use |
| `mv.surf` | slabs, surface energies, Wulff shapes, adsorption |
| `mv.thermo` | convex hull, reactions, chemical potential windows |
| `mv.dft` | first-principles inputs out, results back in |
| `mv.multi` | the sites axis — one row per atom |
| `mv.exp` | measured data, on the same footing as computed data |
| `mv.screen` | filtering, ranking and Pareto fronts that leave a record |
| `mv.gen` | scoring generated candidates, and enumerating substitutions |
| `mv.model` | property prediction, with splits that do not leak |
| `mv.opt` | design campaigns — what to compute next, and what came back |
| `mv.pl` | plotting, including the periodic-table heatmap |
| `mv.utils` | units, checkpointing, cluster submission, object summaries |

`mv.struct` is the v0.1 name for the structure half of `mv.pp`, kept as
re-exports.

## Slot convention

```
X                       materials x elements — the composition matrix
var                     one row per element — the periodic table
obs                     one row per material; scalars carry _<level>
obsm['structures']      structures, one column per variant
obsm['xrd_pbe']         curves: materials x grid, one block per quantity and level
obsm[...]               descriptors, embeddings
obsp                    pairwise: similarity
uns['grids']            the shared axis each curve is defined on
uns['features']         which featuriser produced which block
uns['levels']           per-level-of-theory provenance
uns['provenance']       operations applied, in order
```

One rule covers every result: **the level of theory is a name suffix**.
`obs['energy_pbe']` and `obsm['xrd_pbe']` read the same way, and a measured
pattern is `obsm['xrd_experiment']` rather than a different kind of thing.

Per-atom results do not fit on this axis, because the number of atoms differs
per material. They get a second object:

```python
sites = mv.multi.sites(md)                    # one row per atom
mv.calc.forces(md, sites, level='emt')        # -> sites.obsm['forces_emt']
mv.multi.aggregate(sites, md, 'force_magnitude_emt', how='max')
```

`sites.X` is the one-hot element, so `var` is the same periodic table and
`mv.tl.rank_elements_groups` runs unchanged on atoms.

Structures are in `obsm`, not `uns`, and the placement is load-bearing. `uns`
does not subset with the object, so `md[mask]` would keep every structure while
dropping rows and each surviving row would point at the wrong one — the exact
failure this substrate exists to prevent. Serialising each structure to JSON is
what lets it live in `obsm`, and has the second benefit of making the object
writable to `h5ad` without special handling. Both properties are covered by
tests, because both were originally claimed and neither originally held.

## Operations deposit; they do not return

```python
mv.pp.standardize(md)     # writes obsm['structures']['primitive'], returns None
```

Structure variants accumulate in one object — `['input', 'primitive',
'conventional', 'relaxed_emt']` — instead of becoming four variables downstream
code has to keep straight. `uns['provenance']` records what ran with its
parameters, so a run replays as code rather than reading as a list of verbs.

The two exceptions are constructors, which have no object to deposit into, and
`mv.pp.filter_materials` / `filter_elements`, because AnnData cannot drop rows or
columns in place.

## A screen leaves its reasoning behind

`mv.screen.filter` deposits a boolean column plus the criteria that produced it,
rather than returning a shorter list:

```python
mv.screen.filter(md, e_above_hull_emt__lt=0.05, n_elements__le=3)
md.uns["screens"]["passes"]
# {'criteria': {'e_above_hull_emt__lt': 0.05, 'n_elements__le': 3},
#  'n_pass': 5, 'n_total': 6}
```

NaN never passes: a candidate whose calculation failed to converge has not met
the criterion, and silently admitting it is how a broken run reaches a
shortlist.

## Levels of theory

`mv.calc` dispatches on `level`, and each level records more than its name.

```python
mv.calc.available()          # what this installation can actually run
mv.calc.check_licenses(md)   # ['mace-omat'] — ASL forbids commercial use
mv.compare_levels(md, 'energy_per_atom')    # one quantity, every level, side by side
```

- **`reference`** — what the level reproduces. A model trained on OMat24 targets
  PBE+U; one trained on MatPES targets r2SCAN. `surrogate: True` alone no longer
  distinguishes them, and `mv.thermo.hull` refuses to build a hull across two
  levels whose references disagree.
- **`license`** — MACE-MP and MACE-MPA are MIT; MACE-OMAT and MACE-MATPES are ASL
  and forbid commercial use; UMA's licence excludes several countries. A
  screening result carries the licence of whatever produced it.
- **`uncertainty`** — where `obs['energy_<level>_std']` came from.
  `mv.calc.committee` produces one; nothing pretends it is calibrated.

matverse ships no default beyond `emt`, which needs nothing extra and is
parameterised only for Al, Cu, Ag, Au, Ni, Pd, Pt, H, C, N, O — enough to
exercise a pipeline honestly. The Matbench Discovery leaders are currently
separated by less than the spread between seeds and the ranking reorders
monthly, so hardcoding "the best model" would be stale on arrival. Register what
you have:

```python
mv.calc.register_calculator("myff", MyCalculator, kind="mlip",
                            method="MyFF", reference="r2SCAN", license="MIT")
mv.calc.relax(md, level="myff")
```

## Agent-readable by construction

Every public function carries a registry entry naming what it consumes and
creates.

```python
mv.find('thermodynamic stability')     # ['mv.thermo.hull', ...]
print(mv.describe('convex hull'))
```

```
mv.thermo.hull(md, level='emt', source='input', references=None, ...)

Build the convex hull of energies at one level of theory and record each
material's distance above it, together with what it would decompose into.

requires:
  obs['energy_{level}']
  obsm['structures']['{source}']

produces:
  obs['e_above_hull_{level}']
  obs['is_stable_{level}']
  ...
```

`requires` and `produces` name state a call consumes and creates, so they only
bind in a library where calls have named state to point at. That is why they are
here and were not usable on `pymatgen-analysis-defects`, where results are
attributes on returned objects and nothing is mutated in common.

**Claims are verified by execution, not asserted.** `produces` is checked by
running the call and looking; `requires` by deleting the slot and confirming the
call fails. The current state is **480/480 probed claims verified**. Of the 139
entries that make a contract claim, 125 are probed; the other 14 need the
network, a real VASP or LOBSTER output, or a tool that is not installed, and
each is named with its reason in `tests/_contract_cases.py` rather than quietly
left out.

That distinction is the point. A rate is only as good as its denominator: for
most of this library's life the harness probed 165 of 491 claims and reported
100%, which was a statement about the test suite rather than about the registry.
`test_no_claim_goes_unprobed` now fails if any entry ships a claim nothing
checks.

Claims are deleted rather than repaired when they fail. The ones that did:

| deleted claim | why it failed |
|---|---|
| `feat.element_stats requires var['Z']` | takes whatever numeric columns `var` has |
| `thermo.hull requires levels[{level}]` | only read when `references=` is given |
| `tl.cluster requires obsp['connectivities']` | true of the leiden route, not kmeans |
| `pp.strain produces structures[{name}]` | template was unresolvable; the default is now a real value |
| `screen.filter requires obs['{column}']` | no parameter holds a column name to interpolate |
| `utils.resume requires obs['{column}']` | an absent column means every row is still to do |
| `utils.job_status requires uns['submissions']` | having submitted nothing is an answer, not an error |
| `dft.status requires obs['dft_directory']` | scans the root directory, not the object |
| `exp.attach requires uns['grids']` | a measured curve may be the first grid |
| `surf.wulff produces uns['wulff']` | never written |
| `iface.build produces obs['nsites']` | never written |

Two are informative beyond their own function. `mv.tl.cluster`'s two routes
consume different state, and the contract vocabulary has one `requires` field
per function rather than one per route — expressible only in the `dispatch`
prose, which a caller reads but a tool cannot check. `mv.screen.filter` is the
sharper case: its consumed columns are encoded in the *keys* of `**criteria`, so
there is no parameter for a slot template to name. An API whose inputs are
keyword names puts them beyond what this vocabulary can say, and no wording
fixes it.

**Which object a slot lands on.** On a library where every operation takes one
object, naming a container is enough. matverse has operations that take two — a
material axis and a sites, bands or interface axis — so a container may be
qualified with the parameter that receives it:

```python
produces={"sites.obs": ["coordination_number"]}     # mv.env.coordination
```

This was the one place the omicverse vocabulary needed extending, and it is a
gap rather than a disagreement: the seven slots are unchanged, the claim simply
had no way to say *whose* `obs`. `mv.mag.ground_state` is the case that forces
it — it writes four columns to the parent and a fifth back onto the orderings,
and an unqualified `obs` said they arrived together.

## Interoperability

Reads what already exists rather than replacing it.

| Ecosystem | How it maps |
|---|---|
| [pymatgen](https://pymatgen.org/) | structures are pymatgen objects throughout |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | `data.from_ase` / `to_ase`; `mv.calc` uses ASE calculators |
| [matminer](https://hackingmaterials.lbl.gov/matminer/) | `data.from_matminer` / `to_matminer` round-trip |
| [dscribe](https://singroup.github.io/dscribe/) | `mv.feat.soap` |
| [Materials Project](https://next-gen.materialsproject.org/) | `data.from_mp`, `thermo.references_from_mp` |

ASE's `db` converged on the same split independently — `key_value_pairs`
restricted to scalars, large feature blocks moved to their own table — which is
exactly `obs` versus `obsm`.

## Status

v0.1.36. **176 functions across 27 namespaces**, every one carrying a registry
entry whose claims are verified by execution — currently **480/480**, with
every claim-making entry either probed or exempted for a stated reason. See
[DESIGN.md](DESIGN.md) for the full plan and
[Release notes](matverse_guide/docs/Release_notes.md) for what changed.

Landed:

- `X` as the composition matrix, `var` as the periodic table, and `mv.tl` on top
- structures moved to `obsm`, which fixed both the subsetting bug and the h5ad
  claim — neither of which had ever held
- levels carry `reference`, `license` and `uncertainty`; the hull refuses to mix
- `mv.thermo.hull` takes real reference phases, so `e_above_hull` can be absolute
- curves as `obsm` blocks on a shared grid, which collapsed the two
  level-of-theory conventions into one and removed the need for MuData
- the sites axis, answering the ragged per-atom problem v0.1 flagged as open
- `mv.exp` — experiment as a level, needing no new machinery
- `mv.pp.harmonize` — cross-database energies as a batch effect
- `mv.gen.validate` on LeMat-GenBench's definitions rather than a variant
- `mv.model` with splits that group by composition, prototype or held-out
  element, and report how much a random split was flattering the model
- `mv.opt` — pool-based campaigns that record every round
- `mv.pl`, including the periodic-table heatmap
- `mv.utils` — units, checkpoints, Slurm scripts, object summaries
- OPTIMADE as the primary connector: one protocol, ~20 providers
- `mv.dft` — the boundary between a screen and a directory tree of calculations
- phonons, and the vibrational thermodynamics that make a 0 K hull a hull at
  temperature; validated against the zero-point energy of copper and the
  Dulong-Petit limit rather than against a stored number
- streaming construction and chunked compute, so a corpus larger than memory is
  a loop rather than a wall
- molecular dynamics, migration barriers, surfaces and adsorption
- magnetic ordering enumeration, lattice thermal conductivity, electronic
  structure ingestion and charged-defect thermodynamics
- the registry and the probe harness that verifies it: **480/480 claims**,
  with a coverage check that fails if an entry ships a claim nothing probes

Still open:

- **Scale is chunked, not lazy.** `mv.data.from_iterable` and
  `mv.utils.map_chunks` mean a corpus larger than memory can be processed, but
  the object itself is still materialised. A zarr-backed `obs` and on-demand
  structure resolution are the next step, and the one capability no competing
  package has.
- **`harmonize` is fitted, not validated.** It recovers an injected offset
  exactly on synthetic anchors. Whether it improves a real
  MP-versus-OQMD-versus-Alexandria hull is unmeasured.
- **No models ship.** `mv.calc`, `mv.model` and `mv.feat.embed` are registration
  interfaces; matverse vendors no interatomic potential, no graph network and no
  embedder. That is a choice — weights are hundreds of megabytes with their own
  licences, and the leaderboard reorders monthly — but it means a fresh install
  screens with EMT until you register something better.
- **Workflow submission stays delegated** to atomate2, quacc and AiiDA. `mv.dft`
  writes inputs and harvests outputs; it does not run jobs.
- **Phonons are gamma-point frozen displacements**, which samples only the
  q-points commensurate with the supercell. phonopy does the full job and
  exploits symmetry; this is the version that needs no extra dependency.
- **The materials axis still suits single-system depth badly.** The sites axis
  and grid blocks help; they do not make this the right object for one material's
  full phonon band structure.

Design disagreement is welcome, particularly on the axis choice and on whether
`X` as composition earns its coupling. The test that would kill it is in the
suite: `test_rank_elements_groups_recovers_the_obvious_chemistry`.

## Install

```bash
pip install -e .
pip install -e ".[analysis,descriptors,mp,mlip]"    # optional backends
```

Core dependencies are `anndata`, `numpy`, `pandas`, `scipy`, `pymatgen` and
`ase`. Everything else is an optional extra imported inside the function that
needs it — this ecosystem's version conflicts are real and current, and
`pip install matverse` must never be the thing that breaks an environment.

MIT.

# matverse

A materials analysis ecosystem on the [AnnData](https://anndata.readthedocs.io/)
substrate. One object carries a screening pipeline end to end — query,
standardise, featurise, relax, rank — keeping the structures, the annotations,
the descriptors and the results together, and writing to `h5ad`/`zarr` without
a new file format.

```python
import matverse as mv

md = mv.data.from_cif('candidates/')     # or from_mp / from_ase / from_matminer
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
| `mv.data` | build a dataset — CIF, Materials Project, matminer, ASE, pymatgen |
| `mv.pp` | standardisation, symmetry, quality control, filtering, deduplication |
| `mv.feat` | descriptors into `obsm` |
| `mv.tl` | ordination, clustering, element enrichment, novelty |
| `mv.calc` | energies and relaxation, tagged by level of theory |
| `mv.thermo` | convex hull, energy above hull, decomposition products |
| `mv.screen` | filtering, ranking and Pareto fronts that leave a record |

`mv.struct` is the v0.1 name for the structure half of `mv.pp`, kept as
re-exports.

## Slot convention

```
X                       materials x elements — the composition matrix
var                     one row per element — the periodic table
obs                     one row per material
obsm['structures']      structures, one column per variant
obsm[...]               descriptors, embeddings
obsp                    pairwise: similarity
uns['features']         which featuriser produced which block
uns['levels']           per-level-of-theory provenance
uns['provenance']       operations applied, in order
```

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
call fails. The current state is **68/68 probed claims verified** across 39
entries and 115 claims — and four claims were deleted rather than repaired when
they failed their probe:

| deleted claim | why it failed |
|---|---|
| `feat.element_stats requires var['Z']` | takes whatever numeric columns `var` has |
| `thermo.hull requires levels[{level}]` | only read when `references=` is given |
| `tl.cluster requires obsp['connectivities']` | true of the leiden route, not kmeans |
| `pp.strain produces structures[{name}]` | template was unresolvable; the default is now a real value |

The third is the informative one. `mv.tl.cluster`'s two routes consume different
state, and the contract vocabulary has one `requires` field per function rather
than one per route. That dependency is expressible only in the `dispatch` prose,
which a caller reads but a tool cannot check — the same shape of limitation as
the `pymatgen` transfer boundary, one level down.

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

v0.1.1. A screening pipeline that works end to end, and is honest about what it
does not do yet. See [DESIGN.md](DESIGN.md) for the full plan.

Landed since v0.1.0:

- `X` as the composition matrix, `var` as the periodic table, and `mv.tl` on top
- structures moved to `obsm`, which fixed both the subsetting bug and the h5ad
  claim — neither of which had ever held
- levels carry `reference`, `license` and `uncertainty`; the hull refuses to mix
- `mv.thermo.hull` takes real reference phases, so `e_above_hull` can be absolute
- quality control, deduplication, Pareto screening
- the registry and the probe harness that verifies it

Still open:

- **Scale.** Everything here assumes the dataset fits in memory. Alexandria is
  5.06M entries and OMat24 is ~110M calculations; the lazy zarr-backed path is
  the v0.3.x priority and the one capability no competing package has.
- **The materials axis suits single-system depth badly.** A sites axis and grid
  modalities under MuData are designed but not built. One material's full phonon
  band structure is not what this object is for, and the docs should keep saying
  so.
- **Two level-of-theory conventions.** Arrays get a `layer`, scalars get a name
  suffix, because `obs` has no layers. It is a wart forced by the container.
- No DFT I/O, no generative models, no experimental data. `mv.gen.validate`
  adopting LeMat-GenBench's metric definitions verbatim is the highest-value
  next piece, because every generative paper currently reinvents them.

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

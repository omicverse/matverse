# matverse

A materials analysis ecosystem on the [AnnData](https://anndata.readthedocs.io/)
substrate. One object carries a screening pipeline end to end — query,
standardise, featurise, relax, rank — keeping the structures, the annotations,
the descriptors and the results together, and writing to `h5ad`/`zarr` without
a new file format.

```python
import matverse as mv

md = mv.data.from_matminer(df)          # or from_mp / from_ase / from_structures
mv.struct.standardize(md)               # primitive + conventional + symmetry
mv.feat.composition(md)                 # -> obsm['X_composition']
mv.calc.relax(md, level='emt')          # -> uns['structures']['relaxed_emt']
mv.thermo.hull(md, level='emt')         # -> obs['e_above_hull_emt']
mv.screen.filter(md, e_above_hull_emt__lt=0.05)

md[md.obs['passes']]                    # still an ordinary AnnData
```

## Two problems it addresses

**A DataFrame loses its structures.** The community convention — a pandas
DataFrame with a `structure` column and featuriser-appended columns — works
until you merge, filter or featurise twice, at which point the correspondence
between rows, structures and descriptor blocks is maintained by hand. AnnData's
`obs` / `obsm` / `obsp` axes maintain it by construction: subset the object and
every aligned annotation subsets with it.

**A number without its level of theory is not a result.** A formation energy
from PBE, from HSE06, and from a machine-learned potential are three different
quantities; in a flat table they are three columns whose names happen to
differ. Here the level of theory is the slot name:

```python
md.obs["energy_pbe"]        md.uns["calc"]["pbe"]    # functional, cutoff, k-points
md.obs["energy_mace"]       md.uns["calc"]["mace"]   # surrogate: True
```

Comparing a surrogate against DFT then requires naming both, instead of
silently averaging them.

## Namespaces

| | |
|---|---|
| `mv.data` | build a dataset — Materials Project, matminer, ASE, pymatgen |
| `mv.struct` | standardisation, symmetry, supercells |
| `mv.feat` | descriptors into `obsm` |
| `mv.calc` | energies and relaxation, tagged by level of theory |
| `mv.thermo` | convex hull, energy above hull |
| `mv.screen` | filtering and ranking that leaves a record |

## Slot convention

```
obs                     one row per material
obsm                    descriptors, embeddings
obsp                    pairwise: similarity
uns['structures']       raw structures, keyed by variant
uns['features']         which featuriser produced which block
uns['calc']             per-level calculator settings
uns['screens']          selection criteria and how many passed
uns['provenance']       operations applied, in order
```

`X` is left empty on purpose. AnnData ties `X`'s width to `var`, so it cannot
be widened in place, and every operation here writes in place. Features
therefore go to `obsm`, whose width is free. This is a constraint found by
testing, not a preference.

## Operations deposit; they do not return

```python
mv.struct.standardize(md)     # writes uns['structures']['primitive'], returns None
```

rather than

```python
prim = SpacegroupAnalyzer(s).get_primitive_standard_structure()   # you decide where it lives
```

Structure variants accumulate in one object — `['input', 'primitive',
'conventional', 'relaxed_emt']` — instead of becoming four variables that
downstream code has to keep straight. `uns['provenance']` records what ran, so
a run is reproducible from the object alone.

## A screen leaves its reasoning behind

`mv.screen.filter` deposits a boolean column plus the criteria that produced
it, rather than returning a shorter list:

```python
mv.screen.filter(md, e_above_hull_emt__lt=0.05, n_elements__le=3)
md.uns["screens"]["passes"]
# {'criteria': {'e_above_hull_emt__lt': 0.05, 'n_elements__le': 3},
#  'n_pass': 5, 'n_total': 6}
```

## Calculators

`mv.calc` dispatches on `level`. `emt` (ASE effective-medium theory) ships
working and needs nothing extra — it is parameterised only for Al, Cu, Ag, Au,
Ni, Pd, Pt, H, C, N, O, which is enough to exercise a pipeline honestly.
`mace` and `chgnet` are wired and require their own installs. Register your own:

```python
mv.calc.register_calculator("myff", MyCalculator, method="MyFF", surrogate=True)
mv.calc.relax(md, level="myff")
```

## Interoperability

Reads what already exists rather than replacing it.

| Ecosystem | How it maps |
|---|---|
| [matminer](https://hackingmaterials.lbl.gov/matminer/) | `data.from_matminer` / `data.to_matminer` round-trip |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | `data.from_ase`; `mv.calc` uses ASE calculators and optimisers |
| [pymatgen](https://pymatgen.org/) | structures are pymatgen objects throughout |
| [Materials Project](https://next-gen.materialsproject.org/) | `data.from_mp` (needs `mp-api` and `MP_API_KEY`) |

ASE's `db` converged on the same split independently — `key_value_pairs`
restricted to scalars, large feature blocks moved to their own table — which is
exactly `obs` versus `obsm`.

## Status

Early, and honest about it.

- `mv.thermo.hull` builds the hull over **the dataset's own compositions**.
  Without elemental reference phases it is a relative statement, recorded as
  `uns['phase_diagram']['closed_system']` rather than hidden. Pulling MP
  references is the next step.
- `mv.feat.matminer` delegates when matminer is installed; `mv.feat.composition`
  is the dependency-free fallback.
- Ragged per-site data (forces, magmoms per atom) is the open design problem.
  `uns['sites']` as a list of frames is honest but loses vectorisation; ASE
  makes the same compromise by keeping per-atom data on the `Atoms` object.
- No DFT I/O yet. VASP/QE input generation and output parsing are the obvious
  next namespace.

Design disagreement is welcome, particularly on the axis choice — "rows are
materials" suits material-level screening (thousands of candidates, one row
each) and suits single-structure detail (forces, bands, phonons for one system)
much less well. Everything else follows from that decision.

## Install

```bash
pip install -e .
pip install -e ".[matminer,mp,mlip]"    # optional backends
```

MIT.

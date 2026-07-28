# Screening, end to end

The pipeline matverse exists for: take a list of candidate structures, remove
the ones that are broken, relax what remains, work out which are
thermodynamically plausible, and shortlist — keeping the reasoning attached to
the object rather than in your head.

Everything here runs on `emt`, which ships with ASE and needs no download.

## Build a dataset

```python
import matverse as mv
from pymatgen.core import Lattice, Structure

def fcc(symbol, a):
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

def l12(host, guest, a):
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

candidates = [
    fcc("Al", 4.05), fcc("Cu", 3.61), fcc("Ni", 3.52),
    l12("Al", "Cu", 3.90),                      # CuAl3
    l12("Cu", "Al", 3.70),                      # AlCu3
    Structure(Lattice.cubic(2.89), ["Al", "Ni"], [[0, 0, 0], [.5, .5, .5]]),
]

md = mv.data.from_structures(candidates)
md
```

```
AnnData object with n_obs × n_vars = 6 × 3
    var: 'Z', 'atomic_mass', 'electronegativity', 'period', ...
    uns: 'features', 'levels', 'provenance', 'X_is'
    obsm: 'structures'
```

Six materials, three elements. `X` is already the composition matrix and `var`
is already the periodic table restricted to Al, Ni and Cu — you did not ask for
either, because composition is intrinsic to a material rather than derived from
it.

In practice you would load from disk or from a database instead:

```python
md = mv.data.from_cif('candidates/')
md = mv.data.from_mp({'elements': ['Al', 'Ni'], 'num_elements': (2, 2)})
```

## Standardise and describe

```python
mv.pp.standardize(md)      # primitive + conventional + spacegroup
mv.pp.describe(md)         # formula, nsites, volume, density, n_elements

md.obs[['formula', 'spacegroup', 'n_elements']]
```

`standardize` deposits two new structure variants rather than replacing the
input. The object now carries `['input', 'primitive', 'conventional']`, and every
later call names which one it wants.

```python
list(md.obsm['structures'].columns)
# ['input', 'primitive', 'conventional']
```

## Throw out what is broken

```python
mv.pp.qc(md)
md.obs[['min_distance', 'is_ordered', 'is_valid', 'qc_reason']]
```

`qc` is the analogue of `scanpy.pp.calculate_qc_metrics`, and it exists for the
same reason: a generated or database structure with atoms 0.1 Å apart is a broken
cell, not an exotic one. Relaxing it wastes the calculator's time and pollutes
the hull.

```python
md = mv.pp.filter_materials(md)
```

This is one of the few calls that returns instead of depositing, because AnnData
cannot drop rows in place. It records how many it dropped:

```python
mv.provenance(md)[-1]
# "pp.filter_materials(flag='is_valid', n_dropped=0)"
```

Duplicates are worth removing before you spend compute on them twice:

```python
mv.pp.dedup(md)
md.uns['dedup']
```

`dedup` blocks on `(reduced formula, space group)` before running pymatgen's
`StructureMatcher` inside each block. An all-pairs comparison is quadratic and
unusable past a few thousand candidates; blocking makes the expensive part local.

## Relax

```python
mv.calc.relax(md, level='emt', fmax=0.05)

md.obs[['energy_emt', 'energy_per_atom_emt', 'relax_converged_emt']]
```

The relaxed geometry becomes its own variant, `relaxed_emt`, rather than
overwriting the input — so "which structure was this energy computed on" stays
answerable from the object alone.

The level records what produced it:

```python
mv.level_info(md, 'emt')
# {'kind': 'classical', 'method': 'EMT', 'reference': None,
#  'surrogate': True, 'license': 'LGPL-2.1', 'uncertainty': None,
#  'source': 'input', 'fmax': 0.05, 'steps': 200, 'n_failed': 0, ...}
```

## Build a hull

```python
mv.thermo.hull(md, level='emt', source='relaxed_emt')

md.obs[['e_above_hull_emt', 'is_stable_emt', 'decomposes_to_emt']]
```

```{warning}
That call emits a warning, and you should read it. With no `references=`, the
hull is built over **this dataset's own compositions**, which makes
`e_above_hull` a statement about which of your candidates is lowest — not about
whether any of them is stable. Screening 40 generated oxides against each other
will happily report several as "on the hull" when all 40 decompose.

`md.uns['phase_diagram']['closed_system']` records which kind of number you have.
```

To make it absolute, pass the known competing phases:

```python
refs = mv.thermo.references_from_mp(['Al', 'Ni', 'Cu'])   # needs mp-api
mv.thermo.hull(md, level='pbe', references=refs)
```

Note the level changed. Materials Project entries are PBE+U with fitted
corrections; putting them on a hull with EMT energies is not a hull of anything,
and `mv.thermo.hull` raises `LevelMismatch` rather than letting you.

## Shortlist

```python
mv.screen.filter(md, e_above_hull_emt__lt=0.05, n_elements__le=2)

md.uns['screens']['passes']
# {'criteria': {'e_above_hull_emt__lt': 0.05, 'n_elements__le': 2},
#  'n_pass': 4, 'n_total': 6}
```

The screen deposits a boolean column and the criteria that produced it. It does
**not** return a shorter list, because which criterion a candidate failed is a
result. Subset when you actually want the short list:

```python
shortlist = md[md.obs['passes']]
```

When more than one objective matters and no single column decides:

```python
mv.screen.pareto(md, {'e_above_hull_emt': 'min', 'density': 'min'})
md.obs[['pareto', 'pareto_rank']]
```

`pareto_rank` keeps the second-best trade-offs reachable instead of discarding
everything that is not optimal.

## What the object remembers

```python
mv.provenance(md)
```

```
['data.from_structures',
 "pp.standardize(source='input', symprec=0.01)",
 "pp.describe(source='input')",
 "pp.qc(source='input', min_distance=0.5, require_charge_balance=False)",
 "pp.filter_materials(flag='is_valid', n_dropped=0)",
 "calc.relax(level='emt', source='input', fmax=0.05)",
 "thermo.hull(level='emt', source='relaxed_emt', n_references=0)",
 "screen.filter(name='passes', e_above_hull_emt__lt=0.05, n_elements__le=2)"]
```

Parameters are recorded with each call, so the history replays as code rather
than reading as a list of verbs.

## Save it

```python
md.write_h5ad('screen.h5ad')
```

Structures, descriptors, level records and provenance all survive, and the file
is an ordinary `h5ad` that anndata can read without matverse installed.

## Next

[Chemical space](chemical_space.md) picks up from this object and asks what
distinguishes the candidates that passed.

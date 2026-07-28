"""Cells for tutorials/from_pymatgen.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Coming from pymatgen

matverse is not a replacement for [pymatgen](https://pymatgen.org/) and does
not try to be. pymatgen is the crystallography, the file formats, the symmetry
and most of the analysis; matverse is a substrate that holds a *dataset* of
materials and the results computed on them, with pymatgen doing the work
underneath.

So the question for a pymatgen user is not "what do I have to learn instead" —
it is "what does putting my structures in an object buy me". This page answers
that with the operation pymatgen users write most often: transformations.

The short version:

```python
# pymatgen
structures = [PrimitiveCellTransformation().apply_transformation(s)
              for s in structures]

# matverse
mv.transform.apply(md, 'PrimitiveCellTransformation')
```

The list comprehension loses the originals, keeps no record, and gives you a
new list you now have to keep aligned with everything else by hand. The call
deposits a variant, keeps the input, and writes down what it did."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Every transformation, by name

There are around forty-five `Transformation` classes in pymatgen. matverse does
not wrap forty-five of them — it wraps the *idea*, and looks the class up by
name at call time."""),

    ("code", """\
found = mv.transform.available()
len(found)"""),

    ("code", """\
pd.DataFrame([
    {"name": name, "group": entry["group"], "signature": entry["signature"][:56]}
    for name, entry in list(found.items())[:12]
])"""),

    ("markdown", """\
Read from pymatgen at call time rather than from a table here, so a
transformation added upstream is available the day it lands.

It searches, which matters when you half-remember the name:"""),

    ("code", """\
list(mv.transform.available(search="supercell"))"""),

    ("markdown", """\
## Loading a dataset

Real oxides, so the oxidation-state machinery later has something to bite on."""),

    ("code", """\
md = mv.datasets.load("oxides")[:3].copy()
mv.pp.describe(md)

md.obs[["name", "formula", "spacegroup", "nsites"]]"""),

    ("markdown", """\
## Applying one"""),

    ("code", """\
mv.transform.apply(md, "PrimitiveCellTransformation")
mv.transform.apply(md, "PerturbStructureTransformation", distance=0.05)

mv.variants(md)"""),

    ("markdown", """\
Three variants where a list comprehension would have left you with one list and
no way back. `input` is still there, and every later call names which one it
wants — which is the reason "what structure was this energy computed on" stays
answerable in matverse and is a matter of memory in a script.

Each application records whether it worked, **per row**:"""),

    ("code", """\
md.obs[["name", "formula", "primitivecell_ok", "perturbstructure_ok"]]"""),

    ("markdown", """\
### A failure on one row is not a failure of the call

A dataset where a transformation applies to some materials and not others is
the normal case, not the exceptional one. Decorating with strontium oxidation
states works for SrTiO₃ and cannot work for TiO₂ or VO₂:"""),

    ("code", """\
mv.transform.apply(md, "OxidationStateDecorationTransformation",
                   oxidation_states={"Sr": 2, "Ti": 4, "O": -2},
                   key_added="sr_decorated")

md.obs[["name", "formula", "sr_decorated_ok"]]"""),

    ("code", """\
md.uns["transform"]["sr_decorated"]["errors"]"""),

    ("markdown", """\
The rows that failed kept their original structure and are flagged `False`, so
a screen can filter on it. The alternative — raising on the whole dataset, or
silently returning a mixture you cannot tell apart — is worse in both
directions.

### The parameters are part of the record"""),

    ("code", """\
md.uns["transform"]["perturbstructure"]"""),

    ("markdown", """\
## One-to-many

Some transformations return several structures per input. `apply` takes the
first; `expand` keeps them all as rows, with `obs['parent']` pointing back —
the same derived-axis shape as `mv.pp.defects`, `mv.mag.orderings` and
`mv.disorder.orderings`."""),

    ("code", """\
from pymatgen.core import Lattice, Structure

alloy = mv.data.from_structures([
    Structure(Lattice.cubic(3.8), [{"Cu": 0.5, "Au": 0.5}] * 4,
              [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
])
mv.pp.describe(alloy)

orderings = mv.transform.expand(
    alloy, "OrderDisorderedStructureTransformation", n=3, no_oxi_states=True)
mv.pp.describe(orderings)

orderings.obs[["parent", "variant_index", "formula", "nsites"]]"""),

    ("markdown", """\
```{note}
Where a namespace already wraps a specific one-to-many transformation, use it.
`mv.disorder.orderings` does this same work and records domain metadata the
generic path cannot know about — whether the Ewald ranking is meaningful, for
instance. `mv.transform.expand` is for the ones nothing wraps yet.
```

## Chaining

A sequence, applied in order, deposited as one variant."""),

    ("code", """\
mv.transform.chain(md, [
    ("PrimitiveCellTransformation", {}),
    ("PerturbStructureTransformation", {"distance": 0.03}),
], key_added="prepared")

md.obs[["name", "prepared_ok"]]"""),

    ("code", """\
md.uns["transform"]["prepared"]["chain"]"""),

    ("markdown", """\
One variant rather than one per step, because the intermediates are usually not
interesting and storing them all would fill the object. The full sequence is in
`uns` and in the provenance, so the result is still reproducible from the
record.

## Oxidation states: the missing prerequisite

This one is worth its own function because it is behind several of pymatgen's
more confusing error messages.

| what you ran | what it said |
|---|---|
| `OrderDisorderedStructureTransformation` | `Element has no attribute oxi_state!` |
| Ewald ranking | *(no error — every structure scores zero)* |
| `DopingTransformation` | `Valences cannot be assigned!` |

All three are asking for the same thing."""),

    ("code", """\
mv.transform.oxidation_states(md)

md.obs[["name", "formula", "oxidation_states_ok", "charge_balanced"]]"""),

    ("code", """\
structure = mv.structures(md, "oxidized")[0]
[(str(site.specie), site.specie.oxi_state) for site in structure][:5]"""),

    ("markdown", """\
Bond-valence analysis read the states off the bond lengths, and the cells came
out charge balanced — which is the check worth running, because an unbalanced
assignment is a wrong assignment.

Three routes, and which one you want depends on what you have:"""),

    ("code", """\
mv.transform.oxidation_states(md, method="guess", key_added="guessed")
mv.transform.oxidation_states(md, method={"Sr": 2, "Ti": 4, "V": 4, "O": -2},
                              key_added="explicit")

mv.variants(md)"""),

    ("markdown", """\
### It fails on metals, and says so per row

Bond-valence analysis reads oxidation states off bond lengths, and for a metal
the concept does not apply. A dataset mixing oxides with alloys is normal, so
that is recorded rather than raised:"""),

    ("code", """\
metals = mv.datasets.metals(["Cu", "Al"])
mv.pp.describe(metals)
mv.transform.oxidation_states(metals)

metals.obs[["name", "oxidation_states_ok"]]"""),

    ("code", """\
metals.uns["oxidation_states"]["note"]"""),

    ("markdown", """\
## What you get for it

Everything above is one pymatgen call per structure underneath. What the object
adds is not new physics — it is that the results stay attached:

```python
md = mv.data.from_cif('candidates/')          # pymatgen parses them
mv.transform.oxidation_states(md)             # pymatgen assigns them
mv.transform.apply(md, 'PrimitiveCellTransformation')
mv.calc.relax(md, level='emt', source='primitivecell')
mv.thermo.hull(md, level='emt', source='relaxed_emt')
mv.screen.filter(md, e_above_hull_emt__lt=0.05)

md.write_h5ad('screen.h5ad')                  # all of it, in one file
```

Six calls, one object, and at the end a file that carries the structures, every
variant, every energy, the level of theory each was computed at, and the record
of how they were produced. Written as a script over lists, the same pipeline is
six lists you keep aligned by index and a comment explaining what `structures2`
was.

## The record"""),

    ("code", """\
for step in mv.provenance(md):
    print(step)"""),

    ("markdown", """\
## What matverse does not do

Worth being explicit, so nobody goes looking:

- **Molecules.** `X` is a composition matrix over the periodic table and `obs`
  is materials. pymatgen's `Molecule`, `fragmenter`, `functional_groups` and
  `bond_dissociation` are out of scope by construction.
- **Visualisation.** Crystal Toolkit and VESTA do that better.
- **File format coverage.** `mv.data` has doors for the common ones and hands
  the rest to `pymatgen.io`, which reads far more than matverse should try to.
- **Running anything.** No DFT, no queue, no workflow engine —
  [Infrastructure](infrastructure.ipynb) explains why that boundary is where it
  is.

```{seealso}
[Getting data in and out](data_io.ipynb) is every door into and out of the
object. [Disorder](disorder.ipynb) and [Interfaces](interfaces.ipynb) are the
namespaces that wrap specific transformation families with their domain
knowledge intact.
```"""),
]

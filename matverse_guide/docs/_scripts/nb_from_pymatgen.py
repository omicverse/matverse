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

## A number you can check against a textbook

Most of what a screening library computes can only be checked against another
calculation. The Madelung energy is different — it has a closed form, and that
makes it the one place the machinery can be verified outright."""),

    ("code", """\
rocksalt = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64),
                                     ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
cscl = Structure(Lattice.cubic(4.11), ["Cs", "Cl"],
                 [[0, 0, 0], [.5, .5, .5]])

ionic = mv.data.from_structures([rocksalt, cscl])
mv.pp.describe(ionic)
mv.transform.oxidation_states(ionic)
mv.prop.electrostatic(ionic, source="oxidized")

ionic.obs[["formula", "electrostatic_energy",
           "electrostatic_per_formula_unit"]].round(4)"""),

    ("markdown", """\
| | matverse | α·e²/4πε₀r |
|---|---|---|
| NaCl, α = 1.747565 | −8.9235 | −8.924 |
| CsCl, α = 1.762675 | −7.1310 | −7.131 |

Two structure types, two different Madelung constants, both exact. Getting one
right could be a fitted coincidence; getting both is not.

```{note}
It needs **oxidation states** — a point-charge sum with no charges is zero — and
a neutral structure returns NaN rather than 0, because a zero would read as "no
electrostatic contribution" rather than "nobody assigned any charges".

It is a point-charge model: no covalency, no polarisation, no short-range
repulsion. The right tool for ranking cation orderings on a fixed lattice, which
is what `mv.disorder.orderings` uses it for, and the wrong one for comparing
different chemistries.
```

## The record"""),

    ("code", """\
for step in mv.provenance(md):
    print(step)"""),

    ("markdown", """\
## What matverse does not do

An earlier version of this page listed three things, and two of them were wrong
— not architectural limits, just untested assumptions. They are worth recording
because the corrections are more informative than the original claims.

**"Molecules are out of scope by construction."** False. A `Molecule` has a
composition, so `X` and `var` build for water exactly as for a crystal. The only
thing that failed was a decoder assuming a lattice. See
[Molecules](molecules.ipynb) — `mv.mol` now does point groups, covalent bonds,
fragments and matching, and one object holds molecules and crystals together.

**"Visualisation — Crystal Toolkit and VESTA do that better."** They do, for
real inspection. That was a reason not to compete, not a reason to have
nothing: `mv.pl.structure` draws either kind of material, interactively when
py3Dmol is installed. It is the quick look you take twenty times a day.

**"Running anything."** This one was half right, and the half that was wrong is
now `mv.utils.submit`, which shells out to `sbatch` and records the job id on
the object. matverse still runs no DFT and is not a workflow manager — atomate2,
quacc and AiiDA do that, and a fourth would be a maintenance liability. What it
adds is the link back: *which job is computing this dataset* is answerable from
the data rather than from shell history.

What is genuinely left outside:

- **File format coverage.** `mv.data` has doors for the common ones and hands
  the rest to `pymatgen.io`, which reads far more than matverse should try to.
- **Being a workflow engine.** Retrying, chaining and monitoring belong to the
  tools built for it.

```{seealso}
[Getting data in and out](data_io.ipynb) is every door into and out of the
object. [Disorder](disorder.ipynb) and [Interfaces](interfaces.ipynb) are the
namespaces that wrap specific transformation families with their domain
knowledge intact.
```"""),
]

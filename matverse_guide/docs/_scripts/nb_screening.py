"""Cells for tutorials/screening.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Screening, end to end

The pipeline matverse exists for: take a list of candidate structures, remove
the ones that are broken, relax what remains, work out which are
thermodynamically plausible, and shortlist — keeping the reasoning attached to
the object rather than in your head.

Everything here runs on `emt`, ASE's effective-medium theory, which ships with
ASE and needs no download. The library is the Al–Ni–Cu system: three elemental
metals from published lattice parameters, plus hypothetical ordered
intermetallics built on the L1₂ prototype — the [Cu₃Au
structure](https://doi.org/10.1103/PhysRev.49.122) that most fcc-based ordered
alloys adopt.

The elementals are real; the intermetallics are candidates. That is what a
screen looks like."""),

    ("code", """\
import matverse as mv
import numpy as np

mv.pl.set_style()"""),

    ("markdown", """\
## Build a dataset

`mv.datasets.metals` gives the elemental fcc metals at their measured
room-temperature lattice parameters."""),

    ("code", """\
elemental = mv.datasets.metals(["Al", "Cu", "Ni"])
elemental.obs[["name", "lattice_parameter"]]"""),

    ("markdown", """\
The candidates are L1₂ orderings of the same three elements, plus a B2 AlNi.
Both prototypes are built by hand, because that is what a candidate is — a
structure nobody has computed yet."""),

    ("code", """\
from pymatgen.core import Lattice, Structure


def l12(host, guest, a):
    \"\"\"Cu3Au prototype: guest on the corner, host on the faces.\"\"\"
    return Structure(Lattice.cubic(a),
                     [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def b2(a_symbol, b_symbol, a):
    \"\"\"CsCl prototype.\"\"\"
    return Structure(Lattice.cubic(a), [a_symbol, b_symbol],
                     [[0, 0, 0], [.5, .5, .5]])


candidates = [
    l12("Al", "Cu", 3.90),      # Al3Cu
    l12("Cu", "Al", 3.70),      # AlCu3
    l12("Al", "Ni", 3.78),      # Al3Ni
    b2("Al", "Ni", 2.89),       # AlNi
]"""),

    ("markdown", """\
Put the published elementals and the candidates into one object. Building it in
a single call keeps the row names sequential, which matters more than it sounds
— every later table is indexed by them."""),

    ("code", """\
md = mv.data.from_structures(mv.structures(elemental) + candidates)
md"""),

    ("markdown", """\
Seven materials, three elements. `X` is already the composition matrix and `var`
is already the periodic table restricted to Al, Ni and Cu — you did not ask for
either, because composition is intrinsic to a material rather than derived from
it.

In practice you would load from disk or from a database instead:

```python
md = mv.data.from_cif('candidates/')
md = mv.data.from_mp({'elements': ['Al', 'Ni'], 'num_elements': (2, 2)})
```

## Standardise and describe"""),

    ("code", """\
mv.pp.standardize(md)      # primitive + conventional + spacegroup
mv.pp.describe(md)         # formula, nsites, volume, density, n_elements

md.obs[["formula", "spacegroup", "n_elements"]]"""),

    ("markdown", """\
`standardize` deposits two new structure variants rather than replacing the
input. Every later call names which one it wants."""),

    ("code", """\
mv.variants(md)"""),

    ("markdown", """\
## Throw out what is broken

`qc` is the analogue of `scanpy.pp.calculate_qc_metrics`, and it exists for the
same reason: a generated or database structure with atoms 0.1 Å apart is a
broken cell, not an exotic one. Relaxing it wastes the calculator's time and
pollutes the hull."""),

    ("code", """\
mv.pp.qc(md)
md.obs[["formula", "min_distance", "is_ordered", "is_valid", "qc_reason"]]"""),

    ("code", """\
md = mv.pp.filter_materials(md)
mv.provenance(md)[-1]"""),

    ("markdown", """\
This is one of the few calls that returns instead of depositing, because AnnData
cannot drop rows in place — and it records how many it dropped.

Duplicates are worth removing before you spend compute on them twice:"""),

    ("code", """\
mv.pp.dedup(md)
md.uns["dedup"]"""),

    ("markdown", """\
`dedup` blocks on `(reduced formula, space group)` before running pymatgen's
`StructureMatcher` inside each block. An all-pairs comparison is quadratic and
unusable past a few thousand candidates; blocking makes the expensive part
local.

## Relax"""),

    ("code", """\
mv.calc.relax(md, level="emt", fmax=0.05)

md.obs[["formula", "energy_emt", "energy_per_atom_emt",
        "relax_converged_emt"]].round(4)"""),

    ("markdown", """\
The relaxed geometry becomes its own variant, `relaxed_emt`, rather than
overwriting the input — so "which structure was this energy computed on" stays
answerable from the object alone.

The level records what produced it:"""),

    ("code", """\
mv.level_info(md, "emt")"""),

    ("markdown", """\
`surrogate: True` and `license: 'LGPL-2.1'` are not decoration. The first is
what `mv.thermo.hull` checks before it agrees to mix two levels; the second is
what `mv.calc.check_licenses` reads when you need to know whether a result can
go in a commercial report.

## Build a hull"""),

    ("code", """\
mv.thermo.hull(md, level="emt", source="relaxed_emt")

md.obs[["formula", "e_above_hull_emt", "is_stable_emt",
        "decomposes_to_emt"]].round(4)"""),

    ("markdown", """\
```{warning}
That call emitted a warning, and you should read it. With no `references=`, the
hull is built over **this dataset's own compositions**, which makes
`e_above_hull` a statement about which of your candidates is lowest — not about
whether any of them is stable. Screening 40 generated oxides against each other
will happily report several as "on the hull" when all 40 decompose.
```

Two things about this particular table are worth saying plainly rather than
leaving for a reader to trip over.

EMT's energy zero is the pure element at its equilibrium lattice constant, so
the elemental energies come out at essentially zero and every intermetallic is
positive by construction. **This hull cannot find a stable alloy.** It is
demonstrating the pipeline, not the Al–Ni–Cu phase diagram.

And Al₃Ni is a real, stable phase — it just is not the L1₂ polymorph built
above; the observed one is orthorhombic. A number attached to a prototype says
nothing about a compound that adopts a different structure. Both of those are
ordinary screening mistakes, and both are visible here because the level of
theory and the structure variant are recorded rather than assumed."""),

    ("code", """\
ax = mv.pl.hull(md, level="emt", x="Al")"""),

    ("markdown", """\
The hull plot labels itself when the hull is closed, because a convex hull drawn
without its competing phases looks identical to one drawn with them and means
something entirely different.

Formation energy against composition is also the plot that makes the previous
paragraph obvious: the two elemental endpoints sit at zero and everything
between them is above the tie-line."""),

    ("code", """\
md.uns["phase_diagram"]["closed_system"]"""),

    ("markdown", """\
The object records which kind of number you have, so a later reader does not
have to guess.

To make it absolute, pass the known competing phases:

```python
refs = mv.thermo.references_from_mp(['Al', 'Ni', 'Cu'])   # needs mp-api
mv.thermo.hull(md, level='pbe', references=refs)
```

Note the level changed. Materials Project entries are PBE+U with fitted
corrections; putting them on a hull with EMT energies is not a hull of anything,
and `mv.thermo.hull` raises `LevelMismatch` rather than letting you.

### The hull is not the only stability question

A material can sit on the solid-state hull and still dissolve the moment it
meets water. Aqueous stability is a **separate** question, and
`mv.thermo.pourbaix` answers it — at a stated pH and applied potential, which
is the only way the question means anything."""),

    ("code", """\
try:
    mv.thermo.pourbaix(md, ph=7.0, potential=0.0)
    print(md.obs[["formula", "pourbaix_decomposition"]].round(3))
except (ValueError, ImportError) as exc:
    print(f"{type(exc).__name__}: {exc}")"""),

    ("markdown", """\
It refuses without a key, and the message says why rather than returning zeros.
Aqueous ion energies are **fitted experimental data** that Materials Project
serves; unlike a hull they cannot be computed from your candidate set, so there
is no honest default to fall back on.

## Shortlist"""),

    ("markdown", """\
Ask for something light and not too far above the hull — two criteria that pull
in different directions, which is what makes it a screen rather than a sort."""),

    ("code", """\
mv.screen.filter(md, e_above_hull_emt__lt=0.12, density__lt=8.0)

md.uns["screens"]["passes"]"""),

    ("code", """\
md.obs[["formula", "e_above_hull_emt", "density", "passes"]].round(4)"""),

    ("code", """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 3.6))
passes = md.obs["passes"].to_numpy(dtype=bool)
ax.bar(md.obs["formula"], md.obs["e_above_hull_emt"],
       color=np.where(passes, "#2b7bba", "#cccccc"))
ax.axhline(0.12, linestyle="--", linewidth=0.9, color="#c1121f")
ax.set_ylabel("E above hull (eV/atom)")
ax.set_title("blue passed both criteria; the line is only one of them")"""),

    ("markdown", """\
Cu and Ni sit at zero and are still grey — they failed on density, not on
stability. That is the argument for depositing a boolean column instead of
returning a shorter list: the reason a candidate was dropped is a result, and a
filtered list throws it away."""),

    ("markdown", """\
The screen deposits a boolean column and the criteria that produced it. It does
**not** return a shorter list, because which criterion a candidate failed is a
result. Subset when you actually want the short list:

```python
shortlist = md[md.obs['passes']]
```

When more than one objective matters and no single column decides:"""),

    ("code", """\
mv.screen.pareto(md, {"e_above_hull_emt": "min", "density": "min"})
md.obs[["formula", "e_above_hull_emt", "density", "pareto",
        "pareto_rank"]].round(4)"""),

    ("markdown", """\
`pareto_rank` keeps the second-best trade-offs reachable instead of discarding
everything that is not optimal.

When one column really does decide, `mv.screen.rank` is the simpler tool."""),

    ("code", """\
mv.screen.rank(md, by="e_above_hull_emt")
md.obs[["formula", "e_above_hull_emt", "rank"]].sort_values("rank").round(4)"""),

    ("markdown", """\
## Is the reaction downhill?

A hull says whether a phase survives against everything in the dataset.
`mv.thermo.reaction` answers a narrower, more practical question: does *this*
combination of reactants make *that* product, and by how much."""),

    ("code", """\
mv.thermo.reaction(md, reactants=["Al", "Ni"], products=["AlNi"], level="emt")"""),

    ("markdown", """\
`favourable: False` and a positive energy — EMT again saying no intermetallic is
stable, for the reason given above. On a level of theory that can see chemical
ordering this is the number that tells you whether a synthesis route is worth
attempting."""),

    ("code", """\
ax = mv.pl.pareto(md, "e_above_hull_emt", "density")
ax.set_title("stability against density")"""),

    ("markdown", """\
## What the object remembers"""),

    ("code", """\
for step in mv.provenance(md):
    print(step)"""),

    ("code", """\
ax = mv.pl.provenance(md)"""),

    ("markdown", """\
Parameters are recorded with each call, so the history replays as code rather
than reading as a list of verbs.

## Save it

```python
md.write_h5ad('screen.h5ad')
```

Structures, descriptors, level records and provenance all survive, and the file
is an ordinary `h5ad` that anndata can read without matverse installed.

```{seealso}
[Chemical space](chemical_space.ipynb) picks up from this object and asks what
distinguishes the candidates that passed.
```"""),
]

"""Cells for tutorials/magnetic_ordering.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Magnetic ordering

A composition does not determine a structure, and a structure does not determine
a magnetic state. Iron, nickel, cobalt and most of their compounds have several
spin arrangements at nearly the same energy, and which one is lowest changes the
formation energy by tenths of an eV per atom — comfortably enough to move a
material on or off the convex hull.

So a stability screen over magnetic materials has a step before it that a screen
over oxides of aluminium does not: enumerate the orderings, and find out which
one you are actually computing.

This is the shortest tutorial here, and it ends by refusing to answer the
question. That refusal is the point."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

Three materials chosen to span the cases: a magnetic element, a magnetic alloy,
and something with no magnetic ion at all."""),

    ("code", """\
from pymatgen.core import Lattice, Structure


def fcc(symbol, a):
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


md = mv.data.from_structures([
    fcc("Ni", 3.524),
    Structure(Lattice.cubic(3.57), ["Ni", "Ni", "Ni", "Al"],
              [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
    fcc("Al", 4.050),
])
mv.pp.standardize(md)
mv.pp.describe(md)

md.obs[["formula", "spacegroup", "nsites"]]"""),

    ("markdown", """\
## Which elements can even carry a moment"""),

    ("code", """\
mv.mag.describe(md)

md.obs[["formula", "magnetic_order", "n_magnetic_species",
        "total_magmom", "absolute_magmom"]].round(3)"""),

    ("markdown", """\
Read those two columns carefully, because they answer different questions.

`magnetic_order` is `unknown` for all three, and the moments are NaN. That is
correct: `mv.mag.describe` reports what the **structure carries**, and a CIF or
a hand-built cell carries no moments at all. It is not saying nickel is
non-magnetic; it is saying nobody has told this structure anything.

`n_magnetic_species` is the column doing the work here, and it comes from the
chemistry rather than from the file — 1 for the two nickel-bearing materials, 0
for aluminium. That is what decides whether enumerating orderings is even a
meaningful thing to do.

The set being tested against is `mv.mag.MAGNETIC_ELEMENTS`: the 3d, 4d and 5d
transition metals plus the lanthanides and actinides."""),

    ("code", """\
sorted(mv.mag.MAGNETIC_ELEMENTS)[:20]"""),

    ("markdown", """\
## Enumerating orderings

`mv.mag.orderings` returns a **new object** whose rows are spin configurations,
with `obs['parent']` pointing back at the material — the same shape as
`mv.pp.defects` and `mv.surf.slabs`."""),

    ("code", """\
orderings = mv.mag.orderings(md, max_orderings=6)
orderings"""),

    ("code", """\
mv.pp.describe(orderings)
orderings.obs[["parent", "formula", "ordering", "ordering_index",
               "total_magmom", "is_magnetic"]].round(3)"""),

    ("markdown", """\
Three things worth reading in that table.

The ferromagnetic ordering has the largest total moment, and the
antiferromagnetic one has **exactly zero** — that is what antiferromagnetic
means, and it is a check rather than a coincidence.

Aluminium comes through as a single `nonmagnetic` row rather than being dropped.
Every input material is still represented, so nothing needs re-joining by hand
afterwards.

And the moments are on the structures themselves, as a site property, so they
survive being written to disk and reach any calculator that knows what to do
with them."""),

    ("code", """\
mv.structures(orderings)[0].site_properties["magmom"]"""),

    ("markdown", """\
```{note}
pymatgen's antiferromagnetic enumeration calls out to **enumlib**, which is not
pip-installable and is absent from most environments. matverse falls back to a
simpler construction when it is missing — and records that it did.

Falling back is fine. Falling back silently is not: the fallback explores fewer
configurations, so a ground state found with it is a weaker claim than one found
with enumlib, and a reader has to be able to tell which they are looking at.
```"""),

    ("code", """\
orderings.uns["magnetic_orderings"]["errors"]"""),

    ("markdown", """\
## Picking the ground state

Compute every ordering, and let the lowest energy win. The winner's energy lands
back on the parent material."""),

    ("code", """\
mv.calc.energy(orderings, level="emt")
mv.mag.ground_state(orderings, md, level="emt")

md.obs[["formula", "magnetic_ordering_emt", "magnetic_spread_emt",
        "energy_per_atom_emt"]].round(6)"""),

    ("markdown", """\
## The spread is zero, and that is the answer

`magnetic_spread` is the energy range across the orderings of one material, and
here it is **exactly zero** for both magnetic materials.

That is not a bug. EMT has no notion of spin at all, so every ordering of nickel
is the same set of atoms in the same positions and gets the same energy. The
calculator cannot distinguish them, and the object says so in a number rather
than by producing a confident arbitrary answer.

Which is the useful behaviour: `magnetic_ordering_emt` says `fm` for nickel, and
`magnetic_spread_emt` says that claim is worth nothing."""),

    ("code", """\
md.uns["magnetic"]["emt"]"""),

    ("markdown", """\
```{warning}
**This is where the tutorial stops, because the calculator that ships with
matverse cannot go further.** Resolving a magnetic ground state needs a
spin-polarised method — DFT with initialised moments, or a machine-learned
potential trained on magnetic configurations.

The enumeration above is real and reusable. The energies are not.
```

With such a calculator registered, the rest is unchanged:

```python
from mace.calculators import mace_mp

mv.calc.register_calculator("mace-mpa", lambda: mace_mp(model="medium-mpa-0"),
                            kind="mlip", method="MACE-MPA-0",
                            reference="PBE+U", license="MIT")

orderings = mv.mag.orderings(md, max_orderings=8)
mv.calc.relax(orderings, level="mace-mpa")
mv.mag.ground_state(orderings, md, level="mace-mpa")
```

or by writing the orderings out for DFT:

```python
mv.dft.write_inputs(orderings, code='vasp', preset='relax',
                    directory='orderings/')
```

## Why this belongs before the hull

The energy `mv.mag.ground_state` writes back is under the **ordinary** column
name the calculator would have produced — `energy_per_atom_emt`, not
`energy_per_atom_emt_magnetic`."""),

    ("code", """\
[c for c in md.obs.columns if c.startswith("energy")]"""),

    ("markdown", """\
That is deliberate. `mv.thermo.hull` needs no special case for magnetism: it
sees a normal energy column that happens to be the magnetic ground state, and a
pipeline written without magnetism in mind keeps working when magnetism is
added in front of it.

The alternative — a specially-named column — would mean every downstream
function needed to know about magnetic ordering, and the ones that did not know
would silently use the wrong energy.

```{seealso}
[Screening, end to end](screening.ipynb) is the pipeline this step goes in front
of. [Defects and diffusion](defects_and_diffusion.ipynb) has the same shape:
enumerate configurations, compute them all, let the object record which won.
```"""),

    ("markdown", """\
## The other thing a d shell does

Magnetic ordering is one consequence of a partly filled d shell. Distortion is
the other, and it is structural rather than magnetic: a degenerate electronic
ground state in an octahedral site lowers its energy by distorting the
octahedron.

That is not a detail. It is why LaMnO₃ is orthorhombic rather than cubic, and
why manganese spinel cathodes fade on cycling."""),

    ("code", """\
from pymatgen.core import Lattice, Structure

def perovskite(a_site, b_site, a):
    return Structure(Lattice.cubic(a), [a_site, b_site, "O", "O", "O"],
                     [[0, 0, 0], [.5, .5, .5], [.5, .5, 0],
                      [.5, 0, .5], [0, .5, .5]])

perovskites = mv.data.from_structures([perovskite("La", "Mn", 3.9),
                                       perovskite("Sr", "Ti", 3.905),
                                       perovskite("La", "Ni", 3.85)])
mv.pp.describe(perovskites)
mv.mag.jahn_teller(perovskites)

perovskites.obs[["formula", "jahn_teller_active", "jahn_teller_strength",
                 "jahn_teller_species"]]"""),

    ("markdown", """\
Mn³⁺ and Ni³⁺ come out **strong** and Ti⁴⁺ inactive, which is the textbook
answer: both of the first two put an electron in a doubly degenerate e_g level,
and Ti⁴⁺ is d⁰ with no degeneracy to lift.

The column that matters is `jahn_teller_species` — knowing a material distorts
is not useful without knowing which site is doing it. The ligand bond lengths,
which are the distortion itself rather than a label for it, stay in `uns`.

```{warning}
`strong` means an e_g degeneracy and `weak` a t₂_g one, and the difference is
large: a weak Jahn-Teller distortion usually does not survive room temperature.
Reading `weak` as `distorted` is the common mistake.

The answer also depends on the spin state, which is guessed only if you ask
with `guess_spin=True`. A structure that already carries oxidation states from
`mv.transform.oxidation_states` is used as given rather than re-assigned.
```"""),
]

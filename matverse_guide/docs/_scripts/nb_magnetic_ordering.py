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

    ("markdown", """\
## How far above room temperature does it stay a magnet?

Which ordering is lowest says whether a material is a ferromagnet or an
antiferromagnet. It says nothing about how hot it can get before it stops being
one — and that is usually the question a screen is really asking.

Mapping the ordering energies onto a Heisenberg Hamiltonian gives the exchange
couplings, and a mean-field estimate of the ordering temperature follows:"""),

    ("code", """\
from pymatgen.core import Lattice, Structure

def iron(moments):
    st = Structure(Lattice.cubic(2.87), ["Fe", "Fe"],
                   [[0, 0, 0], [.5, .5, .5]])
    st.add_site_property("magmom", list(moments))
    return st

# stand-in for two spin-polarised total energies, 80 meV apart
spins = mv.data.from_structures([iron([1.0, 1.0]), iron([1.0, -1.0])])
spins.obs_names = ["ferromagnetic", "antiferromagnetic"]
spins.obs["energy_pbe"] = [-0.08, 0.08]

mv.mag.exchange(spins, level="pbe", cutoff=3.0)

spins.obs[["energy_pbe", "exchange_pbe", "ordering_temperature_pbe"]].round(3)"""),

    ("markdown", """\
The ferromagnetic arrangement is lower, so the coupling is positive and the
material orders ferromagnetically — and it does so up to roughly 620 K on this
estimate.

```{warning}
**Read that temperature as an upper bound.** Mean-field theory ignores exactly
the fluctuations that destroy magnetic order, so it overestimates Curie and Néel
temperatures systematically — often by a third to a half. It is good for ranking
candidates and for ruling things out ("nowhere near room temperature"); it is
not a prediction of a measurement.

The couplings come back in meV under pymatgen's convention, which counts per
site rather than per bond. Ratios between materials are convention-free; the
absolute number is not.
```"""),

    ("markdown", """\
### When the fit has nothing to fit

This only means anything for energies from a **spin-polarised** calculation. A
potential that does not distinguish spin returns the same energy for every
ordering, and the Heisenberg fit is then degenerate. matverse says so instead of
reporting a small coupling:"""),

    ("code", """\
import warnings

flat = spins.copy()
flat.obs["energy_pbe"] = [-1.0, -1.0]      # what a spin-blind potential gives

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    mv.mag.exchange(flat, level="pbe", cutoff=3.0)

print(str(caught[-1].message)[:150])
flat.uns["exchange"]["pbe"]["error"]"""),

    ("markdown", """\
A NaN and a stated reason, rather than a number near zero that would read as
"weakly coupled" when it actually means "not calculated with spin"."""),

    ("markdown", """\
## What the moments cost in symmetry

Putting moments on a lattice breaks some of its symmetry and leaves the rest.
Which is which is what a magnetic space group records — and pymatgen ships all
1651 of them but **no analyser that reads one off a structure**. There is no
`MagneticSpaceGroupAnalyzer` beside `SpacegroupAnalyzer`.

`mv.mag.symmetry` computes the underlying quantity instead of naming the group:
every operation of the non-magnetic parent is applied to the moments as the
axial vectors they are, once plainly and once with time reversal, and an
operation survives if either version maps the arrangement onto itself."""),

    ("code", """\
from pymatgen.core import Lattice, Structure

def iron_with(moments):
    st = Structure(Lattice.cubic(2.87), ["Fe", "Fe"],
                   [[0, 0, 0], [.5, .5, .5]])
    st.add_site_property("magmom", list(moments))
    return st

arrangements = mv.data.from_structures([
    iron_with([0.0, 0.0]), iron_with([2.2, 2.2]), iron_with([2.2, -2.2])])
arrangements.obs_names = ["no moments", "ferromagnetic", "antiferromagnetic"]

mv.mag.symmetry(arrangements)
arrangements.obs[["parent_symmetry_order", "magnetic_symmetry_order",
                  "magnetic_symmetry_fraction"]]"""),

    ("markdown", """\
bcc iron has **96** operations. With no moments all 96 survive. With any
collinear ordering along *z*, only the **32** that leave that axis alone do —
a third of the crystal's symmetry, gone, from adding a property that changes no
atomic position.

That fraction is worth having in a screen: one means the moments cost nothing,
a small number means the ordering has broken most of the symmetry, and that is
where to expect magnetic anisotropy and where two orderings at the same energy
are not the same state.

```{note}
**Ferromagnet against antiferromagnet is deliberately not reported here.** The
clean statement of that is whether the magnetic space group contains pure time
reversal, and counting primed operations is not the same thing — a collinear
ferromagnet has primed operations too, picked up from rotations that reverse
its axis. Getting it right needs the group type, which needs the analyser that
does not exist. `mv.mag.describe` gives the net moment, which answers the
practical question without dressing itself up as a symmetry classification.
```"""),
]
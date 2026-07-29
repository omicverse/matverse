"""Cells for tutorials/molecules.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Molecules

The design note used to say molecules were out of scope "by construction".

That was wrong, and checking it took one line. A `Molecule` has a composition,
so `X` and `var` build for water exactly as they do for a crystal — H:2, O:1.
The only thing that failed was a decoder that assumed a lattice.

So molecules live on the same axes as everything else: `obs` is one row per
species, `X` is still the composition matrix, the sites axis is still one row
per atom, and `obs['is_periodic']` tells the two apart. **One object can hold
both**, which is what a study of a catalyst and its adsorbates, an electrolyte
and its salt, or a MOF and its guest actually needs.

What genuinely differs is the analysis: a point group rather than a space
group, covalent bonds rather than a coordination polyhedron, fragments rather
than defects."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

Four molecules with textbook symmetry, so every answer below can be checked
against what a chemistry course would tell you."""),

    ("code", """\
from pymatgen.core import Molecule

water = Molecule(["O", "H", "H"],
                 [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
methane = Molecule(["C", "H", "H", "H", "H"],
                   [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                    [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])
ammonia = Molecule(["N", "H", "H", "H"],
                   [[0, 0, 0.12], [0, 0.94, -0.27],
                    [0.81, -0.47, -0.27], [-0.81, -0.47, -0.27]])
ethanol = Molecule(
    ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
    [[-1.2, 0.2, 0], [0.0, -0.6, 0], [1.1, 0.3, 0],
     [-1.2, 0.9, 0.9], [-1.2, 0.9, -0.9], [-2.1, -0.4, 0],
     [0.0, -1.2, 0.9], [0.0, -1.2, -0.9], [1.9, -0.2, 0]])

md = mv.mol.from_molecules([water, methane, ammonia, ethanol])
md"""),

    ("code", """\
pd.DataFrame(md.X.toarray(), index=["H2O", "CH4", "NH3", "EtOH"],
             columns=md.var_names)"""),

    ("markdown", """\
The composition matrix, built without being asked, exactly as for a crystal.
Nothing about `X` cares whether a formula unit repeats."""),

    ("code", """\
mv.pp.describe(md)
md.obs[["formula", "nsites", "molecular_weight", "volume", "density",
        "is_periodic"]].round(3)"""),

    ("markdown", """\
`volume` and `density` are NaN, because they are properties of a *cell* and a
molecule has none. They become NaN rather than absent so a dataset mixing
crystals with molecules stays one table:"""),

    ("code", """\
mixed = mv.data.from_structures(
    [water, mv.structures(mv.datasets.metals(["Cu"]))[0]])
mv.pp.describe(mixed)

mixed.obs[["formula", "nsites", "molecular_weight", "volume", "is_periodic"]]"""),

    ("markdown", """\
## Symmetry

A molecule has a point group where a crystal has a space group."""),

    ("code", """\
mv.mol.point_group(md)

md.obs[["formula", "point_group", "symmetry_order", "is_chiral",
        "can_be_polar"]]"""),

    ("markdown", """\
C2v for water, **Td** for methane, C3v for ammonia, Cs for ethanol — the
textbook answers, with the group orders to match (4, 24, 6, 2).

Two consequences come free, and the methane row is the interesting one.

**Methane cannot carry a dipole.** Not "does not" — *cannot*. Td contains
operations that map any candidate dipole vector onto its own negative, so the
only vector consistent with the symmetry is zero. A dipole survives only in C1,
Cs, Cn and Cnv; everywhere else it is forbidden. That is a hard selection rule,
and it is why methane has no permanent dipole while water and ammonia do.

**Chirality** is the other one: a molecule is chiral exactly when its point
group contains no improper operation. All four here have a mirror plane, so
none has an enantiomer."""),

    ("code", """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(md.obs["formula"], md.obs["symmetry_order"],
       color=["#4c72b0" if p else "#cccccc" for p in md.obs["can_be_polar"]])
ax.set_ylabel("order of the point group")
ax.set_title("blue can be polar; grey is forbidden by symmetry")"""),

    ("markdown", """\
## Bonds

The molecular counterpart of `mv.env.bonds`, landing in the **same slot** —
`obsp` on the sites object — so a graph algorithm does not need to know which
kind of material it was handed."""),

    ("code", """\
sites = mv.multi.sites(md)
mv.mol.bonds(md, sites)

sites.obsp["bonds"].shape, sites.obsp["bonds"].nnz, sites.uns["bonds"]["kind"]"""),

    ("markdown", """\
Different *perception*, though. A crystal's neighbours come from Voronoi solid
angles; a molecule's bonds come from covalent radii, because a molecule has no
periodic environment to take a solid angle over."""),

    ("code", """\
mv.mol.BOND_STRATEGIES"""),

    ("markdown", """\
## Descriptors that need the graph"""),

    ("code", """\
mv.mol.descriptors(md)

md.obs[["formula", "heavy_atoms", "n_bonds", "n_rings", "rotatable_bonds",
        "radius_of_gyration"]].round(3)"""),

    ("markdown", """\
Ethanol has **two rotatable bonds** — C–C and C–O. The C–H bonds are terminal
and do not count, because rotating about one moves nothing but hydrogens.

That number is what decides how many conformers a search has to cover, and it
grows the cost of everything downstream exponentially.

```{note}
`n_rings` is the cyclomatic number — `edges - nodes + components` — which counts
independent cycles rather than the chemist's smallest set of smallest rings.
They agree for ordinary molecules and differ for fused polycyclics.
```

## Fragments

Break every acyclic bond in turn and keep the pieces. This is what a
bond-dissociation study or a degradation-pathway search enumerates."""),

    ("code", """\
one = mv.mol.from_molecules([ethanol])
mv.pp.describe(one)

fragments = mv.mol.fragments(one)
mv.pp.describe(fragments)

fragments.obs[["broken_bond", "fragment_index", "fragment_formula",
               "fragment_size"]].head(10)"""),

    ("code", """\
fragments.obs.groupby("fragment_index")["fragment_size"].sum().to_dict()"""),

    ("markdown", """\
Every cut splits nine atoms into pieces that still total nine — mass is
conserved, which is the cheapest possible check that the graph surgery is
right.

Breaking C0–C1 gives CH₃ and CH₃O, which is the C–C scission an ethanol
combustion mechanism starts with.

```{warning}
Note `fragment_formula` next to `formula`. pymatgen's **reduced** formula
applies the diatomic convention, so a single hydrogen atom reads `H2` and a
hydroxyl reads `H2O2`. That is right for a stoichiometry and actively
misleading for a fragment, which is a specific set of atoms — so the unreduced
formula is recorded alongside.
```"""),

    ("code", """\
fragments.obs[fragments.obs["fragment_size"] == 1][
    ["broken_bond", "fragment_formula", "formula"]].head(3)"""),

    ("markdown", """\
Ring bonds are not cut, because one cut cannot separate a ring — the molecule
stays connected. A ring needs two, and `fragments` says so rather than
returning nothing without explanation.

Compute the fragments and the parent at the same level and the difference is a
bond dissociation energy. matverse does not take that step for you, because
which reference state you subtract is a choice.

## Deduplication

The molecular counterpart of `mv.pp.dedup`, and it exists for the same reason:
a database query returns the same species many times, and relaxing each costs
the same as relaxing something new."""),

    ("code", """\
moved = Molecule(["O", "H", "H"], np.asarray(water.cart_coords) + 5.0)
library = mv.mol.from_molecules([water, moved, ethanol, methane])
mv.pp.describe(library)
mv.mol.match(library)

library.obs[["formula", "molecule_group", "is_duplicate"]]"""),

    ("code", """\
library.uns["molecule_match"]"""),

    ("markdown", """\
The translated copy of water is recognised: matching is on the reduced formula
and the heavy-atom distance spectrum, which is invariant to rotation,
translation and relabelling.

## Looking at them

`mv.pl.structure` draws either kind — a molecule or a crystal — and picks an
interactive viewer when one is installed."""),

    ("code", """\
ax = mv.pl.structure(one, 0, backend="matplotlib")
ax.set_title("ethanol, projected along its thinnest axis")"""),

    ("markdown", """\
Crude, and it needs nothing extra. With `py3Dmol` installed the default backend
is an interactive viewer instead; neither replaces VESTA or Crystal Toolkit for
real inspection, and that is not what they are for. This is the quick look you
take twenty times a day to catch a slab built upside down or a molecule that
came out of a parser inside out.

## What still applies

Almost everything, because the substrate did not change."""),

    ("code", """\
mv.pp.qc(md)
mv.feat.element_stats(md)
mv.calc.energy(md, level="emt")

md.obs[["formula", "is_valid", "energy_per_atom_emt"]].round(4)"""),

    ("markdown", """\
QC, composition descriptors and even a calculator run unmodified. EMT is
parameterised for C, N, O and H, so those numbers mean something — not much,
but something.

## What refuses, and why

A periodic operation on a molecule would have to invent a lattice, so it
refuses instead:"""),

    ("code", """\
try:
    mv.mol.point_group(mv.datasets.metals(["Cu"]))
except ValueError as exc:
    print(f"ValueError: {exc}")"""),

    ("markdown", """\
```{seealso}
[Environments and bands](structure_and_bands.ipynb) is the crystalline half of
the same bonding question. [Coming from pymatgen](from_pymatgen.ipynb) lists
what matverse still does not do.
```"""),

    ("markdown", """\
## Are the bonds the right length?

`mv.pp.qc` catches atoms sitting on top of one another. It cannot catch the
subtler failure a generated molecule makes: bonds that exist and are the wrong
length. A 1.9 Å C–C bond is not a strained conformer — it is not a molecule —
and no minimum-distance check will say so, because 1.9 Å is a perfectly
ordinary distance between two atoms that are *not* bonded."""),

    ("code", """\
import numpy as np

exact_water = Molecule(["O", "H", "H"],
                       [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
stretched = Molecule([site.specie for site in ethanol],
                     np.asarray(ethanol.cart_coords) * 1.25)

check = mv.mol.from_molecules([exact_water, ethanol, stretched])
mv.pp.describe(check)
mv.mol.bond_lengths(check)

check.obs[["formula", "n_bonds_measured", "mean_bond_deviation",
           "max_bond_deviation", "n_unusual_bonds", "bond_lengths_ok"]].round(3)"""),

    ("markdown", """\
Water comes out at **zero** deviation, because it was built at exactly the
tabulated 0.96 Å O–H. Ethanol is within a few hundredths.

The stretched copy is the row to read, and **`n_bonds_measured` is the column
that matters** — not the deviation. Bonds are found by covalent radius, so a 25%
stretch does not produce long bonds; it produces almost *no* bonds. Eight become
one. A deviation computed from the single survivor is nearly meaningless, and
the count is the loud signal.

```{note}
Lengths are compared against the **single-bond** value throughout, so a double
or triple bond reads as short by design: C=C at 1.34 Å against a tabulated 1.54
is a deviation of 0.2. Read `n_unusual_bonds` as "worth looking at" rather than
"wrong".
```"""),

    ("markdown", """\
## Two questions that both mean "the same molecule"

`mv.mol.match` groups molecules by superposing them — Kabsch for the rotation,
Hungarian for the atom labels. That answers "is this the same *shape*".

A **conformer** is the case where the two questions come apart. Rotate ethanol's
hydroxyl hydrogen about the C–O bond: every bond length is preserved and the
geometry moves."""),

    ("code", """\
conformer = ethanol.copy()
conformer.rotate_sites(
    indices=[8], theta=1.2,
    axis=np.array(ethanol[2].coords) - np.array(ethanol[1].coords),
    anchor=ethanol[2].coords)

pair = mv.mol.from_molecules([ethanol, conformer])
mv.pp.describe(pair)

for how in ("geometry", "topology"):
    mv.mol.match(pair, method=how)
    print(how, "->", pair.uns["molecule_match"]["n_unique"], "unique")"""),

    ("markdown", """\
**Geometry says two molecules. Topology says one.**

Neither is wrong. `method='geometry'` superposes and measures RMSD — this
conformer sits 0.29 Å away, outside the default tolerance of 0.1.
`method='topology'` compares the bond graph and ignores shape entirely.

Which you want depends on whether conformers are the thing you are
deduplicating or the thing you are studying. That is why it is a `dispatch`
rather than a default: the registry entry names both routes, so an agent
choosing between them can see that the choice exists."""),

    ("markdown", """\
## Free energies, and the mode you trust least

A rigid-rotor harmonic-oscillator entropy diverges as $1/\\omega$. That is fine
for a stiff molecule and a disaster for a floppy one: a hindered rotation at
10 cm⁻¹ contributes more entropy than every stiff mode put together, and it is
precisely the mode whose frequency the calculation got least right.

Grimme's quasi-RRHO interpolates those modes onto a free-rotor entropy below a
cutoff. `mv.mol.quasirrho` deposits **both** numbers, so the size of the
correction is visible rather than buried:"""),

    ("code", """\
from pymatgen.core import Molecule

water = Molecule(["O", "H", "H"],
                 [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
thermo = mv.mol.from_molecules([water, water])
thermo.obs_names = ["stiff", "one soft mode"]
thermo.obs["e_dft"] = [-76.4, -76.4]

# identical molecule and energy; the second has one 12 cm-1 mode
mv.mol.quasirrho(thermo,
                 [[1595.0, 3657.0, 3756.0],
                  [12.0, 1595.0, 3657.0]],
                 energy="e_dft")

thermo.obs[["entropy_harmonic", "entropy_quasirrho",
            "free_energy_quasirrho"]].round(4)"""),

    ("markdown", """\
Same molecule, same energy, one mode moved. The stiff case agrees to four
decimals — nothing below the cutoff, nothing to correct. The soft case has the
harmonic entropy **overshooting by 3.6 cal/(mol·K)**, which is about 1 kcal/mol
in $-T\\Delta S$ at room temperature, from a single mode at a frequency no
method pins down well.

That is the whole argument for quasi-RRHO: the error it removes is largest
exactly where the input is weakest.

```{note}
**Frequencies are an argument, in cm⁻¹**, one sequence per row — the same
arrangement as `mv.md.rdf` taking a trajectory. matverse's own calculators are
metals potentials and would give molecular frequencies not worth correcting, so
bring them from the quantum-chemistry run that produced the energy, and bring
the energy in Hartree.

Imaginary modes are dropped and counted in
`uns['quasirrho'][...]['n_imaginary']`. A negative frequency means the geometry
is a saddle rather than a minimum, and a thermochemical correction to a
structure that is not a minimum corrects nothing — better to see the count than
to have it folded in silently.
```"""),
]
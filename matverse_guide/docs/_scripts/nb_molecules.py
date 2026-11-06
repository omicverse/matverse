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
ax = mv.pl.scatter(md, "formula", "symmetry_order", kind="bar",
                   color="can_be_polar")
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
## Which bond breaks first

The bond dissociation energy is the energy of the pieces minus the energy of the
whole. It decides which bond in a battery electrolyte oxidises first and which
C–H a combustion mechanism abstracts, and `mv.mol.fragments` has already done
the breaking — what is left is arithmetic on two energies:"""),

    ("code", """\
from pymatgen.core import Molecule

alcohol = Molecule(
    ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
    [[-1.1, 0.2, 0.0], [0.2, -0.5, 0.0], [1.3, 0.4, 0.0], [-1.0, 1.3, 0.0],
     [-1.7, -0.1, 0.9], [-1.7, -0.1, -0.9], [0.3, -1.2, 0.9],
     [0.3, -1.2, -0.9], [2.1, -0.1, 0.0]])

whole = mv.mol.from_molecules([alcohol])
whole.obs_names = ["ethanol"]
pieces = mv.mol.fragments(whole, depth=1)

# stand-in energies: a real screen would put a calculator here
whole.obs["energy_x"] = [-100.0]
pieces.obs["energy_x"] = [-10.0 * n for n in pieces.obs["fragment_size"]]

bde = mv.mol.dissociation(pieces, whole, level="x")
bde.obs[["broken_bond", "bond_dissociation_energy_x", "n_fragments"]]"""),

    ("markdown", """\
Eight bonds, one row each, every one named. With the placeholder energies above
they all come out equal, which is the point of choosing them that way — every
cut of ethanol leaves nine atoms between the fragments, so at a fixed energy per
atom the arithmetic has to give the same answer eight times. Put a real
calculator in and the spread appears.

```{warning}
**The fragments are radicals**, and that is where the accuracy goes. A radical
has an unpaired electron, and a method that is fine for closed-shell molecules
can be badly wrong for one — universal potentials especially, since they are
trained mostly on closed-shell equilibrium structures. Read the *ordering* of
bonds within a molecule as the useful output; treat the absolute numbers as
indicative unless the level was chosen for open shells.

Geometries are used as they are, so this is the **vertical** dissociation
energy. Relax the fragments first for the adiabatic one; the two differing is
the fragment relaxation energy, which is information rather than a problem.
```"""),

    ("markdown", """\
## What is actually on the molecule

A formula does not say what a molecule does. C₂H₆O is ethanol or dimethyl
ether depending on where the oxygen sits, and those behave nothing alike. A
screen over candidate molecules almost always wants *"the ones with a
carboxylic acid"* rather than *"the ones with two carbons"*.

`mv.mol.functional_groups` walks out from the heteroatoms and reports what it
finds:"""),

    ("code", """\
from pymatgen.core import Molecule

ethanol = Molecule(
    ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
    [[-1.1, 0.2, 0.0], [0.2, -0.5, 0.0], [1.3, 0.4, 0.0], [-1.0, 1.3, 0.0],
     [-1.7, -0.1, 0.9], [-1.7, -0.1, -0.9], [0.3, -1.2, 0.9],
     [0.3, -1.2, -0.9], [2.1, -0.1, 0.0]])
water = Molecule(["O", "H", "H"],
                 [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])

groups = mv.mol.from_molecules([ethanol, water])
groups.obs_names = ["ethanol", "water"]

try:
    mv.mol.functional_groups(groups)
    result = groups.obs[["functional_groups", "n_functional_groups"]]
except ImportError as exc:
    result = str(exc)[:200]      # openbabel is an optional extra
result"""),

    ("markdown", """\
Ethanol comes back as `[CH3];[OH]` — a methyl and a hydroxyl, which is what it
is. The column is a sorted, semicolon-separated **string** on purpose, so it
can go straight into `mv.screen.filter` with a `contains` and be read without
unpacking anything. The counts per group live in `uns`, where a dict belongs.

```{warning}
This needs openbabel's Python bindings, and openbabel needs **libXrender** at
runtime. `pip install openbabel-wheel` gets the bindings; without libXrender
they fail to import with a `ValueError` raised from openbabel's format table —
an error that names neither openbabel nor the missing library, and costs an
afternoon to trace.

`conda install -c conda-forge xorg-libxrender`, or your distribution's
`libxrender1`, fixes it. The error matverse raises says all of this, which is
the only reason it is worth writing down here.
```"""),

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

    ("markdown", """\
## Conformers under restraints

Everything above takes the geometry as given. Often the geometry is what you
need, and you already know something about it: an NMR experiment says two
protons are within 3 Å (a NOE), a ³J coupling pins a torsion through the Karplus
relation, a crystal structure or a pharmacophore fixes a distance. Free
conformer generators do not know any of that, and filtering a free ensemble
afterwards fails exactly when it matters — when the restrained region is a
small corner of conformational space.

Restrained **distance geometry** builds the knowledge in from the start.
`mv.mol` has three functions for it, on RDKit molecules rather than a dataset,
because a conformer search is a per-molecule computation whose result is
coordinates, and RDKit is where stereochemistry, force fields and distance
geometry live:

| function | what it does |
|---|---|
| `mv.mol.restraint_bounds` | the lower/upper interatomic distance matrices the restraints imply |
| `mv.mol.embed_restrained` | conformers that satisfy the restraints and keep the molecule's graph and stereochemistry |
| `mv.mol.vendor_restraint_engine` | copy the engine (numpy + RDKit only) into a package that cannot import matverse |

They need RDKit: `pip install "matverse[conformers]"`.

### Distance geometry in one paragraph

A conformation is fixed, up to its mirror image, by its interatomic distances
(Crippen & Havel, 1988). The molecular graph already bounds most of them: a
bond length (1-2 pairs), a bond angle (1-3), a torsion between cis and trans
(1-4), van der Waals contact as a lower bound for everything further apart.
Those bounds are made mutually consistent by **triangle smoothing** —
`u_ij ≤ u_ik + u_kj` and `l_ij ≥ l_ik − u_kj` for every triple, iterated —
then distances are drawn inside the bounds, turned into coordinates by an
eigendecomposition of the metric matrix, and refined. RDKit's ETKDG adds
experimental torsion preferences on top
([Riniker & Landrum, 2015](https://doi.org/10.1021/acs.jcim.5b00654)).
A restraint is simply a tighter entry in the bounds matrix *before* smoothing —
and smoothing then propagates it to pairs you never mentioned."""),

    ("code", """\
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms

# Ac-Ala-NHMe, the alanine dipeptide: the standard test case of
# conformational analysis. AddHs appends the hydrogens after the heavy atoms,
# so the SMILES order of atoms 0-9 is kept.
dipeptide = Chem.AddHs(Chem.MolFromSmiles("CC(=O)N[C@@H](C)C(=O)NC"))

PHI = [1, 3, 4, 6]          # C(=O)-N-CA-C
PSI = [3, 4, 6, 8]          # N-CA-C(=O)-N
H_N1, H_N2 = 13, 18         # the two amide hydrogens
H_METHYL = [15, 16, 17]     # the alanine side-chain methyl hydrogens

pd.DataFrame([{"index": a.GetIdx(), "element": a.GetSymbol(),
               "bonded to": [n.GetIdx() for n in a.GetNeighbors()]}
              for a in dipeptide.GetAtoms() if a.GetIdx() in range(10) or a.GetIdx() in (13, 18)])"""),

    ("markdown", """\
Its backbone has two free torsions, φ and ψ, and the familiar Ramachandran
regions are combinations of them:

| region | φ | ψ |
|---|---|---|
| right-handed α helix (αR) | −63° | −43° |
| β strand | −120° | +130° |
| polyproline II (PPII) | −75° | +145° |
| C7eq (γ-turn, intramolecular H-bond) | −83° | +73° |
| left-handed helix (αL) | +60° | +45° |

### What the graph alone allows

With no restraints, `restraint_bounds` returns RDKit's smoothed topological
bounds. A bond is pinned to a few hundredths of an ångström, an angle almost
as tightly, a 1-4 pair spans its whole cis-to-trans range, and anything further
is loose:"""),

    ("code", """\
def bounds_table(lower, upper, pairs):
    return pd.DataFrame([{"pair": name, "atoms": (i, j), "lower Å": lower[i, j],
                          "upper Å": upper[i, j]} for name, (i, j) in pairs.items()]).round(3)

PAIRS = {"1-2  C1–N3 (bond)": (1, 3), "1-3  C1–CA4 (angle)": (1, 4),
         "1-4  C1–C6 (φ)": (1, 6), "1-4  N3–N8 (ψ)": (3, 8),
         "HN1–HN2 (sequential amide NOE)": (H_N1, H_N2)}
lower0, upper0 = mv.mol.restraint_bounds(dipeptide)
bounds_table(lower0, upper0, PAIRS)"""),

    ("markdown", """\
### A torsion restraint becomes a 1-4 interval — without its sign

A bounds matrix holds distances, and a torsion enters only through the 1-4
distance it implies:

`d₁₄² = a² + b² + c² − 2ab cos α − 2bc cos β + 2ac (cos α cos β − sin α sin β cos φ)`

That depends on **cos φ**. φ = +60° and φ = −60° are mirror images with the
same 1-4 distance, so no bounds matrix can tell them apart. `restraint_bounds`
evaluates this expression over the whole box of bond lengths and angles the
matrix allows, by interval arithmetic with subdivision and outward rounding, so
the interval is guaranteed to contain every 1-4 distance the restraint permits:"""),

    ("code", """\
rows = []
for label, (lo, hi) in {"φ in [+55, +65]": (55, 65), "φ in [-65, -55]": (-65, -55),
                        "φ in [-130, -110]": (-130, -110)}.items():
    lower, upper = mv.mol.restraint_bounds(
        dipeptide, {"torsions": [{"atoms": PHI, "lower": lo, "upper": hi}]})
    rows.append({"restraint": label, "C1–C6 lower Å": lower[1, 6], "C1–C6 upper Å": upper[1, 6]})
pd.DataFrame(rows).round(3)"""),

    ("markdown", """\
The first two rows are identical, as the formula says they must be. The sign is
fixed later, in the embedding: after ETKDG has produced coordinates,
`embed_restrained` chooses the torsion branch and orients the molecule onto it
before refinement.

### Smoothing propagates a restraint

A NOE between the two amide hydrogens of 2.8 Å or less is characteristic of
helical backbones. Restraining that one pair moves entries nobody restrained:"""),

    ("code", """\
noe = {"distances": [{"atoms": [H_N1, H_N2], "lower": 1.8, "upper": 2.8}]}
lower1, upper1 = mv.mol.restraint_bounds(dipeptide, noe)

changed = (np.abs(upper1 - upper0) > 1e-6) | (np.abs(lower1 - lower0) > 1e-6)
print(f"one restraint changed {int(changed.sum() // 2)} of "
      f"{dipeptide.GetNumAtoms() * (dipeptide.GetNumAtoms() - 1) // 2} pairs")
bounds_table(lower1, upper1, PAIRS)"""),

    ("markdown", """\
The restraint itself is in (1.8–2.8 Å), and so are its consequences — but look
at the 1-4 rows: their lower bounds fell below 1 Å. Once any restraint is
present, every pair beyond 1-3 starts from a loose `[0, far]` rather than from
RDKit's topological preferences, and is narrowed only by the restraints and by
smoothing. The restrained matrix is a feasibility region for the search, not a
description of plausible geometry; bond lengths, angles, clashes and
stereochemistry are enforced afterwards, on the embedded conformer. The 1-2
and 1-3 pairs keep their topological values; `one_bond_slack` and
`two_bond_slack` widen them, which the force-field refinement below needs room
for.

## Embedding under restraints

`embed_restrained` returns a **new** molecule — the input is never modified —
whose conformers each pass an admissibility check: every requested coordinate
within its tolerance, the input's CIP and E/Z labels reproduced from 3D, bond
lengths and angles within the 1-2/1-3 bounds, and no severe clash. Under the
hood: ETKDG on the restrained bounds matrix with a hashed per-attempt seed,
the torsion sign fixed, then MMFF94s refinement with flat-bottom restraints.

Torsions are **signed degrees on a circle**. An interval with `lower > upper`
is the arc that wraps through 180°: `[150, -170]` is the 40° arc from +150°
through ±180° to −170°. Only the literal `[-180, 180]` is the full circle."""),

    ("code", """\
SLACK = dict(one_bond_slack=0.1, two_bond_slack=0.15,
             distance_tol=0.03, angle_tol=2.0, torsion_tol=2.0)

REGIONS = {"αR":   ((-75, -55), (-55, -35)),
           "β":    ((-140, -110), (115, 145)),
           "PPII": ((-85, -65), (150, -170)),     # ψ wraps through 180
           "αL":   ((50, 70), (30, 50))}


def phi_psi(mol, conf_id=0):
    conf = mol.GetConformer(conf_id)
    return (rdMolTransforms.GetDihedralDeg(conf, *PHI),
            rdMolTransforms.GetDihedralDeg(conf, *PSI))


def cip(mol, conf_id=0):
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(probe, confId=conf_id)
    return probe.GetAtomWithIdx(4).GetProp("_CIPCode")


rows, folded = [], {}
for name, (phi, psi) in REGIONS.items():
    out = mv.mol.embed_restrained(dipeptide, {"torsions": [
        {"atoms": PHI, "lower": phi[0], "upper": phi[1]},
        {"atoms": PSI, "lower": psi[0], "upper": psi[1]}]}, random_seed=7, **SLACK)
    folded[name] = out
    f, s = phi_psi(out)
    rows.append({"region": name, "φ requested": phi, "φ": f, "ψ requested": psi, "ψ": s,
                 "CA label": cip(out)})
pd.DataFrame(rows).round(1)"""),

    ("markdown", """\
Every conformer is in its box, and the alanine stays (S) — L-alanine — in all
four, including αL, which mirror symmetry would otherwise reach most cheaply by
inverting the stereocentre.

The values sit at the edges of their intervals. That is the flat-bottom
refinement doing what it is told: inside the interval there is no restraint
force, so the force field relaxes toward its own preference until it meets the
wall. A restraint is an interval, not a target; if you want the middle, ask
for a narrow interval."""),

    ("code", """\
fig, ax = plt.subplots(figsize=(4.5, 4.5))
for k, (name, (phi, psi)) in enumerate(REGIONS.items()):
    colour = f"C{k}"
    for lo, hi in ([psi] if psi[0] <= psi[1] else [(psi[0], 180), (-180, psi[1])]):
        ax.add_patch(plt.Rectangle((phi[0], lo), phi[1] - phi[0], hi - lo,
                                   alpha=0.25, color=colour))
    f, s = phi_psi(folded[name])
    ax.plot(f, s, "o", color=colour, label=name)
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xlabel("φ (degrees)")
ax.set_ylabel("ψ (degrees)")
ax.set_title("requested boxes and embedded conformers")
ax.legend(loc="lower left")"""),

    ("markdown", """\
### A hydrogen bond restraint, and what it does not decide

The C7 conformers close a seven-membered ring with a hydrogen bond from the
second amide N–H to the acetyl oxygen. A distance restraint O2···H18 of
1.8–2.2 Å asks for it — and there are **two** C7 minima, C7eq and C7ax, at
roughly opposite signs of φ and ψ. A distance cannot choose between them. Ask
for three conformers and look:"""),

    ("code", """\
hbond = {"distances": [{"atoms": [2, H_N2], "lower": 1.8, "upper": 2.2}]}
three = mv.mol.embed_restrained(dipeptide, hbond, num_confs=3, random_seed=7, **SLACK)
pd.DataFrame([{"conformer": c, "φ": phi_psi(three, c)[0], "ψ": phi_psi(three, c)[1],
               "O···H Å": np.linalg.norm(three.GetConformer(c).GetPositions()[2]
                                          - three.GetConformer(c).GetPositions()[H_N2])}
              for c in range(3)]).round(2)"""),

    ("markdown", """\
Both minima appear, and two of the three conformers are the same structure:
without `min_rmsd`, nothing asks the ensemble to be diverse. To get C7eq
specifically, add what the distance cannot say — the sign of φ:"""),

    ("code", """\
c7eq = mv.mol.embed_restrained(dipeptide, {
    **hbond, "torsions": [{"atoms": PHI, "lower": -100, "upper": -60}]},
    random_seed=7, **SLACK)
print("C7eq: φ = {:.1f}, ψ = {:.1f}".format(*phi_psi(c7eq)))"""),

    ("markdown", """\
### Ambiguous restraints: an OR over pairs

An NOE to a methyl group is the textbook ambiguity: the three methyl protons
are equivalent on the NMR timescale, so a cross-peak says that *one of them*
is close, not which (Nilges, 1995).
`ambiguous_distances` takes the alternatives explicitly; each entry must be
satisfied by at least one of its pairs, and separate entries must all hold.

A pairwise bounds matrix cannot hold an OR, so a genuine OR group adds nothing
to `restraint_bounds` — it would be wrong to tighten any single pair.
`embed_restrained` instead branches over the alternatives in a fixed order:"""),

    ("code", """\
methyl_noe = {"ambiguous_distances": [
    {"alternatives": [[H_N2, h] for h in H_METHYL], "lower": 1.8, "upper": 2.7}]}

with_or = {**noe, **methyl_noe}               # the helical NOE plus the OR group
same = all(np.array_equal(a, b) for a, b in
           zip(mv.mol.restraint_bounds(dipeptide, noe), mv.mol.restraint_bounds(dipeptide, with_or)))
print("bounds unchanged by adding the OR group:", same)

out = mv.mol.embed_restrained(dipeptide, methyl_noe, random_seed=7, **SLACK)
pos = out.GetConformer(0).GetPositions()
{f"H{H_N2}–H{h}": round(float(np.linalg.norm(pos[H_N2] - pos[h])), 2) for h in H_METHYL}"""),

    ("markdown", """\
One methyl proton is at the limit, the other two are where the methyl's
geometry puts them. That is the right answer to an ambiguous restraint.

### When the restraints cannot be met

There are three different reasons a restrained embedding can come back empty,
and they call for different responses, so they are different exceptions — all
subclasses of `mv.mol.RestraintError`:

| exception | meaning | response |
|---|---|---|
| `RestraintValidationError` | the request is malformed: an index out of range, an angle whose atoms are not a bonded path, `lower > upper` on a distance | fix the input |
| `RestraintSearchError` | nothing admissible was found within `max_attempts` — not a proof that nothing exists | widen tolerances, raise the budget, or look for a contradiction |
| `RestraintInfeasibleError` | *your* `proof` hook certified that the restraints contradict | report it; searching longer will not help |

The engine never claims a proof it does not have: an impossible restraint
without a `proof` hook is a search error. The hook is how a caller who *can*
prove infeasibility says so. Here, a 1-4 distance of at least 5 Å is
impossible — three bonds of at most ~1.6 Å cannot span it — and a short proof
says exactly that: no distance can exceed the summed bond lengths of the
shortest bonded path between its atoms."""),

    ("code", """\
def chain_length_proof(mol, restraints):
    # a certificate (truthy) if any distance exceeds the bonded path between its atoms
    for entry in restraints.get("distances", []):
        i, j = entry["atoms"]
        path = Chem.GetShortestPath(mol, i, j)
        if entry["lower"] > 1.6 * (len(path) - 1):
            return {"pair": (i, j), "path_bonds": len(path) - 1, "lower": entry["lower"]}
    return None


attempts = {
    "angle on a non-path": ({"angles": [{"atoms": [0, 3, 4], "lower": 100, "upper": 120}]}, {}),
    "impossible 1-4 distance": ({"distances": [{"atoms": [1, 6], "lower": 5.0, "upper": 5.5}]}, {}),
    "same, with a proof": ({"distances": [{"atoms": [1, 6], "lower": 5.0, "upper": 5.5}]},
                           {"proof": chain_length_proof}),
}
for label, (restraints, extra) in attempts.items():
    try:
        mv.mol.embed_restrained(dipeptide, restraints, max_attempts=5, **SLACK, **extra)
    except mv.mol.RestraintError as exc:
        print(f"{label:25s} {type(exc).__name__}: {exc}")"""),

    ("markdown", """\
### Ensembles

NMR structure determination, docking and conformer-averaged properties want an
**ensemble**, not one structure. `num_confs` asks for several; `min_rmsd` makes
them distinct — admissible conformers are diversified by rotating unrestrained
acyclic rotors and selected greedily so that every pair is at least `min_rmsd`
apart in symmetry-corrected heavy-atom RMSD. Under the helical NOE from above:"""),

    ("code", """\
ensemble = mv.mol.embed_restrained(dipeptide, noe, num_confs=5, random_seed=9,
                                   min_rmsd=0.5, max_attempts=200, **SLACK)
heavy = Chem.RemoveHs(ensemble)
n = ensemble.GetNumConformers()
rmsd = np.array([[0.0 if a == b else rdMolAlign.GetBestRMS(heavy, heavy, a, b)
                  for b in range(n)] for a in range(n)])
print("φ/ψ:", [tuple(round(x) for x in phi_psi(ensemble, c)) for c in range(n)])
print("HN–HN Å:", [round(float(np.linalg.norm(ensemble.GetConformer(c).GetPositions()[H_N1]
                                            - ensemble.GetConformer(c).GetPositions()[H_N2])), 2)
                   for c in range(n)])
print(f"smallest pairwise heavy-atom RMSD: {rmsd[~np.eye(n, dtype=bool)].min():.2f} Å")
pd.DataFrame(rmsd).round(2)"""),

    ("markdown", """\
Every member satisfies the NOE, and they are pairwise at least 0.5 Å apart.
Some share φ and ψ: RMSD counts every heavy atom, so conformers that differ only
in the terminal acetyl and N-methyl groups count as distinct. If backbone
diversity is what you need, restrain or compare the backbone.

The search is **deterministic for a seed**: each attempt's RDKit seed is a hash
of `random_seed` and the attempt number, so the same call gives the same
conformers on any machine with the same RDKit, and changing an unrelated
restraint does not reshuffle the attempts."""),

    ("code", """\
again = mv.mol.embed_restrained(dipeptide, noe, num_confs=5, random_seed=9,
                                min_rmsd=0.5, max_attempts=200, **SLACK)
print("same seed, same coordinates:",
      all(np.allclose(ensemble.GetConformer(c).GetPositions(),
                      again.GetConformer(c).GetPositions()) for c in range(n)))"""),

    ("markdown", """\
### Taking the engine with you

Sometimes the code that needs restrained conformers has to run where matverse
is not installed — a plugin for another program, a package checked in a clean
environment. The engine imports nothing from matverse (numpy and RDKit only),
and `vendor_restraint_engine` copies it into a directory for a relative import.
Below, the copy is imported in a fresh interpreter in which `import matverse`
is made to fail:"""),

    ("code", """\
import subprocess
import sys
import tempfile
from pathlib import Path

pkg = Path(tempfile.mkdtemp()) / "mypkg"
path = mv.mol.vendor_restraint_engine(pkg)
(pkg / "__init__.py").write_text("from ._restrained_embedding import embed_restrained, RestraintError\\n")

probe = (
    "import sys; sys.modules['matverse'] = None\\n"
    "import mypkg\\n"
    "from rdkit import Chem\\n"
    "m = mypkg.embed_restrained(Chem.AddHs(Chem.MolFromSmiles('CCO')), num_confs=2,\\n"
    "                           one_bond_slack=0.1, two_bond_slack=0.15)\\n"
    "print('conformers:', m.GetNumConformers())\\n")
run = subprocess.run([sys.executable, "-c", probe], cwd=pkg.parent,
                     capture_output=True, text=True, check=True)
print(Path(path).name, "->", run.stdout.strip())"""),

    ("markdown", """\
Wrap the copied functions in the target's own API: translate each `Restraint*`
error into that contract's exception class, and pass its numeric rules as the
keywords (`distance_envelope`, `torsion_tol`, `one_bond_slack`, …) rather than
editing the file.

```{admonition} Pitfalls
:class: warning
- **Atom indices refer to the molecule you pass.** Add hydrogens *before*
  choosing indices for restraints on hydrogens; `Chem.AddHs` appends them, so
  heavy-atom indices from the SMILES survive.
- **Zero slack is strict.** With the defaults the admissibility check demands
  RDKit's exact topological 1-2/1-3 bounds, which a force-field-refined
  structure rarely meets; small slacks (0.1 Å, 0.15 Å here) are realistic.
- **A torsion interval is an arc.** `[-170, 150]` is 320° wide; the 40° arc
  through 180° is `[150, -170]`.
- **Angles and torsions follow bonds.** An angle `i-j-k` needs bonds `i-j` and
  `j-k`; anything else is a validation error, not a geometric request.
- **A search error is not a proof.** Only your `proof` hook can turn "not
  found" into "does not exist".
```

**References.** G. M. Crippen, T. F. Havel, *Distance Geometry and Molecular
Conformation* (Research Studies Press, 1988) · J. M. Blaney, J. S. Dixon,
*Rev. Comput. Chem.* **5**, 299 (1994) · S. Riniker, G. A. Landrum,
*J. Chem. Inf. Model.* **55**, 2562 (2015) · T. A. Halgren,
*J. Comput. Chem.* **20**, 720 (1999) — MMFF94s · M. Nilges,
*J. Mol. Biol.* **245**, 645 (1995) — ambiguous distance restraints ·
K. Wüthrich, *NMR of Proteins and Nucleic Acids* (Wiley, 1986) — NOE distances."""),
]

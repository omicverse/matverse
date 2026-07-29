"""Cells for tutorials/disorder.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Disorder

Everything else in matverse assumes a structure is ordered: this atom is here,
that one is there. A large fraction of real materials are not.

A solid solution, a doped semiconductor, a high-entropy alloy and a cathode
part-way through charging all have sites that are **fractionally occupied** —
and a first-principles code cannot take one as input. It needs a specific
arrangement of specific atoms.

Bridging that gap is a bigger problem than it looks. A disordered cell stands
for an *ensemble*, and any single ordered cell you pick is one sample from it.
Which sample you pick changes the answer, and choosing wrongly is one of the
easier ways to compute something confidently meaningless."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

Three cases spanning the problem: a binary solid solution, a five-component
equiatomic alloy of the kind called *high-entropy*, and an ordered metal for
contrast."""),

    ("code", """\
from pymatgen.core import Lattice, Structure


def mixed(occupancies, a=3.7):
    \"\"\"An fcc cell whose every site is shared between species.\"\"\"
    return Structure(Lattice.cubic(a), [dict(occupancies)] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


md = mv.data.from_structures([
    mixed({"Cu": 0.5, "Au": 0.5}, a=3.8),
    mixed({"Cu": 0.2, "Ni": 0.2, "Co": 0.2, "Fe": 0.2, "Mn": 0.2}),
    mv.structures(mv.datasets.metals(["Cu"]))[0],
])
mv.pp.describe(md)

md.obs[["formula", "nsites", "density"]].round(3)"""),

    ("markdown", """\
Note that `mv.pp.describe` reports a formula for all three — a disordered cell
is a perfectly good structure, it just is not one you can hand to VASP.

## How disordered, and where"""),

    ("code", """\
mv.disorder.describe(md, temperature=1500.0)

md.obs[["formula", "is_ordered", "n_disordered_sites", "max_site_disorder",
        "configurational_entropy", "entropy_term_300K"]].round(5)"""),

    ("markdown", """\
`max_site_disorder` is how far the worst site is from having a single owner:
0.5 for an even two-way split, 0.8 for an even five-way split, 0 for an ordered
cell.

## The entropy a zero-temperature hull leaves out

`configurational_entropy` is the ideal mixing entropy per atom. For an
equiatomic mixture of *m* species it has a closed form, `k_B ln m`, which is
worth checking rather than trusting:"""),

    ("code", """\
predicted = mv.disorder.KB_EV * np.log([2, 5, 1])
pd.DataFrame({
    "formula": md.obs["formula"],
    "computed": md.obs["configurational_entropy"].to_numpy(dtype=float),
    "k_B ln(m)": predicted,
}).round(8)"""),

    ("markdown", """\
Exact. And the consequence is the interesting part."""),

    ("code", """\
import matplotlib.pyplot as plt

temperatures = np.linspace(0, 2000, 200)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for formula, entropy in zip(md.obs["formula"],
                            md.obs["configurational_entropy"]):
    ax.plot(temperatures, -temperatures * entropy, label=formula)
ax.axhline(-0.05, linestyle="--", color="#c1121f", linewidth=1.0,
           label="a typical 50 meV hull cut")
ax.set_xlabel("temperature (K)")
ax.set_ylabel("-TS (eV/atom)")
ax.set_title("what a 0 K hull is missing")
ax.legend(fontsize=8)"""),

    ("markdown", """\
At room temperature the entropy term is tens of meV and usually ignorable. At a
synthesis temperature of 1500 K the five-component alloy gets **−0.21 eV/atom**
— four times the 50 meV that a screen typically calls "close to the hull".

That is the whole argument for high-entropy alloys: a material can sit well
above the convex hull at 0 K and still be the phase that forms, for no reason
other than that there are a lot of ways to arrange it.

```{warning}
This is an **upper bound**. Real mixing is never ideal, and short-range order —
which is the norm rather than the exception in concentrated alloys — reduces
the entropy below `k_B ln m`. Treating the ideal value as the true one
overstabilises exactly the materials it is used to justify.
```

## Making something a DFT code can read

`mv.disorder.orderings` enumerates specific arrangements and returns them as an
ordinary materials object — the same shape as `mv.mag.orderings` and
`mv.pp.defects`, because it is the same move: enumerate the configurations,
compute them all, let the object record which won."""),

    ("code", """\
ordered = mv.disorder.orderings(md, n=3)
mv.pp.describe(ordered)

ordered.obs[["parent", "ordering_index", "formula", "nsites",
             "ewald_energy"]]"""),

    ("code", """\
all(s.is_ordered for s in mv.structures(ordered))"""),

    ("markdown", """\
Every cell is now ordered, and the compositions survived: Cu₀.₅Au₀.₅ became
CuAu, not something else.

The already-ordered copper passed through as a single row rather than being
dropped, so every input material is still represented and nothing needs
re-joining by hand.

### The ranking is arbitrary here, and it says so"""),

    ("code", """\
ordered.uns["orderings"]"""),

    ("markdown", """\
Arrangements are ranked by **Ewald energy**, which needs oxidation states.
Without them every arrangement scores zero and the order is arbitrary — which
is honest for a metallic alloy, where electrostatics is not what decides
anything, and would be wrong for an oxide. Pass `use_oxidation=True` there.

The object records that rather than presenting an arbitrary order as a result.

```{note}
`n=3` returned three arrangements. There are far more: the count of distinct
arrangements grows combinatorially with cell size, and `uns['orderings']` says
this is a subset. A screen over "the orderings" of a concentrated alloy is
always a screen over a sample.
```

## When ordering is the wrong tool

Here is the trap. `orderings` ranks arrangements and gives you the best ones —
that is, the **ordered ground state**. A solid solution is the opposite of an
ordered ground state.

If you want a cell that behaves like a random alloy, you want a *special
quasirandom structure*: the small cell whose correlation functions best match a
genuinely random one."""),

    ("code", """\
try:
    mv.disorder.sqs(md, scaling=1, search_time=0.05)
except ImportError as exc:
    print(f"ImportError: {exc}")"""),

    ("markdown", """\
It needs ATAT's `mcsqs`, a separate Fortran program, and it refuses rather than
quietly handing back the ordered ground state under a different name. That
substitution would be a plausible-looking answer to a different question, which
is the worst kind of wrong.

## Doping

`mv.disorder.dope` is the dilute case: put a small amount of a dopant onto the
sites its charge and ionic radius allow, and compensate the charge. That is
different from `mv.gen.substitute`, which swaps *every* atom of an element."""),

    ("code", """\
oxide = mv.datasets.load("oxides")[:1].copy()
mv.pp.describe(oxide)
oxide.obs[["name", "formula", "nsites"]]"""),

    ("code", """\
try:
    doped = mv.disorder.dope(oxide, "Nb5+", min_length=5.0)
    print(doped.obs[["parent", "dopant", "formula"]])
except ValueError as exc:
    print(f"ValueError: {exc}")"""),

    ("markdown", """\
That message is the point of the function existing at all.

pymatgen's `DopingTransformation` enumerates internally with **enumlib**, a
separate Fortran program. When enumlib is missing it does not raise — it
returns an **empty list**, so a doping study runs to completion and produces
nothing, and the only symptom is a dataset with fewer rows than you expected.

matverse checks for the empty result and tells you which of the two things
happened: a missing install, or a dopant that genuinely fits nowhere.

## Where this sits

Disorder touches the rest of the library in one specific place: the hull.

```python
mv.disorder.describe(md, temperature=1500.0)
mv.thermo.hull(md, level='pbe')

# a candidate is really "on the hull at temperature" if
md.obs['e_above_hull_pbe'] + md.obs['entropy_term_300K'] <= 0
```

A screen that ignores `-TS` will reject exactly the high-entropy phases that
motivated the field.

## What it remembers"""),

    ("code", """\
for step in mv.provenance(ordered):
    print(step)"""),

    ("markdown", """\
```{seealso}
[Magnetic ordering](magnetic_ordering.ipynb) is the same enumerate-then-compute
shape applied to spin. [Screening, end to end](screening.ipynb) is the hull this
entropy term belongs in.
```"""),

    ("markdown", """\
## Did the disorder work?

Everything above produces a cell that is *supposed* to stand for a solid
solution. Nothing so far checks whether it does. The measure for that is the
Warren–Cowley short-range order parameter, one per ordered pair of elements:

$$\\alpha_{AB} = 1 - \\frac{P(B \\mid A)}{c_B}$$

where $P(B\\mid A)$ is the chance that a neighbour of an A atom is a B atom and
$c_B$ is B's overall fraction. **Zero is random.** Negative means A prefers B —
ordering. Positive means A avoids B, so A clusters with its own kind.

The structure that fixes the signs in your head is B2 brass, where every nearest
neighbour of a copper is a zinc:"""),

    ("code", """\
from pymatgen.core import Lattice, Structure

b2 = Structure(Lattice.cubic(2.95), ["Cu", "Zn"], [[0, 0, 0], [.5, .5, .5]])

# the same lattice, same composition, atoms thrown on at random
base = b2.copy()
base.make_supercell([4, 4, 4])
pick = np.random.RandomState(0).permutation(len(base))
solution = Structure(
    base.lattice,
    ["Cu" if pick[i] < len(base) // 2 else "Zn" for i in range(len(base))],
    base.frac_coords)

alloys = mv.data.from_structures([b2, solution])
alloys.obs_names = ["B2 CuZn", "random CuZn"]

mv.disorder.sro(alloys)
pd.DataFrame(alloys.obsm["sro_shell1"], index=alloys.obs_names,
             columns=alloys.uns["sro"]["shell1"]["pairs"]).round(3)"""),

    ("markdown", """\
**+1 for the like pairs and −1 for the unlike ones** on B2, which is what
perfect ordering means, and 0.03 across the board for the random arrangement —
zero to within the noise of a 128-site cell.

Two properties worth knowing are visible in that table. The matrix is symmetric
here because the two concentrations are equal, and each row satisfies
$\\sum_B c_B \\alpha_{AB} = 0$: the neighbours of a copper are *some* element,
whatever the ordering. That sum rule is the one identity that catches a
normalisation mistake, and it is checked in the test suite rather than left to
the eye.

The shell matters, and it is the second argument:"""),

    ("code", """\
for k in (1, 2, 3):
    mv.disorder.sro(alloys, shell=k, key_added=f"shell{k}")

pd.DataFrame(
    {f"shell {k}": alloys.obsm[f"sro_shell{k}"][0] for k in (1, 2, 3)},
    index=alloys.uns["sro"]["shell1"]["pairs"]).round(3)"""),

    ("markdown", """\
The sign **inverts** between the first shell and the second. In bcc the eight
nearest neighbours sit along $\\langle 111\\rangle$ and in B2 they are the other
species; the six second neighbours sit along $\\langle 100\\rangle$ and are the
same species. So $\\alpha_{\\mathrm{CuCu}}$ goes from $+1$ to $-1$. A short-range
order parameter quoted without its shell is half a number.

```{note}
Computed from the definition rather than through pymatgen's
`analysis.disorder`, whose `get_warren_cowley_parameters` returns the same value
for every pair — on this B2 cell it gives −1 for the like pairs as well as the
unlike ones. A wrapper around output that cannot be reproduced from the
definition is worse than the gap.
```

`obs['sro_rms_shell1']` is the single number to sort on: near zero for a good
solid solution, large for anything that has ordered or clustered."""),
]
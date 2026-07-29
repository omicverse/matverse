"""Cells for tutorials/defects_and_diffusion.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Defects and diffusion

A perfect crystal conducts nothing and ages not at all. Everything a material
does slowly — creep, sintering, ionic conduction, radiation damage, the
degradation of a battery cathode over a thousand cycles — happens because atoms
move, and atoms move through defects.

This tutorial builds a vacancy, computes what it costs to make and what it costs
to move, and ends at the quantity a defect chemist actually reports: formation
energy against Fermi level.

The migration barrier at the end comes out at **0.754 eV** against a literature
value of about 0.70 eV for vacancy migration in fcc copper, and the formation
energy at **1.314 eV** against roughly 1.3 eV measured by positron annihilation.
Both from a potential that took no download and no training set."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

Copper, at its measured room-temperature lattice parameter. One material is
enough: a defect calculation is about one host and the many ways to break it."""),

    ("code", """\
copper = mv.datasets.metals(["Cu"])
mv.pp.describe(copper)

copper.obs[["name", "formula", "nsites", "lattice_parameter"]]"""),

    ("markdown", """\
### Two ways to deform a cell before you break it

`mv.pp.supercell` and `mv.pp.strain` both deposit a **variant** rather than
replacing the input, so the original stays available. A supercell is what gives
a defect room not to interact with its own periodic image; a strained cell is
what elastic constants and deformation potentials are computed on."""),

    ("code", """\
mv.pp.supercell(copper, (2, 2, 2))
mv.pp.strain(copper, amount=0.02)

mv.variants(copper)"""),

    ("code", """\
[len(s) for s in mv.structures(copper, "supercell_2x2x2")], \
    round(mv.structures(copper, "strained")[0].lattice.a, 4)"""),

    ("markdown", """\
## Enumerating the defects

`mv.pp.defects` builds a supercell and removes one atom from each
**symmetrically inequivalent** site. In fcc copper every site is equivalent, so
a 2×2×2 supercell of the conventional cell — 32 atoms — yields exactly one
vacancy, not 32."""),

    ("code", """\
defects = mv.pp.defects(copper, supercell=(2, 2, 2), kinds=("vacancy",))

mv.pp.describe(defects)
defects.obs[["parent", "defect", "site", "removed", "nsites"]]"""),

    ("markdown", """\
```{note}
That deduplication is worth checking rather than trusting. An earlier version of
this function imported `SpacegroupAnalyzer` in the caller rather than the helper,
so the helper raised `NameError`, a bare `except` swallowed it, and the symmetry
reduction silently did nothing — 32 vacancies instead of 1, and 32× the compute.
The test suite now asserts the count.
```

The returned object is an ordinary AnnData whose rows are defective cells, with
`obs['parent']` pointing back at the host. Everything else in matverse works on
it unchanged.

## What a vacancy costs to make"""),

    ("code", """\
mv.calc.relax(copper, level="emt", fmax=0.02)
mv.calc.relax(defects, level="emt", fmax=0.05, steps=200, cell=False)

defects.obs[["defect", "energy_emt", "relax_converged_emt"]].round(4)"""),

    ("markdown", """\
```{note}
`cell=False` on the defective supercell, and it is not a detail. A 32-atom cell
with one vacancy is a stand-in for a *dilute* defect in an infinite crystal.
Letting its lattice relax lets the whole crystal contract around a vacancy
concentration of 1 in 32, which is not the quantity anyone wants; the host
lattice is what holds the defect open. So the host relaxes its cell and the
defect relaxes only its ions inside that cell.

Before v0.1.17 this line needed no argument, because `mv.calc.relax` never
moved a cell at all. It was right here for the wrong reason and wrong
everywhere else for the same one.
```"""),

    ("markdown", """\
### Two kinds that are not a site you already have

A vacancy or a substitution is a site you can point at. An **interstitial** has
to be *found* — it is a hole in the structure, and the Voronoi construction
locates them. An **antisite** is the cross product of the species present, which
for a quaternary is more combinations than anyone enumerates by hand.

Both go through `pymatgen-analysis-defects`, which also picks the supercell
itself — targeting a minimum image distance rather than a fixed multiple, so the
`supercell=` argument does not apply to these two."""),

    ("code", """\
cathode = mv.datasets.load("battery_cathodes")[:1].copy()
mv.pp.describe(cathode)

antisites = mv.pp.defects(cathode, kinds=("antisite",))
mv.pp.describe(antisites)
antisites.obs[["defect", "removed", "added", "nsites"]].head(6)"""),

    ("markdown", """\
Twenty-four antisites from a four-species cathode, each an ordered swap — Fe on
the Li site is a different defect from Li on the Fe site, and in LiFePO₄ the
Li/Fe antisite is *the* defect that blocks the one-dimensional lithium channel.

Interstitials need to know what to insert:"""),

    ("code", """\
interstitials = mv.pp.defects(cathode, kinds=("interstitial",),
                              interstitial_species=["Li"])
mv.pp.describe(interstitials)
interstitials.obs[["defect", "added", "nsites"]].head(4)"""),

    ("markdown", """\
One more atom than the host supercell, which is what an interstitial is.

```{note}
Ask for a supercell window nothing can satisfy and it says so — the generator
targets a minimum image distance as well as an atom count, so a small
`max_atoms` can leave no legal cell. It used to report "no defect was
generated", which blames the chemistry for an argument problem.
```"""),

    ("markdown", """\
The raw energy of a cell with one atom missing is not a formation energy. A
defect changes the number of atoms, so what the missing atom is *worth* has to
come from somewhere else — and it cannot be derived from the defective cell
alone.

That "somewhere else" is the chemical potential, and it is a choice about
synthesis conditions rather than a constant. `mv.thermo.chempot_limits` reports
the range the phase diagram allows."""),

    ("code", """\
mu_cu = float(copper.obs["energy_per_atom_emt"].iloc[0])
n_host = int(copper.obs["nsites"].iloc[0]) * 8
e_host = float(copper.obs["energy_per_atom_emt"].iloc[0]) * n_host
e_defect = float(defects.obs["energy_emt"].iloc[0])

formation = e_defect - e_host + mu_cu
round(formation, 3)"""),

    ("markdown", """\
**1.314 eV**, against roughly 1.3 eV measured for copper by positron
annihilation. That is the number that sets the equilibrium vacancy concentration
through `exp(-E_f / kT)`, and landing within a couple of percent of it from
effective-medium theory is better than this calculation deserves — EMT was
fitted to exactly this kind of metallic environment.

It was 1.25 eV until v0.1.17, when `mv.calc.relax` began relaxing the host's
lattice. The host now sits at EMT's own equilibrium volume instead of copper's
measured room-temperature one, and the defect is held in that lattice — which is
the combination the formula assumes.

## Formation energy against Fermi level

For a semiconductor or insulator the vacancy can also be charged, and then the
formation energy depends on where the electron came from. That makes it a
**line** in the Fermi level, not a number — the standard defect-chemistry plot,
and the reason `mv.thermo.defect_formation` deposits a curve."""),

    ("code", """\
mv.thermo.defect_formation(defects, host=copper, level="emt",
                           chempot={"Cu": mu_cu}, band_gap=1.5,
                           charges=(-2, -1, 0, 1, 2))

defects.obsm["formation_vs_fermi_emt"].shape, mv.grid_of(
    defects, "formation_vs_fermi").shape"""),

    ("code", """\
import matplotlib.pyplot as plt

fermi = mv.grid_of(defects, "formation_vs_fermi")
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(fermi, defects.obsm["formation_vs_fermi_emt"][0], linewidth=1.6)
ax.set_xlabel("Fermi level above VBM (eV)")
ax.set_ylabel("formation energy (eV)")
ax.set_title("the lower envelope over charge states")"""),

    ("code", """\
defects.obs[["defect", "defect_formation_energy_emt",
             "stable_charge_emt"]].round(3)"""),

    ("markdown", """\
Each straight segment is one charge state, and its slope is the charge; the
kinks are the transition levels where the stable charge changes. `stable_charge`
records which state wins where.

The reported 1.117 eV is *not* the 1.314 eV computed by hand above, and the
difference is the point: 1.314 eV is the neutral vacancy, while this is the lower
envelope over all five charge states evaluated at the Fermi level. They coincide
only where the neutral state is the stable one.

```{warning}
Copper is a metal, so a band gap of 1.5 eV is fiction and the charged states are
meaningless here — this is the machinery, exercised on the only material EMT can
run. Use it on the oxides and halides where a Fermi level means something.

`mv.thermo.defect_formation` **warns and returns NaN** when no chemical
potential is given, rather than assuming one. A defect creates or destroys
atoms, and what they cost is not derivable from the defective cell.
```

## What it costs to move

Formation says how many vacancies there are. Migration says how fast they move,
and the two together give the diffusion coefficient.

`mv.neb.hop_endpoints` builds the two ends of a hop: a vacancy, and the same
cell with a neighbour moved into it."""),

    ("code", """\
mv.neb.hop_endpoints(copper, species="Cu", supercell=(2, 2, 2))

copper.obs[["hop_distance", "hop_species"]].round(4)"""),

    ("markdown", """\
2.556 Å is the nearest-neighbour distance in copper, `a/√2`, which is what a
vacancy hop in fcc actually is.

```{warning}
Getting that number right is the whole calculation. The destination has to be the
nearest periodic **image** of the vacancy, not the coordinates as stored. Using
the stored ones sent the atom the long way round the cell — 7.66 Å instead of
2.55 — and the NEB then measured the cost of dragging an atom straight through
its neighbours, giving barriers of 5–7 eV. It looked like a physics result.

`hop_endpoints` now checks that the distance travelled matches the distance
reported and raises if they disagree.
```

Both endpoints need relaxing, and they need relaxing **separately** — a NEB needs
two distinct minima."""),

    ("code", """\
mv.calc.relax(copper, level="emt", source="hop_initial", key_added="start",
              fmax=0.05, steps=80, cell=False)
mv.calc.relax(copper, level="emt", source="hop_final", key_added="end",
              fmax=0.05, steps=80, cell=False)

mv.variants(copper)"""),

    ("markdown", """\
`key_added` matters here. Without it the second relaxation would overwrite the
first and the band would run from a structure to itself.

`cell=False` again, and for a second reason on top of the dilute-defect one: a
band is interpolated between two endpoints, and endpoints with different
lattices are not two points on one path.

## The nudged elastic band"""),

    ("code", """\
mv.neb.barrier(copper, initial="start", final="end", level="emt",
               n_images=7, fmax=0.05, steps=200)

copper.obs[["barrier_emt", "barrier_reverse_emt", "reaction_energy_emt",
            "neb_converged_emt"]].round(4)"""),

    ("markdown", """\
**0.754 eV against a literature value of about 0.70 eV.** For a potential that
took no download and no training set, that is a genuinely useful number.

```{note}
The band uses the **improved-tangent** formulation of Henkelman and Jónsson
(2000), not ASE's default. ASE's own warning describes that default as "an
unpublished, custom implementation ... not recommended as it frequently results
in very poor bands", which is not something to leave on and hope for.
```

Two internal checks come free with a symmetric hop: forward and reverse barriers
must agree, and the reaction energy must be zero. Both hold to 0.02 eV. If they
did not, the band had not converged and the barrier would be an artefact —
worth reading every time, because a NEB that has not converged still reports a
number."""),

    ("code", """\
ax = mv.pl.spectra(copper, "neb_profile", levels=("emt",), rows=[0])
ax.set_title("the minimum energy path")
ax.set_ylabel("energy relative to start (eV)")"""),

    ("markdown", """\
The path rises to a single saddle and comes back down, which is what a
single-hop mechanism looks like. A profile with two humps means an intermediate
minimum and a mechanism you have not modelled.

```{note}
The default interpolation is **IDPP**, not linear. Linear interpolation moves
every atom along a straight line in Cartesian space, which drives the hopping
atom directly through its neighbours; the band then has to climb out of that,
and often does not. IDPP interpolates interatomic distances instead.
```

## From a barrier to a diffusion coefficient

The two numbers combine in the classical way — a jump attempt at the Debye
frequency, times the probability of clearing the barrier, times the probability
that a neighbouring site is vacant."""),

    ("code", """\
kB = 8.617333e-5                       # eV/K
nu = 1e13                              # attempt frequency, s^-1
a = float(copper.obs["hop_distance"].iloc[0]) * 1e-10     # m
barrier = float(copper.obs["barrier_emt"].iloc[0])

T = np.array([300.0, 500.0, 800.0, 1000.0, 1300.0])
c_vac = np.exp(-formation / (kB * T))
D = a ** 2 * nu * c_vac * np.exp(-barrier / (kB * T))

pd.DataFrame({"T (K)": T, "vacancy fraction": c_vac, "D (m^2/s)": D})"""),

    ("code", """\
fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.semilogy(1000 / T, D, "o-")
ax.set_xlabel("1000 / T (1/K)")
ax.set_ylabel("D (m²/s)")
ax.set_title("Arrhenius: the slope is formation + migration")"""),

    ("markdown", """\
A straight line on this plot is the definition of Arrhenius behaviour, and its
slope is the **sum** of the two energies — which is why a measured activation
energy alone cannot tell you whether a material diffuses quickly because
vacancies are cheap or because they are mobile. Computing them separately can.

At 1300 K, just under copper's melting point of 1358 K, this gives 1×10⁻¹⁴ m²/s
against a measured self-diffusion coefficient of order 10⁻¹³ — within an order
of magnitude, from two energies each good to about 10%, which is what
exponentiating small errors does to you.

At room temperature it is 10⁻⁴⁰ m²/s: twenty-six orders of magnitude slower, and
the reason a copper wire does not visibly age. The vacancy fraction alone falls
from 10⁻⁵ to 10⁻²¹ over that range."""),

    ("markdown", """\
## Before the barrier: can the ion get out at all?

A barrier is the cost of **one** hop. It is silent on whether that hop repeated
ever carries an ion across the crystal. A material can have a beautifully low
barrier between two sites that form a closed pair and conduct precisely nothing,
and no amount of NEB will tell you so — the calculation you ran was about those
two sites.

`mv.neb.percolation` asks the other question, and asks it from geometry alone,
so it costs nothing and can run first. Starting from one mobile site, which
periodic *images of that same site* can be reached by hops no longer than the
cutoff? Reaching a different site one cell over only says two sites are
connected. Reaching your own site in the next cell says the network repeats —
the ion can keep going.

The two canonical lithium cathodes make the point:"""),

    ("code", """\
from pymatgen.core import Lattice, Structure

# Fd-3m in pymatgen is origin choice 1, so the tetrahedral 8a site is at the
# origin, not at (1/8, 1/8, 1/8). Put Li at (1/8, 1/8, 1/8) here and you
# silently build Li2MnO4 on a 16-fold site instead of spinel.
spinel = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(8.24), ["Li", "Mn", "O"],
    [[0, 0, 0], [0.625] * 3, [0.3875] * 3])
layered = Structure.from_spacegroup(
    "R-3m", Lattice.hexagonal(2.82, 14.05), ["Li", "Co", "O"],
    [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.2395]])

cathodes = mv.data.from_structures([spinel, layered])
cathodes.obs_names = ["LiMn2O4", "LiCoO2"]
cathodes.obs["formula"] = [s.composition.reduced_formula
                           for s in mv.structures(cathodes, "input")]

mv.neb.percolation(cathodes, species="Li", cutoff=3.6)
cathodes.obs[["formula", "percolation_sites_Li",
              "percolation_dimensionality_Li",
              "percolation_threshold_Li"]].round(3)"""),

    ("markdown", """\
**3 against 2**, which is the textbook difference between the two: spinel
conducts lithium in three dimensions through a network of corner-sharing
tetrahedral sites, and a layered oxide conducts it only within the planes
between the CoO₂ slabs. Both percolate; only one of them percolates everywhere.

The thresholds are exact geometry — 3.568 Å is a·√3/4, the nearest-neighbour
distance on spinel's diamond-like 8a sublattice, and 2.820 Å is simply the
hexagonal *a* of the layered cell. That is the bottleneck hop: the shortest
length at which anything percolates at all, and the one the ion cannot avoid.

Now the part worth being careful about. Dimensionality is a property of the
network **at a hop length**, not a property of the material:"""),

    ("code", """\
for cutoff in (2.5, 3.0, 3.6, 5.0):
    mv.neb.percolation(cathodes, species="Li", cutoff=cutoff,
                       key_added=f"at_{cutoff}")

cathodes.obs[["formula"] + [f"percolation_dimensionality_at_{c}"
                            for c in (2.5, 3.0, 3.6, 5.0)]]"""),

    ("markdown", """\
Allow a 5 Å hop and the layered oxide becomes three-dimensional too, because
that is long enough to jump the slab. The answer did not change; the question
did. Quoting a dimensionality without the hop length it was measured at is
quoting half a result — which is why the cutoff is recorded in
`uns['percolation']` alongside it.

Read this as a filter, not a verdict. A percolating network is a **necessary**
condition for conduction and never a sufficient one: it says the geometry does
not forbid transport, and says nothing about what it costs. That is what the
barrier above is for. Run percolation on a thousand candidates in seconds, then
spend the NEB budget on the ones that survive."""),

    ("markdown", """\
## What the object remembers"""),

    ("code", """\
for step in mv.provenance(copper):
    print(step)"""),

    ("markdown", """\
```{seealso}
[Surfaces and adsorption](surfaces_and_adsorption.ipynb) does the same thing for
the other place atoms move — the surface. [Dynamics](dynamics.ipynb) reaches
diffusion the other way, by simply watching atoms move.
```"""),
]

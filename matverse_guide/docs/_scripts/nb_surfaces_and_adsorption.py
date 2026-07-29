"""Cells for tutorials/surfaces_and_adsorption.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Surfaces and adsorption

Catalysis, corrosion, crystal growth and sintering all happen at the surface,
and none of them can be predicted from the bulk. A surface calculation asks two
things: which faces a crystal chooses to expose, and what sticks to them.

Both are answered here on copper, with the calculator that ships with matverse.
The surface energies come out in the literature ordering — (111) < (100) <
(110) — and oxygen binds most strongly in the hollow site, which is the standard
result for a close-packed metal surface."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

Copper and platinum, at their measured lattice parameters. Two metals is enough
to show that the answer is a property of the material rather than of the
method."""),

    ("code", """\
bulk = mv.datasets.metals(["Cu", "Pt"])
mv.pp.describe(bulk)
mv.calc.energy(bulk, level="emt")

bulk.obs[["name", "lattice_parameter", "energy_per_atom_emt"]].round(4)"""),

    ("markdown", """\
The bulk energy is not optional bookkeeping — a surface energy is the cost of
the atoms in a slab *relative to the same atoms in the bulk*, so without it
there is nothing to subtract. `mv.surf.surface_energy` refuses rather than
guessing.

## Cutting slabs

`mv.surf.slabs` enumerates the symmetrically distinct terminations of every
Miller index up to `max_index`, and returns them as a **new object** whose rows
are facets."""),

    ("code", """\
facets = mv.surf.slabs(bulk, max_index=1, min_slab=8.0, min_vacuum=10.0)
facets"""),

    ("code", """\
mv.pp.describe(facets)
facets.obs[["parent", "miller", "termination", "slab_area", "nsites",
            "is_polar"]].head(10).round(3)"""),

    ("markdown", """\
More rows out than in, with `obs['parent']` pointing back at the bulk material —
the same shape as `mv.pp.defects`. A facet is not a material, so it does not
belong in the materials object; it is its own axis with its own rows, and
`parent` is the join.

`min_slab` and `min_vacuum` are the two convergence parameters that matter. Too
thin a slab and the two surfaces feel each other; too little vacuum and the slab
feels its own periodic image. Both are recorded."""),

    ("code", """\
facets.uns["slabs"]"""),

    ("markdown", """\
## Surface energies"""),

    ("code", """\
mv.calc.energy(facets, level="emt")
mv.surf.surface_energy(facets, bulk=bulk, level="emt")

frame = facets.obs[facets.obs["parent"] == "0"]
frame.groupby("miller", observed=True)["surface_energy_emt"].min().round(3)"""),

    ("markdown", """\
**(111) < (100) < (110)**, which is the literature ordering for copper and for
fcc metals generally. The close-packed face has the fewest broken bonds per unit
area, so it costs the least to make.

EMT gets the magnitude wrong by roughly a factor of two against the measured
~1.8 J/m² for Cu(111) — but it gets the ordering right, and the ordering is what
decides crystal shape."""),

    ("code", """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for name, row in zip(bulk.obs["name"], bulk.obs_names):
    part = facets.obs[facets.obs["parent"] == row]
    by_facet = part.groupby("miller", observed=True)[
        "surface_energy_emt"].min()
    ax.plot(by_facet.index, by_facet.to_numpy(), "o-", label=name)
ax.set_ylabel("surface energy (J/m²)")
ax.set_xlabel("Miller index")
ax.set_title("close-packed is cheapest, for both metals")
ax.legend()"""),

    ("markdown", """\
## When a slab is not the bulk formula

Everything above rests on an assumption worth making explicit: that the slab is
the bulk formula times a whole number. Subtracting `N × e_bulk` only removes the
bulk cost if that holds. It does for an elemental metal however you cut it, and
it does for most default cuts of a compound, because `SlabGenerator` chooses
shifts that keep the composition.

It stops holding the moment you ask for symmetric slabs. `symmetrize=True`
deletes sites until the two faces are equivalent, and on an ordered alloy that
routinely leaves the slab off-stoichiometry."""),

    ("code", """\
from pymatgen.core import Lattice, Structure

cu3au = mv.data.from_structures([Structure.from_spacegroup(
    "Pm-3m", Lattice.cubic(3.75), ["Au", "Cu"], [[0, 0, 0], [0.5, 0.5, 0]])])
mv.calc.energy(cu3au, level="emt")

alloy_facets = mv.surf.slabs(cu3au, max_index=1, min_slab=8.0,
                             min_vacuum=10.0, symmetrize=True)
mv.calc.energy(alloy_facets, level="emt")

alloy_facets.obs["formula"] = [s.composition.formula
                               for s in mv.structures(alloy_facets, "input")]
alloy_facets.obs[["miller", "formula"]]"""),

    ("markdown", """\
Four of these five are not Cu₃Au. The leftover atoms have to come from somewhere
or go somewhere, and their energy depends on that reservoir — so there is no
single surface energy to report. `mv.surf.surface_energy` says so rather than
returning a number that looks like one:"""),

    ("code", """\
mv.surf.surface_energy(alloy_facets, bulk=cu3au, level="emt")

alloy_facets.obs[["miller", "formula", "surface_energy_emt",
                  "surface_energy_emt_off_stoichiometry"]].round(4)"""),

    ("markdown", """\
### Referencing the surplus to a reservoir

What those slabs have is not a surface energy but a line:

$$\\gamma(\\Delta\\mu) = \\gamma_0 - \\Gamma\\,\\Delta\\mu$$

where $\\Gamma$ is the surface excess of the element in surplus and
$\\Delta\\mu$ is how far its reservoir sits below the pure element.
`mv.surf.surface_energy_chempot` needs elemental structures to measure that
against, at the same level of theory as everything else."""),

    ("code", """\
fcc = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
reservoirs = mv.data.from_structures(
    [Structure(Lattice.cubic(a), [e] * 4, fcc)
     for e, a in (("Cu", 3.61), ("Au", 4.08))])
mv.calc.energy(reservoirs, level="emt")

mv.surf.surface_energy_chempot(alloy_facets, bulk=cu3au, refs=reservoirs,
                               level="emt", key_added="gamma_au_rich")
mv.surf.surface_energy_chempot(alloy_facets, bulk=cu3au, refs=reservoirs,
                               level="emt", chempot={"Au": -0.5},
                               key_added="gamma_au_poor")

alloy_facets.obs[["miller", "formula", "surface_excess_Au_emt",
                  "gamma_au_rich", "gamma_au_poor"]].round(4)"""),

    ("markdown", """\
Two things to read off. The two (100) terminations carry equal and opposite Au
excess — one face is Au-rich exactly as much as the other is Au-poor, which is
what cutting one plane two ways has to mean.

And the ordering **inverts**. In the Au-rich limit the Au-rich termination is the
cheaper one; pull the gold reservoir down by half an electronvolt and the Au-poor
termination wins instead. Which facet a crystal of this alloy shows is not a fact
about the alloy — it is a fact about the atmosphere you grew it in. A Wulff shape
built from one of these columns is a shape at one chemical potential, and saying
which one is part of the result.

The stoichiometric (111) slab gets the same number from both routes, which is the
consistency check worth doing whenever two methods overlap."""),

    ("code", """\
stoich = alloy_facets.obs["miller"] == "1_1_1"
alloy_facets.obs.loc[stoich, ["surface_energy_emt", "gamma_au_rich",
                              "gamma_au_poor"]].round(6)"""),

    ("markdown", """\
## The equilibrium crystal shape

A crystal at equilibrium minimises total surface energy at fixed volume, and the
construction that solves that is Wulff's: each facet sits at a distance from the
centre proportional to its surface energy, and the cheap faces grow large while
the expensive ones shrink out of existence."""),

    ("code", """\
mv.surf.wulff(facets, bulk=bulk, level="emt")

bulk.obs[["name", "wulff_effective_radius_emt",
          "wulff_shape_factor_emt"]].round(4)"""),

    ("code", """\
facets.obs[facets.obs["parent"] == "0"][
    ["miller", "surface_energy_emt", "wulff_area_fraction_emt"]].round(4)"""),

    ("markdown", """\
(111) takes **two thirds** of the surface, (100) most of the rest, and (110)
barely 5% — a 19% spread in surface energy turns into a 12× spread in area,
because the construction is a competition and small differences compound.

That is a prediction about particle shape you can check under an electron
microscope, and it is why fcc metal nanoparticles are cuboctahedra: (111) faces
truncated by (100), with (110) surviving only as narrow edges.

The anisotropy — 0.053 here — is how far the shape departs from a sphere. Copper
is a nearly isotropic metal, which is why the particles are round-ish
polyhedra rather than sharp plates. A layered material would give a number
orders of magnitude larger, and flakes."""),

    ("code", """\
bulk.uns["wulff"]["emt"]["0"]"""),

    ("code", """\
fig, ax = plt.subplots(figsize=(6, 3.6))
part = facets.obs[facets.obs["parent"] == "0"]
grouped = part.groupby("miller", observed=True)[
    "wulff_area_fraction_emt"].max()
ax.bar(grouped.index, grouped.to_numpy(), color="#4c72b0")
ax.set_ylabel("fraction of the surface")
ax.set_title("what a copper particle actually shows the world")"""),

    ("markdown", """\
## Adsorption

The other half of a surface calculation: what sticks to it, where, and how
strongly. Start from a single clean facet — Cu(111), the one the Wulff
construction says dominates."""),

    ("code", """\
copper = mv.datasets.metals(["Cu"])
mv.pp.describe(copper)
mv.calc.energy(copper, level="emt")

clean = mv.surf.slabs(copper, miller=(1, 1, 1), min_slab=8.0, min_vacuum=12.0)
mv.calc.energy(clean, level="emt")

mv.pp.describe(clean)
clean.obs[["miller", "termination", "slab_area", "nsites",
           "energy_emt"]].round(4)"""),

    ("markdown", """\
### Sites are enumerated, not guessed

Which site binds most strongly is the question, so several are placed and all of
them are relaxed. This is the AdsorbML protocol, and it exists because the site
you would have guessed is often not the one that wins."""),

    ("code", """\
configs = mv.surf.adsorption_sites(clean, "O", height=1.8)

configs.obs[["parent", "site_kind", "site_index"]].head(10)"""),

    ("code", """\
configs.obs["site_kind"].value_counts()"""),

    ("markdown", """\
On-top, bridge and hollow — the three coordination environments a close-packed
surface offers. Relax them all and see which one oxygen prefers."""),

    ("code", """\
mv.calc.relax(configs, level="emt", fmax=0.1, steps=60)
mv.surf.adsorption_energy(configs, clean=clean, reference=0.0, level="emt")

configs.obs.groupby("site_kind", observed=True)[
    "adsorption_energy_emt"].min().round(4)"""),

    ("markdown", """\
**Hollow is lowest, then bridge, then on-top** — more neighbours, more bonding.
That is the standard result for oxygen on a close-packed metal, and it is why
the site has to be enumerated rather than assumed: on-top is the site that looks
obvious in a drawing and it is the weakest of the three.

All three values are *positive*, which does not mean nothing binds. It means
`reference=0.0` put the free oxygen atom at zero energy, and EMT's isolated
oxygen atom is not at zero. Only the **differences** between sites are
meaningful here, so plot them that way."""),

    ("code", """\
by_site = configs.obs.groupby("site_kind", observed=True)[
    "adsorption_energy_emt"].min()
order = [k for k in ("ontop", "bridge", "hollow") if k in by_site.index]
relative = by_site[order] - by_site[order].max()

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.bar(order, relative.to_numpy(), color="#c1121f")
ax.set_ylabel("energy relative to on-top (eV)")
ax.set_title("more coordination, stronger binding")
ax.axhline(0, linewidth=0.8, color="#333")"""),

    ("markdown", """\
About 170 meV between on-top and hollow. That is the size of difference that
decides a reaction mechanism, and it is why the enumeration is not optional."""),

    ("markdown", """\
```{warning}
`reference=0.0` above sets the energy zero for the free adsorbate, and it is a
**choice you have to make and state**. Oxygen adsorption is conventionally
referenced to ½O₂, not to an isolated O atom, and the two differ by half the
dissociation energy — about 2.6 eV. A binding energy quoted without its
reference is not a number anyone else can use.

EMT also has no meaningful description of a free oxygen molecule, so the
absolute values here are not comparable with DFT. The *ordering* between sites
is what this example is for.
```

## What the objects remember

Two objects, two histories. The facets carry theirs and the bulk carries its
own, which is what keeps "which slab, cut how, from which bulk" answerable."""),

    ("code", """\
for step in mv.provenance(facets):
    print(step)"""),

    ("markdown", """\
```{seealso}
[Defects and diffusion](defects_and_diffusion.ipynb) covers the other place
atoms move. [Dynamics](dynamics.ipynb) puts the whole thing at temperature.
```"""),
]

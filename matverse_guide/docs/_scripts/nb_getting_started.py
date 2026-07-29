"""Cells for tutorials/getting_started.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Screening real materials with matverse

Materials screening is the problem of narrowing a large candidate list to a
small one you can afford to study properly. The narrowing is cheap and
approximate; the studying is expensive and exact; and almost all the difficulty
lies in keeping track of which is which.

matverse holds a screen in a single [AnnData](https://anndata.readthedocs.io/)
object. Structures, descriptors, computed properties and the record of what
produced them stay together, so a filtered dataset cannot silently point at the
wrong structure and a surrogate energy cannot be quietly averaged with a
first-principles one.

This notebook runs a screen end to end on published structures. It needs no
network, no API key and no downloaded model."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

`mv.datasets` ships real structures rather than cells built in code. The
crystallographic data comes from the set [pymatgen](https://pymatgen.org/)
distributes, so nothing is downloaded and the provenance is a package you
already have installed.

We use the olivine and NASICON cathodes:

- **LiFePO₄** — the olivine cathode of [Padhi et al.
  (1997)](https://doi.org/10.1149/1.1837571), the paper that made
  phospho-olivines a cathode class rather than a curiosity
- **NaFePO₄** — the sodium analogue of the same framework, the reason
  sodium-ion work keeps returning to olivines (Moreau et al., *Chem. Mater.*
  2010)
- **Li₃V₂(PO₄)₃** — a NASICON-framework cathode, a different way of building
  open channels out of phosphate polyanions (Yin et al., *J. Am. Chem. Soc.*
  2003)

Three materials is not a screen. It is enough to see what the object does with
them, which is what this page is for."""),

    ("code", """\
mv.datasets.available()"""),

    ("code", """\
md = mv.datasets.load("battery_cathodes")
md"""),

    ("markdown", """\
`X` is the composition matrix and `var` is the periodic table. Neither was asked
for: composition is intrinsic to a material rather than derived from it, so both
are built at construction.

Materials × elements is the same shape as cells × genes — sparse, non-negative,
mostly zero — which is what lets the ordination and enrichment machinery apply
to chemical space without being rewritten."""),

    ("code", """\
pd.DataFrame(md.X.toarray(), index=md.obs["name"], columns=md.var_names)"""),

    ("code", """\
md.var[["Z", "electronegativity", "period", "is_transition_metal"]]"""),

    ("markdown", """\
Because `var` is the periodic table, the natural display of anything
element-indexed is the periodic table itself — here, how many of the three
cathodes each element appears in."""),

    ("code", """\
md.var["n_materials"] = (md.X > 0).sum(axis=0).A1

ax = mv.pl.periodic_table(md, color="n_materials", label="materials containing")
ax.set_title("the chemistry of this library")"""),

    ("markdown", """\
And you can just look at one, which catches the mistakes a number will not
show you — a slab built upside down, a cell that came out of a parser inside
out."""),

    ("code", """\
ax = mv.pl.structure(md, "LiFePO4", backend="matplotlib")
ax.set_title("LiFePO$_4$, projected along its thinnest axis")"""),

    ("markdown", """\
With `py3Dmol` installed the default backend is an interactive viewer instead.
Neither replaces VESTA for real inspection — this is the quick look you take
twenty times a day.

Counts come from the **reduced** formula, so a supercell and its primitive cell
occupy the same point in chemical space. Cell size lives in `obs['nsites']`.

## Describing and screening the structures

`mv.pp` is the preprocessing namespace, shaped after `scanpy.pp` because the
operations correspond: `qc` is `calculate_qc_metrics`, `filter_materials` is
`filter_cells`."""),

    ("code", """\
mv.pp.describe(md)
mv.pp.qc(md)

md.obs[["name", "spacegroup", "formula", "nsites", "density", "is_valid"]]"""),

    ("code", """\
mv.pp.prototype(md)
md.obs[["name", "spacegroup", "prototype_mineral", "strukturbericht"]]"""),

    ("code", """\
mv.pp.symmetry(md)
md.obs[["name", "crystal_system", "point_group", "n_wyckoff", "wyckoff"]]"""),

    ("code", """\
mv.prop.frontier_orbitals(md)
md.obs[["name", "homo_element", "homo_orbital", "lumo_element",
        "lumo_orbital", "likely_metal"]]"""),

    ("markdown", """\
The cheapest possible statement about electronic structure: line up the atomic
orbital energies of the elements present, and see which sits highest occupied
and which lowest unoccupied. No calculator, no structure — only the composition.

What it tells you is **which element controls each band edge**, and that is a
real design handle. On SrTiO₃ it says O 2p and Ti 3d, the textbook perovskite
answer, and it says exactly the same for BaTiO₃ — which is why substituting on
the A site of a perovskite barely moves the gap and substituting on the B site
moves it a lot.

```{warning}
`orbital_gap_estimate` is a **difference of atomic orbital energies**, not a
band gap. No hybridisation, no crystal field, no Madelung potential, no
structure — two polymorphs get the same answer, and NaCl comes out near 6 eV
against a measured 8.5. Use it to sort a list before computing, never as a
number.

`likely_metal` is the more trustworthy column: "the frontier orbitals overlap"
survives a lot of approximation.
```"""),

    ("markdown", """\
`spacegroup` gives the symbol. This gives what the symbol *implies* — and
`wyckoff` is the part worth reading.

Wyckoff positions say how many **distinct** sites a structure has, and that
number decides how much work everything downstream is: how many vacancies
`mv.pp.defects` will enumerate, how many NMR environments to expect, how many
independent parameters a refinement has. LiFePO₄ has eight, which is why the
defect tutorial gets six inequivalent vacancies out of a 28-atom cell rather
than twenty-eight.

Two structures in the same space group with different Wyckoff sets are
different structures."""),

    ("markdown", """\
A space group says which symmetries a structure has. A **prototype** says which
structure it *is* — and the two are not the same question: Fm-3m covers
rocksalt, face-centred cubic and half-Heusler alike, so a screen that groups by
space group puts them in one bin.

This is also what makes a generative model's output legible. "Novel composition
in a known prototype" and "novel prototype" are different claims, and the second
is rare.

An unmatched structure gets an empty string rather than a guess — the AFLOW
library is large but finite, and "not in it" is worth keeping distinct from a
wrong label."""),

    ("markdown", """\
`spacegroup` is determined from the deposited coordinates, not copied from a
label — which is why LiFePO₄ comes out as P2₁/c rather than the Pnma the olivine
is usually reported in. The framework is the same; the cell in this file is a
lower-symmetry setting of it. `mv.pp.standardize` puts both into a conventional
cell when that matters.

That is the point of shipping published structures rather than a cubic cell
someone typed in: the awkwardness is real and you would meet it anyway.

## Simulating diffraction patterns

`mv.prop.xrd` needs no calculator, so it is the first real property available for
any structure. Peaks are broadened onto a shared 2θ grid, because a peak list
cannot be compared across materials — no two share peak positions."""),

    ("code", """\
mv.prop.xrd(md, two_theta=(10, 60), step=0.02)

md.obsm["xrd_calc"].shape"""),

    ("code", """\
ax = mv.pl.spectra(md, "xrd", rows=[0, 1, 2], offset=110)
ax.set_title("simulated powder patterns, Cu Kα")"""),

    ("markdown", """\
LiFePO₄ and NaFePO₄ are the same framework with a different alkali, and the
patterns show it: the same families of reflections, shifted.

## Identifying a phase

Feed one pattern back as though it had been measured, and ask which candidate it
is. This is phase identification against the library you already have."""),

    ("code", """\
measured = md.obsm["xrd_calc"][1]
mv.exp.match_xrd(md, measured, mv.grid_of(md, "xrd"))

md.obs[["name", "xrd_match", "xrd_match_rank"]].sort_values("xrd_match_rank")"""),

    ("markdown", """\
Row 1 comes back at 1.00 and the runner-up at 0.06, which is the margin you want
— NaFePO₄ and LiFePO₄ share a framework, so a scoring function that could not
tell them apart would be no use on the cases that matter.

```{warning}
`match_xrd` scores against the candidates in **this object** and nothing else,
and records that in `uns['xrd_match']['scored_against']`. A high score means
"the best of what you gave it", not "identified" — the true phase can be absent
from your library entirely.
```

## Choosing a calculator

Energies need a calculator, and which one is part of the result rather than an
implementation detail. `mv.calc.available()` reports what this installation can
run."""),

    ("code", """\
mv.calc.available()["emt"]"""),

    ("markdown", """\
ASE's effective-medium theory is the only calculator matverse ships working. It
is parameterised for Al, Cu, Ag, Au, Ni, Pd, Pt, H, C, N and O — which excludes
the Fe, P and V in the cathodes above.

That is a real constraint, not a tutorial convenience, so the screen switches to
materials EMT can run. For anything else, register a machine-learned potential:

```python
from mace.calculators import mace_mp

mv.calc.register_calculator("mace-mpa", lambda: mace_mp(model="medium-mpa-0"),
                            kind="mlip", method="MACE-MPA-0",
                            reference="PBE+U", license="MIT")
```"""),

    ("code", """\
metals = mv.datasets.metals()
metals.obs[["name", "lattice_parameter"]]"""),

    ("markdown", """\
Published room-temperature lattice parameters for the seven face-centred cubic
metals. Relax them:"""),

    ("code", """\
mv.pp.describe(metals)
mv.calc.relax(metals, level="emt", fmax=0.02)

metals.obs[["name", "energy_per_atom_emt", "relax_converged_emt"]]"""),

    ("code", """\
mv.variants(metals)"""),

    ("markdown", """\
The relaxed geometry becomes its own **variant** rather than replacing the
input, so "which structure was this energy computed on" stays answerable from
the object alone.

## Computing properties

Elastic constants by finite strain, phonons by frozen displacement, and thermal
conductivity from both."""),

    ("code", """\
mv.prop.elastic(metals, level="emt", source="relaxed_emt")
mv.prop.phonon(metals, level="emt", source="relaxed_emt", supercell=(1, 1, 1))
mv.prop.thermal_conductivity(metals, level="emt")

metals.obs[["name", "bulk_modulus_emt", "debye_temperature_emt",
            "thermal_conductivity_emt", "dynamically_stable_emt"]].round(1)"""),

    ("markdown", """\
Worth comparing against measured values rather than accepting:

| | EMT | experiment |
|---|---|---|
| bulk modulus, Cu | 135 | 140 GPa |
| bulk modulus, Ni | 176 | 180 GPa |
| bulk modulus, Ag | 100 | 101 GPa |
| bulk modulus, Al | 40 | 76 GPa |
| Debye temperature, Au | 176 | 165 K |
| Debye temperature, Ag | 254 | 225 K |
| Debye temperature, Al | 420 | 428 K |

Good for the transition metals EMT was fitted to, and a factor of two out on
aluminium's bulk modulus. That is the shape of the error a cheap potential
makes, and the reason a screen ranked on it is a shortlist rather than an
answer.

```{note}
Before v0.1.17 the copper entry read 123 GPa and the nickel entry 157. The
difference is not a better potential: `mv.calc.relax` moved only the atomic
positions, and the forces on an fcc metal vanish by symmetry, so nothing moved
at all and every one of these properties was computed at the published
room-temperature lattice parameter rather than at EMT's own minimum. Relaxing
the cell moved five of the seven metals closer to experiment. Aluminium did not
move much, because EMT's aluminium is wrong for a different reason.
```

The conductivities look far too small next to the 400 W/m/K a copper wire is
sold on — but `mv.prop.thermal_conductivity` returns the **lattice**
contribution from the Slack model, and in a metal almost all the heat is carried
by electrons. A few W/m/K is the right magnitude for the phonons alone.

Every phonon mode is real, so all seven are dynamically stable, as elemental fcc
metals at their equilibrium lattice parameter should be. That check matters
because a composition can sit on the convex hull and still be a structure that
will not hold together."""),

    ("code", """\
import matplotlib.pyplot as plt

measured = {"Al": 76, "Cu": 140, "Ni": 180, "Ag": 101, "Au": 180,
            "Pd": 180, "Pt": 230}
names = list(metals.obs["name"])
predicted = metals.obs["bulk_modulus_emt"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(names))
ax.bar(x - 0.2, predicted, 0.4, label="EMT")
ax.bar(x + 0.2, [measured[n] for n in names], 0.4, label="experiment")
ax.set_xticks(x, names)
ax.set_ylabel("bulk modulus (GPa)")
ax.set_title("where the potential holds, and where it does not")
ax.legend()"""),

    ("markdown", """\
Aluminium is the outlier, platinum is high, and the middle five track closely.
A plot makes that visible in a way the table does not — EMT is not uniformly
wrong, it is wrong about particular elements, which is a different thing to
know.

## Two routes to the same number

A bulk modulus is a curvature. `mv.prop.elastic` takes it in strain, from the
stiffness tensor; `mv.prop.eos` takes it in volume, by compressing the cell and
fitting an equation of state. They are the same quantity, so they have to
agree — and because both live on the same object, checking is a subtraction."""),

    ("code", """\
mv.prop.eos(metals, level="emt", source="relaxed_emt")

comparison = metals.obs[["name", "bulk_modulus_emt", "bulk_modulus_eos_emt",
                         "bulk_modulus_derivative_emt",
                         "equilibrium_volume_emt"]].round(2)
comparison["disagreement_%"] = (
    100 * (comparison["bulk_modulus_eos_emt"] - comparison["bulk_modulus_emt"])
    / comparison["bulk_modulus_emt"]).round(2)
comparison"""),

    ("markdown", """\
Within 1.1% across seven metals and two independent implementations, and under
0.4% for six of them.

That agreement is worth more as a diagnostic than as a reassurance. While
`mv.calc.relax` was leaving the cell alone, these two columns differed by 9–12%
and always in the same direction — the stiffness was being measured about a
geometry under residual tensile stress, which softens it. Neither number looked
wrong on its own. **The disagreement was the only visible symptom**, and it was
visible only because both quantities sit on one object under names that say what
they are.

`mv.prop.eos` also returns what the elastic route cannot: the pressure
derivative `B0'`, near 4 for most solids, and the equilibrium volume read off
the fitted curve rather than off the optimiser.

Aluminium's `B0'` comes out at 1.95 against a measured 4.4. Taken with a bulk
modulus half the experimental value, that says EMT has the wrong *shape* for
aluminium's energy–volume curve rather than merely the wrong curvature at one
point — which is more than the bulk modulus alone could tell you.

The curves themselves are an `obsm` block on a shared scale-factor grid, so
they plot like any other curve."""),

    ("code", """\
ax = mv.pl.spectra(metals, "eos", levels=("emt",), rows=[0, 1, 4])
ax.set_xlabel("volume scale factor $V/V_0$")
ax.set_ylabel("energy (eV/atom)")
ax.set_title("energy-volume curves")"""),

    ("markdown", """\
## Off the zero-kelvin hull

A hull at 0 K is a hull at 0 K. `mv.prop.quasiharmonic` reuses that same
energy-volume curve to get to finite temperature: the Debye model supplies a
vibrational free energy at each volume, and the Gibbs free energy is minimised
over volume at every temperature. The cell expands because the minimum
moves."""),

    ("code", """\
mv.prop.quasiharmonic(metals, level="emt", source="relaxed_emt",
                      t_max=900, poisson=0.34)

metals.obs[["name", "thermal_expansion_qha_emt", "gruneisen_emt",
            "debye_temperature_qha_emt"]].round(6)"""),

    ("markdown", """\
| volumetric expansion, /K | matverse | experiment |
|---|---|---|
| Cu | 4.5e-5 | 5.0e-5 |
| Ag | 5.1e-5 | 5.7e-5 |
| Al | 4.9e-5 | 6.9e-5 |

```{warning}
That column is **not** what pymatgen's `QuasiharmonicDebyeApprox` reports for
the optimum volume. Its Gruneisen parameter is right — 1.91 for copper against
a measured 1.96 — but the volume minimum it finds moves twelve times too
little, giving 4.3e-6 /K for copper.

`mv.prop.quasiharmonic` computes the expansion from the thermodynamic identity
instead:

$$\\alpha_V = \\frac{\\gamma\\, C_V}{B\\, V}$$

with the model's own Gruneisen parameter, the bulk modulus fitted from the same
E(V) points, and the Debye heat capacity. That is the version that agrees with
experiment.

The disagreement was findable because `B` from `mv.prop.eos` and `gamma` from
the Debye model sit on one object under names that say what they are — the same
reason the relaxation bug two sections up was findable.
```"""),

    ("code", """\
ax = mv.pl.spectra(metals, "phonon_dos", levels=("emt",), rows=[0, 1, 4],
                   offset=0.3)
ax.set_title("phonon density of states")"""),

    ("markdown", """\
The phonons are also everything thermodynamics needs. `mv.prop.free_energy`
integrates the density of states into a vibrational free energy and entropy at
a temperature you name — the term that decides which polymorph wins when two
sit within a few meV at zero kelvin."""),

    ("code", """\
mv.prop.free_energy(metals, level="emt", temperature=300.0)

metals.obs[["name", "vibrational_free_energy_emt",
            "vibrational_entropy_emt"]].round(4)"""),

    ("markdown", """\
## Screening

`mv.screen.filter` takes criteria as `column__operator=value` and **deposits** a
boolean column plus the criteria that produced it, rather than returning a
shorter list — because which criterion a candidate failed is a result."""),

    ("code", """\
mv.screen.filter(metals,
                 thermal_conductivity_emt__gt=3.0,
                 bulk_modulus_emt__gt=50.0,
                 name="conductive_and_stiff")

metals.uns["screens"]["conductive_and_stiff"]"""),

    ("code", """\
metals.obs[["name", "thermal_conductivity_emt", "bulk_modulus_emt",
            "conductive_and_stiff"]].round(1)"""),

    ("markdown", """\
Subset when you actually want the short list — the object is an ordinary
AnnData, so `metals[metals.obs['conductive_and_stiff']]` works and carries
everything with it.

## Which chemistry passed?

The question that follows every screen. Because `X` is the composition matrix,
this is `rank_genes_groups` with the nouns changed."""),

    ("code", """\
mv.tl.rank_elements_groups(metals, "conductive_and_stiff")

metals.uns["rank_elements_groups"]["True"][
    ["element", "n_in_group", "frac_in_group", "odds_ratio", "pval"]]"""),

    ("markdown", """\
With seven elementals this is a formality. On a library of thousands it is the
operation that turns a pass/fail column into a statement about chemistry, and it
is the reason `X` holds composition rather than being left empty.

## What the object remembers"""),

    ("code", """\
for step in mv.provenance(metals):
    print(step)"""),

    ("code", """\
ax = mv.pl.provenance(metals)"""),

    ("markdown", """\
Parameters are recorded with each call, so the history replays as code rather
than reading as a list of verbs. Saving keeps all of it:

```python
metals.write_h5ad("screen.h5ad")
```

Structures, descriptors, level records and provenance all survive, and the file
is an ordinary `h5ad` that anndata reads without matverse installed.

```{seealso}
[Screening, end to end](screening.ipynb) covers the same pipeline in more detail;
[Chemical space](chemical_space.ipynb) picks up where `rank_elements_groups` left
off; [Beyond one number](beyond_one_number.ipynb) covers curves, per-atom results
and measured data.
```"""),
]


# Tutorials

Each tutorial is a working pipeline rather than a feature tour, and sixteen of
the seventeen are **executed notebooks** — every number and figure on those pages is the
real output of the code above it, produced when the documentation was built.
They run on a small library and a calculator that ships with matverse, so you
can execute them before deciding whether to install a machine-learned potential.

Download any of them with the {octicon}`download;1em;` button and run it.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Getting started {octicon}`rocket;1em;`
:link: getting_started
:link-type: doc

A runnable notebook on published structures — LiFePO₄, NaFePO₄ and the fcc
metals — with no network, no API key and no downloaded model.
:::

:::{grid-item-card} Screening, end to end {octicon}`beaker;1em;`
:link: screening
:link-type: doc

Load candidates, standardise them, throw out the broken ones, relax, build a
hull, and shortlist — with the reasoning left in the object.
:::

:::{grid-item-card} Chemical space {octicon}`graph;1em;`
:link: chemical_space
:link-type: doc

What `X` being the composition matrix buys: ordination, clustering, and finding
which elements distinguish the candidates that passed.
:::

:::{grid-item-card} Beyond one number {octicon}`pulse;1em;`
:link: beyond_one_number
:link-type: doc

Curves on a shared grid, per-atom results on their own axis, measurements as a
level of theory, and scoring generated candidates.
:::

:::{grid-item-card} Models and campaigns {octicon}`rocket;1em;`
:link: models_and_campaigns
:link-type: doc

Predicting what you have not computed, with splits that do not leak; choosing
what to compute next; and plotting the result.
:::

:::{grid-item-card} Scale and first principles {octicon}`server;1em;`
:link: scale_and_dft
:link-type: doc

Corpora larger than memory, screens that outlive a walltime limit, and handing
the survivors to VASP or Quantum ESPRESSO.
:::
::::

## Physics

The four above are about the shape of the object. These four are about what you
compute with it — the parts of materials science that are not a scalar per
material.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Defects and diffusion {octicon}`git-branch;1em;`
:link: defects_and_diffusion
:link-type: doc

Vacancies, formation energy against Fermi level, and a migration barrier that
lands at 0.754 eV against a literature 0.70 for copper.
:::

:::{grid-item-card} Surfaces and adsorption {octicon}`stack;1em;`
:link: surfaces_and_adsorption
:link-type: doc

Slabs, surface energies in the literature ordering, the Wulff shape, and where
oxygen actually binds on Cu(111).
:::

:::{grid-item-card} Dynamics {octicon}`pulse;1em;`
:link: dynamics
:link-type: doc

Temperature: equilibration you can see, thermal expansion from motion alone,
and a melt-quench with its known failure mode.
:::

:::{grid-item-card} Magnetic ordering {octicon}`north-star;1em;`
:link: magnetic_ordering
:link-type: doc

Enumerate the spin states before the hull — and read the number that says your
calculator cannot tell them apart.
:::

:::{grid-item-card} Environments and bands {octicon}`telescope;1em;`
:link: structure_and_bands
:link-type: doc

The two questions composition cannot answer: what an atom's neighbourhood looks
like, and what the electrons can do.
:::

:::{grid-item-card} Interfaces {octicon}`versions;1em;`
:link: interfaces
:link-type: doc

Will the lattices match, will the contact survive, and what cell do you
actually run — three separate questions about two materials at once.
:::

:::{grid-item-card} Disorder {octicon}`shuffle;1em;`
:link: disorder
:link-type: doc

Fractional occupancy, ordered approximants, and the entropy term that lets a
high-entropy alloy sit above the hull and form anyway.
:::

:::{grid-item-card} Molecules {octicon}`beaker;1em;`
:link: molecules
:link-type: doc

Point groups, covalent bonds and fragments — on the same axes as crystals,
because the composition matrix never cared whether a formula unit repeats.
:::
::::

## Plumbing

Getting data in and out, and getting a campaign to finish.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Getting data in and out {octicon}`database;1em;`
:link: data_io
:link-type: doc

Every door into the object and back out, queried against a **live** OQMD over
OPTIMADE — where 7 of 15 hits turn out to be duplicates.
:::

:::{grid-item-card} Infrastructure {octicon}`gear;1em;`
:link: infrastructure
:link-type: doc

Units, checkpoints, corpora larger than memory, Slurm scripts, and handing the
shortlist to VASP.
:::

:::{grid-item-card} Coming from pymatgen {octicon}`arrow-switch;1em;`
:link: from_pymatgen
:link-type: doc

Every pymatgen transformation by name, and what putting your structures in an
object actually buys you.
:::
::::

## Suggested order

**Getting started** is the shortest complete pass through the library. Start
there.

**Screening**, **Chemical space** and **Beyond one number** build on each other
and are worth reading in sequence — screening produces the object the other two
pick up, and each rebuilds it in one cell so you can also read them alone.

The last two are independent. Read **Models and campaigns** when a screen has
more candidates than you can afford to compute, and **Scale and first
principles** when it has more candidates than fit in memory, or when the
shortlist needs real DFT. That last page stays prose rather than a notebook: it
is about corpora larger than memory and jobs handed to VASP, and a notebook of
it would be a page of code nobody could execute.

The two **plumbing** pages are reference more than story, and between them they
exercise every function in the registry — which is checked, not asserted: the
documentation build fails if a registered function appears in no notebook.

The four **physics** tutorials are independent of each other and of everything
above. Each answers a question that a scalar per material cannot: how fast do
atoms move, what does the crystal show the world, what happens at temperature,
and which spin state are you actually computing.

```{toctree}
:hidden: true
:maxdepth: 1

getting_started
screening
chemical_space
beyond_one_number
models_and_campaigns
scale_and_dft
defects_and_diffusion
surfaces_and_adsorption
dynamics
magnetic_ordering
structure_and_bands
interfaces
disorder
molecules
from_pymatgen
data_io
infrastructure
```

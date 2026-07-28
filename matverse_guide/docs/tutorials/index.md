# Tutorials

Each tutorial is a working pipeline rather than a feature tour, and the first
five are **executed notebooks** — every number and figure on those pages is the
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

```{toctree}
:hidden: true
:maxdepth: 1

getting_started
screening
chemical_space
beyond_one_number
models_and_campaigns
scale_and_dft
```

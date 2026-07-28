# Tutorials

Each tutorial is a working pipeline rather than a feature tour. They run on a
small library and a calculator that ships with matverse, so you can execute them
before deciding whether to install a machine-learned potential.

::::{grid} 1 2 2 2
:gutter: 2

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
::::

```{toctree}
:hidden: true
:maxdepth: 1

screening
chemical_space
```

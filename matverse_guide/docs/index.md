# matverse Documentation

## One object for a materials screening campaign

**matverse** is a materials analysis framework on the
[AnnData](https://anndata.readthedocs.io/) substrate. One object carries a
screening pipeline end to end — query, standardise, featurise, relax, rank —
keeping the structures, the annotations, the descriptors and the results
together, and writing to `h5ad`/`zarr` without a new file format.

```python
import matverse as mv

md = mv.data.from_cif('candidates/')
mv.pp.standardize(md)
mv.pp.qc(md)
md = mv.pp.filter_materials(md)
mv.calc.relax(md, level='emt')
mv.thermo.hull(md, level='emt')
mv.screen.filter(md, e_above_hull_emt__lt=0.05)
mv.tl.rank_elements_groups(md, 'passes')
```

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation {octicon}`plug;1em;`
:link: Installation_guide
:link-type: doc

Set up matverse and choose which optional backends you actually need.
:::

:::{grid-item-card} Tutorials {octicon}`play;1em;`
:link: tutorials/index
:link-type: doc

A screening campaign end to end, and the chemical-space toolchain on top of it.
:::

:::{grid-item-card} API Reference {octicon}`book;1em;`
:link: api/index
:link-type: doc

Every public function, generated from the registry that describes them.
:::

:::{grid-item-card} Design {octicon}`telescope;1em;`
:link: Design
:link-type: doc

Why the object has this shape, what matverse deliberately does not build, and
what is still open.
:::

:::{grid-item-card} Benchmark {octicon}`checklist;1em;`
:link: Benchmark
:link-type: doc

matverse-bench: goal-not-API tasks graded on end state, by a grader with no
model calls in it.
:::

:::{grid-item-card} Release notes {octicon}`tag;1em;`
:link: Release_notes
:link-type: doc

What changed, and which claims were deleted because they failed their probe.
:::

:::{grid-item-card} GitHub {octicon}`mark-github;1em;`
:link: https://github.com/matverse/matverse

Find a bug? Interested in contributing? Check out the repository.
:::
::::

## Three conventions carry the design

**Operations deposit; they do not return.** `mv.pp.standardize(md)` writes
`obsm['structures']['primitive']` and returns `None`. Structure variants
accumulate in one object instead of becoming four variables downstream code has
to keep straight, and `uns['provenance']` records what ran with its parameters —
so a run replays as code rather than reading as a list of verbs.

**A result carries its level of theory in the slot name.** `obs['energy_emt']`
and `obs['energy_pbe']` are different quantities, and `uns['levels'][level]`
records what produced each: the method, what it reproduces, its licence, and
where its uncertainty came from. `mv.thermo.hull` refuses to build a hull across
two levels whose references disagree, because that hull is not a hull of
anything.

**`X` is the composition matrix and `var` is the periodic table.** Materials by
elements is the same shape as cells by genes — sparse, non-negative, mostly zero,
since almost every material draws on five or fewer of 118 columns. That
correspondence is what lets ordination, clustering and differential enrichment
apply to chemical space without being rewritten.

```{seealso}
`mv.tl.rank_elements_groups` is `rank_genes_groups` with the nouns changed: it
answers "which elements distinguish the candidates that passed from the ones
that failed", the question that follows every screen.
```

## Agent-readable by construction

Every public function carries a registry entry naming what it consumes and
creates, so a tool can find it by intent rather than by remembering its name.

```python
mv.find('thermodynamic stability')     # ['mv.thermo.hull', ...]
print(mv.describe('convex hull'))
```

`requires` and `produces` name state a call consumes and creates, so they bind
only in a library where calls *have* named state to point at — which is why they
work here and did not transfer to libraries whose results are attributes on
returned objects. Every claim is verified by execution rather than asserted: see
[the contract section](Developer_guide.md#contract-claims-are-verified-by-execution).

```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

Installation_guide
tutorials/index
api/index
Design
Benchmark
Release_notes
Developer_guide
GitHub <https://github.com/matverse/matverse>
```

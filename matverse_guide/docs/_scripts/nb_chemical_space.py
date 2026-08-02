"""Cells for tutorials/chemical_space.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Chemical space

This tutorial is about the one design decision that most distinguishes matverse
from a DataFrame with a `structure` column: **`X` is the composition matrix and
`var` is the periodic table**.

It picks up where [Screening, end to end](screening.ipynb) left off — same
Al–Ni–Cu library, same `passes` column — and asks the question that follows
every screen: *what chemistry is on the shortlist?*"""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

Rebuilding the screening object in one cell: three elemental fcc metals at their
published lattice parameters, four hypothetical intermetallics, relaxed and
screened with `emt`."""),

    ("code", """\
from pymatgen.core import Lattice, Structure


def l12(host, guest, a):
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


elemental = mv.datasets.metals(["Al", "Cu", "Ni"])
md = mv.data.from_structures(mv.structures(elemental) + [
    l12("Al", "Cu", 3.90), l12("Cu", "Al", 3.70), l12("Al", "Ni", 3.78),
    Structure(Lattice.cubic(2.89), ["Al", "Ni"], [[0, 0, 0], [.5, .5, .5]]),
])

mv.pp.describe(md)
mv.calc.relax(md, level="emt", fmax=0.05)
mv.thermo.hull(md, level="emt", source="relaxed_emt")
mv.screen.filter(md, e_above_hull_emt__lt=0.12, density__lt=8.0)

md.obs[["formula", "e_above_hull_emt", "density", "passes"]].round(3)"""),

    ("markdown", """\
## Why elements are the right axis

A composition matrix is materials × elements — sparse, non-negative, and almost
entirely zero, because nearly every material draws on five or fewer of 118
columns. That is structurally a cells × genes matrix."""),

    ("code", """\
pd.DataFrame(md.X.toarray(), index=md.obs["formula"], columns=md.var_names)"""),

    ("markdown", """\
Counts come from the **reduced** formula, so a supercell and its primitive cell
occupy the same point in chemical space. Cell size belongs in `obs['nsites']`,
not in the chemistry.

`var` carries the periodic table, which is what makes element-level results
interpretable without a lookup somewhere else:"""),

    ("code", """\
md.var[["Z", "electronegativity", "period", "is_transition_metal"]]"""),

    ("markdown", """\
```{note}
Economic properties — price, supply risk, criticality — are deliberately **not**
shipped. They are jurisdiction- and date-dependent and no authoritative open
table exists, so inventing numbers would make `mv.screen.filter(md,
supply_risk__lt=0.3)` return confident nonsense. Attach your own instead.
```"""),

    ("code", """\
mv.elements.annotate(md, pd.DataFrame(
    {"price_usd_kg": [2.4, 18.0, 9.5]}, index=["Al", "Ni", "Cu"]))

md.var[["price_usd_kg"]]"""),

    ("markdown", """\
## The dividend inside the library

Because `X` holds the composition and `var` holds the element properties, the
classic composition descriptor — the weighted mean, spread and range of element
properties across a formula — is a matrix product of two things the object is
already carrying."""),

    ("code", """\
mv.feat.element_stats(md)
md.obsm["X_element_stats"].shape"""),

    ("markdown", """\
Check one by hand. For Al₃Cu the weighted mean electronegativity should be
exactly `0.75 × 1.61 + 0.25 × 1.90`:"""),

    ("code", """\
names = list(md.uns["features"]["X_element_stats"]["names"])
j = names.index("electronegativity_mean")
row = list(md.obs["formula"]).index("Al3Cu")

md.obsm["X_element_stats"][row, j], 0.75 * 1.61 + 0.25 * 1.90"""),

    ("markdown", """\
Those descriptors are also a distance, which makes the library a matrix you can
look at."""),

    ("code", """\
import matplotlib.pyplot as plt

mv.feat.similarity(md)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(md.obsp["similarity_X_element_stats"], cmap="magma")
labels = list(md.obs["formula"])
ax.set_xticks(range(len(labels)), labels, rotation=90)
ax.set_yticks(range(len(labels)), labels)
ax.set_title("descriptor similarity")
fig.colorbar(im, ax=ax, shrink=0.8)"""),

    ("markdown", """\
The two Al–Cu orderings sit next to each other and the elementals separate,
which is the expected structure and therefore worth checking — a similarity
matrix with no block structure means the descriptor is not carrying chemistry.

`min`, `max` and `range` are taken over the elements that are *present at all*,
so "the most electronegative element in this compound" is the quantity it sounds
like rather than an amount-weighted blend.

Composition descriptors cannot tell two polymorphs apart, because polymorphs have
the same composition. When that matters, reach for a structure descriptor:"""),

    ("code", """\
try:
    mv.feat.soap(md)
    result = md.obsm["soap"].shape
except ImportError as exc:
    result = str(exc)[:170]     # needs matverse[descriptors]
result"""),

    ("markdown", """\
A site-averaged SOAP vector per structure, which separates two polymorphs of
one composition where a composition descriptor cannot.

```{note}
Needs **dscribe**, which imports `sparse`, which imports `numba`, which pins
numpy below 2.5. On a newer numpy the install resolves and the import does not,
so this cell reports rather than fails.
```

### Descriptors matverse does not ship

`mv.feat.register_embedder` is the door for anything that turns a structure into
a vector — a pretrained graph network, a fingerprint from another library, a
scheme of your own. `mv.feat.embed` runs it and records where the numbers came
from."""),

    ("code", """\
def mass_and_volume(structures):
    \"\"\"A deliberately trivial embedder.

    It takes the whole list and returns one row per structure — the batch
    contract, so a model that runs on a GPU is called once rather than n times.
    \"\"\"
    return np.array([[s.composition.weight / len(s), s.volume / len(s)]
                     for s in structures])


mv.feat.register_embedder("toy", mass_and_volume, method="mass and volume",
                          license="n/a")
mv.feat.embed(md, model="toy")

pd.DataFrame(md.obsm["X_toy"], index=md.obs["formula"],
             columns=["mean mass", "volume per atom"]).round(3)"""),

    ("code", """\
md.uns["features"]["X_toy"]"""),

    ("markdown", """\
Which model produced the block, and under what licence, is recorded next to it
in `uns['features']` — the same slot `mv.feat.element_stats` writes to. Two
descriptor blocks from different models are therefore distinguishable after the
fact, which matters the moment two people in a group use different checkpoints
of the same network.

```{note}
Note that this is `uns['features']`, not `uns['levels']`. A descriptor is not a
level of theory: it makes no claim about energy and cannot be mixed into a hull,
so it does not need the machinery that stops that happening. The rule is that
the record goes where the thing being described lives.
```

## Mapping the space"""),

    ("code", """\
mv.pp.normalize_composition(md)     # -> layers['fraction']
mv.tl.pca(md, n_comps=2)
mv.tl.neighbors(md, n_neighbors=3)
mv.tl.cluster(md, method="kmeans", n_clusters=2)

md.obs[["formula", "cluster"]]"""),

    ("markdown", """\
`normalize_composition` writes atomic fractions to a **layer** rather than
replacing `X`, because a hull needs the counts and a chemical-space map needs
the fractions. `mv.tl.pca` defaults to the fraction layer when it exists: raw
counts scale with the size of the formula unit, so `Fe2O3` and `Fe4O6` would
otherwise sit at different points despite being the same chemistry.

PCA here is an exact SVD. The element axis is at most 118 columns wide, so there
is nothing to approximate and no randomised solver to configure."""),

    ("code", """\
ax = mv.pl.embedding(md, color="cluster")
ax.set_title("chemical space, coloured by cluster")"""),

    ("markdown", """\
## The question that follows every screen

You have a shortlist. What chemistry is on it?"""),

    ("code", """\
mv.tl.rank_elements_groups(md, "passes")

md.uns["rank_elements_groups"]["True"]"""),

    ("markdown", """\
This is `rank_genes_groups` with the nouns changed, and it is the operation that
justifies the whole design choice. Answering the same question from a DataFrame
of averaged Magpie descriptors means writing the contingency table and the test
by hand, every time, for every screen.

Two routes, because "which elements" has two meanings — how *often* an element
appears in the group, and how *much* of it there is."""),

    ("code", """\
mv.tl.rank_elements_groups(md, "passes", method="fraction")   # Wilcoxon

md.uns["rank_elements_groups"]["True"][
    ["element", "mean_frac_in_group", "mean_frac_rest", "diff", "pval", "qval"]]"""),

    ("markdown", """\
Both report an uncorrected p-value and a Benjamini–Hochberg q-value. With seven
materials nothing here is significant, and the table says so — which is the point
of reporting both.

The natural display is the periodic table, in the way a dot plot is the natural
display for differential expression."""),

    ("code", """\
mv.tl.rank_elements_groups(md, "passes")
ax = mv.pl.rank_elements_groups(md, group="True")
ax.set_title("elements enriched on the shortlist")"""),

    ("markdown", """\
The bar chart when you want the numbers; the periodic table when you want the
layout."""),

    ("code", """\
frame = md.uns["rank_elements_groups"]["True"]
scores = frame.set_index("element")["log2_odds"].reindex(md.var_names)

ax = mv.pl.periodic_table(md, values=scores.to_numpy(),
                          label="log2 odds, shortlist", center=0.0)"""),

    ("markdown", """\
Aluminium comes out grey rather than dark. Its odds ratio is infinite — it is in
every shortlisted material and the log is undefined — and `mv.pl.periodic_table`
leaves non-finite values uncoloured instead of clipping them to the top of the
scale, so a cell that is grey means "no number", never "zero enrichment".

With seven materials that will happen often. On a real library it means one
group is empty and the contingency table has a structural zero, which is worth
seeing rather than having smoothed away.

### Dropping elements nothing uses

A library assembled from a database query often carries columns for elements
that appear in one material, or none. `mv.pp.filter_elements` narrows the
periodic table to what is actually present, which matters once `var` is
thousands of columns wide rather than three."""),

    ("code", """\
narrowed = mv.pp.filter_elements(md, min_materials=2)
md.n_vars, narrowed.n_vars, list(narrowed.var_names)"""),

    ("markdown", """\
## Is this candidate actually new?

Novelty is distance in composition space to the nearest known material. Here the
"known" set is the three elementals; in practice it is a database."""),

    ("code", """\
known = mv.data.from_structures(mv.structures(elemental))
mv.pp.describe(known)
mv.tl.novelty(md, reference=known)

md.obs[["formula", "novelty_distance", "nearest_reference"]].round(3)"""),

    ("code", """\
ax = mv.pl.scatter(md, "formula", "novelty_distance")
ax.set_ylabel("distance to nearest known")
ax.set_title("novelty in composition space")"""),

    ("markdown", """\
```{warning}
Composition-space distance answers "is this chemistry new", not "is this
structure new". A novel polymorph of a known composition scores zero — which is
exactly what the three elementals do above. For structural novelty, concatenate
against the reference and run `mv.pp.dedup`.
```

## Does the bet hold?

`X` as the composition matrix is a bet, and the test that would kill it is in
the suite:

```
tests/test_pipeline.py::TestTools::test_rank_elements_groups_recovers_the_obvious_chemistry
```

It groups materials by whether they contain Al and asserts that the top ranked
element is Al. If element-level enrichment ever stops recovering chemistry a
domain expert would recognise, the design falls back to the width-zero `X` of
v0.1 — `mv.data.from_structures(..., build_X=False)` — and nothing else changes.

```{seealso}
[Beyond one number](beyond_one_number.ipynb) covers the results that are not a
single number per material: curves, per-atom values and measurements.
```"""),
]

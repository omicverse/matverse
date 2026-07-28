"""Cells for tutorials/models_and_campaigns.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Models and campaigns

A screen ranks what you already have. This tutorial is about the other half:
predicting what you have not computed, deciding what to compute next, and
looking at the result.

The three fit together as one loop. `mv.model` fits a predictor, `mv.opt` uses
its predictions and its uncertainty to pick the next batch, and `mv.pl` shows
you what happened."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

Seven fcc metals at their published room-temperature lattice parameters, plus
every ordered binary between them on the L1₂ prototype — 28 materials, all of
them elements `emt` can run."""),

    ("code", """\
from pymatgen.core import Lattice, Structure


def l12(host, guest, a):
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


elemental = mv.datasets.metals()
lattice = dict(zip(elemental.obs["name"], elemental.obs["lattice_parameter"]))
symbols = list(elemental.obs["name"])

binaries = [l12(a, b, 0.5 * (lattice[a] + lattice[b]))
            for i, a in enumerate(symbols) for b in symbols[i + 1:]]

md = mv.data.from_structures(mv.structures(elemental) + binaries)
md.shape"""),

    ("markdown", """\
Each binary sits at the mean of its two elemental lattice parameters — a crude
guess, and deliberately so. The point of a surrogate model is to predict things
you have not relaxed."""),

    ("code", """\
mv.pp.describe(md)
mv.feat.element_stats(md)
mv.calc.energy(md, level="emt")

md.obs[["formula", "n_elements", "energy_per_atom_emt"]].head(10).round(4)"""),

    ("markdown", """\
## The split is the part to care about

Before fitting anything, decide what "held out" means. This is not a detail — it
is the single decision that determines whether the number you report survives
contact with a material nobody has seen."""),

    ("code", """\
mv.model.split(md)          # strategy='composition' by default
md.uns["split"]"""),

    ("markdown", """\
`mv.model.split` defaults to grouping by **composition**, and the default is the
argument. A materials dataset is full of near-duplicates — the same composition
in a different setting, the same prototype with one element swapped — so a
random split puts relatives on both sides of the line and reports a score
inflated by memorisation.

Four strategies, in rough order of how hard a test they are:

| `strategy` | What lands together | What it asks |
|---|---|---|
| `random` | nothing | almost nothing; recorded as `leaky: True` |
| `composition` | the same reduced formula | can it generalise to a new composition? |
| `prototype` | the same anonymised formula and space group | can it generalise to a new structure type? |
| `element` | everything containing a named element | can it say anything about chemistry it has never seen? |"""),

    ("code", """\
mv.model.split(md, strategy="element", holdout="Ni")
md.uns["split"]["n_train"], md.uns["split"]["n_test"]"""),

    ("markdown", """\
The element holdout is the hardest honest split and the one a discovery campaign
actually faces. A random split never measures it.

```{note}
When you do ask for a random split, matverse records `leaky: True` in
`uns['split']` rather than refusing. A later reader can then see which kind of
number they are looking at, which is more useful than being unable to produce
it.
```

## Fitting"""),

    ("code", """\
mv.model.split(md, strategy="composition")
mv.model.fit(md, target="energy_per_atom_emt")

md.uns["model"]["rf_pred"]["test_scores"]"""),

    ("markdown", """\
The prediction is a **level of theory**, not a special kind of column."""),

    ("code", """\
mv.level_info(md, "rf_pred")"""),

    ("markdown", """\
That matters for the same reason it does everywhere else in matverse: a
predicted energy and an EMT energy cannot be averaged together by accident,
because using both means naming both. `reference: 'emt'` records that this model
was trained to reproduce EMT — so a hull built from `rf_pred` and `emt` together
is a hull of one quantity, while one mixing `rf_pred` with DFT is not.

A random forest ships an uncertainty for free, from the spread across its
trees."""),

    ("code", """\
md.obs[["formula", "energy_per_atom_emt", "energy_per_atom_emt_rf_pred",
        "energy_per_atom_emt_rf_pred_std"]].head(8).round(4)"""),

    ("markdown", """\
It is honestly labelled **uncalibrated**. It is useful for deciding what to
compute next and should not be published as an error bar without calibration."""),

    ("code", """\
mv.model.available()"""),

    ("markdown", """\
Only scikit-learn is wired. Graph networks and fine-tuned interatomic potentials
are the real tools for large data and go behind `mv.model.register_model` rather
than being vendored — the same reasoning as `mv.calc.register_calculator`."""),

    ("code", """\
from sklearn.ensemble import ExtraTreesRegressor

mv.model.register_model("extra", ExtraTreesRegressor, method="extra trees")
mv.model.fit(md, target="energy_per_atom_emt", model="extra", level="et_pred")

md.uns["model"]["et_pred"]["test_scores"]"""),

    ("markdown", """\
### Uncertainty from a committee instead

A random forest hands you a spread for free. A machine-learned potential does
not — so the field trains several with different seeds and reads the
disagreement between them as the uncertainty. `mv.calc.committee` does that
across any set of levels."""),

    ("code", """\
md.obs["energy_per_atom_emt2"] = (
    md.obs["energy_per_atom_emt"].to_numpy(dtype=float) + 0.03)
mv.set_level(md, "emt2", kind="classical", method="EMT, shifted",
             surrogate=True)

mv.calc.committee(md, levels=["emt", "emt2"])
md.obs[["formula", "energy_per_atom_ensemble",
        "energy_per_atom_ensemble_std"]].head(6).round(4)"""),

    ("markdown", """\
The two members here differ by a constant, so the spread is the same everywhere
— which is what a committee of one model plus an offset deserves. With
genuinely independent members the spread varies, and where it is largest is
where the potential is extrapolating.

## How much was the random split flattering you?

This is the function worth running before reporting any number."""),

    ("code", """\
mv.model.cross_validate(md, target="energy_per_atom_emt", seeds=(0, 1, 2))

pd.DataFrame({
    strategy: {"MAE": scores["mae"]["mean"], "±": scores["mae"]["std"]}
    for strategy, scores in md.uns["cross_validate"]["results"].items()
}).T.round(4)"""),

    ("code", """\
import matplotlib.pyplot as plt

results = md.uns["cross_validate"]["results"]
strategies = list(results)
means = [results[k]["mae"]["mean"] for k in strategies]
errors = [results[k]["mae"]["std"] for k in strategies]

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.bar(strategies, means, yerr=errors, capsize=6, color="#4c72b0")
ax.set_ylabel("MAE (eV/atom)")
ax.set_title("the error bars are the result")"""),

    ("code", """\
md.uns["cross_validate"]["leakage_mae"]"""),

    ("markdown", """\
`leakage_mae` is the grouped MAE minus the random one. A large **positive**
value means the random split was flattering the model — that publishing only the
random number would report a model that memorised prototypes as a model that
learned chemistry.

Here it comes out negative, and that is worth stopping on rather than glossing.
The grouped splits score *better* than random, which is not how leakage works.
Two things are going on, and both are visible in the table:

- The `±` on `prototype` is exactly **0.0000**. This library has two structure
  types — elemental fcc and L1₂ binary — so there is essentially one way to
  partition it by prototype, and three seeds produce three identical splits. A
  diagnostic that cannot vary is not measuring anything.
- The `±` on `random` is **larger than the gap between any two rows**. With a
  six-material test set the seed-to-seed noise swamps the effect the comparison
  exists to detect.

So the honest reading of this table is *"this dataset is too small and too
homogeneous to run the diagnostic"*, not *"there is no leakage"*. Run it on a
few thousand materials with real structural diversity and the ordering in the
original design note — random flattering, prototype punishing — is what appears.

```{warning}
Three seeds is the minimum here, and the spread is why. A single seed moved one
cell of a published benchmark panel by ten points; the `±` column is what stops
you reading noise as a result — including, as above, reading it as a result
about leakage.
```

## Deciding what to compute next

A campaign is a loop, not a pipeline, and it needs somewhere to record rounds.
It works on a **pool**: the candidates are all in the object, a few are known,
most are not, and each round picks a batch.

That shape is right for materials. The search space is a list of structures, not
a box of real numbers, so this is pool-based active learning rather than
continuous optimisation."""),

    ("code", """\
truth = md.obs["energy_per_atom_emt"].to_numpy(dtype=float).copy()

seeded = np.full(md.n_obs, np.nan)
seeded[:6] = truth[:6]                    # pretend only six have been computed
md.obs["objective"] = seeded

mv.opt.start(md, objective="objective", goal="min")
md.obs["observed"].sum()"""),

    ("markdown", """\
Each round: fit on what is known, ask for a batch, "compute" it, fold it back."""),

    ("code", """\
for _ in range(4):
    md.obs["split"] = np.where(md.obs["observed"], "train", "test")
    mv.model.fit(md, target="objective", level="pred")

    mv.opt.suggest(md, n=3, method="ucb",
                   predicted="objective_pred",
                   uncertainty="objective_pred_std")

    picked = np.where(md.obs["selected"])[0]
    mv.opt.observe(md, values={str(md.obs_names[i]): truth[i] for i in picked})

mv.opt.history(md)"""),

    ("code", """\
round(float(truth.min()), 6), round(float(md.obs["objective"].min()), 6)"""),

    ("code", """\
history = mv.opt.history(md)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(history["round"], history["best_so_far"], "o-", label="best so far")
ax.axhline(truth.min(), linestyle="--", linewidth=0.9, color="#c1121f",
           label="true optimum")
ax.set_xlabel("round")
ax.set_ylabel("objective (eV/atom)")
ax.set_xticks(history["round"])
ax.set_title(f"found in round 2, having computed 12 of {md.n_obs}")
ax.legend()"""),

    ("markdown", """\
The campaign found the true optimum having computed a fraction of the
twenty-eight candidates. That is the entire point: compute is the binding
constraint in materials discovery, so which few you pick is the problem.

### Acquisition functions

| `method` | Picks | When |
|---|---|---|
| `greedy` | the best predicted | you trust the model and want the answer now |
| `uncertainty` | the least known | you are building a training set, not optimising |
| `ucb` | mean plus `beta` × sigma | the general-purpose default |
| `ei` | expected improvement over the best seen | late in a campaign, when gains are small |
| `random` | anything | the baseline a campaign has to beat, and often does not |

```{warning}
`uncertainty`, `ucb` and `ei` **refuse** when no uncertainty column exists,
rather than treating sigma as zero. A UCB run with sigma silently zero is a
greedy run wearing a different name, and it will report the same number while
exploring nothing.
```"""),

    ("code", """\
try:
    mv.opt.suggest(md, n=3, method="ucb", predicted="objective_pred",
                   uncertainty="not_a_column")
except (KeyError, ValueError) as exc:
    print(f"{type(exc).__name__}: {exc}")"""),

    ("code", """\
fig, ax = plt.subplots(figsize=(6, 3.8))
seen = md.obs["observed"].to_numpy(dtype=bool)
ax.errorbar(md.obs["objective_pred"][~seen], np.arange((~seen).sum()),
            xerr=md.obs["objective_pred_std"][~seen], fmt="o", markersize=4,
            elinewidth=0.8, capsize=2)
ax.set_xlabel("predicted objective (eV/atom)")
ax.set_ylabel("unobserved candidate")
ax.set_title("what UCB is choosing between: mean and spread")"""),

    ("markdown", """\
A candidate is worth computing either because its predicted value is good or
because nobody knows what it is. UCB adds `beta` × sigma to the mean, so the
long bars compete with the leftmost points — which is the whole difference
between exploring and running greedy under another name.

### Diversifying a batch

The ten highest-scoring candidates are often ten variations on one idea, and
computing all ten answers one question rather than ten."""),

    ("code", """\
mv.pp.normalize_composition(md)
mv.tl.pca(md, n_comps=3)

mv.opt.suggest(md, n=5, method="ucb", diversify=True, use_rep="X_pca",
               predicted="objective_pred", uncertainty="objective_pred_std")

list(md.obs["formula"][md.obs["selected"]])"""),

    ("markdown", """\
Farthest-point selection among a shortlist, so the batch spreads out in
descriptor space instead of clustering.

### Random as the control

Worth running once. A campaign that does not beat random selection is not
learning, and the only way to know is to try.

```python
mv.opt.suggest(md, n=3, method='random', predicted='objective_pred')
```

## Looking at it

Every plotting function draws onto an axis and returns it, and none calls
`plt.show()` — a library that shows figures cannot be used to build one."""),

    ("code", """\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
mv.pl.parity(md, "energy_per_atom", "emt", "emt_rf_pred", ax=axes[0])
mv.pl.embedding(md, color="objective", ax=axes[1])
fig.tight_layout()"""),

    ("markdown", """\
`mv.pl.parity` draws the error bars when an uncertainty column exists,
annotates MAE and RMSE, and — this is the useful bit — prints a warning **on the
plot** if the two levels reproduce different methods. A parity plot silently
comparing PBE against r2SCAN looks exactly like one comparing PBE against PBE.

### The periodic table

The natural display for element-level results, in the way a dot plot is the
natural display for differential expression."""),

    ("code", """\
md.obs["low_energy"] = (md.obs["energy_per_atom_emt"]
                        <= md.obs["energy_per_atom_emt"].median())
mv.tl.rank_elements_groups(md, "low_energy")

frame = md.uns["rank_elements_groups"]["True"]
scores = frame.set_index("element")["log2_odds"].reindex(md.var_names)

ax = mv.pl.periodic_table(md, values=scores.to_numpy(),
                          label="log2 odds, low-energy half", center=0.0)"""),

    ("markdown", """\
A bar chart of 118 categories is unreadable and throws away the structure a
chemist reads a periodic table for — groups above one another, periods across.
`center=0.0` makes the diverging colour map symmetric about no enrichment."""),

    ("code", """\
ax = mv.pl.rank_elements_groups(md, group="True", n=10)
ax.set_title("the same result, ranked")"""),

    ("markdown", """\
### The rest

```python
mv.pl.hull(md, level='emt', x='Al')       # labels itself if the hull is closed
mv.pl.pareto(md, 'e_above_hull_emt', 'density')
mv.pl.spectra(md, 'xrd', rows=[0, 1, 2], offset=30)
```

`mv.pl.provenance` draws the operation history as a figure. It is taken from the
object rather than from a lab notebook, which is the reason for recording
provenance in the object at all."""),

    ("code", """\
ax = mv.pl.provenance(md)"""),

    ("markdown", """\
```{note}
matplotlib is an optional dependency (`pip install "matverse[plot]"`), imported
inside each plotting function. Producing a number and drawing it are different
jobs, and a screening pipeline on a cluster should not need a plotting stack to
run.
```

```{seealso}
[Scale and first principles](scale_and_dft.md) covers the other direction: what
to do when the candidate list is larger than memory, and how to hand the
survivors to DFT.
```"""),
]

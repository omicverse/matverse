# Models and campaigns

A screen ranks what you already have. This tutorial is about the other half:
predicting what you have not computed, deciding what to compute next, and looking
at the result.

The three fit together as one loop. `mv.model` fits a predictor, `mv.opt` uses
its predictions and its uncertainty to pick the next batch, and `mv.pl` shows you
what happened.

Everything here runs on `emt`, on a 28-material library of seven fcc metals and
every ordered binary between them.

```python
import numpy as np
import matverse as mv
from pymatgen.core import Lattice, Structure

def fcc(symbol, a):
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

def l12(host, guest, a):
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

elements = ["Al", "Cu", "Ni", "Ag", "Au", "Pd", "Pt"]
structures = [fcc(e, 3.5 + 0.1 * i) for i, e in enumerate(elements)]
for i, a in enumerate(elements):
    for b in elements[i + 1:]:
        structures.append(l12(a, b, 3.6 + 0.02 * i))

md = mv.data.from_structures(structures)
mv.pp.describe(md)
mv.feat.element_stats(md)
mv.calc.energy(md, level="emt")

md.shape        # (28, 7)
```

## The split is the part to care about

Before fitting anything, decide what "held out" means. This is not a detail — it
is the single decision that determines whether the number you report survives
contact with a material nobody has seen.

```python
mv.model.split(md)          # strategy='composition' by default
md.uns['split']
```

```
{'strategy': 'composition', 'leaky': False,
 'n_train': 22, 'n_test': 6, 'seed': 0,
 'n_groups': 28, 'n_test_groups': 6}
```

`mv.model.split` defaults to grouping by **composition**, and the default is the
argument. A materials dataset is full of near-duplicates — the same composition
in a different setting, the same prototype with one element swapped — so a random
split puts relatives on both sides of the line and reports a score inflated by
memorisation.

Four strategies, in rough order of how hard a test they are:

| `strategy` | What lands together | What it asks |
|---|---|---|
| `random` | nothing | almost nothing; recorded as `leaky: True` |
| `composition` | the same reduced formula | can it generalise to a new composition? |
| `prototype` | the same anonymised formula and space group | can it generalise to a new structure type? |
| `element` | everything containing a named element | can it say anything about chemistry it has never seen? |

```python
mv.model.split(md, strategy="element", holdout="Ni")
md.uns['split']['n_train'], md.uns['split']['n_test']    # (21, 7)
```

The element holdout is the hardest honest split and the one a discovery campaign
actually faces. A random split never measures it.

```{note}
When you do ask for a random split, matverse records `leaky: True` in
`uns['split']` rather than refusing. A later reader can then see which kind of
number they are looking at, which is more useful than being unable to produce it.
```

## Fitting

```python
mv.model.split(md, strategy="composition")
mv.model.fit(md, target="energy_per_atom_emt")

md.uns['model']['rf_pred']['test_scores']
# {'n': 6, 'mae': 0.194, 'rmse': 0.252, 'r2': 0.341}
```

The prediction is a **level of theory**, not a special kind of column:

```python
mv.level_info(md, 'rf_pred')
# {'kind': 'model', 'method': 'random forest', 'reference': 'emt',
#  'surrogate': True, 'uncertainty': 'spread across trees (uncalibrated)',
#  'trained_on': 'energy_per_atom_emt', 'split_strategy': 'composition', ...}
```

That matters for the same reason it does everywhere else in matverse: a predicted
energy and an EMT energy cannot be averaged together by accident, because using
both means naming both. `reference: 'emt'` records that this model was trained to
reproduce EMT — so a hull built from `rf_pred` and `emt` together is a hull of
one quantity, while one mixing `rf_pred` with DFT is not.

A random forest ships an uncertainty for free, from the spread across its trees:

```python
md.obs[['energy_per_atom_emt_rf_pred', 'energy_per_atom_emt_rf_pred_std']]
```

It is honestly labelled **uncalibrated**. It is useful for deciding what to
compute next and should not be published as an error bar without calibration.

```python
mv.model.available()
# {'rf': {...}, 'ridge': {...}, 'gbr': {...}}
```

Only scikit-learn is wired. Graph networks and fine-tuned interatomic potentials
are the real tools for large data and go behind `mv.model.register_model` rather
than being vendored — the same reasoning as `mv.calc.register_calculator`.

```python
from sklearn.ensemble import ExtraTreesRegressor
mv.model.register_model("extra", ExtraTreesRegressor, method="extra trees")
mv.model.fit(md, target="energy_per_atom_emt", model="extra", level="et_pred")
```

## How much was the random split flattering you?

This is the function worth running before reporting any number.

```python
mv.model.cross_validate(md, target="energy_per_atom_emt", seeds=(0, 1, 2))

for strategy, scores in md.uns['cross_validate']['results'].items():
    print(f"{strategy:12s} MAE {scores['mae']['mean']:.4f} ± {scores['mae']['std']:.4f}")
```

```
random       MAE 0.1167 ± 0.0385
composition  MAE 0.1383 ± 0.0401
prototype    MAE 0.2602 ± 0.0000
```

```python
md.uns['cross_validate']['leakage_mae']    # 0.0217
```

Read the third row. Holding out a whole **structure type** more than doubles the
error against a random split — the model had been learning the prototype, and a
random split could not tell you that because relatives of every test material
were in training.

`leakage_mae` is the grouped MAE minus the random one. A large positive value
means the random split was flattering the model. Publishing only the random
number is how a model that memorised prototypes gets reported as a model that
learned chemistry.

```{warning}
Three seeds is the minimum here, and the spread is why. A single seed moved one
cell of a published benchmark panel by ten points; the `± std` column is what
stops you reading noise as a result.
```

## Deciding what to compute next

A campaign is a loop, not a pipeline, and it needs somewhere to record rounds. It
works on a **pool**: the candidates are all in the object, a few are known, most
are not, and each round picks a batch.

That shape is right for materials. The search space is a list of structures, not
a box of real numbers, so this is pool-based active learning rather than
continuous optimisation.

```python
truth = md.obs['energy_per_atom_emt'].to_numpy(dtype=float).copy()

seeded = np.full(md.n_obs, np.nan)
seeded[:6] = truth[:6]                    # pretend only six have been computed
md.obs['objective'] = seeded

mv.opt.start(md, objective='objective', goal='min')
```

Each round: fit on what is known, ask for a batch, "compute" it, fold it back.

```python
for _ in range(4):
    md.obs['split'] = np.where(md.obs['observed'], 'train', 'test')
    mv.model.fit(md, target='objective', level='pred')

    mv.opt.suggest(md, n=3, method='ucb',
                   predicted='objective_pred',
                   uncertainty='objective_pred_std')

    picked = np.where(md.obs['selected'])[0]
    mv.opt.observe(md, values={str(md.obs_names[i]): truth[i] for i in picked})
```

```python
mv.opt.history(md)
```

```
 round  n_selected  n_observed  best_so_far method
     1           3           9    -0.089397    ucb
     2           3          12    -0.109608    ucb
     3           3          15    -0.109608    ucb
     4           3          18    -0.109608    ucb
```

The true optimum is −0.1096, and the campaign reached it in round 2 having
computed twelve of twenty-eight candidates. That is the entire point: compute is
the binding constraint in materials discovery, so which few you pick is the
problem.

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
```

### Diversifying a batch

The ten highest-scoring candidates are often ten variations on one idea, and
computing all ten answers one question rather than ten.

```python
mv.pp.normalize_composition(md)
mv.tl.pca(md, n_comps=3)

mv.opt.suggest(md, n=5, method='ucb', diversify=True, use_rep='X_pca',
               predicted='objective_pred', uncertainty='objective_pred_std')
```

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
`plt.show()` — a library that shows figures cannot be used to build one.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
mv.pl.parity(md, 'energy_per_atom_emt', 'emt', 'rf_pred', ax=axes[0])
mv.pl.embedding(md, color='objective', ax=axes[1])
fig.tight_layout()
```

`mv.pl.parity` draws the error bars when an uncertainty column exists, annotates
MAE and RMSE, and — this is the useful bit — prints a warning **on the plot** if
the two levels reproduce different methods. A parity plot silently comparing PBE
against r2SCAN looks exactly like one comparing PBE against PBE.

### The periodic table

The natural display for element-level results, in the way a dot plot is the
natural display for differential expression.

```python
md.obs['low_energy'] = md.obs['energy_per_atom_emt'] <= \
    md.obs['energy_per_atom_emt'].median()
mv.tl.rank_elements_groups(md, 'low_energy')

frame = md.uns['rank_elements_groups']['True']
scores = frame.set_index('element')['log2_odds'].reindex(md.var_names)

mv.pl.periodic_table(md, values=scores.to_numpy(),
                     label='log2 odds, low-energy half', center=0.0)
```

A bar chart of 118 categories is unreadable and throws away the structure a
chemist reads a periodic table for — groups above one another, periods across.
`center=0.0` makes the diverging colour map symmetric about no enrichment.

```python
mv.pl.rank_elements_groups(md, group='True', n=10)
```

is the same result as a ranked bar chart, when you want the numbers rather than
the layout.

### The rest

```python
mv.pl.hull(md, level='emt', x='Al')       # labels itself if the hull is closed
mv.pl.pareto(md, 'e_above_hull_emt', 'density')
mv.pl.spectra(md, 'xrd', rows=[0, 1, 2], offset=30)
mv.pl.provenance(md)                       # the pipeline, from the object
```

`mv.pl.provenance` draws the operation history as a figure. It is taken from the
object rather than from a lab notebook, which is the reason for recording
provenance in the object at all.

```{note}
matplotlib is an optional dependency (`pip install "matverse[plot]"`), imported
inside each plotting function. Producing a number and drawing it are different
jobs, and a screening pipeline on a cluster should not need a plotting stack to
run.
```

## Next

[Scale and first principles](scale_and_dft.md) covers the other direction: what
to do when the candidate list is larger than memory, and how to hand the
survivors to DFT.

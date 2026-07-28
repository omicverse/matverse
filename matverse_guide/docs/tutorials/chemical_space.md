# Chemical space

This tutorial is about the one design decision that most distinguishes matverse
from a DataFrame with a `structure` column: **`X` is the composition matrix and
`var` is the periodic table**.

## Why elements are the right axis

A composition matrix is materials × elements — sparse, non-negative, and almost
entirely zero, because nearly every material draws on five or fewer of 118
columns. That is structurally a cells × genes matrix.

```python
import matverse as mv

md = mv.data.from_structures(candidates)   # from the screening tutorial

md.X.toarray()
# array([[0., 0., 1.],      Al
#        [1., 0., 0.],      Cu
#        [0., 1., 0.],      Ni
#        [1., 0., 3.],      CuAl3
#        [3., 0., 1.],      AlCu3
#        [0., 1., 1.]])     AlNi
list(md.var_names)
# ['Al', 'Ni', 'Cu']
```

Counts come from the **reduced** formula, so a supercell and its primitive cell
occupy the same point in chemical space. Cell size belongs in `obs['nsites']`,
not in the chemistry.

`var` carries the periodic table, which is what makes element-level results
interpretable without a lookup somewhere else:

```python
md.var[['Z', 'electronegativity', 'period', 'is_transition_metal']]
```

```{note}
Economic properties — price, supply risk, criticality — are deliberately **not**
shipped. They are jurisdiction- and date-dependent and no authoritative open
table exists, so inventing numbers would make `mv.screen.filter(md,
supply_risk__lt=0.3)` return confident nonsense. Attach your own:

```python
import pandas as pd
mv.elements.annotate(md, pd.DataFrame(
    {"price_usd_kg": [2.4, 18.0, 9.5]}, index=["Al", "Ni", "Cu"]))
```
```

## The dividend inside the library

Because `X` holds the composition and `var` holds the element properties, the
classic composition descriptor — the weighted mean, spread and range of element
properties across a formula — is a matrix product of two things the object is
already carrying:

```python
mv.feat.element_stats(md)
md.obsm['X_element_stats'].shape
```

For `CuAl3`, the weighted mean electronegativity is exactly
`0.75 × 1.61 + 0.25 × 1.90`. `min`, `max` and `range` are taken over the elements
that are present at all, so "the most electronegative element in this compound"
is the quantity it sounds like rather than an amount-weighted blend.

Composition descriptors cannot tell two polymorphs apart, because polymorphs have
the same composition. When that matters, reach for a structure descriptor:

```python
mv.feat.soap(md)          # needs matverse[descriptors]
```

## Mapping the space

```python
mv.pp.normalize_composition(md)     # -> layers['fraction']
mv.tl.pca(md, n_comps=2)
mv.tl.neighbors(md, n_neighbors=3)
mv.tl.cluster(md, method='kmeans', n_clusters=2)
```

`normalize_composition` writes atomic fractions to a **layer** rather than
replacing `X`, because a hull needs the counts and a chemical-space map needs the
fractions. `mv.tl.pca` defaults to the fraction layer when it exists: raw counts
scale with the size of the formula unit, so `Fe2O3` and `Fe4O6` would otherwise
sit at different points despite being the same chemistry.

PCA here is an exact SVD. The element axis is at most 118 columns wide, so there
is nothing to approximate and no randomised solver to configure.

## The question that follows every screen

You have a shortlist. What chemistry is on it?

```python
mv.tl.rank_elements_groups(md, 'passes')

md.uns['rank_elements_groups']['True']
```

```
  element  n_in_group  n_rest  frac_in_group  frac_rest  odds_ratio      pval
0      Al           3       0            1.0        0.0         inf  0.047619
1      Cu           2       1            0.667      0.5         2.00  1.000000
2      Ni           0       1            0.0        1.0         0.00  0.400000
```

This is `rank_genes_groups` with the nouns changed, and it is the operation that
justifies the whole design choice. Answering the same question from a DataFrame
of averaged Magpie descriptors means writing the contingency table and the test
by hand, every time, for every screen.

Two routes, because "which elements" has two meanings:

```python
mv.tl.rank_elements_groups(md, 'passes', method='presence')   # how often (Fisher exact)
mv.tl.rank_elements_groups(md, 'passes', method='fraction')   # how much (Wilcoxon)
```

Both report an uncorrected p-value and a Benjamini–Hochberg q-value. With six
materials nothing here is significant, and the table says so — which is the point
of reporting both.

## Is this candidate actually new?

```python
known = mv.data.from_mp({'chemsys': 'Al-Ni-Cu'})     # needs mp-api
mv.tl.novelty(md, reference=known)

md.obs[['novelty_distance', 'nearest_reference']]
```

```{warning}
Composition-space distance answers "is this chemistry new", not "is this
structure new". A novel polymorph of a known composition scores zero. For
structural novelty, concatenate against the reference and run `mv.pp.dedup`.
```

## Does the bet hold?

`X` as the composition matrix is a bet, and the test that would kill it is in the
suite:

```
tests/test_pipeline.py::TestTools::test_rank_elements_groups_recovers_the_obvious_chemistry
```

It groups materials by whether they contain Al and asserts that the top ranked
element is Al. If element-level enrichment ever stops recovering chemistry a
domain expert would recognise, the design falls back to the width-zero `X` of
v0.1 — `mv.data.from_structures(..., build_X=False)` — and nothing else changes.

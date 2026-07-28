# Beyond one number per material

The screening tutorial produced scalars: an energy, a distance above the hull, a
pass/fail. Plenty of results are not scalars. A diffraction pattern is a curve. A
force is one vector per atom, and the number of atoms differs from material to
material. A measurement is a number somebody took off an instrument rather than
out of a calculator.

This tutorial covers where each of those lives, and why.

## Curves live in `obsm` on a shared grid

```python
import matverse as mv

mv.prop.xrd(md, two_theta=(10, 90), step=0.02, fwhm=0.1)

md.obsm['xrd_calc'].shape        # (n_materials, n_grid_points)
mv.grid_of(md, 'xrd').shape      # the two-theta axis, stored once
```

pymatgen returns a peak list, and peak lists cannot be compared across materials
because no two share the same peak positions. Broadening each reflection onto a
common grid is what turns a set of patterns into a matrix — and a matrix is
something you can take a distance on, cluster, or subtract a measurement from.

The block is named `'<quantity>_<level>'`, exactly like a scalar:

```
obs['energy_pbe']        a number per material, level in the suffix
obsm['xrd_pbe']          a curve per material, level in the suffix
obsm['xrd_experiment']   the same curve, measured
```

```{note}
This is one convention, not two. An earlier draft of the design had curves using
an AnnData `layer` for the level of theory while scalars used a name suffix, and
listed the split as a wart. Putting curves in `obsm` removed it — and removed the
need for MuData along with it, since a `materials × grid` matrix is aligned to
the material axis and that is what `obsm` is for.
```

`mv.prop.rdf` is the other curve that ships, and it earns its place by doing
something composition cannot:

```python
mv.prop.rdf(md, r_max=10.0)
```

Two polymorphs have the same composition, so `X` cannot tell them apart and
neither can any composition descriptor. A radial distribution function can, and
unlike SOAP it needs no extra dependency.

## Per-atom results get their own axis

Forces are ragged. A six-material library with 4, 4, 4, 4, 4 and 2 atoms has 22
force vectors and no way to put them in a column of a six-row table.

```python
sites = mv.multi.sites(md)

sites
# AnnData object with n_obs × n_vars = 22 × 3
#     obs: 'material', 'material_index', 'site_index', 'element'
#     obsm: 'X_frac', 'X_cart'
```

One row per atom, with `obs['material']` pointing back at the parent. Per-atom
results are a matrix again:

```python
mv.calc.forces(md, sites, level='emt')

sites.obsm['forces_emt'].shape           # (22, 3)
sites.obs['force_magnitude_emt']
```

`mv.calc.forces` takes **both** objects, and the signature is honest about why:
forces need the structures, which live on the material axis, and produce one row
per atom, which lives on the sites axis.

```{warning}
The sites object is a snapshot of one structure variant. If you build it from
`'input'` and then ask for forces on `'relaxed_emt'`, the atoms will not line up
— so matverse refuses rather than silently misaligning them. Rebuild with
`mv.multi.sites(md, source='relaxed_emt')`.
```

### Getting back to the material axis

A per-site result is not screenable until it is summarised:

```python
mv.multi.aggregate(sites, md, 'force_magnitude_emt', how='max')

mv.screen.filter(md, force_magnitude_emt_max__lt=0.05)
```

The detail stays on the sites object where it can still be inspected; the summary
lands on the materials object where a screen can reach it.

### The element axis is shared

`sites.X` is the one-hot element indicator, so `sites.var` is the same periodic
table the parent carries. Everything written for element-level questions works on
atoms without modification:

```python
sites.obs['strained'] = sites.obs['force_magnitude_emt'] > 0.1
mv.tl.rank_elements_groups(sites, 'strained')
```

That is `rank_genes_groups` answering "which elements sit in the
highest-force environments", and it needed no new function.

### One object, if you want one

```python
mdata = mv.multi.to_mudata(md, sites)     # needs matverse[multi]
```

Optional throughout. matverse's operations take `AnnData`, and the sites object
is useful without ever being assembled.

## Experiment is a level of theory

This needed no new machinery, which is the argument for having typed the level of
theory in the first place.

```python
mv.exp.measure(md, 'band_gap', measured_values, instrument='UV-Vis')

mv.compare_levels(md, 'band_gap')
#      pbe   hse06  experiment
# 0   0.61    1.12        1.05
# 1   1.84    2.41        2.33
```

Three numbers, three levels, one table, and nobody had to decide which one is
"the" band gap. `uns['levels']['experiment']` records the instrument where
`uns['levels']['pbe']` records the functional.

### Phase identification

The compelling case is a measured pattern against a candidate library:

```python
mv.prop.xrd(md)
mv.exp.match_xrd(md, measured_intensity, measured_two_theta)

md.obs[['xrd_match', 'xrd_match_rank']].sort_values('xrd_match_rank').head()
md.uns['xrd_match']['best']
```

Both patterns are baseline-shifted and unit-normalised before the dot product, so
the score reflects peak positions and relative heights rather than exposure time.

```{warning}
`match_xrd` scores against the candidates in this object and nothing else, and
records that fact in `uns['xrd_match']['scored_against']`. A high score means
"the best of what you gave it", not "identified" — the true phase can be absent
from your library entirely.
```

### Attaching a full measured curve

```python
mv.exp.attach(md, 'xrd', measured_patterns, instrument_two_theta,
              instrument='Bruker D8')
mv.prop.compare_grids(md, 'xrd', 'calc', 'experiment')

md.obs[['xrd_cosine_calc_vs_experiment', 'xrd_overlap_calc_vs_experiment']]
```

Measurements are resampled onto the existing grid, because two curves on
different grids cannot be subtracted. Points outside the measured range become
NaN rather than zero, and the comparison runs over the overlap — a measurement
covering a narrower range than the calculation is the normal case, not an error.

```{warning}
Attach a measurement at its own resolution or better. Diffraction peaks are
narrow, and resampling onto a grid coarser than the peak width discards them
permanently — no later interpolation brings them back. There is a test in the
suite pinning this behaviour so nobody "fixes" it.
```

## Reconciling databases

If your candidates came from more than one source, their energies are not
directly comparable. Materials Project, OQMD and Alexandria differ in
pseudopotentials, cutoffs and correction schemes, and the resulting offsets have
compositional structure — which makes them a batch effect.

```python
mv.pp.harmonize(md, batch_key='database',
                energy_key='energy_per_atom_dft', reference='mp')

md.uns['harmonize']['offsets']['oqmd']      # one value per element
md.uns['harmonize']['diagnostics']['oqmd']
# {'n_anchors': 412, 'rmse_before': 0.087, 'rmse_after': 0.019, ...}
```

The model is the one the field already uses by hand — a per-element reference
offset — fitted by least squares on the compositions two databases share.

```{warning}
It cannot repair a disagreement that is not linear in composition. Two databases
that differ because one relaxed with a different functional will differ structure
by structure, and a compositional offset absorbs only the average of that. Read
`rmse_after`: it is what is left.

`harmonize` also warns and does nothing when the databases share no composition,
because with no anchors there is nothing to fit.
```

## Scoring generated candidates

```python
mv.gen.validate(candidates, reference=known, level='pbe')

candidates.uns['gen_validate']['rates']
# {'valid': 0.94, 'unique': 0.88, 'novel': 0.71, 'stable': 0.03,
#  'sun': 0.02, 'msun': 0.14}
```

Validity, uniqueness, novelty and stability use LeMat-GenBench's definitions
rather than a variant, and every parameter is recorded next to the rates:

```python
candidates.uns['gen_validate']['definitions']
```

That matters more than it looks. Until those definitions were pinned, the same
metric name meant different things depending on which reference set, stability
threshold and matching tolerance a paper used — so the numbers were not
comparable even when the names were identical.

```{warning}
Stability is reported as **not assessed** rather than zero when no level is given
or when the hull was built over the dataset's own compositions. A closed hull
cannot say whether anything is stable; check
`uns['gen_validate']['not_assessed']`.
```

Novelty means "absent from the reference set you named", which is a weaker claim
than it sounds. A 2026 stress test found that neither MatterGen nor DiffCSP++
recovered the experimentally observed structures of the newly synthesised
GdNiSn₄ and LuNiSn₄, despite both being built from known motifs — current models
recombine compositions within known structural families. A high novelty rate
measured against a database does not contradict that.

The baseline worth measuring against is cheap:

```python
candidates = mv.gen.substitute(md, {'Al': ['Ga', 'In']}, charge_balanced=True)
```

Substitution within a known structure type is what several generative models were
found to be doing implicitly.

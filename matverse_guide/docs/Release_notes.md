# Release notes

## v0.1.2

The second and third axes, measured data on the same footing as computed data,
and two things the design document had claimed but never built.

### Grid-shaped results, and one convention instead of two

A curve — a diffraction pattern, a density of states, a phonon spectrum — is
`materials × grid`. It goes into `obsm` as `'<quantity>_<level>'`, with the
shared axis recorded once in `uns['grids'][quantity]`.

That resolves an open problem rather than adding a feature. The design had
array-shaped results using a `layer` for the level of theory and scalars using a
name suffix, and listed the split as a wart forced by `obs` having no layers.
Putting grids in `obsm` means the suffix works everywhere: `obs['energy_pbe']`
and `obsm['xrd_pbe']` read the same way, and a measured pattern is
`obsm['xrd_experiment']` rather than a different kind of thing.

It also removed a dependency. Grids no longer need a modality, so they no longer
need MuData.

- `mv.prop.xrd` — powder patterns, broadened onto a common grid
- `mv.prop.rdf` — radial distribution functions, a structural fingerprint that
  separates polymorphs without needing dscribe
- `mv.prop.compare_grids` — cosine and RMSE between two levels, over the points
  where both are defined, with the overlap recorded

### The sites axis

Per-atom results — forces, charges, moments — are ragged: the number of atoms
differs per material. v0.1 stored them as records in `uns` and called it the open
design problem, which was accurate.

`mv.multi.sites(md)` returns a second `AnnData` whose rows are atoms, with
`obs['material']` as the foreign key. Per-atom results become a matrix again.

- `mv.calc.forces` writes `sites.obsm['forces_{level}']`
- `mv.multi.aggregate` reduces a per-atom column back onto the material axis, so
  a per-site result becomes something a screen can filter on
- `mv.multi.to_mudata` assembles both into one MuData, and nothing requires it

`X` on the sites object is the one-hot element, so `var` is the same periodic
table the parent carries and `mv.tl.rank_elements_groups` runs unchanged on
atoms — "which elements carry the largest forces" needed no new function.

### Experiment is a level of theory

This turned out to need no new machinery, which is the argument for having typed
the level of theory in the first place.

- `mv.exp.measure` — a measured scalar becomes a level, so
  `mv.compare_levels(md, 'band_gap')` puts PBE, HSE06 and the spectrometer in one
  table without anyone deciding which is the band gap
- `mv.exp.attach` — a measured curve, resampled onto the computed grid
- `mv.exp.match_xrd` — rank every candidate against one measured pattern. It
  records that it scored against this object's candidates and nothing else, so a
  high score reads as "the best of what you gave it" rather than "identified"

### `mv.pp.harmonize`

Formation energies from Materials Project, OQMD and Alexandria carry systematic
offsets from differing pseudopotentials, cutoffs and correction schemes. That is
a batch effect with compositional structure, and `harmonize` fits it the way the
field already does by hand — as per-element reference offsets, by least squares
on the compositions two databases share.

It recovers an injected offset exactly and drives the cross-database residual to
zero on synthetic anchors. It cannot repair a disagreement that is not linear in
composition, reports the residual so the size of what is left is visible, and
warns rather than silently doing nothing when the databases share no composition.

### `mv.gen`

- `mv.gen.validate` — validity, uniqueness, novelty and stability using
  LeMat-GenBench's definitions rather than a variant, with every parameter it
  used recorded in `uns['gen_validate']['definitions']`. Stability is reported as
  **not assessed** rather than zero when no level is given or when the hull was
  closed, because a hull over a dataset's own compositions cannot say whether
  anything is stable.
- `mv.gen.substitute` — element substitution enumeration with an optional
  charge-balance filter. Substitution within a known structure type is what
  several generative models were found to be doing implicitly, so it is the
  baseline worth beating.

### Registry

52 entries, 157 contract claims, **contract-verified rate 101/101**.

A second limitation of the contract vocabulary surfaced, and is recorded rather
than worked around: `mv.calc.forces` writes into the `sites` object passed as its
second argument, and `produces` describes slots on one object only. Those writes
are documented in prose and go unprobed. Together with the route-conditional
`requires` on `mv.tl.cluster`, that is two places the vocabulary has run out.

### Fixed

- `mv.prop.compare_grids` let a single undefined point turn a whole comparison
  into NaN. A measurement covering a narrower range than the calculation is the
  normal case, so comparison now runs over the overlap and records how many
  points it used.

### Known caveat

Attach a measurement at its own resolution or better. Diffraction peaks are
narrow, and resampling onto a grid coarser than the peak width discards them
permanently — no later interpolation brings them back. There is a test pinning
this so nobody "fixes" it.

## v0.1.1

A screening pipeline that works end to end, and the substrate decisions that
make the rest of the roadmap possible.

### The composition matrix

`X` is now materials × elements and `var` is the periodic table, built at
construction rather than deposited by a featuriser — AnnData ties `X`'s width to
`var`, so it cannot be widened in place, and composition is intrinsic to a
material rather than derived from it.

What it unlocks:

- `mv.tl.pca`, `neighbors`, `cluster` on chemical space
- `mv.tl.rank_elements_groups` — which elements distinguish one group of
  materials from another, the question that follows every screen
- `mv.tl.novelty` — distance to the nearest known composition
- `mv.feat.element_stats` as a matrix product of `X` and `var`, rather than a
  featuriser that re-derives the periodic table

`build_X=False` restores the width-zero `X` of v0.1.0.

### Two claims that had never held

Both were fixed by moving structures from `uns` to `obsm`, and both now have
tests.

- **Subsetting silently misaligned structures.** `uns` does not subset with the
  object, so `md[mask]` kept every structure while dropping rows and each
  surviving row pointed at the wrong one — the exact failure this substrate
  exists to prevent.
- **The object could not be written to `h5ad`.** A list of pymatgen objects in
  `uns` is not something anndata can serialise, so the central interoperability
  claim failed on contact. Structures are now JSON strings in an `obsm` frame,
  which is aligned by construction and writes without special handling.

### Levels of theory

`uns['calc']` became `uns['levels']` and gained three fields that 2026 made
load-bearing:

- `reference` — what the level reproduces. A model trained on OMat24 targets
  PBE+U, one trained on MatPES targets r2SCAN; `surrogate: True` alone no longer
  distinguishes them.
- `license` — MACE-MP and MACE-MPA are MIT, MACE-OMAT and MACE-MATPES are ASL and
  forbid commercial use, UMA's licence excludes several countries.
  `mv.calc.check_licenses(md)` reads it back.
- `uncertainty` — where `_std` came from. `mv.calc.committee` produces one and
  says plainly that it is uncalibrated.

`mv.thermo.hull` now raises `LevelMismatch` rather than building a hull from two
levels whose references disagree. `mv.compare_levels` lines one quantity up
across every level that computed it.

### The hull can be absolute

v0.1.0 built the hull over the dataset's own compositions, which makes
`e_above_hull` a statement about which candidate is lowest rather than whether
any is stable. `references=` now accepts competing phases — a list of entries or
another matverse object — and `mv.thermo.references_from_mp` fetches them.
`uns['phase_diagram']['closed_system']` records which kind of number you have,
and a closed hull warns.

Also new: `formation_energy_{level}` when elemental references are present, and
`decomposes_to_{level}` — *what* a material decomposes into, not just how far
above the hull it sits.

### New namespaces

- **`mv.pp`** absorbs `mv.struct` and adds `qc`, `filter_materials`,
  `filter_elements`, `normalize_composition`, `dedup`, `rattle` and `strain`.
  `mv.struct` remains as re-exports.
- **`mv.tl`** — the borrowed analysis layer described above.

`mv.pp.dedup` blocks on `(reduced formula, space group)` before running
`StructureMatcher` inside each block; an all-pairs comparison is quadratic and
unusable past a few thousand candidates.

`mv.screen` gains `pareto` for multi-objective screens, and `filter` now
excludes NaN — a candidate whose calculation failed to converge has not met the
criterion, and silently admitting it is how a broken run reaches a shortlist.

### The registry and its probe

39 entries, 115 contract claims, all carrying a description and a runnable
example. The registry is vendored into matverse rather than imported, so the core
dependency list stays at six packages.

Claims are verified by execution: `produces` by running the call and looking,
`requires` by deleting the slot and confirming failure. The current
**contract-verified rate is 68/68**, and four claims were deleted rather than
repaired when they failed:

| deleted claim | why |
|---|---|
| `feat.element_stats requires var['Z']` | takes whatever numeric columns `var` has |
| `thermo.hull requires levels[{level}]` | only read when `references=` is given |
| `tl.cluster requires obsp['connectivities']` | true of the leiden route, not kmeans |
| `pp.strain produces structures[{name}]` | template was unresolvable; the default is now a real value |

The third is a finding rather than a defect: `requires` has one field per
function, not one per dispatch route, so a route-conditional dependency is
expressible only in prose a tool cannot check.

### Fixed

- `mv.screen.pareto` reduced domination over the wrong axis, marking points that
  dominated others as dominated themselves.
- `mv.pp.standardize` no longer aborts a whole dataset when spglib fails on one
  disordered cell.

### Compatibility

- Requires Python 3.10.
- `uns['calc']` → `uns['levels']`; objects written by v0.1.0 need the key renamed.
- `uns['structures']` → `obsm['structures']`, serialised as JSON.
- `mv.feat.composition` is now `mv.feat.element_stats`, with the old name kept as
  an alias. It deposits `obsm['X_element_stats']`, not `obsm['X_composition']`.

## v0.1.0

The initial skeleton: six namespaces, 17 functions, and the two conventions that
survived — operations deposit rather than return, and a result carries its level
of theory in the slot name.

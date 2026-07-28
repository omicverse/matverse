# Release notes

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

# Release notes

## v0.1.12

### The tutorials are executed notebooks

Four more tutorials — screening, chemical space, beyond one number, models and
campaigns — were prose with unexecuted code fences. They are now notebooks that
run when the documentation is built, which means the code in them is **tested**
rather than merely written. A tutorial whose examples only ever lived in a
markdown fence rots silently; this one fails the build.

`_scripts/build_notebooks.py` builds all five and exits non-zero if any cell
raised. `scale_and_dft` stays prose deliberately: it is about corpora larger
than memory and jobs handed to VASP, and a notebook of it would be a page of
code nobody could execute.

Executing them found five things the prose had asserted and the code did not do.

**Forces on a perfect cubic cell are zero.** The per-atom section computed
forces on unrelaxed fcc, L1₂ and B2 cells and presented the result as ragged
per-atom data. Every value was exactly zero — by symmetry, not by accident. The
notebook now rattles the structures first, which is both what makes the example
work and what force-training sets for machine-learned potentials actually are.

**`harmonize` had nothing to fit on.** The database-reconciliation example split
one library into two halves by alternating rows, so the two "databases" shared
no composition and the fit was silently skipped. It now duplicates every
material into both databases, and the fitted offsets recover the ones put in to
within 4e-17.

**`compare_grids` reports a count, not a fraction.** The text called the
`overlap` column a fraction of the grid. It is the number of grid points both
curves cover.

**The leakage diagnostic did not say what the tutorial claimed.** The prose
asserted that holding out a structure type roughly doubles the error against a
random split. On a 28-material library it does the opposite, and the reason is
in the table: the `prototype` spread is exactly zero across three seeds, because
the library has two structure types and only one way to partition it. The
notebook now reads the table honestly — "too small and too homogeneous to run
this diagnostic" — rather than quoting numbers from a library that no longer
exists.

**EMT cannot find a stable alloy.** Its energy zero is the pure element at
equilibrium, so every intermetallic is above the hull by construction. The
screening notebook now says so, along with the reminder that L1₂ Al₃Ni being
unstable says nothing about Al₃Ni, which is a real phase in a different
structure.

### `mv.set_level`

Now exported. Every operation that computes at a level records one, but a user
who computes a column themselves had no supported way to type it — which made
the level system a thing the library did to you rather than a thing you could
use. `mv.compare_levels` works on the result like any other level.

### Fixes surfaced by executing the tutorials

- `mv.data.from_structures` and `mv.multi.sites` named their rows *after*
  handing the frame to AnnData, so every dataset construction emitted an
  `ImplicitModificationWarning` about coercing an integer index — and that
  warning landed in the output of the first cell of every notebook.
- `mv.opt.suggest` raised a bare `KeyError` naming the column when
  `predicted=` or `uncertainty=` named one that did not exist. It now says what
  the column was for and lists the available ones.
- The documentation pointed at `github.com/matverse/matverse`, which does not
  exist. Every "edit this page" and "download" button was dead.

## v0.1.11

### `mv.pl.set_style`

One call at the top of a notebook, and every later plot matches: resolution,
font size, spines, figure size. It touches only `rcParams`, so anything set
afterwards still wins and a figure built by hand is unaffected.

This is the counterpart of `ov.plot_set` in omicverse, and it exists for the
same reason — styling nine plots one at a time is how a notebook ends up with
nine different-looking plots.

### The getting-started notebook, rewritten

The first version read as an essay with code in it. It now follows the shape
omicverse's notebooks use, which readers of that ecosystem already know:

- a title, then the scientific context, with the structures **cited** — the
  olivine cathode is [Padhi et al. (1997)](https://doi.org/10.1149/1.1837571),
  not an anonymous CIF;
- a first cell that is imports plus `mv.pl.set_style()`;
- an explicit *Loading a dataset* section naming where the data comes from,
  rather than a call whose provenance the reader has to take on trust;
- short cells, mostly one operation, ending with the object so it renders;
- brief markdown between cells saying what the next step is for.

The pipeline it runs is unchanged, and it is still executed rather than
illustrative.

## v0.1.10

### `mv.datasets` — real materials to work on

Every tutorial built its structures in code, which meant the documentation
taught the library's own test fixtures rather than the domain. This ships real
published structures instead: LiFePO₄ as it is actually reported in P2₁/c,
Li₁₀GeP₂S₁₂ as the 58-atom solid electrolyte people actually study.

- `load` — five curated sets, each a coherent scenario rather than a grab-bag,
  and each saying what it is *for*: `battery_cathodes`, `solid_electrolytes`,
  `oxides`, `semiconductors`, `simple`.
- `metals` — the seven fcc metals from published room-temperature lattice
  parameters. EMT is the only calculator that ships working and is parameterised
  for exactly these plus H, C, N and O, so they are the materials on which every
  example runs end to end without downloading a model.
- `fetch` — Materials Project or any OPTIMADE provider, cached to disk so the
  second call is free. Honours `MATVERSE_DATA`, which is the one to set on a
  cluster: a home directory is small, NFS-backed and shared, and a downloaded
  corpus does not belong there.

The bundled structures are **read from the set pymatgen already ships** rather
than re-distributed. pymatgen is a hard dependency so they are always present,
which keeps the package small and the provenance of the crystallographic data
clear.

### A notebook that actually runs

`tutorials/getting_started.ipynb` is generated by a script and **executed**, so
the outputs in the documentation are real rather than illustrative — myst-nb is
configured with `execution_mode = "off"`, and a notebook committed without
outputs renders as a page of code and no results.

It loads real cathodes, computes their diffraction patterns, feeds one back as a
measured pattern and identifies it, then switches to the metals for the
calculator path — because the cathodes contain Fe, P and V, which EMT is not
parameterised for. That constraint is in the notebook rather than designed
around, since it is the constraint a reader will hit.

## v0.1.9

Tier 2 of the gap analysis: the four capabilities the survey called important
rather than essential. **115 functions across 20 namespaces.**

### `mv.mag` — magnetic order, and why a hull needs it

Every energy computed before this assumed the magnetic configuration handed to
the calculator was the right one. For anything containing iron, cobalt, nickel,
manganese or chromium that is usually wrong, and the ferromagnetic and
antiferromagnetic states can differ by hundreds of meV per atom — so a hull
built from whichever ordering the input file carried is a hull of the wrong
quantity, systematically.

`orderings` enumerates the candidates, `ground_state` collapses them onto the
parent keeping the lowest, and `magnetic_spread` records how far apart they
were. That last number is the one people skip and the one that says whether the
guess would have mattered.

pymatgen's antiferromagnetic strategies call out to **enumlib**, Fortran
executables that are not pip-installable and absent from most environments —
and it raises rather than returning the ferromagnetic state it could have
produced. matverse falls back to generating sign assignments on the magnetic
sublattice directly, which is coarser and dependency-free, and records that it
did.

The winning energy is written under the ordinary `energy_per_atom_<level>` name,
so `mv.thermo.hull` needs no special case — it sees a normal column that happens
to be the magnetic ground state.

### `mv.prop.thermal_conductivity`

κ_L by the Slack model from the phonon and elastic data already computed, with
the Debye temperature and Grüneisen parameter it needs. Debye temperatures land
within about 15% of the literature on fcc metals — Au 161 against 165, Ag 239
against 225 — and the ordering is right.

An order-of-magnitude model, not a solution of the Boltzmann transport equation,
and it says so. Matbench Discovery weights thermal conductivity at 40% of its
combined score precisely because it is the property cheap methods get wrong, so
a screen ranked on this is a shortlist for phono3py rather than an answer.

### `mv.dft.read_dos`

The density of states onto a Fermi-referenced grid, plus the scalars a screen
filters on: gap, whether it is direct, band edges, and the density of states at
the Fermi level. Cheap, because `mv.dft` already parses the same files.

The level records that a PBE gap is roughly half the experimental one — which is
why `is_metal` from a PBE run is trustworthy while a small gap is not. PBE turns
narrow-gap semiconductors into metals, so the false negatives all point the same
way.

### `mv.thermo.defect_formation`

Defect enumeration existed; the thermodynamics did not. Formation energy depends
on where the Fermi level sits and on the chemical potential of whatever was
added or removed, so it is a **line rather than a number** — stored on the grid
axis against the Fermi level, with the lowest charge state at each point giving
the stable charge.

Two things it refuses to do quietly. Without a chemical potential it produces
NaN and warns, because what an added or removed atom costs is not derivable from
the defective cell alone. And no image-charge correction is applied — a charged
defect in a periodic cell interacts with its own images — so the result is
declared uncorrected and points at doped and pydefect for the real thing.

### Registry

115 entries, 354 contract claims, **contract-verified rate 141/141**.

## v0.1.8

The three capabilities a 2026 survey of the field named as must-haves and this
library did not have. **107 functions across 19 namespaces.**

### `mv.md` — molecular dynamics

Everything before this described a structure sitting still at zero kelvin.
Diffusion, ionic conductivity, thermal expansion, melting and the structure of a
glass are all properties of a system that is moving, which is why this is the
module the rest of the missing half hangs off.

- `run` — NVT or NPT at one temperature, depositing mean energy, mean
  temperature, MSD and diffusivity. **Trajectories are not stored**: observables
  are computed as the run goes, because a library that materialises a trajectory
  per candidate stops working at a few hundred materials.
- Per-element diffusivity goes to a **layer**, since it is materials × elements
  — the same shape as `X`. For an ionic conductor the number that matters is the
  mobile species' diffusivity; averaging lithium with its framework describes
  nothing.
- `sweep` — a temperature series stored on the **condition axis**, using the same
  grid mechanism as a diffraction pattern. Nothing new was needed to express
  "the same property at different conditions". Gives thermal expansion and an
  Arrhenius activation energy.
- `conductivity` — Nernst–Einstein, with its assumption stated.
- `melt_quench` — with the fixed-volume quench that survives the 2026 finding
  that all eight tested universal potentials produce catastrophically
  under-dense amorphous structures under a naive NPT quench.

**The thermostat now tells you when it failed.** A weakly coupled thermostat
takes tens of picoseconds to reach its target, so a short run samples a
temperature that is not the one requested — 69 K for a requested 300 K in one
test. The achieved temperature was always recorded; a run that misses by more
than 20% now warns, because every observable belongs to the temperature actually
reached.

### `mv.neb` — migration barriers

`hop_endpoints` builds the endpoints for a vacancy-mediated hop; `barrier` runs
climbing-image NEB and records the forward and reverse barriers, the reaction
energy, and the energy profile on the grid axis.

Validated against physics: vacancy migration in fcc copper comes out at
**0.762 eV against a literature ~0.70**, the forward and reverse barriers match
for a symmetric hop, and the profile is a clean symmetric arc.

Two things had to be right for that, and both were wrong first:

- **The destination must be the nearest periodic image of the vacancy**, not its
  stored coordinates. Those differ whenever the shortest hop crosses a cell
  boundary, and using the stored ones sent the atom the long way round — a
  2.55 Å hop became a 7.66 Å traverse, and the NEB measured the cost of dragging
  an atom through the lattice. `hop_endpoints` now verifies the displacement it
  built matches the hop it reported.
- **Interpolation defaults to IDPP, not linear.** A straight-line path drives the
  hopping atom through its neighbours; the band starts with overlapping atoms and
  converges, if at all, on several eV of repulsion.

Every surrogate level records the caveat that machine-learned potentials soften
the potential energy surface and under-predict barriers — roughly 0.07–0.08 eV
MAE without transition metals, about 0.20 eV with them. The error has a sign, so
a screen ranked on these promotes candidates DFT would reject.

### `mv.surf` — surfaces, shapes and adsorption

This used to sit outside a screening library because a surface energy meant a
DFT slab per facet. Universal potentials changed the arithmetic.

- `slabs` — every distinct low-index facet and termination, returned as a new
  dataset with `obs['parent']` back to the material.
- `surface_energy` — requires the bulk object rather than guessing it, because
  a wrong reference shifts every value by a constant: invisible in a ranking,
  fatal in a Wulff construction. Recovers the literature ordering for copper,
  (111) < (100) < (110).
- `wulff` — the equilibrium crystal shape, with per-facet area fractions on the
  facets and the shape summary on the material.
- `adsorption_sites` / `adsorption_energy` — enumerate atop, bridge and hollow
  rather than guessing one, relax all, keep the minimum. That is the AdsorbML
  protocol, and it reproduces hollow-binds-strongest for oxygen on Cu(111). The
  adsorbate reference is a required argument with no default, since an energy
  against half an H₂ and one against an isolated H atom differ by about 2.3 eV
  and both conventions are in use.

### Fixed

`mv.calc.relax` always wrote `relaxed_<level>`, the same name every time, so
relaxing two variants at one level silently overwrote the first. NEB needs two
relaxed endpoints, which is how it surfaced. `key_added` now names the output.

Miller indices were spelled two ways inside `mv.surf` — `'111'` on the facets and
`'1.1.1'` in the Wulff summary. Both are now `'1_1_1'`, with underscores because
a bare join makes `(1, -1, 1)` into `'1-11'`.

## v0.1.7

### Fixed

`mv.data.from_ase` put per-atom arrays — magnetic moments, charges, tags — into
`uns['sites']` as one record per material. That was wrong twice over, and the
two failures are the same two the substrate has taught before:

- **It did not subset.** `uns` is untouched by `md[mask]`, so filtering a dataset
  left every surviving row pointing at another material's atoms.
- **It did not save.** A list of dicts is not writable to `h5ad` at all, so any
  object built with `from_ase` failed on `write_h5ad`.

No test caught it because none combined `from_ase` with a save or a subset.
Three now do.

Per-atom arrays are now attached as **site properties on the structure itself**.
They travel with it, serialise with it, and appear as columns the moment someone
calls `mv.multi.sites` — no extra plumbing, and alignment is automatic rather
than maintained. The function's note about a site axis being "the open design
problem" was also stale: `mv.multi.sites` shipped in v0.1.2.

### Documented

The Developer guide gains the rule that decides where a result lives: **put it
in `uns` only if it is not aligned to an axis.** `uns` is a first-class part of
the model and holds much more than it gets credit for — scalars, arrays,
DataFrames and nested dicts of those all serialise — and matverse uses it
heavily and correctly for levels, grids, screens and provenance, none of which
are per-material. What it will not do is subset, so anything of length `n_obs`
belongs in `obs`, `obsm` or `obsp`.

## v0.1.6

The last concrete gaps the roadmap named. **96 functions across 16 namespaces.**

- **`mv.dft` is no longer VASP-only.** `code='espresso'` writes Quantum ESPRESSO
  input. Pseudopotentials are named, not shipped: which set a run used is part of
  the level of theory — SSSP and PSLibrary disagree for the same functional — so
  guessing a filename would put a silent choice into a result the object claims
  to record.
- **`mv.pp.defects`** enumerates vacancies and substitutions in a supercell,
  one per symmetry-inequivalent site. Without that deduplication a 32-atom
  elemental supercell yields 32 identical vacancies and wastes a calculator on
  31 of them. Defect *formation energies* need charge states, a chemical
  potential and a finite-size correction, which is what doped and pydefect exist
  for; this builds the structures.
- **`mv.thermo.pourbaix`** gives distance from aqueous stability at a pH and
  potential. A material on the solid-state hull can still dissolve, which is why
  it is a separate question rather than a column of the same one. Needs
  Materials Project's fitted ion energies and cannot be computed from a
  candidate set alone.
- **`mv.feat.embed`** takes a pretrained model's latent vectors, through the
  same registration interface as `mv.calc.register_calculator`. matverse ships
  no embedder — weights are hundreds of megabytes with their own licences — and
  the docstring notes the 2026 finding that plain CGCNN and ALIGNN generalise
  better out of distribution than fine-tuned foundation embeddings.

### Fixed

`mv.pp.defects` silently skipped its own symmetry deduplication.
`SpacegroupAnalyzer` was imported inside the calling function, so the helper
raised `NameError`, hit a bare `except`, and fell back to enumerating every
site. The test that caught it asserts the physics — one distinct vacancy in an
elemental fcc supercell — rather than a stored count.

## v0.1.5

Working past memory.

### Scale

Alexandria is 5.06M entries and OMat24 is roughly 110M calculations. A
constructor that takes a list rules both out before anything else does.

- **`mv.data.from_iterable`** builds a dataset from a stream in blocks,
  concatenating as it goes. The element axis is unioned across blocks, so a block
  containing an element no earlier block had widens the axis rather than failing.
- **`mv.data.from_ase_file`** reads extended-XYZ and ASE databases with ASE's own
  iterator, so a hundred-million-frame training corpus can be sampled without
  being read.
- **`mv.utils.map_chunks`** applies an expensive operation block by block and
  merges back the obs columns, feature blocks and structure variants each block
  produced. `skip_if=` skips blocks already finished, so a re-run after a killed
  job continues rather than restarts, and `checkpoint_to=` writes between blocks.
- **`mv.structures(md, rows=...)`** decodes a window rather than everything.
  Decoding is what costs at scale — five million serialised structures are a few
  gigabytes of strings and several times that as objects.

`map_chunks` deliberately does **not** merge `uns`. A per-block `uns` entry is a
statement about that block, and quietly keeping the last one would be wrong: a
screen's criteria and a hull's reference count both mean something different per
block than they do overall.

This is chunking, not laziness. The object is still materialised; a zarr-backed
`obs` with on-demand structure resolution is the next step.

## v0.1.4

First principles at the boundary, phase equilibria beyond the hull, and lattice
dynamics. **88 functions across 16 namespaces**, contract-verified rate 142/142.

### `mv.dft` — inputs out, results back in

matverse runs no DFT and submits no jobs. Workflow management has three good
answers already — atomate2 with jobflow-remote, quacc, AiiDA — and a fourth would
be a liability.

What is not solved is the boundary: a screen lives in one object, DFT lives in a
directory tree, and the correspondence is normally maintained by a naming
convention and someone's memory.

```python
mv.dft.write_inputs(md, 'runs/', preset='relax')   # one directory per row
# ... sbatch, atomate2, quacc, a week ...
mv.dft.read_outputs(md, 'runs/', level='pbe')      # back onto the same rows
```

Each directory carries a manifest recording which row it came from, so a
directory renamed by a workflow manager still resolves — the usual reason a
hand-rolled harvest attaches results to the wrong material. Presets (`relax`,
`static`, `bands`, `scan`, `hse`) record what each reproduces, so a run tagged
`scan` arrives as r2SCAN and `mv.thermo.hull` will not mix it with PBE.

Rows whose run is missing or unconverged get NaN and a reason rather than being
dropped: which candidates failed is a result, and a systematically failing corner
of composition space is worth seeing.

### `mv.prop.phonon` — the check a hull cannot make

Gamma-point frozen phonons on a supercell, giving the phonon density of states on
a shared frequency grid, the zero-point energy, and a count of imaginary modes.

A composition can sit on the convex hull and still be a structure that will not
hold together. Copper's stable phase is fcc; bcc copper has the same composition
and is dynamically unstable, and only a phonon calculation says so. Generated
structures fail this far more often than they fail the hull.

`mv.prop.free_energy` derives the harmonic vibrational free energy, entropy and
heat capacity from that DOS, which is where a hull built at 0 K starts to become
a hull at temperature.

Both are validated against physics rather than against a stored number: the
zero-point energy of copper (0.029 eV/atom against a literature 0.03) and the
Dulong–Petit limit (heat capacity converging to 3k_B per atom above the Debye
temperature). A regression test that compares only to last week's output cannot
tell a refactoring from a sign error.

### `mv.prop.elastic`

Elastic stiffness by finite strains with Voigt–Reuss–Hill moduli, Poisson ratio
and a Born stability flag.

### `mv.thermo` — beyond the hull distance

- `reaction` balances a reaction between compositions in the dataset and computes
  its energy. A reaction energy is not a synthesis route: it says a product is
  downhill, not that anything gets there.
- `chempot_limits` reports the chemical potential window over which each stable
  phase remains on the hull — the conditions it could be grown under — and says
  plainly when a closed hull makes that window a statement about the dataset
  rather than about chemistry.

### Registry

88 entries, 250 contract claims, **contract-verified rate 142/142**.

## v0.1.3

The namespaces the design named and the library did not have: plotting,
supervised models, design campaigns, and the infrastructure underneath them.
**80 functions across 15 namespaces**, all decorated, all probed.

### `mv.pl` — plotting

Every function draws onto an axis and returns it, and none calls `plt.show`; a
library that shows figures cannot be used to build one.

- **`periodic_table`** — the display for `rank_elements_groups`, in the way a dot
  plot is the display for differential expression. A bar chart of 118 categories
  is unreadable and throws away the structure a chemist reads a periodic table
  for.
- `rank_elements_groups`, `hull` (which labels itself when the hull is closed),
  `parity` (with error bars when an uncertainty is recorded, and a warning on the
  plot when the two levels reproduce different methods), `pareto`, `embedding`,
  `spectra`, `provenance`.

### `mv.model` — supervised prediction, honestly split

A prediction is a level of theory: it gets a record saying what it was trained
on and how it was split, so a predicted number and a DFT number cannot be
averaged together by accident.

**`mv.model.split` defaults to grouping by composition.** Random train/test
splits are the field's most common silent methodological failure — a materials
dataset is full of near-duplicates, so a random split puts relatives on both
sides and reports a number that will not survive a genuinely new material.
Strategies: `composition`, `prototype` (anonymised formula plus space group),
`element` (hold out everything containing one element), and `random`, which is
recorded as `leaky: True`.

**`mv.model.cross_validate`** scores under several strategies at once and reports
`leakage_mae` — the gap between the grouped number and the random one. On the
test library that gap is 1.7, which is the size of the leak a random split would
have hidden.

### `mv.opt` — design campaigns

Pool-based active learning, which is the right shape for materials: the search
space is a list of structures, not a box of real numbers.

`start` / `suggest` / `observe` / `history`, with greedy, uncertainty, UCB,
expected-improvement and random acquisition. Batches can be diversified by
farthest-point selection, because ten highest-scoring candidates are often ten
variations on one idea and computing all ten answers one question.

Methods needing an uncertainty **refuse** when none exists rather than pretending
sigma is zero.

### `mv.utils` — units, checkpoints, cluster

- `convert` / `set_units` / `check_units`. eV, meV, kJ/mol, kcal/mol, Rydberg,
  Hartree; angstrom, nm, pm, bohr. Conversions deposit beside the original
  rather than overwriting, and `check_units` fills in what matverse itself
  produced.
- `resume` reports which rows an operation has not filled, so a screen killed by
  a walltime limit continues rather than restarts. `checkpoint` writes and
  records.
- `slurm_script` writes a batch script rather than submitting it — submitting is
  a side effect on a shared machine.
- `summary` renders what an object contains, including warnings about a closed
  hull or a non-commercial level.

### `mv.prop.elastic`

Elastic stiffness by finite strains at any level, with Voigt–Reuss–Hill bulk,
shear and Young's moduli, Poisson ratio, and a Born stability flag. On relaxed
EMT metals it recovers the right ordering (Ni > Cu > Al).

### `mv.data` — OPTIMADE

`from_optimade` queries any OPTIMADE-compliant provider with one filter
expression; eight base URLs ship, and any endpoint works via `base_url=`. One
protocol against roughly twenty providers beats twenty bespoke clients, which is
why this is now the primary connector.

`from_optimade_response` parses an already-fetched payload — separate on purpose,
because parsing is deterministic and testable while fetching is neither.

### Registry: derived names versus claimed ones

Adding `mv.pl` surfaced a real distinction the registry had collapsed.
`mv.pl.hull` mirroring `mv.thermo.hull` is a convention worth keeping, the way
scanpy pairs `pl` with `tl` — but both derive the bare name `hull`.

The rule is now:

- an **explicit alias** is a claim, and two functions claiming one still raises;
- a **derived name** is not, so an explicit alias outranks it, and two derived
  names that collide **withdraw** the key from exact lookup rather than awarding
  it to whichever module imported first.

`mv.describe('rank_elements_groups')` now answers "names more than one function;
say which", which is the honest response to an ambiguous question.

### Fixed

- A list of dicts in `uns` cannot be written to `h5ad` — anndata turns it into an
  object array and h5py refuses it. `uns['checkpoints']` and the campaign's
  rounds now use an ordered, writable record store. This is the same class of
  failure as structures in `uns` in v0.1.1, found the same way: by trying to
  save.
- `mv.model.cross_validate` swallowed fit failures and returned an empty table,
  which reads as "the model scored nothing" rather than "nothing was fitted". It
  now re-raises the underlying error.
- `mv.prop.compare_grids` gained an overlap count alongside cosine and RMSE.

### Registry

**80 entries, 223 contract claims, contract-verified rate 128/128.** Four claims
were adjusted rather than kept when probing showed them unresolvable or untrue,
including two on `mv.opt` whose slot names are chosen at `mv.opt.start` and so
have nothing in a later call's arguments to resolve against — recorded in the
function's notes as a third place the contract vocabulary runs out.

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

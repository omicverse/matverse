# matverse — design

*Status: proposal. Written against the v0.1 skeleton (6 namespaces, 17 functions,
~716 lines) and a July 2026 survey of the materials software ecosystem.*

This document answers two questions: **what shape should the object be**, and
**what should the library contain**. It is opinionated, and it says what
matverse should *not* build at least as often as what it should.

---

## 1. What this is for

matverse is for **populations of materials**. Thousands to millions of
candidates, each carrying structures, descriptors, computed properties at
several levels of theory, and eventually measurements — held in one object that
subsets, saves and reloads without losing the correspondence between them.

It is not a replacement for pymatgen (structure algorithms), ASE (atoms and
calculators), atomate2 (DFT workflow submission), or the MLIP trainers. It is
the layer above them that most materials projects currently reimplement as a
folder of scripts and a DataFrame of dicts.

The honest scope for v1 is **crystalline inorganic solids**. Molecules,
polymers, amorphous phases, composites and mesoscale structure are a different
axis choice and should not be promised. (Meta's 2025–26 releases went sideways
into exactly those domains — OMol25, OMC25, OPoly26 — so the expansion path
exists; it is a v2 decision, not a v1 hedge.)

**The competitive claim.** JARVIS-Tools is the closest existing attempt at
unification, and pymatgen+ASE+matminer is the de facto stack. What none of them
has is a *canonical dataset object*: an in-memory and on-disk representation
that survives subsetting, merging, and a year on disk. That is the wedge.
Everything else in this document follows from it.

---

## 2. The substrate

### 2.1 Three axes, not one

The v0.1 README already names the central problem and is right about it: "rows
are materials" suits screening and suits single-system detail much less well.
The fix is not to abandon the materials axis but to admit that a materials study
has **three** natural axes, and to give each one a real home.

| Axis | Rows are | Lives in | Example content |
|---|---|---|---|
| **materials** | one candidate | the primary `AnnData` | formula, spacegroup, energy, e_above_hull, descriptors |
| **sites** | one atom in one material | a modality with its own `obs` | forces, Bader charges, magmoms, SOAP, coordination number |
| **grid** | one point on a shared axis | one modality per grid | XRD 2θ, DOS energy, phonon frequency, wavelength, PDF *r* |

`MuData` holds all three. The materials axis is the primary `obs`; the sites
modality carries a `material_id` foreign key; each grid modality is a
`materials × grid` matrix sharing the primary `obs`.

```
mdata
├── obs                          n_materials — the primary axis
├── mod['struct']  (AnnData)     X = composition,   var = elements
├── mod['sites']   (AnnData)     obs = atoms,       obs['material_id'] → parent
├── mod['xrd']     (AnnData)     var = 2θ grid,     layers = {'pbe', 'experiment'}
├── mod['dos']     (AnnData)     var = energy grid, layers = {'pbe', 'hse06'}
└── mod['phonon']  (AnnData)     var = frequency grid
```

Two things fall out of this that are worth the complexity:

- **Per-site descriptors and per-site results share an axis.** SOAP vectors are
  per-site, and so are the forces and charges an MLIP returns. Putting them on
  the same axis means a training set for a fitted potential is a subset
  operation, not a script.
- **A measured spectrum and a computed one become two layers of one matrix.**
  `mod['xrd'].layers['experiment']` against `mod['xrd'].layers['pbe']` is the
  compute-versus-experiment comparison that the field currently does by
  exporting both to CSV.

Trajectories (relaxation paths, MD) deliberately do **not** get an axis. OMat24
alone is ~110M frames; these belong in a zarr store referenced from `uns`, read
lazily, never materialised into the object.

### 2.2 `X` should be the composition matrix

v0.1 leaves `X` empty and says so on purpose. That was the right call for a
skeleton and is the wrong call for the library, because it throws away `var` and
the entire borrowed toolchain along with it.

**Elements are the genes of materials.** A composition matrix is
`n_materials × n_elements`, sparse (almost every material uses ≤5 of 118),
non-negative, and count-like. That is structurally identical to a cells × genes
matrix, and the identity is not a cute analogy — it makes real operations work
without being rewritten:

| scanpy operation | materials meaning |
|---|---|
| `normalize_total` | atom counts → atomic fractions |
| `filter_genes` | drop elements absent from this candidate library |
| `highly_variable_genes` | elements that actually vary across the library |
| `pca` / `neighbors` / `umap` / `leiden` | a chemical-space map of the library |
| **`rank_genes_groups`** | **which elements are enriched in the stable cluster** |
| `ingest` / label transfer | project new candidates onto a reference database |
| **`harmony` / batch correction** | **reconcile MP vs OQMD vs Alexandria energies** |

The last row is the one to build first, and it is not a stretch. Formation
energies from MP, OQMD and Alexandria carry systematic offsets from differing
pseudopotentials, cutoffs and correction schemes. The field currently handles
this with hand-fitted elemental reference corrections. **It is a batch effect,
and the LeMat-Bulk effort (5.3M materials deduplicated across all three) exists
because nobody had a principled way to do it.** `mv.pp.harmonize(mdata,
batch_key='database')` is a genuine contribution, not a wrapper.

And `var` becomes the periodic table — one row per element, with columns for
electronegativity, radius, group, period, mass, natural abundance, **price,
supply risk and toxicity**. Those last three make `mv.screen.filter(mdata,
supply_risk__lt=0.3)` a one-liner, which no current package supports well and
every industrial screen needs.

If this bet fails — if composition-space clustering turns out to carry no signal
worth the coupling — the fallback is v0.1's empty `X`, and nothing else in the
design changes.

### 2.3 Level of theory is the type system

v0.1's convention (`obs['energy_pbe']` + `uns['calc']['pbe']`) is the best idea
in the skeleton and should be extended rather than merely kept.

**Where results are array-shaped, the level is a `layer`, not a name suffix.**
`mod['dos'].layers['pbe']` and `mod['dos'].layers['hse06']` are the same shape
and different quantities — exactly what layers are for. Scalars stay as
suffixed `obs` columns because `obs` has no layers. Two conventions, but each
one is forced by the container, and the boundary is mechanical: array → layer,
scalar → suffix.

**`uns['levels'][level]` needs more fields than v0.1 gives it.** The 2026
ecosystem makes three of them load-bearing:

```python
uns['levels']['mace-omat-0'] = {
    'kind':       'mlip',            # dft | mlip | classical | experiment | model
    'method':     'MACE',
    'checkpoint': 'MACE-OMAT-0',
    'reference':  'PBE+U (OMat24)', # what it was trained to reproduce
    'surrogate':  True,
    'license':    'ASL',             # ← non-commercial. See §6.3
    'uncertainty': 'ensemble',       # how obs['energy_*_std'] was produced
}
```

`license` is there because model weights in 2026 are not uniformly open. MACE-MP
and MACE-MPA are MIT; MACE-OMAT-0 and MACE-MATPES are ASL (non-commercial); UMA
is under a licence excluding several countries. A screening library that
silently produces a commercial result with a non-commercial checkpoint is a
liability, and the object is the right place to record it.

`reference` matters because "surrogate: True" is no longer enough information.
An OAM-trained model reproduces PBE+U; a MatPES-trained one reproduces r2SCAN.
Mixing them is the same error as mixing PBE with HSE, one level up.

**Experiment is a level of theory.** `uns['levels']['experiment'] = {'kind':
'experiment', 'instrument': ..., 'uncertainty': 'measured'}`. This is nearly
free — the machinery already exists — and it is what lets a measured band gap
and three computed ones sit in one object without anyone deciding which is "the"
band gap. It is also the hook §5 hangs the closed loop on.

**Uncertainty is a first-class output, not an afterthought.** Every `mv.calc`
entry point that can produce a variance deposits `obs['energy_{level}_std']`.
Active learning (§4.4) is unbuildable without it, and it is the single most
common thing screening pipelines drop on the floor.

---

## 3. Namespaces

Two organising principles, applied where each fits. Where the data is
matrix-shaped, borrow scanpy's shape (`pp`/`tl`/`pl`) so anyone from scverse is
immediately oriented. Where the data is physics, name it after the physics,
because materials scientists think in domains.

### Substrate

**`mv.data`** — get data in and out.
- OPTIMADE as the primary connector (one protocol, ~20 providers) rather than N
  bespoke API clients. Bespoke clients only where the payload is richer than
  OPTIMADE exposes: `from_mp` (note: MP moved to Delta tables on S3 in the
  v2026.04.13 release, `mp-api ≥ 0.46.2` returns a PyArrow-backed `MPDataset` —
  this breaks pre-2026 ingestion code and is an *opportunity*, see §5),
  `from_jarvis`, `from_alexandria`.
- `from_lemat_bulk` — 5.3M materials already deduplicated across MP/OQMD/
  Alexandria with PBE/PBEsol/SCAN. The best single ingestion target that exists.
- Files: CIF directories, POSCAR, extxyz, ASE `.db`, LAMMPS data.
- Training corpora as first-class: MPtrj, OMat24, sAlex, MatPES. These are how
  people fine-tune, and they arrive as trajectories, so they exercise §2.1's
  lazy path immediately.
- Round trips: `to_ase`, `to_pymatgen`, `to_matminer`, `to_h5mu`.

**`mv.pp`** — everything that happens before analysis. Absorbs v0.1's `struct`.
- `standardize` (primitive/conventional/Niggli, spglib), `symmetrize`
- `qc` — the exact analogue of `scanpy.pp.calculate_qc_metrics`: nsites,
  density, min interatomic distance, is_ordered, partial occupancy, charge
  neutrality, spacegroup. Then `filter_materials` / `filter_elements`.
- `dedup` — structure matching. Must not be O(n²) `StructureMatcher`; needs a
  fingerprint plus locality-sensitive hashing to survive 10⁶ candidates.
- **`harmonize`** — cross-database energy reconciliation (§2.2). Build early.
- Derivative structures: `supercell`, `slab`, `interface`, `strain`,
  `defects` (vacancy/substitution/interstitial enumeration), `sqs` (disorder),
  `rattle` (perturbation for MLIP training sets).

**`mv.feat`** — descriptors into `obsm`.
- Composition: Magpie-style element statistics, oxidation-state features,
  learned element embeddings (mat2vec, universal atomic embeddings).
- Structure: SOAP / ACSF / MBTR via **dscribe** (actively maintained), Voronoi,
  CrystalNN fingerprint, RDF/ADF, Valle-Oganov.
- Learned: latent vectors pulled from a pretrained MLIP. This is the
  scGPT/Geneformer analogue and should be a one-liner —
  `mv.feat.embed(mdata, level='sevennet-omni')`.
- **matminer is optional, not foundational.** Last release April 2024; treat as
  maintenance-mode. Delegate to it when installed, never depend on it.

**`mv.tl`** — the borrowed layer. `pca`, `neighbors`, `umap`, `leiden`,
`rank_elements_groups`, prototype/structure-type assignment, novelty scoring
against a reference database (distance to nearest known material — the honest
version of "is this new?"), Pareto fronts.

**`mv.pl`** — plotting with publication defaults. Convex hull, phase diagram,
band structure + DOS, phonon dispersion, XRD overlay (computed vs measured),
parity plots *with* uncertainty, Pareto front, chemical-space UMAP, Wulff shape,
Pourbaix diagram, and a **periodic-table heatmap** — the materials analogue of
the dotplot, and the natural display for `rank_elements_groups`.

### Physics

**`mv.calc`** — energies, forces, relaxation, MD, tagged by level.
- Dispatch over a registry, not a hardcoded list. The Matbench Discovery top
  three are separated by 0.003 CPS — that is noise, the ranking churns monthly,
  and any package that hardcodes "the best model" is stale on arrival. Ship
  `mv.calc.available()` and let `level=` name a registered entry.
- Sensible 2026 defaults, all OAM-trained: SevenNet-Omni, GRACE-3L-OAM,
  MACE-MPA-0, eSEN-30M-OAM. `emt` stays as the dependency-free smoke test.
- **Execution goes through TorchSim, not an ASE loop.** Batched, GPU-resident
  relaxation and MD across MACE/fairchem/SevenNet/ORB/MatterSim, reported ~100×
  over ASE. For a library whose entire premise is *thousands of candidates at
  once*, this is not an optimisation — it is the difference between the object
  being useful and being a demo.
- DFT: input-set generation and output parsing, submission delegated (§6.1).

**`mv.prop`** — derived physical properties, each depositing into `obs` or a
grid modality: elastic tensor and moduli; phonons via phonopy → phonon DOS
modality, free energy, heat capacity; thermal conductivity; electronic structure
→ DOS/band modalities, band gap, effective mass; carrier transport (AMSET,
BoltzTraP2); dielectric and optical absorption; magnetic moment and ordering;
surface energy and work function; simulated XRD; Bader charges; COHP bonding
(lobsterpy).

**`mv.thermo`** — stability and phase equilibria. First job is fixing v0.1's
honestly-flagged limitation: the hull is currently built over the dataset's own
compositions, which makes `e_above_hull` a relative statement. Pull competing
phases from MP/Alexandria and make it absolute. Then: chemical potential
diagrams, Pourbaix (aqueous stability), decomposition products (*what* it
decomposes into, not just how far above), reaction energies and reaction
networks, finite-temperature corrections with vibrational entropy, CALPHAD
bridge via pycalphad.

### Discovery

**`mv.model`** — supervised ML. Common `fit`/`predict` depositing into `obs`.
- Fine-tuning a universal MLIP on your own data is the single most-requested
  2026 workflow; MatterTune already provides a unified interface over several
  backbones and is the right thing to wrap.
- Keep CGCNN/ALIGNN-class models available. 2026 OOD studies find they
  generalise *better* out of distribution than fine-tuned foundation embeddings,
  attributed to representation collapse. "Foundation model always wins" is not
  what the evidence says.
- **`mv.model.split` — leakage-aware splits.** Grouping by composition,
  prototype, or held-out element rather than at random. Random splits are the
  field's most common silent methodological failure and this is cheap to fix.
- Uncertainty and calibration throughout, feeding §2.3's `_std` convention.

**`mv.gen`** — generation and search.
- Backends: MatterGen (MIT, code+weights+data public — the reference open
  model), DiffCSP/DiffCSP++, symmetry-first transformers (WyFormer,
  CrystalFormer) on the compute-efficiency frontier, CrystaLLM.
- Symmetry-constrained random generation (pyxtal) + MLIP relaxation, which is
  a strong and often-underrated baseline.
- Substitution enumeration with data-mined substitution probabilities and SMACT
  charge-balance filtering.
- **`mv.gen.validate` — adopt LeMat-GenBench's definitions verbatim.** Every
  generative paper reinvents validity/uniqueness/novelty/stability with a
  different reference set, threshold and fingerprint; LeMat-GenBench pinned all
  three and published a leaderboard. Implementing *their* metric, citing it, and
  refusing to invent a variant is worth more than any new model wrapper.
- Ship the caveat with the tool: a 2026 stress test found neither MatterGen nor
  DiffCSP++ recovers the experimentally observed structure of newly synthesised
  GdNiSn₄/LuNiSn₄ — current models recombine compositions within known
  structural families rather than inventing structure types. `mv.gen` should
  make that easy to see, not easy to forget.

**`mv.screen`** — v0.1's namespace, kept as designed. Filtering and ranking that
deposits a boolean column plus its criteria. Add multi-objective Pareto ranking
and cost/supply-risk constraints (§2.2).

**`mv.opt`** — the closed loop. Bayesian optimisation over a candidate pool
(Ax/BoTorch backend, which is the field default), batch and multi-objective
acquisition, uncertainty-driven active learning for MLIP training sets, and
`uns['campaign']` recording rounds — provenance for iterative design rather than
for a single pipeline.

### Reality

**`mv.exp`** — measured data, which almost nothing unifies.
- XRD: simulate patterns for every candidate in the object, match against a
  measured pattern, rank candidates by fit. That is phase identification, and it
  is a natural fit for an object that already holds both the candidates and the
  grid modality.
- PDF, XAS, Raman/IR (derivable from phonons), XPS.
- Synthesis: text-mined recipes, synthesizability scoring.
- Everything here uses `level='experiment'` (§2.3), so no new machinery.

A calibration note for anything built here: A-Lab's Nature paper received an
Author Correction in January 2026 — 40 successes reduced to 36, "novel" changed
to "inorganic", and "new" clarified to mean new to the prediction platform
rather than new to science. Cite the corrected numbers.

### Infrastructure

**`mv.utils`** — units (a `pint`-backed contract on every deposit; eV vs
kJ/mol vs Ry silently mixed is a whole class of bug), HPC execution (Slurm-aware
batch submission), caching and checkpointing, and the function registry (§7).

---

## 4. What to build first

Tiered by whether the library is *usable* without it.

**v0.1.x — a screening pipeline that works end to end.**
`mv.data` (OPTIMADE + MP + local files), `mv.pp` (standardize, qc, dedup,
supercell), `mv.feat` (composition + dscribe SOAP + MLIP embedding), `mv.calc`
(TorchSim + OAM-model dispatch), `mv.thermo.hull` with real reference phases,
`mv.screen`, basic `mv.pl`. Every function decorated at authoring time (§7).

**v0.2.x — the object earns its keep.**
The MuData multi-axis layout (§2.1), `X` as composition and `mv.tl` on top of it
including `harmonize`, `mv.model` with leakage-aware splits and fine-tuning,
`mv.prop` for phonons/elastic/electronic, `mv.gen.validate`.

**v0.3.x — the loop closes.**
`mv.opt` campaigns, `mv.exp` with XRD matching, defect thermodynamics,
out-of-core at 10⁶–10⁸ (§5), and `matverse-bench`.

---

## 5. Scale is the moat

GNoME is 2.2M structures. Alexandria is 5.06M. OMat24 is ~110M calculations.
LeMat-Bulk is 5.3M deduplicated. **A materials library that assumes its dataset
fits in memory is already obsolete**, and every incumbent — pymatgen, matminer,
ASE `.db` — makes that assumption.

Two facts make this tractable rather than aspirational:

1. Materials Project's v2026.04.13 release moved core data products to Delta
   tables on S3, and `mp-api` now returns a PyArrow-backed `MPDataset`. The
   upstream ecosystem has already moved to a lazy columnar substrate.
2. AnnData/MuData have a zarr-backed lazy path, and this repository sits beside
   `anndataoom`, `mudataoom` and `anndata-rs` — out-of-core work that already
   exists and has no equivalent anywhere in materials.

Concretely: `obs` as a lazy Arrow table, `obsm` descriptor blocks as chunked
zarr, structures resolved on access rather than held as pymatgen objects, and
trajectories never materialised. A screen over 5M candidates that runs on a
laptop is a capability no competing package has.

---

## 6. What matverse must not build

The failure mode for a "unifying" package is reimplementing its dependencies.
The rule: **matverse owns the object, the orchestration, and the borrowed
analysis layer. It owns no physics algorithm.**

| Concern | Delegate to |
|---|---|
| Structures, symmetry, I/O | pymatgen, spglib |
| Atoms objects, calculators | ASE |
| Batched GPU MLIP execution | TorchSim |
| DFT workflow submission | atomate2 + jobflow-remote, quacc, AiiDA |
| MLIP training and fine-tuning | mace / nequip / sevennet trainers, MatterTune |
| Local-environment descriptors | dscribe |
| Phonons | phonopy |
| Defects | doped, pydefect, py-sc-fermi |
| Bayesian optimisation | Ax / BoTorch |
| Generative models | mattergen, diffcsp, … as backends |
| Database access | mp-api, OPTIMADE clients |

### 6.1 Dependency discipline is a hard requirement

This ecosystem's dependency conflicts are documented and current: AMSET requires
Python ≥3.9 while pymatgen now requires ≥3.11; `reaction-network`'s PyPI release
is from 2024 while its GitHub is alive; matminer has not released since April
2024.

The rule that follows: **core dependencies are `anndata`, `mudata`, `numpy`,
`pandas`, `pymatgen`, `ase` and nothing else.** Everything above is an optional
extra, imported lazily *inside* the function that needs it, failing with an
error that names the extra to install. For the worst offenders, subprocess
isolation rather than a shared environment. `pip install matverse` must never be
the thing that breaks someone's environment.

### 6.2 Do not pin to a leaderboard

Model rankings churn monthly and the current top three differ by less than
noise. Registry-based dispatch, `mv.calc.available()`, no default that a paper
can make wrong.

### 6.3 Record licences

MACE-MP/MPA are MIT; MACE-OMAT and MACE-MATPES are ASL (non-commercial); UMA's
licence excludes several countries. Record it in `uns['levels'][level]['license']`
and warn on use.

---

## 7. Agent-readability is the research contribution

This is already the thesis of `DEV_PROMPT.md` and nothing here changes it — the
design above is chosen partly *because* it makes the contract vocabulary bind.

The Beacon registry's `requires`/`produces` fields failed to bind on
`pymatgen-analysis-defects` because pymatgen adds and removes no named state.
matverse's deposit convention creates named state on every call, so the fields
have referents:

```python
@register_function(
    requires=["uns['structures'][source]", "obs['energy_{level}']"],
    produces=["obs['e_above_hull_{level}']", "uns['levels'][level]"],
    prerequisites=["mv.calc.energy"],
)
```

Three ecosystem findings sharpen this:

- **MatTools found that LLM-generated documentation retrieves better than either
  the codebase or human-written docs**, and that simple retrieval plus
  self-reflection beats Agentic RAG and GraphRAG. A structured registry is the
  extreme version of "generated documentation" — this is direct evidence for the
  approach, from the benchmark this project already sits beside.
- There is **no official Materials Project MCP server** and no materials tool-
  registry standard; the existing implementations are third-party. A registry
  designed for this from the start has an open lane.
- `mv.calc.energy` remains the genuine dispatch case, and the templated slot
  names (`obs['energy_{level}']`) are a design question omicverse's static keys
  never faced. Resolve it once, make the probe agree, and document the choice.

Two invariants from §2 must be verified by execution, not asserted: `produces`
by observing object state after a real call, `requires` by deleting the key and
confirming failure. Report a contract-verified rate alongside AFS.

---

## 8. Open problems

1. **`X` as composition is a bet.** If chemical-space structure carries no
   signal worth the coupling, fall back to v0.1's empty `X`. Nothing else
   changes. Test it before committing: does `rank_elements_groups` on a real
   screen recover chemistry a domain expert would recognise?
2. **The materials axis still fits single-system depth badly.** The sites axis
   and grid modalities help. They do not make this the right object for one
   material's full phonon band structure, and the docs should say so rather than
   overclaim.
3. **Two level-of-theory conventions** (layer for arrays, name suffix for
   scalars) is a wart. It is forced by `obs` having no layers. Worth one more
   attempt at unification before it ossifies.
4. **Scope discipline.** Fourteen namespaces is a lot of surface. v0.1.x is seven
   of them, and shipping a screening pipeline that genuinely works beats
   fourteen half-namespaces.
5. **Benchmark credibility.** Per `DEV_PROMPT.md`: write task specifications
   against scientific goals *first*, then check which functions they need. Do
   not co-design a task around a function you just wrote.
6. **Ragged per-site data.** The sites-axis modality is the proposed answer to
   the problem v0.1 flagged as open. It should be prototyped early enough to
   fail cheaply.

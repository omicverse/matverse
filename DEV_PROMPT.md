# matverse development prompt

Hand this to a coding agent working in `/scratch/users/steorra/analysis/omicverse_dev/matverse`.

---

## What you are building

`matverse` is a materials-analysis library on the AnnData substrate. It already
exists as a skeleton: six namespaces, 17 public functions, ~716 lines. Read
`README.md` and `matverse/_core.py` first — they state the two invariants every
namespace obeys, and both are load-bearing:

1. **Operations deposit; they do not return.** `mv.struct.standardize(md)` writes
   `uns['structures']['primitive']` and returns `None`.
2. **A result carries its level of theory in the slot name.** `obs['energy_pbe']`
   and `obs['energy_mace']` are different quantities; `uns['calc'][level]` holds
   the parameters that produced each.

Your job is to grow it into a library that is **agent-readable by construction**,
and to build the benchmark that measures whether that worked.

## Why this library, and why now

This is not a general "make a materials library" task. It closes a specific,
identified hole.

We evaluated the Beacon registry (`@register_function` with seven components:
aliases, description, `requires`, `produces`, `prerequisites`, examples,
`related`, dispatch) on `omicverse` and then on a third-party materials
benchmark, MatTools, over `pymatgen-analysis-defects`. It transferred — 18.4 % →
32.7 % task accuracy against the benchmark authors' own 7,192-document corpus at
24.5 % — **but two of the seven slots did not bind.**

The reason is structural, not incidental. `pymatgen` adds and removes no named
state: results are attributes on returned objects, and nothing is mutated in
common. So `requires`/`produces`, which name state a call consumes and creates,
had nothing to point at, and an object-centric `key_results` slot had to be
substituted. Reviewers correctly read that as the limit of the generality claim.

`matverse` puts materials on AnnData — one mutated object carrying named
containers — which is the same state model `omicverse` has. **It is therefore the
library where the contract vocabulary transfers unchanged into a second
scientific domain.** That is the point of the exercise. Do not quietly design
away the mutated-object convention to make some individual function nicer; it is
the thing under test.

## Ground rules

**Decorate at authoring time, not retroactively.** Every new public function
lands with its `@register_function` entry in the same commit. On `omicverse`, 445
entries were retrofitted and the review pass cost an afternoon; matverse has 17
functions today, so the cost now is close to zero and the resulting claim is
cleaner — a library designed agent-readable from the start rather than annotated
after the fact.

**Every contract claim must be verified by execution before it ships.** Not
asserted. `produces` is confirmed by observing object state after a real call;
`requires` by deleting the key and confirming the call fails; `prerequisites` by
removing a call from the sequence and confirming the downstream call breaks. A
claim that fails its probe is deleted, not repaired by hand. This matters because
the audit metric we published (AFS) credits a field for being *present*, not for
being *true* — on our own hand-written omicverse registry, 51 of 169 contract
claims did not survive probing. Ship the probe harness alongside the library and
report a **contract-verified rate**, not just AFS.

**Empirical priors from the omicverse ablation.** Leave-one-out losses, seed 0,
n = 38, full registry 92.1 % against a 71.1 % baseline: `description` −13.2, the
contract fields jointly −10.5, `examples` −10.5, `aliases` −7.9, `docstring`
−5.3, dispatch entries −2.6. These are upper bounds from one seed and eight of
nine CIs contain zero, so treat them as an ordering, not a ranking. What they
justify: spend your effort on the description and on a runnable example, and do
not skip the contract fields because they are tedious.

Two known failure modes of machine-generated entries, worth designing against
from the start: alias keys collide (46 of 1,162 on omicverse), and prerequisite
chains rarely survive probing (11 did). Make aliases unique at registration time
— raise on collision rather than silently overwriting — and let the probe delete
prerequisite claims it cannot confirm.

**What a docstring will not buy you.** Retrieving the library's real signatures
and docstrings through the identical channel gained +0.0 on `deepseek-v4-flash`.
Retrieving worked usage examples mined from docstrings gained +7.9. The full
registry gained +21.1. So: writing better prose docstrings is not a substitute
for the registry, and adding an example is worth roughly a third of it. Write
both, but do not expect the docstring to carry the load.

## Concrete work

### 1. Decorate the existing 17 functions

`mv.data` — `from_structures`, `from_matminer`, `to_matminer`, `from_ase`,
`from_mp`; `mv.struct` — `standardize`, `supercell`, `describe`; `mv.feat` —
`composition`, `matminer`, `similarity`; `mv.calc` — `register_calculator`,
`energy`, `relax`; `mv.thermo` — `hull`; `mv.screen` — `filter`, `rank`.

The contract slots have obvious referents here, which is the whole point:

```python
@register_function(
    aliases=["convex hull", "energy above hull", "thermodynamic stability"],
    description="Compute the convex hull of formation energies at one level of "
                "theory and record each material's distance above it.",
    requires=["uns['structures'][source]", "obs['energy_{level}']"],
    produces=["obs['e_above_hull_{level}']", "uns['calc'][level]"],
    prerequisites=["mv.calc.energy"],
    examples=["mv.thermo.hull(md, level='emt')"],
    related=["mv.calc.energy", "mv.screen.filter"],
)
def hull(md, level="emt", source="input"): ...
```

`mv.calc.energy` is a genuine **dispatch** case — `level='emt'|'pbe'|'mace'`
routes through `register_calculator` to different backends, and each route should
be separately retrievable, the way `ov.utils.cluster[method=leiden]` is. Do not
collapse the levels into one entry.

Note the slot names are templated on `level`. Decide early whether `produces`
holds the literal template or the resolved key, and make the probe agree with the
choice. This is a real design question the omicverse registry never faced, since
its keys are static.

### 2. Grow the library, decorated as you go

Gaps worth filling, in rough order of how much a screening pipeline needs them:
structure matching and deduplication; symmetry-aware supercells; more featurisers
into `obsm`; phase-diagram construction beyond the hull distance; a proper
provenance replay (`uns['provenance']` currently records operation names only);
and `to_ase` / `to_pymatgen` round trips to match the `from_*` constructors.

Each one arrives with its decorator entry, its probe, and a test.

### 3. Build `matverse-bench`

Model it on `ovagent-bench`, whose design rules held up under review:

- **Goal-not-API prompts.** A task names a scientific objective and a storage
  target, never an API path. "Screen these 40 candidates for thermodynamic
  stability at the EMT level and put the pass/fail in `obs`" — not "call
  `mv.thermo.hull`".
- **End-state grading.** The grader loads the AnnData the agent produced and
  checks whether it is scientifically valid, not whether the right function name
  appeared. A task passes only when every declared check passes.
- **Zero model calls in the grader.** `bench/grader.py` in ovagent-bench contains
  none, and that is why the pass criterion is arguable from the code. Keep it.
- **Accept correct answers that are spelled differently.** Match `obs` columns by
  regex where the name is not part of the specification.
- **Layer the tasks** by what they demand — single call, compose two, full
  pipeline — so failures localise to discovery versus composition.

The arms to run, at minimum: baseline, `doc_RAG` over the library's own
signatures and docstrings on the same retrieval channel, and the full registry.
Add `prose_equivalent` — the seven slots rendered as prose — if you want to
separate the schema from the field syntax; on omicverse that residual was +2.2 pp
and not resolved.

Run at least three seeds. A single seed moved one `gpt-5.5` cell by 10.6 points
in the omicverse panel, which was enough to produce a published inconsistency.

**Do not co-design a task around a function you just wrote.** The credibility
problem with a self-authored benchmark is real and was raised by two reviewers.
Write the task specs against the scientific goal first, then check which
functions they happen to need.

### 4. Report the negative results

Beacon does not help an arbitrary model: `qwen3.5-9b` gained +4.4 despite having
the weakest baseline, because ~30 % of its trajectories exhausted the turn budget
in both arms. A lookup channel costs turns and cannot pay off for a model that
cannot finish unaided. Expect the same floor here and report it rather than
dropping the model.

If a slot turns out not to bind on matverse either, say so and say why. The
`requires`/`produces` failure on `pymatgen` is the most informative single result
in the whole line of work.

## Environment

Python env: `/scratch/users/steorra/env/omicdev/bin/python`. Never run heavy work
on the Sherlock login node — `sh_dev` or `sbatch`. Job I/O to `$SCRATCH`.

The decorator lives in `omicverse.utils.register_function`; the standalone
framework is in `../beacon/` and `../awe-decoratr/`. Reference registries to read
before writing your own: `../beacon-rebuttal/results/mattools_registry/` holds 98
entries for `pymatgen-analysis-defects`, including
`decorations_pymatgen.py`, which is that registry rendered as decorator calls —
note that it has no `requires`/`produces`, for the reason given above. matverse's
should.

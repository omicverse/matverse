# Developer guide

## The rules a new function obeys

**Deposit; do not return.** An operation writes into named slots and returns
`None`. The only exceptions are constructors, which have no object to deposit
into, and `mv.pp.filter_materials` / `filter_elements`, because AnnData cannot
drop rows or columns in place.

**Tag results with their level of theory.** Anything a different method could
also compute gets `_<level>` in its slot name, and the method gets a record in
`uns['levels'][level]` through `matverse._core.set_level`.

**Record the call.** End with `record(md, "pp.thing", param=value)` so
`uns['provenance']` replays as code.

**Fail with the fix.** An error should name the function that would have
produced the missing slot. `matverse._core.require` does this by asking the
registry, so the message stays correct as the library changes.

**Put it in `uns` only if it is not aligned to an axis.** This is the rule that
decides where a result lives, and getting it wrong has cost this library two
bugs.

`uns` is a first-class part of the model and holds far more than it is usually
given credit for — scalars, strings, arrays, DataFrames, and arbitrarily nested
dicts of those all write to `h5ad` without complaint. matverse uses it heavily
and correctly: `uns['levels']` is keyed by level, `uns['grids']` by quantity,
`uns['screens']` by screen name, `uns['phase_diagram']` describes the object.
None of those is per-material, which is exactly why `uns` is right for them.

What `uns` will not do is subset. `md[mask]` drops rows and leaves `uns`
untouched, so **anything with one entry per material must live in `obs`,
`obsm` or `obsp`** — otherwise every surviving row silently points at the wrong
entry. Structures were in `uns` until v0.1.1 and had precisely that bug;
`mv.data.from_ase` kept per-atom arrays there until v0.1.7 and had it too. If a
value has length `n_obs`, it does not belong in `uns`.

Two smaller `uns` rules, both learned by trying to save:

- **Heterogeneous lists do not serialise.** A list of dicts, a list of ragged
  arrays, and a mixed-type tuple each fail on write. Use
  `matverse._core.append_record` / `records`, which store an ordered list as a
  dict keyed by a zero-padded index and write cleanly.
- **Per-atom data belongs on the structure.** pymatgen site properties travel
  with the structure, serialise with it, and appear as columns the moment
  someone calls `mv.multi.sites`. No extra plumbing, and alignment is automatic.

**Decorate in the same commit.** Every public function lands with its
`@register_function` entry. Retrofitting a registry onto a grown library is an
afternoon of work per few hundred functions; doing it as you go costs nothing and
produces a cleaner claim — a library designed to be agent-readable rather than
annotated afterwards.

## Writing a registry entry

```python
@register_function(
    aliases=["convex hull", "energy above hull", "thermodynamic stability"],
    category="thermo",
    description="Build the convex hull of energies at one level of theory and "
                "record each material's distance above it.",
    requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
    produces={"obs": ["e_above_hull_{level}", "is_stable_{level}"]},
    prerequisites=["mv.calc.energy"],
    examples=["mv.thermo.hull(md, level='emt')"],
    related=["mv.calc.energy", "mv.screen.filter"],
)
def hull(md, level="emt", source="input", references=None): ...
```

Spend your effort on `description` and on a runnable `example`. Those are the two
fields that move retrieval most, and neither is a place to be terse.

### Slot templates hold the template, not the resolved key

`produces={"obs": ["e_above_hull_{level}"]}` — not `e_above_hull_emt`. The
template is what holds for every call; a resolved key is true only for the one
call that produced it, and the pattern is the part worth teaching. The probe
resolves templates against each call's bound arguments before checking them, so
the two agree by construction.

If a template cannot resolve — because the parameter defaults to `None` and the
real key is computed — either give the parameter a real default or accept that
the claim is only checkable when the caller passes one. `mv.pp.strain` took the
first route after its probe failed.

### Aliases must be unique

Registration raises `AliasCollision` on a duplicate rather than overwriting.
Machine-generated registries collide constantly, and a silent overwrite makes one
function unreachable through the exact channel the registry exists to provide.

## Contract claims are verified by execution

A registry field is easy to write and easy to get wrong, and the usual audit
metric credits a field for being *present*, not for being *true*. matverse ships
the probe next to the registry and reports a **contract-verified rate**.

```bash
pytest tests/test_contracts.py tests/test_contract_coverage.py -q -s -k rate
```

```
contract-verified rate: 268/268 = 100.0%
  produces       187/187
  requires       81/81
```

| Claim | How it is tested |
|---|---|
| `produces` | run the call on a real dataset, then look |
| `requires` | delete the slot, run the call, confirm it fails |
| `prerequisites` | omit the upstream call, confirm the downstream one breaks |

### The denominator is the part that has to be defended

A rate is only as good as what it is a rate *of*. The probe battery grew
alongside the first six namespaces; the library grew to twenty-nine and the
battery did not follow. For several versions the harness probed **165 of 491
claims and reported 100%** — which is the same kind of statement the audit
metric is criticised for, one level up.

`tests/test_contract_coverage.py::TestCoverage::test_no_claim_goes_unprobed`
fails if any entry ships a claim nothing checks. **A new function may not
introduce a claim without a probe.** If a claim genuinely cannot be decided
offline — it needs the network, a real VASP output, or a tool that is not
installed — name it in `UNPROBEABLE` in `tests/_contract_cases.py` with the
reason. That list is checked too: an entry on it must still exist, must carry a
reason, and must not in fact be probed.

A claim whose backend is missing is reported as **undecided** rather than
failed, and excluded from the denominator. `mv.elec.transport` needs BoltzTraP2,
which does not build in every environment; counting it as a failure would report
the environment as a defect in the registry.

**A claim that fails its probe is deleted, not repaired by hand.** A registry
whose claims are all earned is worth more than a larger one whose claims are
aspirational. Eleven have been deleted so far:

| deleted claim | why it failed |
|---|---|
| `feat.element_stats requires var['Z']` | takes whatever numeric columns `var` has |
| `thermo.hull requires levels[{level}]` | only read when `references=` is given |
| `tl.cluster requires obsp['connectivities']` | true of the leiden route, not kmeans |
| `pp.strain produces structures[{name}]` | template was unresolvable |
| `screen.filter requires obs['{column}']` | no parameter holds a column name |
| `utils.resume requires obs['{column}']` | an absent column means every row is to do |
| `utils.job_status requires uns['submissions']` | having submitted nothing is an answer |
| `dft.status requires obs['dft_directory']` | scans the root, not the object |
| `exp.attach requires uns['grids']` | a measured curve may be the first grid |
| `surf.wulff produces uns['wulff']` | never written |
| `iface.build produces obs['nsites']` | never written |

### A slot may name the object it lands on

Most operations deposit on the object they were handed. Some take two — a
material axis and a sites, bands or interface axis — and deposit on the second.
A container may be qualified with the parameter that receives it:

```python
requires={"sites.obs": ["coordination_number"]},     # mv.env.summarise
produces={"md.obs": ["mean_coordination", ...]},
```

Unqualified means the first parameter, so the single-object case is unchanged.
The qualifier is validated against the signature at import: one that names no
parameter resolves to no object and could never be probed, so registration
refuses it.

`mv.mag.ground_state` is the case that forces this. It writes four columns to
the parent and a fifth back onto the orderings; an unqualified `obs` claimed all
five arrive on the same object, which would send an agent to the wrong one.

### Probing a call that returns a new dataset

`mv.disorder.orderings`, `mv.surf.slabs` and `mv.mol.fragments` build a *new*
object — one row per ordering, per slab, per fragment — and deposit there. Both
kinds obey "operations deposit"; they differ only in which object receives it.
Probe them with `returns='new'`, which looks at the return value:

```python
probe_call(mv.surf.slabs, one_metal, max_index=1, returns="new")
```

Getting this wrong makes a true claim look false, which is worse than not
probing it.

### The probe's own options are named so they cannot collide

Everything after the dataset factory is forwarded to the call, so `probe_call`'s
own arguments are `entry_name` and `returns`. `name` was the original spelling
of the first, and `name` is a real parameter of `mv.pp.supercell` and
`mv.screen.filter` — passing it sent the value to the probe instead of the
function, the entry came back empty, and **the call was silently not probed at
all**. If you add an option to the harness, make sure no matverse function takes
a parameter by that name.

### The one that is a finding, not a bug

`mv.tl.cluster` carries **no** `requires` claim, and the omission is deliberate.
Its two routes consume different state — leiden reads the neighbour graph, kmeans
reads the embedding directly — and the contract vocabulary has one `requires`
field per function, not one per route. That dependency is expressible only in the
`dispatch` prose, which a caller can read but a tool cannot check.

`tests/test_contracts.py::TestPrerequisites::test_cluster_has_no_unconditional_prerequisite`
fails loudly if anyone adds one back.

This is the same shape of limitation as the boundary found when the contract
vocabulary was carried to a library whose results are attributes on returned
objects — `requires`/`produces` bind to named mutated state, and where a call's
state depends on a runtime branch, one field per function is not enough.

## Adding a calculator

Levels are registered, not hardcoded. `factory` is called with no arguments and
returns an ASE calculator; deferring construction matters because a foundation
model checkpoint costs seconds and hundreds of megabytes to instantiate.

```python
mv.calc.register_calculator(
    "myff", MyCalculator,
    kind="mlip", method="MyFF",
    reference="r2SCAN",          # what it reproduces, not just that it is a surrogate
    surrogate=True,
    license="MIT",               # read back by mv.calc.check_licenses
    uncertainty=None)
```

`reference` is load-bearing. A model trained on OMat24 targets PBE+U and one
trained on MatPES targets r2SCAN; mixing them is the same class of error as
mixing PBE with HSE06, and `surrogate: True` alone no longer distinguishes them.
`mv.thermo.hull` compares references before it will build a hull across two
levels.

Do not add a default. The Matbench Discovery leaders are separated by less than
the spread between seeds, the ranking reorders monthly, and a library that
hardcodes "the best model" is stale on arrival.

## Testing

```bash
pytest -q
```

The suite runs against a real calculator, on a six-material Al–Cu–Ni library
chosen so `emt` — parameterised only for Al, Cu, Ag, Au, Ni, Pd, Pt, H, C, N, O —
applies to all of it. EMT is not a good potential; it is a real one, which is
what a pipeline test needs.

Two tests are load-bearing beyond their own subject:

- `test_subsetting_carries_the_structures` — structures live in `obsm` precisely
  so `md[mask]` cannot silently misalign them. This is the reason for being on
  this substrate at all.
- `test_rank_elements_groups_recovers_the_obvious_chemistry` — the test that
  would kill the `X`-as-composition design if element enrichment stopped
  recovering chemistry a domain expert would recognise.

## What matverse must not build

The failure mode of a unifying package is reimplementing its dependencies.
matverse owns the object, the orchestration and the borrowed analysis layer. It
owns no physics algorithm.

| Concern | Delegate to |
|---|---|
| Structures, symmetry, I/O | pymatgen, spglib |
| Atoms objects, calculators | ASE |
| DFT workflow submission | atomate2 + jobflow, quacc, AiiDA |
| MLIP training and fine-tuning | the model authors' trainers |
| Local-environment descriptors | dscribe |
| Phonons | phonopy |
| Defects | doped, pydefect |
| Database access | mp-api, OPTIMADE |

New dependencies go in `[project.optional-dependencies]` and are imported inside
the function that needs them, failing with an error that names the extra.

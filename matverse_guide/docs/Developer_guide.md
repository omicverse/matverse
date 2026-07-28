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
pytest tests/test_contracts.py -q -s -k rate
```

```
contract-verified rate: 68/68 = 100.0%
  produces       50/50
  requires       18/18
```

| Claim | How it is tested |
|---|---|
| `produces` | run the call on a real dataset, then look |
| `requires` | delete the slot, run the call, confirm it fails |
| `prerequisites` | omit the upstream call, confirm the downstream one breaks |

**A claim that fails its probe is deleted, not repaired by hand.** A registry
whose claims are all earned is worth more than a larger one whose claims are
aspirational. Four were deleted while building v0.1.1:

| deleted claim | why it failed |
|---|---|
| `feat.element_stats requires var['Z']` | takes whatever numeric columns `var` has |
| `thermo.hull requires levels[{level}]` | only read when `references=` is given |
| `tl.cluster requires obsp['connectivities']` | true of the leiden route, not kmeans |
| `pp.strain produces structures[{name}]` | template was unresolvable |

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

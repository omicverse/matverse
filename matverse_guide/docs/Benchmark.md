---
myst:
  html_meta:
    description: "matverse-bench: goal-not-API tasks graded on end state, with a grader that contains no model calls."
---

# matverse-bench

Twelve tasks that state a scientific objective and a storage target, graded on
the object an agent produced rather than on how it got there.

```bash
python -m bench.run                    # the reference solutions
python -m bench.run --layer pipeline
python -m bench.run --task stability --verbose
```

```
matverse-bench: 12/12 tasks = 100.0%
  single    5/5
  compose   4/4
  pipeline  3/3
  checks    28/28
```

## Goal, not API

A prompt never names a function.

> Screen this candidate set for thermodynamically plausible materials. Throw out
> anything physically broken first, relax what survives with the effective-medium
> calculator, work out how far each sits above the convex hull, and mark which
> ones are within 50 meV per atom of it. Leave the criteria you used somewhere I
> can read them.

Not "call `mv.thermo.hull`". A benchmark that names the function measures
whether a model can read, not whether it can work — and a test asserts that no
prompt contains `mv.`.

## Three layers

| Layer | What it demands | What a failure localises to |
|---|---|---|
| `single` | one operation | discovery — did it find the function at all? |
| `compose` | two, the second consuming what the first deposited | composition — did it know the order? |
| `pipeline` | a full screen | either; the per-check report says which |

A task passes only when **every** check passes. Partial credit would let a
pipeline that produced half an answer look half right, and half a screen is not
half a result.

## The grader contains no model calls

That is what makes a pass arguable from the code rather than delegated to a
black box, and a test reads `bench/grader.py`'s source to keep it that way.

Columns are matched by regular expression wherever the name is not part of the
specification, because it usually is not. A model that writes `stable` where the
reference writes `passes` has not made a mistake, and a benchmark that says
otherwise measures conformity.

```python
Check("obs", pattern=r"e_?above_?hull|hull_?dist", dtype="numeric",
      finite_fraction=1.0, within=(-1e-6, 1e9),
      describe="a non-negative distance above the hull")
```

The `within` bound is doing real work there: it catches a sign error, which is
the most common way that quantity comes out wrong and is invisible to a
presence check.

## Proving it discriminates

Two things need proving about a benchmark and only one is usually checked.

**That it is passable.** Reference solutions score 12/12, so a zero from some
arm is a real zero rather than a criterion written wrong — the failure mode
where every arm scores nothing and the number just looks like a hard benchmark.

**That it fails when it should**, which is the half that matters. Most of the
benchmark's own test suite feeds the grader wrong answers on purpose:

| Wrong answer | What catches it |
|---|---|
| doing nothing at all | 0/12 |
| a validity flag marking everything valid | the count: six of seven, not seven |
| a sign error in the hull distance | the `within` bound |
| a half-computed column | `finite_fraction` |
| forgetting to save the file | the file is opened and its row count checked |
| a constant offset posing as a cross-database reconciliation | a purpose-built agreement check |

The last is the interesting one. Both a real per-element reconciliation and
subtracting a constant produce a column, and only one makes the two databases
agree on a composition whose correction differs from the average. Nothing about
the column's shape distinguishes them.

It is also shown to **accept** answers it should: differently-named columns, and
an agent that computes a duplicate flag by hand rather than calling
`mv.pp.dedup`. The trajectory is not graded.

## Adding an arm

A solver takes `(task, md, workdir)` and returns the object it produced.

```python
from bench import grader, run

def my_agent(task, md, workdir):
    ...            # hand task.prompt to a model, let it work on md
    return md

results = run.run(solver=my_agent)
print(grader.report(results))
```

The grader does not care how the object got there.

## Layout

```
bench/
├── tasks.py       task specifications and check definitions
├── fixtures.py    starting datasets, all EMT-compatible
├── grader.py      end-state grading — no model calls
├── reference.py   one solution per task, to prove it is passable
└── run.py         the runner
```

Every fixture uses elements EMT is parameterised for, so a run needs no
downloaded model and no network. That constrains the chemistry to metals, which
constrains what the tasks can ask about, and the constraint is worth naming: a
benchmark whose tasks only run on one machine measures that machine.

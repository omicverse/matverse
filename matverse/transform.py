"""``mv.transform`` — pymatgen's transformations, on the object.

pymatgen has around forty-five ``Transformation`` classes: make a supercell,
take the primitive cell, guess oxidation states, perturb the sites, build a
grain boundary, substitute a species. Each takes one structure and returns one
or several.

Wrapping forty-five of them as forty-five matverse functions would be a bad
trade — most are one line, the registry would be mostly noise, and every new
pymatgen release would need another wrapper. So there is one function that
takes the transformation by name and applies it to every row:

```python
mv.transform.apply(md, 'PrimitiveCellTransformation')
mv.transform.apply(md, 'PerturbStructureTransformation', distance=0.1)
mv.transform.apply(md, 'CubicSupercellTransformation', min_length=10)
```

which is also the migration path. A pymatgen script that reads

```python
structures = [PrimitiveCellTransformation().apply_transformation(s)
              for s in structures]
```

becomes one call that deposits a variant, keeps the originals, and records what
it did.

The result goes to a **structure variant**, not over the top of the input, for
the reason every other namespace does the same: "which structure was this
computed on" has to stay answerable from the object alone.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
from anndata import AnnData

from ._core import deposit_structures, record, structures
from ._registry import register_function

#: The modules searched for a transformation, in order.
MODULES = (
    "pymatgen.transformations.standard_transformations",
    "pymatgen.transformations.advanced_transformations",
    "pymatgen.transformations.site_transformations",
)


def _find(name: str):
    """Locate a transformation class by name across pymatgen's modules."""
    import importlib

    wanted = str(name).strip()
    candidates = {}
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:                               # pragma: no cover
            continue
        for attribute, obj in vars(module).items():
            if not inspect.isclass(obj) or attribute.startswith("_"):
                continue
            if not attribute.endswith("Transformation"):
                continue
            candidates[attribute] = obj
            candidates[attribute.replace("Transformation", "")] = obj

    if wanted in candidates:
        return candidates[wanted], wanted
    lowered = {k.lower(): k for k in candidates}
    if wanted.lower() in lowered:
        key = lowered[wanted.lower()]
        return candidates[key], key

    close = sorted(k for k in candidates
                   if k.endswith("Transformation")
                   and wanted.lower()[:5] in k.lower())
    raise KeyError(
        f"no pymatgen transformation named {wanted!r}. "
        + (f"Did you mean one of {close}? " if close else "")
        + "mv.transform.available() lists them all.")


@register_function(
    aliases=["available transformations", "list transformations",
             "what transformations", "pymatgen transformations"],
    category="transform",
    description="List every pymatgen transformation matverse can apply, with "
                "the arguments each takes.",
    examples=["mv.transform.available()",
              "mv.transform.available(search='supercell')"],
    related=["mv.transform.apply", "mv.transform.chain"],
    notes="Read from pymatgen at call time rather than from a table, so a "
          "transformation added upstream is available here the day it lands "
          "and a removed one disappears rather than failing later with a "
          "confusing message.",
)
def available(search: str | None = None) -> dict:
    """Transformation names mapped to their signatures."""
    import importlib

    out = {}
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:                               # pragma: no cover
            continue
        short = module_name.rsplit(".", 1)[-1].replace("_transformations", "")
        for attribute, obj in sorted(vars(module).items()):
            if (not inspect.isclass(obj) or attribute.startswith("Abstract")
                    or not attribute.endswith("Transformation")):
                continue
            if obj.__module__ != module_name:
                continue
            try:
                signature = str(inspect.signature(obj))
            except (TypeError, ValueError):               # pragma: no cover
                signature = "(...)"
            out[attribute] = {"group": short, "signature": signature,
                              "doc": (obj.__doc__ or "").strip().split("\n")[0]}

    if search:
        needle = search.lower()
        out = {k: v for k, v in out.items()
               if needle in k.lower() or needle in v["doc"].lower()}
    return out


@register_function(
    aliases=["apply transformation", "transform structures", "run "
             "transformation", "pymatgen transformation"],
    category="transform",
    description="Apply any pymatgen transformation to every material, "
                "depositing the result as a new structure variant.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{key_added}"],
              "obs": ["{key_added}_ok"], "uns": ["transform"]},
    dispatch="name= is any pymatgen Transformation class; keyword arguments "
             "are passed to its constructor. mv.transform.available() lists "
             "them with their signatures.",
    examples=["mv.transform.apply(md, 'PrimitiveCellTransformation')",
              "mv.transform.apply(md, 'PerturbStructureTransformation', "
              "distance=0.1)",
              "mv.transform.apply(md, 'CubicSupercellTransformation', "
              "min_length=10.0)"],
    related=["mv.transform.available", "mv.transform.chain",
             "mv.pp.supercell", "mv.pp.standardize"],
    notes="Deposits a **variant** rather than replacing the input, so the "
          "original is still there and 'which structure was this computed on' "
          "stays answerable — the same rule mv.pp.standardize and "
          "mv.calc.relax follow.\n\n"
          "A transformation that fails on one row leaves that row's original "
          "structure in place and records False in "
          "``obs['{key_added}_ok']``. A screen can then filter on it, which is "
          "better than either raising on the whole dataset or silently "
          "returning a mixture you cannot tell apart.\n\n"
          "One-to-many transformations return their first result here; use "
          "``mv.transform.expand`` when you want all of them as rows.",
)
def apply(md: AnnData, name: str, source: str = "input",
          key_added: str | None = None, **params) -> None:
    """Apply a transformation to every row. Deposits; returns ``None``."""
    cls, resolved = _find(name)
    key = key_added or resolved.replace("Transformation", "").lower()

    try:
        transformation = cls(**params)
    except TypeError as exc:
        raise TypeError(
            f"{resolved} does not take {params!r}: {exc}. Its signature is "
            f"{inspect.signature(cls)}") from exc

    built, ok, failures = [], [], []
    for i, structure in enumerate(structures(md, source)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = transformation.apply_transformation(structure)
            if isinstance(result, list):
                result = result[0]["structure"] if result else None
            if result is None:
                raise ValueError("the transformation returned nothing")
            built.append(result)
            ok.append(True)
        except Exception as exc:
            built.append(structure)
            ok.append(False)
            failures.append(f"{i}: {type(exc).__name__}: {exc}")

    deposit_structures(md, key, built)
    md.obs[f"{key}_ok"] = ok
    md.uns.setdefault("transform", {})[key] = {
        "transformation": resolved, "params": {k: str(v)
                                               for k, v in params.items()},
        "source": source, "n_ok": int(sum(ok)),
        "n_failed": len(failures), "errors": failures[:10],
    }
    record(md, "transform.apply", name=resolved, source=source,
           key_added=key, n_ok=int(sum(ok)))


@register_function(
    aliases=["expand transformation", "all results", "one to many",
             "ranked transformation"],
    category="transform",
    description="Apply a one-to-many transformation and keep every result as "
                "its own row, rather than only the first.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["parent", "variant_index"], "structures": ["input"]},
    examples=["out = mv.transform.expand(md, "
              "'OrderDisorderedStructureTransformation', n=4, "
              "no_oxi_states=True)"],
    related=["mv.transform.apply", "mv.disorder.orderings",
             "mv.mag.orderings"],
    notes="Returns an ordinary materials object with ``obs['parent']`` "
          "pointing back — the same derived-axis shape as mv.pp.defects, "
          "mv.mag.orderings and mv.disorder.orderings, because it is the same "
          "move.\n\n"
          "Where a namespace already wraps a specific one-to-many "
          "transformation, prefer it: mv.disorder.orderings and "
          "mv.mag.orderings do the same work and record domain metadata this "
          "generic path cannot know about.",
)
def expand(md: AnnData, name: str, source: str = "input", n: int = 4,
           **params) -> AnnData:
    """Every result of a one-to-many transformation. Returns a new object."""
    from .data import from_structures

    cls, resolved = _find(name)
    transformation = cls(**params)
    labels = [str(x) for x in md.obs.get("name", md.obs_names)]

    built, parents, indices, failures = [], [], [], []
    for i, structure in enumerate(structures(md, source)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = transformation.apply_transformation(
                    structure, return_ranked_list=n)
        except Exception as exc:
            failures.append(f"{labels[i]}: {type(exc).__name__}: {exc}")
            continue

        if not isinstance(results, list):
            results = [{"structure": results}]
        for k, entry in enumerate(results[:n]):
            got = entry["structure"] if isinstance(entry, dict) else entry
            built.append(got)
            parents.append(labels[i])
            indices.append(k)

    if not built:
        raise ValueError(
            f"{resolved} produced nothing for any row. "
            f"Several pymatgen transformations enumerate with enumlib and "
            f"return an empty list rather than raising when it is missing."
            + (f" Errors: {failures[:3]}" if failures else ""))

    out = from_structures(built)
    out.obs["parent"] = parents
    out.obs["variant_index"] = indices
    out.uns["transform"] = {
        "transformation": resolved,
        "params": {k: str(v) for k, v in params.items()},
        "source": source, "n_requested": int(n),
        "n_failed": len(failures), "errors": failures[:10],
    }
    record(out, "transform.expand", name=resolved, source=source, n=n)
    return out


@register_function(
    aliases=["chain transformations", "transformation pipeline",
             "sequence of transformations", "apply several"],
    category="transform",
    description="Apply a sequence of transformations in order, depositing "
                "only the final result but recording every step.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{key_added}"], "obs": ["{key_added}_ok"]},
    examples=["mv.transform.chain(md, ["
              "('PrimitiveCellTransformation', {}), "
              "('PerturbStructureTransformation', {'distance': 0.05})])"],
    related=["mv.transform.apply", "mv.utils.summary"],
    notes="A chain is one variant, not one per step, because the intermediates "
          "are usually not interesting and storing them all would fill the "
          "object. The full sequence is in ``uns['transform']`` and in the "
          "provenance, so the result is still reproducible from the record.",
)
def chain(md: AnnData, steps, source: str = "input",
          key_added: str = "transformed") -> None:
    """Apply transformations in sequence. Deposits; returns ``None``."""
    if not steps:
        raise ValueError("steps is empty; pass [(name, params), ...]")

    prepared = []
    for step in steps:
        name, params = step if isinstance(step, tuple) else (step, {})
        cls, resolved = _find(name)
        prepared.append((resolved, cls(**params), params))

    built, ok, failures = [], [], []
    for i, structure in enumerate(structures(md, source)):
        current, good = structure, True
        for resolved, transformation, _ in prepared:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = transformation.apply_transformation(current)
                current = result[0]["structure"] if isinstance(result, list) \
                    else result
            except Exception as exc:
                failures.append(f"{i} at {resolved}: "
                                f"{type(exc).__name__}: {exc}")
                good = False
                break
        built.append(current if good else structure)
        ok.append(good)

    deposit_structures(md, key_added, built)
    md.obs[f"{key_added}_ok"] = ok
    md.uns.setdefault("transform", {})[key_added] = {
        "chain": [{"transformation": r, "params": {k: str(v)
                                                   for k, v in p.items()}}
                  for r, _, p in prepared],
        "source": source, "n_ok": int(sum(ok)),
        "n_failed": len(failures), "errors": failures[:10],
    }
    record(md, "transform.chain",
           steps=[r for r, _, _ in prepared], source=source,
           key_added=key_added)


@register_function(
    aliases=["oxidation states", "guess valences", "assign charges",
             "add oxidation states", "bond valence"],
    category="transform",
    description="Assign oxidation states to every structure, which several "
                "other operations need before they will run.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{key_added}"],
              "obs": ["oxidation_states_ok", "charge_balanced"]},
    dispatch="method='bva' uses bond-valence analysis on the geometry; "
             "'guess' uses composition-based electronegativity ordering; a "
             "dict assigns them directly.",
    examples=["mv.transform.oxidation_states(md)",
              "mv.transform.oxidation_states(md, method='guess')",
              "mv.transform.oxidation_states(md, method={'Fe': 2, 'O': -2})"],
    related=["mv.disorder.orderings", "mv.thermo.pourbaix", "mv.pp.qc"],
    notes="This is the missing prerequisite behind several confusing pymatgen "
          "errors. ``OrderDisorderedStructureTransformation`` fails with "
          "``Element has no attribute oxi_state!``, Ewald ranking silently "
          "scores everything zero, and ``DopingTransformation`` refuses with "
          "``Valences cannot be assigned!`` — all three are asking for this.\n\n"
          "Bond-valence analysis reads the states off bond lengths and fails "
          "on metals, where the concept does not apply. That failure is "
          "recorded per row rather than raised, because a mixed dataset of "
          "oxides and alloys is normal.",
)
def oxidation_states(md: AnnData, source: str = "input",
                     method="bva", key_added: str = "oxidized") -> None:
    """Assign oxidation states. Deposits a variant; returns ``None``."""
    from pymatgen.transformations.standard_transformations import (
        AutoOxiStateDecorationTransformation,
        OxidationStateDecorationTransformation)

    if isinstance(method, dict):
        transformation = OxidationStateDecorationTransformation(method)
        label = "explicit"
    elif method == "bva":
        transformation = AutoOxiStateDecorationTransformation()
        label = "bond valence analysis"
    elif method == "guess":
        transformation = None
        label = "composition guess"
    else:
        raise ValueError(f"method must be 'bva', 'guess' or a dict, got "
                         f"{method!r}")

    built, ok, balanced, failures = [], [], [], []
    for i, structure in enumerate(structures(md, source)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if transformation is None:
                    got = structure.copy()
                    got.add_oxidation_state_by_guess()
                else:
                    got = transformation.apply_transformation(structure)
            built.append(got)
            ok.append(True)
            balanced.append(abs(float(got.charge)) < 1e-6)
        except Exception as exc:
            built.append(structure)
            ok.append(False)
            balanced.append(False)
            failures.append(f"{i}: {type(exc).__name__}: {exc}")

    deposit_structures(md, key_added, built)
    md.obs["oxidation_states_ok"] = ok
    md.obs["charge_balanced"] = balanced
    md.uns["oxidation_states"] = {
        "method": label, "source": source, "n_ok": int(sum(ok)),
        "n_failed": len(failures), "errors": failures[:10],
        "note": "bond-valence analysis fails on metals, where an oxidation "
                "state is not a meaningful quantity; that is recorded per row "
                "rather than raised",
    }
    record(md, "transform.oxidation_states", method=label, source=source,
           key_added=key_added)


__all__ = ["MODULES", "available", "apply", "expand", "chain",
           "oxidation_states"]

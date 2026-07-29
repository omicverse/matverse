"""Verify contract claims by executing them.

A registry field is easy to write and easy to get wrong. The audit metric in
common use credits a field for being *present*, not for being *true* — on a
hand-written registry of 169 contract claims, 51 did not survive probing. So
matverse ships the probe alongside the registry and reports a
**contract-verified rate**, not just a coverage score.

What each probe does
--------------------
``produces``
    Run the call on a real dataset, then look. A slot that is not there
    afterwards is a false claim.

``requires``
    Delete the slot, run the call, and see whether it fails. A call that
    succeeds without something it claims to need did not need it.

``prerequisites``
    Omit the upstream call and see whether the downstream one breaks. This is
    the claim that survives least often — 11 of a much larger set on omicverse —
    because most "prerequisites" are conventions rather than dependencies.

Where a slot is expected to land
--------------------------------
Most operations deposit on the object they were handed. A minority build a new
dataset from an old one — one row per ordering, per slab, per fragment — and
deposit there instead. Both obey "operations deposit"; they differ only in
which object receives it, so ``probe_call(..., returns='new')`` looks at the
return value. Getting this wrong makes a true claim look false, which is worse
than not probing it at all.

Claims that fail their probe are meant to be **deleted from the decorator**, not
repaired by hand. A registry whose claims are all verified is worth more than a
larger one whose claims are aspirational.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from ._registry import (get_registry, resolve_slot, slot_template_fields,
                        split_container)


@dataclass
class ClaimResult:
    """One contract claim, and what happened when it was tested."""
    function: str
    kind: str                 # produces | requires | prerequisites
    container: str
    slot: str
    resolved: str
    verified: bool
    detail: str = ""
    skipped: bool = False     # the environment could not test it either way

    def __str__(self) -> str:
        mark = "skip" if self.skipped else ("ok  " if self.verified else "FAIL")
        return (f"{mark} {self.function} {self.kind} "
                f"{self.container}[{self.resolved!r}]"
                + (f" — {self.detail}" if self.detail else ""))


@dataclass
class ProbeReport:
    """Every claim tested, and the rate that survived."""
    results: list[ClaimResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def tested(self) -> list[ClaimResult]:
        """Claims the environment could actually decide. The rate's denominator.

        A claim whose backend is not installed is neither true nor false here;
        counting it either way would misreport the registry rather than the
        environment.
        """
        return [r for r in self.results if not r.skipped]

    @property
    def verified(self) -> list[ClaimResult]:
        return [r for r in self.tested if r.verified]

    @property
    def failed(self) -> list[ClaimResult]:
        return [r for r in self.tested if not r.verified]

    @property
    def untestable(self) -> list[ClaimResult]:
        return [r for r in self.results if r.skipped]

    @property
    def rate(self) -> float:
        return len(self.verified) / len(self.tested) if self.tested else 0.0

    def by_kind(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.tested:
            slot = out.setdefault(r.kind, [0, 0])
            slot[1] += 1
            slot[0] += int(r.verified)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def summary(self) -> str:
        lines = [f"contract-verified rate: {len(self.verified)}/"
                 f"{len(self.tested)} = {self.rate:.1%}"]
        for kind, (ok, total) in sorted(self.by_kind().items()):
            lines.append(f"  {kind:14s} {ok}/{total}")
        if self.failed:
            lines.append("")
            lines.append("failed claims (delete these from the decorator):")
            lines += [f"  {r}" for r in self.failed]
        if self.untestable:
            lines.append("")
            lines.append(f"claims this environment cannot decide "
                         f"({len(self.untestable)}):")
            lines += [f"  {r}" for r in self.untestable]
        if self.skipped:
            lines.append("")
            lines.append(f"not probed ({len(self.skipped)}): "
                         + ", ".join(sorted(set(self.skipped))))
        return "\n".join(lines)


_MISSING = object()


def _members(value) -> list[str] | None:
    """The names a collection-valued argument stands for, or None if it is one."""
    if isinstance(value, dict):
        return [str(k) for k in value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value]
    return None


def _expand_slot(slot: str, bound: dict) -> list[str]:
    """One claim per member when a template field binds to a collection.

    ``mv.screen.pareto(md, {'e_above_hull_emt': 'min', 'density': 'max'})``
    consumes two columns, and the parameter naming them is a dict;
    ``mv.calc.committee(md, ['mace-mpa', 'chgnet'])`` consumes one column per
    level. Plain substitution would resolve those to the ``repr`` of a dict or a
    list — a claim about a column that could not exist — so a field bound to a
    collection stands for as many slots as it has members.
    """
    slots = [slot]
    for field in slot_template_fields(slot):
        members = _members(bound.get(field))
        if members is None:
            continue
        slots = [s.replace("{" + field + "}", m)
                 for s in slots for m in members]
    return [resolve_slot(s, bound) for s in slots]


def _target(container: str, default, bound: dict,
            first: str | None = None) -> tuple[Any, str]:
    """The object a claim lands on, and the container name without its qualifier.

    ``'sites.obs'`` resolves ``sites`` out of the call's bound arguments. An
    unqualified container lands on ``default``, the object the call was made on
    — and so does one that names the first parameter, since that *is* the
    object the call was made on. ``mv.env.summarise(sites, md)`` names both of
    its objects, and only one of them arrives through ``args``.
    """
    param, bare = split_container(container)
    if param is None or param == first:
        return default, bare
    return bound.get(param, _MISSING), bare


def _entry_for(registry, func: Callable, name: str | None):
    """The entry for the function actually passed, not one that shares its name.

    ``mv.pl.pareto`` and ``mv.screen.pareto`` are both ``pareto``, and looking a
    probe up by bare name silently returns the wrong one — which then binds the
    wrong parameters and reports a claim the function never made. The qualified
    name is unambiguous, so try it first.
    """
    if name is not None:
        return registry.get(name)
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", "")
    if module and qualname:
        entry = registry.get(f"{module}.{qualname}")
        if entry is not None:
            return entry
    return registry.get(getattr(func, "__name__", ""))


def _first_parameter(func: Callable) -> str | None:
    try:
        return next(iter(inspect.signature(func).parameters))
    except (TypeError, ValueError, StopIteration):
        return None


def _holder(md, container: str):
    """The mapping a contract container actually names, or None."""
    if container == "structures":
        from ._core import STRUCTURE_KEY
        return md.obsm.get(STRUCTURE_KEY)
    if container in ("levels", "features"):
        return md.uns.get(container, {})
    if container == "uns":
        return md.uns
    return getattr(md, container, None)


def _read_slot(md, container: str, key: str) -> tuple[bool, Any]:
    """Is ``container[key]`` present on this object, and what is it?"""
    if container == "X":
        return (getattr(md, "n_vars", 0) > 0, None)
    if container == "files":
        return (True, None)                    # not an object slot; not probed
    holder = _holder(md, container)
    if holder is None:
        return (False, None)
    try:
        return (key in holder, holder[key] if key in holder else None)
    except Exception:
        return (False, None)


def _delete_slot(md, container: str, key: str) -> bool:
    """Remove a slot so a ``requires`` claim can be tested. False if it cannot."""
    try:
        holder = _holder(md, container)
        if holder is None or key not in holder:
            return False
        if container == "structures":
            from ._core import STRUCTURE_KEY
            md.obsm[STRUCTURE_KEY] = holder.drop(columns=[key])
            cache = getattr(md, "_mv_structure_cache", None)
            if isinstance(cache, dict):
                cache.pop(key, None)
            return True
        del holder[key]
        return True
    except Exception:
        return False


def _bind(func: Callable, args: tuple, kwargs: dict) -> dict:
    """Call arguments including defaults, for resolving slot templates."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def probe_call(func: Callable, make_dataset: Callable, *args,
               entry_name: str | None = None, returns: str = "self",
               **kwargs) -> list[ClaimResult]:
    """Test every contract claim on one call.

    ``make_dataset`` returns a fresh dataset each time it is called — the probes
    are destructive, so each one needs its own copy.

    Everything after ``make_dataset`` is forwarded to the call, so the probe's
    own options are named ``entry_name`` and ``returns`` rather than anything a
    matverse function might take. ``name`` in particular is a real parameter of
    ``mv.pp.supercell``, ``mv.screen.filter`` and others; while the lookup
    argument was called ``name``, passing it went to the probe instead of to the
    function, the entry came back empty and the call was silently not probed at
    all — a harness that reports nothing wrong because it tested nothing.

    ``returns`` says where the ``produces`` slots are expected to land. Most
    operations deposit on the object they were handed, which is ``'self'``. A
    minority build a *new* dataset from an old one — one row per ordering, per
    slab, per fragment — and deposit there; those are ``'new'``, and the probe
    looks at the return value instead. ``requires`` is tested on the input
    either way, since that is the object the slot is deleted from.
    """
    registry = get_registry()
    entry = _entry_for(registry, func, entry_name)
    if entry is None:
        return []
    if returns not in ("self", "new"):
        raise ValueError(f"returns must be 'self' or 'new', not {returns!r}")
    label = entry["public_name"]
    bound = _bind(entry["function"], (None,) + args, kwargs)
    first = _first_parameter(entry["function"])
    results: list[ClaimResult] = []

    # produces — run it, then look.
    md = make_dataset()
    before = {}
    if returns == "self":
        for container, slots in entry["produces"].items():
            landed, bare = _target(container, md, bound, first)
            if landed is _MISSING or bare in ("files", "X"):
                continue
            for slot in slots:
                for resolved in _expand_slot(slot, bound):
                    before[(container, resolved)] = _read_slot(
                        landed, bare, resolved)[0]
    missing_backend = False
    try:
        out = func(md, *args, **kwargs)
        ran, why = True, ""
    except ImportError as exc:
        # An optional backend that is not installed says nothing about whether
        # the claim is true, so record it as undecided rather than as a failure.
        out, ran, why, missing_backend = None, False, f"{exc}", True
    except Exception as exc:
        out, ran, why = None, False, f"{type(exc).__name__}: {exc}"

    if returns == "new":
        target = out
        if ran and target is None:
            ran, why = False, "call returned None, so there is no new dataset"
    else:
        target = md

    for container, slots in entry["produces"].items():
        landed, bare = _target(container, target, bound, first)
        for slot, resolved in ((s, r) for s in slots
                               for r in _expand_slot(s, bound)):
            if bare == "files":
                results.append(ClaimResult(label, "produces", container, slot,
                                           resolved, True,
                                           "writes to disk; not probed"))
                continue
            if not ran:
                results.append(ClaimResult(
                    label, "produces", container, slot, resolved, False,
                    (f"backend not installed: {why}" if missing_backend
                     else f"call failed: {why}"),
                    skipped=missing_backend))
                continue
            if landed is _MISSING:
                results.append(ClaimResult(
                    label, "produces", container, slot, resolved, False,
                    "the qualifying parameter was not passed, so there is no "
                    "object to look at"))
                continue
            present, _ = _read_slot(landed, bare, resolved)
            detail = "" if present else "not present after the call"
            if present and before.get((container, resolved)):
                detail = "was already present before the call"
            results.append(ClaimResult(label, "produces", container, slot,
                                       resolved, present, detail))

    # requires — delete it, then see whether the call still works.
    for container, slots in entry["requires"].items():
        param, bare = split_container(container)
        if param == first:
            param = None      # the object the call is made on
        for slot, resolved in ((s, r) for s in slots
                               for r in _expand_slot(s, bound)):
            if bare in ("files", "X"):
                results.append(ClaimResult(label, "requires", container, slot,
                                           resolved, True, "not removable"))
                continue
            md = make_dataset()
            call_args, call_kwargs = args, kwargs
            if param is None:
                stripped = md
            else:
                # The slot lives on another argument, so strip a copy of that
                # one and substitute it in — the caller's object must survive.
                given = bound.get(param, _MISSING)
                if given is _MISSING or given is None:
                    results.append(ClaimResult(
                        label, "requires", container, slot, resolved, False,
                        f"parameter {param!r} was not passed, so the claim "
                        "could not be tested"))
                    continue
                stripped = given.copy()
                call_args, call_kwargs = _substitute(
                    entry["function"], args, kwargs, param, stripped)
            if not _delete_slot(stripped, bare, resolved):
                results.append(ClaimResult(
                    label, "requires", container, slot, resolved, False,
                    "slot was not present on a fresh dataset, so the claim "
                    "could not be tested"))
                continue
            try:
                func(md, *call_args, **call_kwargs)
                results.append(ClaimResult(
                    label, "requires", container, slot, resolved, False,
                    "call succeeded without it"))
            except Exception as exc:
                results.append(ClaimResult(
                    label, "requires", container, slot, resolved, True,
                    f"raised {type(exc).__name__}"))

    return results


def _substitute(func: Callable, args: tuple, kwargs: dict, param: str,
                value: Any) -> tuple[tuple, dict]:
    """The same call with one parameter replaced, wherever it was passed."""
    kwargs = dict(kwargs)
    if param in kwargs:
        kwargs[param] = value
        return args, kwargs
    try:
        names = list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return args, kwargs
    if param in names:
        position = names.index(param) - 1        # args excludes the first
        if 0 <= position < len(args):
            args = args[:position] + (value,) + args[position + 1:]
            return args, kwargs
    kwargs[param] = value
    return args, kwargs


def probe_prerequisite(func: Callable, make_dataset: Callable,
                       upstream: Callable, *args, entry_name: str | None = None,
                       upstream_args: tuple = (), upstream_kwargs: dict | None = None,
                       **kwargs) -> ClaimResult:
    """Test one prerequisite: omit the upstream call and expect a failure."""
    registry = get_registry()
    entry = _entry_for(registry, func, entry_name)
    label = entry["public_name"] if entry else getattr(func, "__name__", "?")
    up_entry = _entry_for(registry, upstream, None)
    up_label = up_entry["public_name"] if up_entry else "upstream"

    md = make_dataset()
    try:
        func(md, *args, **kwargs)
        return ClaimResult(label, "prerequisites", "functions", up_label,
                           up_label, False,
                           "downstream call succeeded without it")
    except Exception as exc:
        detail = f"raised {type(exc).__name__} without it"

    md = make_dataset()
    upstream(md, *upstream_args, **(upstream_kwargs or {}))
    try:
        func(md, *args, **kwargs)
    except Exception as exc:
        return ClaimResult(label, "prerequisites", "functions", up_label,
                           up_label, False,
                           f"still failed after running it: "
                           f"{type(exc).__name__}: {exc}")
    return ClaimResult(label, "prerequisites", "functions", up_label, up_label,
                       True, detail)


def audit(registry=None) -> dict:
    """Field coverage across the registry — how much is claimed, not how much
    is true. Report it beside a probe rate, never instead of one."""
    registry = registry or get_registry()
    entries = registry.entries()
    if not entries:
        return {"n_entries": 0}
    fields = ("aliases", "description", "examples", "related", "requires",
              "produces", "prerequisites", "dispatch")
    filled = {f: sum(1 for e in entries if e.get(f)) for f in fields}
    return {
        "n_entries": len(entries),
        "filled": filled,
        "coverage": {f: filled[f] / len(entries) for f in fields},
        "n_claims": sum(
            len(slots) for e in entries
            for mapping in (e["requires"], e["produces"])
            for slots in mapping.values()),
    }


__all__ = ["probe_call", "probe_prerequisite", "audit", "ProbeReport",
           "ClaimResult"]

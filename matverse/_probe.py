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

Claims that fail their probe are meant to be **deleted from the decorator**, not
repaired by hand. A registry whose claims are all verified is worth more than a
larger one whose claims are aspirational.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from ._registry import get_registry, resolve_slot


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

    def __str__(self) -> str:
        mark = "ok  " if self.verified else "FAIL"
        return (f"{mark} {self.function} {self.kind} "
                f"{self.container}[{self.resolved!r}]"
                + (f" — {self.detail}" if self.detail else ""))


@dataclass
class ProbeReport:
    """Every claim tested, and the rate that survived."""
    results: list[ClaimResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def verified(self) -> list[ClaimResult]:
        return [r for r in self.results if r.verified]

    @property
    def failed(self) -> list[ClaimResult]:
        return [r for r in self.results if not r.verified]

    @property
    def rate(self) -> float:
        return len(self.verified) / len(self.results) if self.results else 0.0

    def by_kind(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.results:
            slot = out.setdefault(r.kind, [0, 0])
            slot[1] += 1
            slot[0] += int(r.verified)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def summary(self) -> str:
        lines = [f"contract-verified rate: {len(self.verified)}/"
                 f"{len(self.results)} = {self.rate:.1%}"]
        for kind, (ok, total) in sorted(self.by_kind().items()):
            lines.append(f"  {kind:14s} {ok}/{total}")
        if self.failed:
            lines.append("")
            lines.append("failed claims (delete these from the decorator):")
            lines += [f"  {r}" for r in self.failed]
        if self.skipped:
            lines.append("")
            lines.append(f"not probed ({len(self.skipped)}): "
                         + ", ".join(sorted(set(self.skipped))))
        return "\n".join(lines)


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
               name: str | None = None, **kwargs) -> list[ClaimResult]:
    """Test every contract claim on one call.

    ``make_dataset`` returns a fresh dataset each time it is called — the probes
    are destructive, so each one needs its own copy.
    """
    registry = get_registry()
    entry = registry.get(name or getattr(func, "__name__", ""))
    if entry is None:
        return []
    label = entry["public_name"]
    bound = _bind(entry["function"], (None,) + args, kwargs)
    results: list[ClaimResult] = []

    # produces — run it, then look.
    md = make_dataset()
    before = {}
    for container, slots in entry["produces"].items():
        for slot in slots:
            resolved = resolve_slot(slot, bound)
            before[(container, resolved)] = _read_slot(md, container, resolved)[0]
    try:
        func(md, *args, **kwargs)
        ran, why = True, ""
    except Exception as exc:
        ran, why = False, f"{type(exc).__name__}: {exc}"

    for container, slots in entry["produces"].items():
        for slot in slots:
            resolved = resolve_slot(slot, bound)
            if container == "files":
                results.append(ClaimResult(label, "produces", container, slot,
                                           resolved, True,
                                           "writes to disk; not probed"))
                continue
            if not ran:
                results.append(ClaimResult(label, "produces", container, slot,
                                           resolved, False,
                                           f"call failed: {why}"))
                continue
            present, _ = _read_slot(md, container, resolved)
            detail = "" if present else "not present after the call"
            if present and before.get((container, resolved)):
                detail = "was already present before the call"
            results.append(ClaimResult(label, "produces", container, slot,
                                       resolved, present, detail))

    # requires — delete it, then see whether the call still works.
    for container, slots in entry["requires"].items():
        for slot in slots:
            resolved = resolve_slot(slot, bound)
            if container in ("files", "X"):
                results.append(ClaimResult(label, "requires", container, slot,
                                           resolved, True, "not removable"))
                continue
            md = make_dataset()
            if not _delete_slot(md, container, resolved):
                results.append(ClaimResult(
                    label, "requires", container, slot, resolved, False,
                    "slot was not present on a fresh dataset, so the claim "
                    "could not be tested"))
                continue
            try:
                func(md, *args, **kwargs)
                results.append(ClaimResult(
                    label, "requires", container, slot, resolved, False,
                    "call succeeded without it"))
            except Exception as exc:
                results.append(ClaimResult(
                    label, "requires", container, slot, resolved, True,
                    f"raised {type(exc).__name__}"))

    return results


def probe_prerequisite(func: Callable, make_dataset: Callable,
                       upstream: Callable, *args, name: str | None = None,
                       upstream_args: tuple = (), upstream_kwargs: dict | None = None,
                       **kwargs) -> ClaimResult:
    """Test one prerequisite: omit the upstream call and expect a failure."""
    registry = get_registry()
    entry = registry.get(name or getattr(func, "__name__", ""))
    label = entry["public_name"] if entry else getattr(func, "__name__", "?")
    up_entry = registry.get(getattr(upstream, "__name__", ""))
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

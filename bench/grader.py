"""matverse-bench grader.

**This module contains no model calls, and must not acquire any.** That is what
makes a pass arguable from the code: anyone can read what the criterion was and
disagree with it. A grader that asks a model whether an answer looks right has
moved the benchmark's ground truth into a black box.

What is graded is the **end state** — the object the agent produced — not the
trajectory. Whether it called ``mv.thermo.hull`` or reimplemented a convex hull
by hand is not the question; whether the object now carries a correct distance
above the hull is.

Columns are matched by regular expression wherever the name is not part of the
specification, because it usually is not. A model that writes ``stable`` where
the reference writes ``passes`` has not made a mistake, and a benchmark that
says otherwise measures conformity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CheckResult:
    passed: bool
    describe: str
    detail: str = ""

    def __str__(self) -> str:
        return ("ok   " if self.passed else "FAIL ") + self.describe + \
            (f" — {self.detail}" if self.detail else "")


@dataclass
class TaskResult:
    task_id: str
    layer: str
    checks: list = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        """A task passes only when every check passes. Partial credit would let
        a pipeline that produced half an answer look half right, and half a
        screen is not half a result."""
        return not self.error and bool(self.checks) and \
            all(c.passed for c in self.checks)

    def summary(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'} {self.task_id} " \
               f"({self.layer})"
        if self.error:
            return f"{head}\n  error: {self.error}"
        return "\n".join([head] + [f"  {c}" for c in self.checks])


def _columns(frame, pattern: str | None, target: str) -> list:
    names = [str(c) for c in frame.columns] if hasattr(frame, "columns") \
        else [str(k) for k in frame]
    if pattern is None:
        return [target] if target in names else []
    regex = re.compile(pattern, re.IGNORECASE)
    return [n for n in names if regex.search(n)]


def _is_boolean(values) -> bool:
    arr = np.asarray(values)
    if arr.dtype == bool:
        return True
    try:
        unique = set(np.unique(arr[~_null(arr)]).tolist())
    except Exception:
        return False
    return bool(unique) and unique <= {0, 1, True, False, 0.0, 1.0}


def _null(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in "fc":
        return ~np.isfinite(arr)
    return np.zeros(len(arr), dtype=bool)


def _is_numeric(values) -> bool:
    return np.asarray(values).dtype.kind in "fiu"


def _check_obs(md, check) -> CheckResult:
    candidates = _columns(md.obs, check.pattern, check.target)
    if not candidates:
        return CheckResult(False, check.describe, "no matching obs column")

    reasons = []
    for name in candidates:
        values = md.obs[name].to_numpy()
        if check.dtype == "boolean" and not _is_boolean(values):
            reasons.append(f"{name}: not boolean")
            continue
        if check.dtype == "numeric" and not _is_numeric(values):
            reasons.append(f"{name}: not numeric")
            continue
        if check.dtype == "string" and _is_numeric(values):
            reasons.append(f"{name}: numeric, expected text")
            continue

        if check.dtype == "boolean":
            n_true = int(np.asarray(values, dtype=bool).sum())
            if check.min_true is not None and n_true < check.min_true:
                reasons.append(f"{name}: {n_true} true, need >= {check.min_true}")
                continue
            if check.max_true is not None and n_true > check.max_true:
                reasons.append(f"{name}: {n_true} true, need <= {check.max_true}")
                continue

        if check.finite_fraction is not None and _is_numeric(values):
            finite = float(np.isfinite(np.asarray(values, dtype=float)).mean())
            if finite < check.finite_fraction:
                reasons.append(f"{name}: {finite:.0%} finite, "
                               f"need {check.finite_fraction:.0%}")
                continue

        if check.within is not None and _is_numeric(values):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            low, high = check.within
            if len(arr) and (arr.min() < low or arr.max() > high):
                reasons.append(f"{name}: range [{arr.min():.3g}, {arr.max():.3g}] "
                               f"outside [{low:g}, {high:g}]")
                continue

        return CheckResult(True, check.describe, f"obs[{name!r}]")

    return CheckResult(False, check.describe, "; ".join(reasons[:3]))


def _check_mapping(md, check, container: str) -> CheckResult:
    holder = getattr(md, container, None)
    if holder is None:
        return CheckResult(False, check.describe, f"no {container}")
    names = [str(k) for k in holder]
    if container == "obsm":
        names = [n for n in names if n != "structures"]
    matched = _match(names, check)
    return CheckResult(bool(matched), check.describe,
                       f"{container}[{matched[0]!r}]" if matched
                       else f"nothing matching in {container}: {names[:6]}")


def _match(names: list, check) -> list:
    if check.pattern is None:
        return [check.target] if check.target in names else []
    regex = re.compile(check.pattern, re.IGNORECASE)
    return [n for n in names if regex.search(n)]


def _check_structures(md, check) -> CheckResult:
    frame = md.obsm.get("structures")
    variants = list(frame.columns) if frame is not None else []
    matched = _match(variants, check)
    return CheckResult(bool(matched), check.describe,
                       f"variant {matched[0]!r}" if matched
                       else f"variants present: {variants}")


def _check_agreement(md, check) -> CheckResult:
    """Rows with the same composition must agree after a correction.

    The check that separates a real per-element reconciliation from subtracting
    a constant offset. Both produce a column; only one makes the two databases
    agree on a composition whose correction differs from the average.
    """
    candidates = _columns(md.obs, check.pattern, check.target)
    if not candidates:
        return CheckResult(False, check.describe, "no matching obs column")
    if md.n_vars == 0:
        return CheckResult(False, check.describe, "no element axis to group by")

    raw = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    keys = [tuple(np.round(row / max(row.sum(), 1e-12), 6))
            for row in np.asarray(raw, dtype=float)]

    tolerance = check.within[1] if check.within else 1e-6
    reasons = []
    for name in candidates:
        values = md.obs[name].to_numpy(dtype=float)
        worst, groups = 0.0, 0
        for key in set(keys):
            members = values[[i for i, k in enumerate(keys) if k == key]]
            members = members[np.isfinite(members)]
            if len(members) < 2:
                continue
            groups += 1
            worst = max(worst, float(members.max() - members.min()))
        if groups == 0:
            reasons.append(f"{name}: no composition appears twice")
            continue
        if worst > tolerance:
            reasons.append(f"{name}: worst disagreement {worst:.3g} > "
                           f"{tolerance:g}")
            continue
        return CheckResult(True, check.describe,
                           f"obs[{name!r}], {groups} shared compositions agree "
                           f"to {worst:.2g}")
    return CheckResult(False, check.describe, "; ".join(reasons[:3]))


def _check_file(md, check, workdir: Path) -> CheckResult:
    path = workdir / check.target
    if not path.exists():
        return CheckResult(False, check.describe, f"{check.target} not written")
    try:
        import anndata
        back = anndata.read_h5ad(path)
    except Exception as exc:
        return CheckResult(False, check.describe,
                           f"written but unreadable: {type(exc).__name__}: {exc}")
    if back.n_obs != md.n_obs:
        return CheckResult(False, check.describe,
                           f"file has {back.n_obs} rows, object has {md.n_obs}")
    return CheckResult(True, check.describe, f"{check.target}, {back.n_obs} rows")


def grade_one(md, task, workdir: Path | None = None) -> TaskResult:
    """Grade one task against the object an agent produced."""
    result = TaskResult(task_id=task.id, layer=task.layer)
    if md is None:
        result.error = "no object was produced"
        return result

    workdir = Path(workdir or ".")
    for check in task.checks:
        try:
            if check.kind == "obs":
                result.checks.append(_check_obs(md, check))
            elif check.kind in ("obsm", "obsp", "varm", "uns"):
                result.checks.append(_check_mapping(md, check, check.kind))
            elif check.kind == "structures":
                result.checks.append(_check_structures(md, check))
            elif check.kind == "levels":
                levels = list(md.uns.get("levels", {}))
                matched = _match(levels, check)
                result.checks.append(CheckResult(
                    bool(matched), check.describe,
                    f"level {matched[0]!r}" if matched
                    else f"levels present: {levels}"))
            elif check.kind == "X":
                result.checks.append(CheckResult(
                    md.n_vars > 0, check.describe,
                    f"{md.n_vars} columns"))
            elif check.kind == "agreement":
                result.checks.append(_check_agreement(md, check))
            elif check.kind == "file":
                result.checks.append(_check_file(md, check, workdir))
            else:
                result.checks.append(CheckResult(
                    False, check.describe, f"unknown check kind {check.kind!r}"))
        except Exception as exc:
            result.checks.append(CheckResult(
                False, check.describe,
                f"grader error: {type(exc).__name__}: {exc}"))
    return result


def grade(results: list) -> dict:
    """Aggregate task results into the numbers a run reports."""
    by_layer: dict[str, list] = {}
    for result in results:
        by_layer.setdefault(result.layer, []).append(result)

    return {
        "n_tasks": len(results),
        "n_passed": sum(1 for r in results if r.passed),
        "accuracy": (sum(1 for r in results if r.passed) / len(results)
                     if results else 0.0),
        "by_layer": {
            layer: {"n": len(group),
                    "passed": sum(1 for r in group if r.passed),
                    "accuracy": sum(1 for r in group if r.passed) / len(group)}
            for layer, group in by_layer.items()},
        "n_checks": sum(len(r.checks) for r in results),
        "n_checks_passed": sum(1 for r in results for c in r.checks if c.passed),
        "failures": [r.task_id for r in results if not r.passed],
    }


def report(results: list) -> str:
    """A readable run report."""
    scores = grade(results)
    lines = [f"matverse-bench: {scores['n_passed']}/{scores['n_tasks']} tasks "
             f"= {scores['accuracy']:.1%}"]
    for layer in ("single", "compose", "pipeline"):
        if layer in scores["by_layer"]:
            block = scores["by_layer"][layer]
            lines.append(f"  {layer:9s} {block['passed']}/{block['n']}")
    lines.append(f"  checks    {scores['n_checks_passed']}/{scores['n_checks']}")
    lines.append("")
    lines += [r.summary() for r in results if not r.passed]
    return "\n".join(lines)


__all__ = ["grade_one", "grade", "report", "TaskResult", "CheckResult"]

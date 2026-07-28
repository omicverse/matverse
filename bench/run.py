"""Run matverse-bench.

    python -m bench.run                    # the reference solutions
    python -m bench.run --layer pipeline   # one layer
    python -m bench.run --task stability

An agent arm plugs in by passing a callable that takes ``(task, md, workdir)``
and returns the object it produced. The grader does not care how it got there.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import fixtures, grader, reference, tasks  # noqa: E402


def run(selection=None, solver=None, workdir=None) -> list:
    """Run tasks and return their results.

    ``solver(task, md, workdir) -> md``. Defaults to the reference solutions,
    which exist to show the benchmark is passable rather than to be a target.
    """
    chosen = selection if selection is not None else tasks.TASKS
    solver = solver or (lambda task, md, wd: reference.solve(task.id, md, wd))

    results = []
    for task in chosen:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(workdir or tmp)
            try:
                md = fixtures.build(task.fixture)
                produced = solver(task, md, root)
            except Exception as exc:
                result = grader.TaskResult(task.id, task.layer)
                result.error = f"{type(exc).__name__}: {exc}"
                results.append(result)
                continue
            results.append(grader.grade_one(produced, task, root))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("single", "compose", "pipeline"))
    parser.add_argument("--task")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.task:
        selection = [tasks.by_id(args.task)]
    elif args.layer:
        selection = tasks.by_layer(args.layer)
    else:
        selection = tasks.TASKS

    results = run(selection)
    print(grader.report(results))
    if args.verbose:
        print()
        for result in results:
            print(result.summary())

    scores = grader.grade(results)
    return 0 if scores["n_passed"] == scores["n_tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

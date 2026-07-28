"""Build and execute the tutorial notebooks.

    python _scripts/build_notebooks.py                 # all of them
    python _scripts/build_notebooks.py screening       # one

Each notebook is generated from a ``nb_<name>.py`` module in this directory and
then executed, so the outputs in the documentation are real. Exits non-zero if
any cell raised, which is what makes the tutorials tested rather than merely
written.
"""

from __future__ import annotations

import importlib
import sys

from _nbbuild import execute_and_write

NOTEBOOKS = [
    "getting_started",
    "screening",
    "chemical_space",
    "beyond_one_number",
    "models_and_campaigns",
]


def main(argv: list[str]) -> int:
    import matplotlib
    matplotlib.use("Agg")

    wanted = argv[1:] or NOTEBOOKS
    unknown = [n for n in wanted if n not in NOTEBOOKS]
    if unknown:
        print(f"unknown: {unknown}; known: {NOTEBOOKS}")
        return 2

    failures = 0
    for name in wanted:
        module = importlib.import_module(f"nb_{name}")
        failures += execute_and_write(module.CELLS, name)
    if failures:
        print(f"[matverse docs] {failures} cell(s) raised")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

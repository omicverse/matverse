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

from _nbbuild import DOCS, REPO, execute_and_write

NOTEBOOKS = [
    "getting_started",
    "screening",
    "chemical_space",
    "beyond_one_number",
    "models_and_campaigns",
    "defects_and_diffusion",
    "surfaces_and_adsorption",
    "dynamics",
    "magnetic_ordering",
    "structure_and_bands",
    "interfaces",
    "disorder",
    "from_pymatgen",
    "molecules",
    "data_io",
    "infrastructure",
]


def check_registry_coverage() -> list[str]:
    """Every registered function must appear in some notebook.

    A registry entry is a promise that a function is part of the public
    surface. If the documentation never calls it, nothing checks that the
    promise still holds — the signature can drift, the slot it writes can be
    renamed, and the first person to find out is a user. So this is enforced
    rather than tracked.
    """
    import json
    import re
    import sys as _sys

    # Running this as a script puts _scripts/ on sys.path, not the repository
    # root, so an editable checkout is not importable without help. The
    # notebooks themselves are unaffected — nbclient starts its kernel with
    # cwd set to the repository.
    if str(REPO) not in _sys.path:
        _sys.path.insert(0, str(REPO))
    import matverse as mv

    registered = {name.replace("matverse.", "mv.")
                  for name in set(mv.registry._index.values())}

    used: set[str] = set()
    for path in (DOCS / "tutorials").glob("*.ipynb"):
        source = "\n".join("".join(cell["source"])
                            for cell in json.loads(path.read_text())["cells"])
        used |= {f"mv.{ns}.{fn}"
                 for ns, fn in re.findall(r"\bmv\.(\w+)\.(\w+)", source)}
    return sorted(registered - used)


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

    if set(wanted) == set(NOTEBOOKS):
        uncovered = check_registry_coverage()
        if uncovered:
            print(f"[matverse docs] {len(uncovered)} registered function(s) "
                  f"appear in no notebook:")
            for name in uncovered:
                print(f"    {name}")
            return 1
        print("[matverse docs] registry coverage: every registered function "
              "is called in a notebook")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

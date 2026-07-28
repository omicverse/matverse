"""Shared machinery for building the tutorial notebooks.

Each tutorial is a module in this directory exposing ``CELLS``, a list of
``(kind, body)`` pairs where *kind* is ``"markdown"`` or ``"code"``. The
notebooks are generated rather than edited by hand, and then **executed**, so the
outputs in the documentation are real: myst-nb runs with
``nb_execution_mode = "off"``, and a notebook committed without outputs renders
as a page of code and no results.

Generating them also means the code in the tutorials is tested. A tutorial whose
examples were only ever typed into a markdown fence rots silently — this one
fails the build.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
REPO = DOCS.parent.parent


def build(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(body) if kind == "markdown"
                else nbf.v4.new_code_cell(body) for kind, body in cells]
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "mystnb": {"execution_mode": "off"},
    })
    return nb


def execute_and_write(cells: list[tuple[str, str]], name: str,
                      timeout: int = 1800) -> int:
    """Run the cells, write the notebook, and report any that raised."""
    from nbclient import NotebookClient

    nb = build(cells)
    target = DOCS / "tutorials" / f"{name}.ipynb"

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            allow_errors=True,
                            resources={"metadata": {"path": str(REPO)}})
    client.execute()
    nbf.write(nb, str(target))

    failed = [i for i, c in enumerate(nb.cells)
              if any(o.get("output_type") == "error"
                     for o in c.get("outputs", []))]
    status = "ok" if not failed else f"{len(failed)} FAILED"
    print(f"[matverse docs] {name}.ipynb — {len(nb.cells)} cells, {status}")
    for i in failed:
        for o in nb.cells[i].get("outputs", []):
            if o.get("output_type") == "error":
                print(f"    cell {i}: {o.get('ename')}: {o.get('evalue')}")
                print("      " + "\n      ".join(
                    ''.join(nb.cells[i]["source"]).splitlines()[:4]))
    return len(failed)

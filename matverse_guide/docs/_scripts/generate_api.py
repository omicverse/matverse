"""Generate ``api/user.md`` from the matverse registry.

The registry already holds everything an API page needs — the description, the
aliases someone would search for, and the slots each call reads and writes — so
the page is generated rather than maintained. A function added with its
decorator appears in the docs on the next build; one added without a decorator
does not appear at all, which is the intended pressure.

Run directly, or let ``conf.py`` call it on every build::

    python _scripts/generate_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
REPO = DOCS.parent.parent

if (REPO / "matverse").is_dir():
    sys.path.insert(0, str(REPO))

import matverse as mv  # noqa: E402

#: Section order and heading for each registry category.
SECTIONS = [
    ("data", "Data IO", "Build a dataset, and get it back out again."),
    ("pp", "Preprocessing",
     "Structure standardisation, quality control, filtering, deduplication and "
     "cross-database harmonisation."),
    ("feat", "Featurisation", "Descriptors into `obsm`."),
    ("tl", "Tools",
     "Analysis on the composition matrix — ordination, clustering, element "
     "enrichment and novelty."),
    ("calc", "Calculators",
     "Energies, forces and relaxation, tagged by level of theory."),
    ("prop", "Properties",
     "Derived properties, including curves stored on a shared grid."),
    ("thermo", "Thermodynamics",
     "Convex hull, energy above hull, decomposition products."),
    ("multi", "Sites axis",
     "Per-atom results, on a companion object whose rows are atoms."),
    ("exp", "Experiment",
     "Measured data, carried as a level of theory like any other."),
    ("screen", "Screening",
     "Filtering, ranking and Pareto fronts that leave a record."),
    ("gen", "Generated candidates",
     "Scoring generated structures, and enumerating substitutions."),
    ("model", "Machine learning",
     "Property prediction, with splits that do not leak."),
    ("opt", "Design campaigns",
     "Choosing what to compute next, and recording each round."),
    ("pl", "Plotting",
     "Publication defaults; every function draws onto an axis and returns it."),
    ("utils", "Infrastructure",
     "Units, checkpointing, cluster submission and object summaries."),
]

HEADER = """# User API

Import matverse as:

```python
import matverse as mv
```

This page is generated from the `@register_function` entries in the matverse
registry, so it lists exactly what the library exposes to a caller — and to an
agent. Every entry names the state it reads and the state it writes, and each of
those claims is verified by execution in `tests/test_contracts.py` rather than
asserted.

Public registry entries listed here: {n_entries}

Look a function up by intent rather than by name:

```python
mv.find('thermodynamic stability')      # ['mv.thermo.hull', ...]
print(mv.describe('convex hull'))       # signature, contract, examples
```

```{{eval-rst}}
.. currentmodule:: matverse
```
"""

CONTRACT_INTRO = """
## What each function writes

The table below is the `produces` half of the registry contract: the slots a
call deposits into the object. Names in braces are templated on the call's own
arguments — `obs['energy_{level}']` becomes `obs['energy_emt']` when you pass
`level='emt'`.
"""


def _relative(entry: dict) -> str:
    """``matverse.thermo.hull`` -> ``thermo.hull`` for autosummary."""
    full = entry["full_name"]
    return full[len("matverse."):] if full.startswith("matverse.") else full


def _render_slot(container: str, slot: str) -> str:
    if container == "structures":
        return f"obsm['structures']['{slot}']"
    if container in ("levels", "features"):
        return f"uns['{container}']['{slot}']"
    if container == "X":
        return "X"
    if container == "files":
        return "written to disk"
    return f"{container}['{slot}']"


def build() -> str:
    registry = mv.registry
    by_category: dict[str, list[dict]] = {}
    for entry in registry.entries():
        by_category.setdefault(entry["category"], []).append(entry)
    for entries in by_category.values():
        entries.sort(key=lambda e: e["short_name"])

    out = [HEADER.format(n_entries=len(registry))]

    for category, title, blurb in SECTIONS:
        entries = by_category.get(category, [])
        if not entries:
            continue
        out.append(f"\n## {title}\n\n{blurb}\n")
        out.append("```{eval-rst}\n.. autosummary::\n   :toctree: reference/\n"
                   "   :nosignatures:\n")
        for entry in entries:
            out.append(f"   {_relative(entry)}")
        out.append("```\n")

    # Anything registered under a category the section list does not name.
    extra = sorted(set(by_category) - {c for c, _, _ in SECTIONS})
    for category in extra:
        out.append(f"\n## {category}\n")
        out.append("```{eval-rst}\n.. autosummary::\n   :toctree: reference/\n"
                   "   :nosignatures:\n")
        for entry in by_category[category]:
            out.append(f"   {_relative(entry)}")
        out.append("```\n")

    out.append(CONTRACT_INTRO)
    out.append("| Function | Writes |")
    out.append("|---|---|")
    for category, _, _ in SECTIONS:
        for entry in by_category.get(category, []):
            slots = [_render_slot(container, slot)
                     for container, values in entry["produces"].items()
                     for slot in values]
            if not slots:
                continue
            cells = ", ".join(f"`{s}`" for s in slots)
            out.append(f"| `{entry['public_name']}` | {cells} |")
    out.append("")

    return "\n".join(out)


def main() -> None:
    target = DOCS / "api" / "user.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(), encoding="utf-8")
    print(f"[matverse docs] wrote {target.relative_to(DOCS)} "
          f"({len(mv.registry)} entries)")


if __name__ == "__main__":
    main()

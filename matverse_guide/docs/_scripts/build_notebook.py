"""Build and execute the getting-started notebook.

The notebook is generated from this script rather than edited by hand, and then
**executed**, so the outputs in the documentation are real. myst-nb is configured
with ``nb_execution_mode = "off"`` — it renders stored outputs rather than
running anything at build time — which means a notebook committed without
outputs renders as a page of code and no results.

Regenerate after changing the library:

    python _scripts/build_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
REPO = DOCS.parent.parent

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Getting started, on real materials

Every other tutorial builds its structures in code. This one does not: it loads
published structures — LiFePO₄ as it is actually reported, Li₁₀GeP₂S₁₂ as the
solid electrolyte people actually study — and runs a screen on them.

It executes end to end with no network, no API key and no downloaded model."""),

    ("code", """\
import matverse as mv

mv.datasets.available()"""),

    ("markdown", """\
Each bundled set is a coherent scenario rather than a grab-bag, and says what it
is *for*. Take the cathodes."""),

    ("code", """\
md = mv.datasets.load("battery_cathodes")
md"""),

    ("markdown", """\
`X` is already the composition matrix and `var` is already the periodic table.
Nobody asked for either — composition is intrinsic to a material rather than
derived from it, so it is built at construction."""),

    ("code", """\
import pandas as pd

pd.DataFrame(md.X.toarray(), index=md.obs["name"], columns=md.var_names)"""),

    ("markdown", """\
Counts come from the **reduced** formula, so a supercell and its primitive cell
land on the same point in chemical space. Cell size lives in `obs['nsites']`.

`var` carries the periodic table restricted to the elements present:"""),

    ("code", """\
md.var[["Z", "electronegativity", "period", "is_transition_metal"]]"""),

    ("markdown", "## What are these?"),

    ("code", """\
mv.pp.describe(md)
mv.pp.qc(md)

md.obs[["name", "spacegroup", "formula", "nsites", "density", "is_valid"]]"""),

    ("markdown", """\
## A diffraction pattern for each

`mv.prop.xrd` needs no calculator, so it runs on anything — which makes it the
first real property available for a set like this."""),

    ("code", """\
mv.prop.xrd(md, two_theta=(10, 60), step=0.02)

md.obsm["xrd_calc"].shape, mv.grid_of(md, "xrd").shape"""),

    ("markdown", """\
The patterns are a `materials × 2θ` block on a shared grid, which is what makes
them comparable. A peak list would not be: no two materials share peak
positions."""),

    ("code", """\
import matplotlib.pyplot as plt

ax = mv.pl.spectra(md, "xrd", rows=[0, 1, 2], offset=110)
ax.set_title("computed powder patterns")
ax.figure.set_size_inches(8, 4)"""),

    ("markdown", """\
LiFePO₄ and NaFePO₄ are the same framework with a different alkali, and the
patterns show it — the same families of reflections, shifted.

## Phase identification

Feed one of them back as if it were measured, and ask which candidate it is."""),

    ("code", """\
measured = md.obsm["xrd_calc"][1]              # pretend we measured row 1
mv.exp.match_xrd(md, measured, mv.grid_of(md, "xrd"))

md.obs[["name", "xrd_match", "xrd_match_rank"]].sort_values("xrd_match_rank")"""),

    ("markdown", """\
```{warning}
`match_xrd` scores against the candidates in this object and nothing else, and
records that in `uns['xrd_match']['scored_against']`. A high score means "the
best of what you gave it", not "identified" — the true phase can be absent from
your library entirely.
```

## Running a calculator

The cathodes contain Fe, P and V, which the only calculator that ships working —
ASE's effective-medium theory — is not parameterised for. That is a real
constraint rather than a tutorial convenience, so the screen switches to the
metals EMT *can* run."""),

    ("code", """\
mv.calc.available()["emt"]"""),

    ("code", """\
metals = mv.datasets.metals()
metals.obs[["name", "lattice_parameter"]]"""),

    ("markdown", "Published room-temperature lattice parameters. Relax them:"),

    ("code", """\
mv.pp.describe(metals)
mv.calc.relax(metals, level="emt", fmax=0.02)

metals.obs[["name", "energy_per_atom_emt", "relax_converged_emt"]]"""),

    ("markdown", """\
The relaxed geometry becomes its own variant rather than replacing the input, so
"which structure was this energy computed on" stays answerable:"""),

    ("code", """\
mv.variants(metals)"""),

    ("markdown", """\
## Properties that need the calculator

Elastic constants by finite strain, and phonons by frozen displacement."""),

    ("code", """\
mv.prop.elastic(metals, level="emt", source="relaxed_emt")
mv.prop.phonon(metals, level="emt", source="relaxed_emt", supercell=(1, 1, 1))
mv.prop.thermal_conductivity(metals, level="emt")

metals.obs[["name", "bulk_modulus_emt", "debye_temperature_emt",
            "thermal_conductivity_emt", "dynamically_stable_emt"]].round(1)"""),

    ("markdown", """\
Compare against experiment: bulk moduli of 140 (Cu), 76 (Al), 180 (Ni) GPa and
Debye temperatures of 343, 428, 165, 225 K for Cu, Al, Au, Ag. EMT gets the
magnitudes roughly and the ordering right, which is what it is for.

Every mode is real, so all seven are dynamically stable — as elemental fcc
metals at their equilibrium lattice parameter should be.

## A screen

Rank on something, and leave the reasoning in the object."""),

    ("code", """\
mv.screen.filter(metals, thermal_conductivity_emt__gt=3.0,
                 bulk_modulus_emt__gt=50.0, name="conductive_and_stiff")

metals.uns["screens"]["conductive_and_stiff"]"""),

    ("code", """\
metals.obs[["name", "thermal_conductivity_emt", "bulk_modulus_emt",
            "conductive_and_stiff"]].round(1)"""),

    ("markdown", """\
The screen **deposits** a boolean column plus the criteria rather than returning
a shorter list, because which criterion a candidate failed is a result. Subset
when you actually want the short list.

## Which chemistry passed?"""),

    ("code", """\
mv.tl.rank_elements_groups(metals, "conductive_and_stiff")
metals.uns["rank_elements_groups"]["True"][
    ["element", "n_in_group", "frac_in_group", "odds_ratio", "pval"]]"""),

    ("markdown", """\
This is `rank_genes_groups` with the nouns changed, and it is the operation that
justifies making `X` the composition matrix. With seven elementals it is a
formality; on a library of thousands it is the question that follows every
screen.

## What the object remembers"""),

    ("code", """\
for step in mv.provenance(metals):
    print(step)"""),

    ("markdown", """\
Parameters are recorded with each call, so the history replays as code rather
than reading as a list of verbs.

```{seealso}
[Screening, end to end](screening.md) walks the same pipeline in more detail;
[Chemical space](chemical_space.md) picks up where `rank_elements_groups` left
off; [Beyond one number](beyond_one_number.md) covers curves, per-atom results
and measured data.
```"""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(body) if kind == "markdown"
                else nbf.v4.new_code_cell(body) for kind, body in CELLS]
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "mystnb": {"execution_mode": "off"},
    })
    return nb


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    target = DOCS / "tutorials" / "getting_started.ipynb"
    nb = build()

    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO)}})
    client.execute()

    nbf.write(nb, str(target))
    failed = [i for i, c in enumerate(nb.cells)
              if any(o.get("output_type") == "error"
                     for o in c.get("outputs", []))]
    print(f"[matverse docs] wrote {target.relative_to(DOCS)} "
          f"({len(nb.cells)} cells)")
    if failed:
        print(f"[matverse docs] ERROR: cells {failed} raised")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

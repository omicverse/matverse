"""Cells for tutorials/infrastructure.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Infrastructure

The parts of a screening campaign that are not physics: what unit a number is
in, how to survive a walltime limit, how to process a corpus that does not fit
in memory, and how to hand the survivors to a real DFT code.

None of this is interesting. All of it is what separates a calculation you ran
once from a campaign that finished."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("code", """\
md = mv.datasets.metals()
mv.pp.describe(md)
mv.calc.energy(md, level="emt")

md.obs[["name", "energy_per_atom_emt"]].round(4)"""),

    ("markdown", """\
## What is this object carrying?

`mv.utils.summary` is the human-readable inventory — every axis, every variant,
every level, in one string."""),

    ("code", """\
print(mv.utils.summary(md))"""),

    ("markdown", """\
## Units

A number without a unit is not a result. matverse works internally in eV and
ångström, records the unit of every column it writes, and converts on request
rather than silently."""),

    ("code", """\
mv.utils.INTERNAL_UNITS"""),

    ("code", """\
mv.utils.check_units(md)"""),

    ("markdown", """\
Every column matverse wrote has a unit; `lattice_parameter` and `nsites` come
back `None` because nothing declared one. A column you compute yourself is in
the same position until you say otherwise."""),

    ("code", """\
md.obs["cohesive_estimate"] = -md.obs["energy_per_atom_emt"].to_numpy(dtype=float)
mv.utils.set_units(md, "cohesive_estimate", "eV/atom")

mv.utils.check_units(md)["cohesive_estimate"]"""),

    ("markdown", """\
`mv.utils.convert` goes the other way — it takes a column in *someone else's*
unit and brings it into matverse's. That is the direction that matters, because
the numbers arriving from a collaborator or a paper are the ones in kJ/mol.

Simulate that: build a column in kJ/mol, declare it, and convert it back."""),

    ("code", """\
md.obs["reported_energy"] = (
    md.obs["energy_per_atom_emt"].to_numpy(dtype=float) / mv.utils.TO_EV["kj/mol"])
mv.utils.set_units(md, "reported_energy", "kJ/mol")

mv.utils.convert(md, "reported_energy", kind="energy")
md.obs[["name", "reported_energy", "reported_energy_ev",
        "energy_per_atom_emt"]].round(4)"""),

    ("markdown", """\
`reported_energy_ev` and `energy_per_atom_emt` agree, which is the check worth
running: a unit conversion that does not round-trip is a unit conversion you
should not trust.

Note that it **deposits a new column** rather than rewriting the old one. A
converted column sitting beside its original is auditable; a silently rewritten
one is the bug this function exists to prevent, one step later. The declared
unit is carried along."""),

    ("code", """\
mv.utils.TO_EV["kj/mol"], mv.utils.TO_ANGSTROM["bohr"]"""),

    ("markdown", """\
## Surviving a walltime limit

A screen that takes six hours on a queue with a four-hour limit needs to be
resumable, and the object is where that state lives.

`mv.utils.checkpoint` writes the object to disk with a note."""),

    ("code", """\
import tempfile
from pathlib import Path

workdir = Path(tempfile.mkdtemp())
path = mv.utils.checkpoint(md, workdir / "screen.h5ad", note="after emt")
Path(path).name, round(Path(path).stat().st_size / 1e3, 1)"""),

    ("markdown", """\
`mv.utils.resume` answers the question a restarted job actually asks: *which
rows still need doing?*"""),

    ("code", """\
import anndata

reloaded = anndata.read_h5ad(path)
reloaded.obs.loc[reloaded.obs_names[:3], "band_gap_pbe"] = [1.2, 0.0, 2.4]

todo = mv.utils.resume(reloaded, "band_gap_pbe")
todo, todo.sum()"""),

    ("markdown", """\
Three rows were done before the job died; four remain. Note that this works on
the **reloaded** object — an object you cannot pick back up is not a
checkpoint.

```{note}
That sentence is load-bearing. h5ad stores `uns['provenance']` as a list and
reads it back as a numpy array, so until v0.1.13 every operation on a reloaded
object failed on its own provenance write. Saving is only useful if the object
can be worked on afterwards, and there is now a test that says so.
```

## Corpora larger than memory

`mv.utils.chunks` slices the object into pieces, yielding the starting row
index with each — so a piece can always be written back to where it came
from."""),

    ("code", """\
for start, piece in mv.utils.chunks(md, size=3):
    print(f"rows {start}-{start + piece.n_obs - 1}: {list(piece.obs['name'])}")"""),

    ("markdown", """\
`map_chunks` runs an operation over each piece, optionally checkpointing as it
goes, and can skip rows already done — which is the resumable-screen pattern in
one call."""),

    ("code", """\
fresh = mv.datasets.metals()
mv.pp.describe(fresh)

report = mv.utils.map_chunks(
    fresh,
    lambda piece: mv.calc.energy(piece, level="emt"),
    size=3,
    checkpoint_to=workdir / "progress.h5ad",
)
report"""),

    ("code", """\
fresh.obs[["name", "energy_per_atom_emt"]].round(4)"""),

    ("markdown", """\
The results landed on the parent object even though the work happened
chunk-by-chunk, and the checkpoint on disk is current.

## Getting onto a cluster

`mv.utils.slurm_script` writes a submission script rather than submitting
anything. Submitting is the scheduler's job and every site does it differently;
generating a correct script is the part that is the same everywhere."""),

    ("code", """\
script = mv.utils.slurm_script(
    "python screen.py --checkpoint screen.h5ad",
    path=workdir / "screen.sbatch",
    partition="normal", hours=8, cpus=16, memory="64GB",
    job_name="alni-screen",
    setup="module load python/3.12\\nsource $SCRATCH/env/bin/activate",
)
print(Path(script).read_text())"""),

    ("markdown", """\
```{warning}
It writes `--time`, `--mem`, `--cpus-per-task` and `--partition` explicitly and
does not rely on defaults. Slurm's defaults are typically one CPU, a few
hundred megabytes and a short walltime, so a job submitted without them fails
in a way that looks like a bug in your code.
```

## Handing over to DFT

EMT and machine-learned potentials narrow the field. The shortlist gets real
DFT, and matverse's job is to generate inputs and harvest results — not to run
VASP, which the queue does."""),

    ("code", """\
mv.dft.presets()"""),

    ("code", """\
shortlist = md[md.obs["name"].isin(["Cu", "Al"])].copy()
written = mv.dft.write_inputs(shortlist, workdir / "runs", code="vasp",
                              preset="relax")
[Path(p).name for p in written]"""),

    ("code", """\
sorted(p.name for p in Path(written[0]).iterdir())"""),

    ("markdown", """\
INCAR, POSCAR, KPOINTS and a POTCAR *specification* — the last is a list of
which potentials to concatenate rather than the potentials themselves, because
those are licensed and cannot be redistributed.

### Where did the jobs get to?"""),

    ("code", """\
mv.dft.status(shortlist, workdir / "runs")"""),

    ("markdown", """\
Nothing has run, so both are missing. Pretend one finished:"""),

    ("code", """\
(Path(written[0]) / "vasprun.xml").write_text("<modeling/>")
mv.dft.status(shortlist, workdir / "runs")"""),

    ("markdown", """\
```{note}
`status` resolves runs through a **manifest** written next to the inputs, not by
matching directory names. A workflow manager that renames `run-000` to
`job-41725` would otherwise silently attach results to the wrong row — which is
the usual way a hand-rolled harvest goes wrong, and it goes wrong quietly.
```

### Harvesting

`read_outputs` parses whatever finished and deposits it at the level you name."""),

    ("code", """\
mv.dft.read_outputs(shortlist, workdir / "runs", level="pbe")

shortlist.obs[["name", "energy_pbe", "dft_error_pbe"]]"""),

    ("markdown", """\
Both are NaN with a reason, because the `<modeling/>` above is not a real
vasprun. **That is the designed behaviour**: a run that failed becomes a NaN
carrying an explanation, not a dropped row.

A dropped row is how a screen quietly becomes a screen over the subset that
happened to converge — and the bias that introduces points exactly the wrong
way, since the calculations that fail are the difficult, interesting ones."""),

    ("code", """\
mv.dft.read_dos(shortlist, workdir / "runs", level="pbe")

[k for k in shortlist.obs if k.endswith("_pbe")]"""),

    ("markdown", """\
`read_dos` fills the density of states onto the shared energy grid, and derives
the band gap and Fermi level from it. Same story: absent runs are recorded as
absent.

With real output files, `obsm['dos_pbe']` would plot with `mv.pl.spectra` like
any other curve, and the gap would be an ordinary `obs` column that
`mv.screen.filter` can reach.

## Licences

A last piece of bookkeeping that is easy to skip and expensive to get wrong.
Levels of theory carry their licence, so the question "can this result go in a
commercial report" has an answer in the object."""),

    ("code", """\
mv.calc.check_licenses(md)"""),

    ("code", """\
mv.check_commercial_use(md)"""),

    ("markdown", """\
```{seealso}
[Getting data in and out](data_io.ipynb) is the other end of the plumbing.
[Scale and first principles](scale_and_dft.md) covers the same ground in prose,
with more on the failure modes of very large corpora.
```"""),
]

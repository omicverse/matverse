"""Cells for tutorials/data_io.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.

This one hits the network — it queries OQMD over OPTIMADE for real structures.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Getting data in and out

Every other tutorial starts from `mv.datasets`, which is convenient and not how
anyone's work actually begins. Real candidates come from a directory of CIFs, a
database query, an ASE trajectory, or a colleague's pickle.

This page is the full set of doors into and out of the object, and it queries a
real database rather than describing how one would. The point is that **matverse
is not a place your data gets stuck**: everything that goes in comes back out."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## From a database, over OPTIMADE

[OPTIMADE](https://www.optimade.org/) is the common query API that most
materials databases now implement. Unlike the Materials Project's own REST API
it needs no key, so this is the door that works out of the box."""),

    ("code", """\
mv.data.optimade_providers()"""),

    ("markdown", """\
We query [OQMD](https://oqmd.org/) — about 1.4 million entries — for every
binary aluminium–nickel structure it holds."""),

    ("code", """\
query = 'elements HAS ALL "Al","Ni" AND nelements=2'

# OQMD first, then two other OPTIMADE providers if it is unreachable. A
# public database being down for an afternoon is not a fact about matverse,
# and a tutorial that dies on it teaches you to ignore the failure. A bug in
# the client still raises: only network-level failures are caught here, and
# whichever provider answered is printed so the numbers below are traceable.
md = None
for provider in ("oqmd", "alexandria", "jarvis"):
    try:
        md = mv.data.from_optimade(query, provider=provider, max_n=15)
        print(f"answered by: {provider}")
        break
    except (RuntimeError, OSError) as exc:
        print(f"{provider} unavailable: {str(exc)[:90]}")

if md is None:
    raise RuntimeError("no OPTIMADE provider answered; check the network")

md"""),

    ("code", """\
mv.pp.describe(md)
md.obs[["optimade_id", "provider", "formula", "nsites", "density"]].round(3)"""),

    ("markdown", """\
Real data, with its real database identifiers in `obs['optimade_id']`, so any
row can be traced back to the record it came from.

```{note}
Provider endpoints go down, and matverse tries to tell you which kind of nothing
you got. Materials Project's OPTIMADE mirror currently answers `200` with
`data_returned=0` to *any* query including an empty filter — so "check your
filter" would be the wrong advice, and the error says so instead.
```"""),

    ("code", """\
try:
    mv.data.from_optimade("nelements=2", provider="mp", max_n=5)
except ValueError as exc:
    print(f"ValueError: {exc}")"""),

    ("code", """\
try:
    mv.data.from_optimade("nelements=2", provider="nowhere")
except (KeyError, ValueError) as exc:
    print(f"{type(exc).__name__}: {exc}")"""),

    ("markdown", """\
### What a real query gives you that a curated one does not

Fifteen entries, all of them AlNi. Databases contain the same compound many
times — different calculations, different cells, different submissions — and
that is exactly what `mv.pp.dedup` is for."""),

    ("code", """\
mv.pp.qc(md)
mv.pp.dedup(md)
md.uns["dedup"]"""),

    ("markdown", """\
**Seven of fifteen were duplicates.** That is not a contrived example — it is
what happens when you query a real database and then pay a calculator to relax
everything you got.

`dedup` blocks on `(reduced formula, space group)` and runs pymatgen's
`StructureMatcher` inside each block, so the quadratic part stays local."""),

    ("code", """\
md.obs[["optimade_id", "formula", "nsites", "density", "is_duplicate"]].round(3)"""),

    ("markdown", """\
### Cached, so the second call is free

`mv.datasets.fetch` wraps the same query with a disk cache."""),

    ("code", """\
# Same provider fallback as above — the cache cannot help with the first call,
# and a provider outage is not a fact about the cache.
gold = None
for provider in ("oqmd", "alexandria", "jarvis"):
    try:
        gold = mv.datasets.fetch('elements HAS ALL "Cu","Au" AND nelements=2',
                                 provider=provider, max_n=8)
        print(f"answered by: {provider}")
        break
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"{provider} unavailable: {str(exc)[:90]}")

if gold is None:
    raise RuntimeError("no OPTIMADE provider answered; check the network")

mv.pp.describe(gold)
gold.obs[["optimade_id", "formula", "nsites"]]"""),

    ("code", """\
mv.datasets.cached()"""),

    ("code", """\
mv.datasets.cache_dir()"""),

    ("markdown", """\
```{warning}
Set **`MATVERSE_DATA`** on a cluster. The default sits under the system
temporary directory, which on a compute node is usually node-local and wiped
when the job ends — so the cache silently never hits. `$SCRATCH` is where a
downloaded corpus belongs; a home directory, which is small, NFS-backed and
shared, is where it does not.
```

### Materials Project directly

`mv.data.from_mp` uses the native API, which returns computed properties
OPTIMADE does not carry — formation energies, band gaps, magnetic moments. It
needs `MP_API_KEY` in the environment.

```python
md = mv.data.from_mp({'elements': ['Li', 'Fe', 'P', 'O'], 'num_elements': 4})
```

### Parsing a payload you already have

If you fetched the JSON yourself — through a proxy, from a cache, from a
provider matverse does not know — `from_optimade_response` takes the parsed
dictionary. No network."""),

    ("code", """\
response = {"data": [
    {"id": "mp-30", "attributes": {
        "chemical_formula_reduced": "Cu",
        "lattice_vectors": [[0.0, 1.8075, 1.8075],
                            [1.8075, 0.0, 1.8075],
                            [1.8075, 1.8075, 0.0]],
        "cartesian_site_positions": [[0.0, 0.0, 0.0]],
        "species_at_sites": ["Cu"],
        "nsites": 1,
    }},
]}

parsed = mv.data.from_optimade_response(response)
mv.pp.describe(parsed)
parsed.obs[["formula", "nsites", "volume"]].round(3)"""),

    ("markdown", """\
## From structures you already have"""),

    ("code", """\
from pymatgen.core import Lattice, Structure

structures = [
    Structure(Lattice.cubic(3.615), ["Cu"] * 4,
              [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
    Structure(Lattice.cubic(4.050), ["Al"] * 4,
              [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
]
local = mv.data.from_structures(structures, obs=pd.DataFrame({
    "sample": ["ICSD-627117", "ICSD-43423"],
    "measured_a": [3.6149, 4.0495],
}))
local.obs"""),

    ("markdown", """\
Pass a DataFrame and it becomes `obs`, aligned by position.

### From an iterable, when the list does not fit

`from_iterable` consumes a generator, so a million structures never all exist
at once."""),

    ("code", """\
def generated():
    \"\"\"Stand-in for a reader that streams from disk.\"\"\"
    for a in (3.5, 3.6, 3.7, 3.8):
        yield Structure(Lattice.cubic(a), ["Cu"] * 4,
                        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


stream = mv.data.from_iterable(generated())
stream.n_obs, mv.provenance(stream)[-1]"""),

    ("markdown", """\
## From ASE

ASE `Atoms` are the other structure object in the ecosystem, and every
calculator speaks them."""),

    ("code", """\
from ase.build import bulk

from_atoms = mv.data.from_ase([bulk("Cu", "fcc", a=3.615, cubic=True),
                               bulk("Ni", "fcc", a=3.524, cubic=True)])
mv.pp.describe(from_atoms)
from_atoms.obs[["formula", "nsites", "density"]].round(3)"""),

    ("markdown", """\
### From a trajectory or structure file

`from_ase_file` reads anything ASE can — `.traj`, `.xyz`, `.cif`, VASP
`POSCAR` — so a molecular-dynamics trajectory becomes a dataset whose rows are
frames."""),

    ("code", """\
import tempfile
from pathlib import Path
from ase.io import write

workdir = Path(tempfile.mkdtemp())
write(workdir / "frames.xyz", [bulk("Cu", "fcc", a=a, cubic=True)
                               for a in (3.55, 3.60, 3.65, 3.70)])

frames = mv.data.from_ase_file(workdir / "frames.xyz", index=":")
mv.pp.describe(frames)
frames.obs[["formula", "volume"]].round(3)"""),

    ("markdown", """\
## CIFs, out and back

A directory of CIFs is how crystallographic data usually travels."""),

    ("code", """\
mv.data.to_cif(md, workdir / "cifs")
sorted(p.name for p in (workdir / "cifs").iterdir())[:6]"""),

    ("code", """\
round_trip = mv.data.from_cif(workdir / "cifs")
mv.pp.describe(round_trip)
round_trip.obs[["formula", "nsites", "density"]].head(5).round(3)"""),

    ("markdown", """\
Out and back with the formulas intact. `to_cif` names each file from its row,
which is what makes the round trip *identifiable* rather than merely possible.

## Out to the rest of the ecosystem"""),

    ("code", """\
atoms = mv.data.to_ase(from_atoms)
atoms[0], atoms[0].get_chemical_symbols()"""),

    ("code", """\
back = mv.data.to_pymatgen(from_atoms)
back[0].composition.reduced_formula, round(back[0].lattice.a, 4)"""),

    ("markdown", """\
And the object itself is an ordinary `AnnData`, so `write_h5ad` is the archive
format — readable by anndata with matverse absent.

## matminer

[matminer](https://hackingmaterials.lbl.gov/matminer/) is the established
featurisation library, and matverse does not try to replace it: `to_matminer`
hands it a DataFrame, `from_matminer` takes one back, and `mv.feat.matminer`
runs its featurisers against this object."""),

    ("code", """\
try:
    frame = mv.data.to_matminer(from_atoms)
    print(frame.head())
except ImportError as exc:
    print(f"matminer is optional and absent here: {exc}")"""),

    ("code", """\
try:
    rebuilt = mv.data.from_matminer(frame)
    print(rebuilt)
except (ImportError, NameError) as exc:
    print(f"needs matminer: {type(exc).__name__}")"""),

    ("code", """\
try:
    mv.feat.matminer(from_atoms, featurizers=["ElementProperty"])
    print(from_atoms.obsm["X_matminer"].shape)
except (ImportError, Exception) as exc:
    print(f"{type(exc).__name__}: {str(exc)[:90]}")"""),

    ("markdown", """\
```{note}
matminer is an optional dependency (`pip install "matverse[matminer]"`), and an
absent optional backend produces a message naming what to install rather than a
traceback from three layers down.
```

## What survives a round trip"""),

    ("code", """\
print(mv.utils.summary(md))"""),

    ("markdown", """\
```{seealso}
[Getting started](getting_started.ipynb) is the pipeline these doors lead into;
[Infrastructure](infrastructure.ipynb) covers units, checkpoints and getting a
screen onto a cluster.
```"""),
]

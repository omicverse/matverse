# Scale and first principles

The screening tutorial used six candidates. This one is about the two things
that change when there are five million, and about what happens when the survivors
need real DFT.

They belong together because they are the same problem seen from two ends: too
much work for one machine, and too much work for one calculator.

## Getting data in without holding it

Alexandria is 5.06M entries. OMat24 is roughly 110M calculations. A constructor
that takes a list rules both out before anything else does.

```python
import matverse as mv

md = mv.data.from_iterable(structure_stream(), chunk_size=5000)
```

`from_iterable` consumes a stream in blocks and concatenates as it goes. The
element axis is **unioned** across blocks, so a block containing an element no
earlier block had widens the axis rather than failing:

```python
md = mv.data.from_iterable(stream, chunk_size=100)
md.shape          # (500, 11)
mv.provenance(md)
# ['data.from_iterable(chunk_size=100, n=500, n_blocks=5)']
```

The history shows one operation because the caller made one call, whether it took
one block or fifty.

For the file formats training corpora actually arrive in:

```python
md = mv.data.from_ase_file('mptrj.extxyz', max_n=50_000)
```

This reads with ASE's own iterator rather than loading the file, so a
hundred-million-frame corpus can be sampled without being read.

### From a database

OPTIMADE is the primary connector — one protocol against roughly twenty
providers beats twenty bespoke API clients:

```python
md = mv.data.from_optimade('elements HAS ALL "Al","Ni"', provider='mp')
md = mv.data.from_optimade('nelements=2', provider='alexandria')

mv.data.optimade_providers()
# {'mp': ..., 'oqmd': ..., 'alexandria': ..., 'cod': ..., 'jarvis': ...}
```

Any OPTIMADE endpoint works via `base_url=`, whether matverse has heard of it or
not. Reach for a provider's own client only when its payload is richer than
OPTIMADE exposes — which for Materials Project it is, hence `mv.data.from_mp`.

```{note}
`from_optimade_response` parses an already-fetched payload. Splitting parsing
from fetching is deliberate: parsing is deterministic and testable, fetching is
neither, and a function that does both can only be tested against a live server.
```

## Running an operation you cannot run all at once

```python
mv.utils.map_chunks(md, lambda block: mv.calc.energy(block, level='emt'),
                    size=500)
# {'size': 500, 'n_processed': 500, 'n_skipped': 0, 'errors': []}
```

Each block is processed in isolation and what it wrote is merged back onto the
parent — obs columns, feature blocks, and structure variants alike:

```python
mv.utils.map_chunks(md, lambda b: mv.calc.relax(b, level='emt'), size=500)
mv.variants(md)              # ['input', 'relaxed_emt']
```

```{note}
`map_chunks` deliberately does **not** merge `uns`. A per-block `uns` entry is a
statement about that block, and quietly keeping the last one would be wrong — a
screen's criteria and a hull's reference count both mean something different per
block than they do overall.
```

### Surviving a walltime limit

A screen over ten thousand candidates outlives a Slurm allocation. Re-running the
same script should continue, not restart.

```python
mv.utils.map_chunks(md, lambda b: mv.calc.energy(b, level='mace-mpa'),
                    size=500,
                    skip_if='energy_mace-mpa',
                    checkpoint_to='run.h5ad')
# {'size': 500, 'n_processed': 200, 'n_skipped': 300, 'errors': []}
```

`skip_if` names an obs column; blocks whose rows are already finite are skipped.
`checkpoint_to` writes the whole object between blocks. Together they make the
script idempotent: run it again after a kill and it picks up where it stopped.

```python
todo = mv.utils.resume(md, 'energy_mace-mpa')      # boolean mask
todo.sum()
```

`resume` returns a mask rather than mutating anything, because what to do about a
half-finished column is the caller's decision — recompute the failures, or skip
them and record how many.

### A failing block does not stop the run

```python
report = mv.utils.map_chunks(md, expensive_thing, size=500)
report['errors']
# ['rows 1500:2000: RuntimeError: calculator diverged']
```

One bad block is recorded and the rest continue. On a long screen the alternative
— losing four hours because block seven of forty had a bad structure — is not
acceptable behaviour.

### Decoding a window

```python
subset = mv.structures(md, rows=[0, 5, 10])        # decodes three, not 5,000,000
```

Decoding is what costs at scale. Five million serialised structures are a few
gigabytes of strings and several times that as pymatgen objects, so anything
that walks a large dataset must be able to ask for a window.

```{warning}
This is **chunking, not laziness**. The object is still materialised in memory;
`map_chunks` means an operation too expensive to run at once becomes a loop, and
`from_iterable` means construction does not need the whole corpus at once. A
zarr-backed `obs` with on-demand structure resolution is the next step and is not
built.
```

### Getting it onto a cluster

```python
mv.utils.slurm_script('screen.py', 'job.sbatch',
                      partition='normal', hours=8, cpus=8, memory='32GB',
                      gpus=1, setup='source ~/env/matverse/bin/activate')
```

It writes the script and stops. Submitting is a side effect on a shared machine,
and a script you can read before running is worth more than one command less to
type. The generated script redirects `HF_HOME`, `XDG_CACHE_HOME` and
`PIP_CACHE_DIR` to scratch, because a model download into a small shared home
directory is a classic way to wedge a cluster account.

### Units, before they bite

```python
mv.utils.check_units(md)
# {'volume': 'angstrom^3', 'density': 'g/cm^3',
#  'energy_per_atom_emt': 'eV/atom', 'expt_value': None}
```

matverse works in eV and angstrom throughout, so `check_units` fills in what it
produced itself and reports `None` for anything it did not. The risk is at the
boundary — a column pasted from a spreadsheet, or read from a database quoting
kJ/mol:

```python
mv.utils.convert(md, 'formation_energy_expt', 'kJ/mol')
md.obs['formation_energy_expt_ev']
```

Conversions **deposit beside** the original rather than overwriting it. A
converted column next to its source is auditable; a silently rewritten one is the
same bug, one step later.

```python
print(mv.utils.summary(md))
```

renders what the object holds — axes, variants, levels, grids, screens — plus
warnings about a closed hull or a non-commercial level. It is what to read first
when you open an `h5ad` you have not seen in a month.

## Handing the survivors to DFT

matverse runs no DFT and submits no jobs. Workflow management has three good
answers already — atomate2 with jobflow-remote, quacc, AiiDA — and a fourth would
be a maintenance liability with no upside.

What is *not* solved is the boundary. A screen lives in one object, DFT lives in
a directory tree, and the correspondence between them is normally maintained by a
naming convention and someone's memory.

```python
shortlist = md[md.obs['passes']].copy()
mv.dft.write_inputs(shortlist, 'runs/', preset='relax',
                    source='relaxed_emt')
```

One directory per material:

```
runs/
├── 3/
│   ├── INCAR
│   ├── KPOINTS
│   ├── POSCAR
│   ├── POTCAR.spec
│   └── matverse.json      ← which row this belongs to
└── 7/ ...
```

The manifest is what makes the round trip work. `read_outputs` finds each row by
**identity**, so a directory renamed by a workflow manager still resolves — the
usual reason a hand-rolled harvest silently attaches results to the wrong
material.

### Presets carry a level of theory

```python
mv.dft.presets()
# {'relax':  {'reference': 'PBE+U',  ...},
#  'static': {'reference': 'PBE+U',  ...},
#  'bands':  {'reference': 'PBE+U',  ...},
#  'scan':   {'reference': 'r2SCAN', ...},
#  'hse':    {'reference': 'HSE06',  ...}}
```

`scan` and `hse` are not settings, they are different levels of theory, and the
preset records which. A run tagged `scan` arrives as r2SCAN, and
`mv.thermo.hull` will refuse to put it on one hull with PBE.

```{warning}
POTCARs are written as a **specification**, not as files. VASP pseudopotentials
are licensed and cannot be redistributed; point pymatgen at your own with
`PMG_VASP_PSP_DIR` and pass `potcar_spec=False`.
```

### Quantum ESPRESSO

```python
mv.dft.write_inputs(shortlist, 'runs/', code='espresso',
                    pseudopotentials={'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.UPF'})
```

Pseudopotentials are **named, not shipped**, and the placeholder filenames are
deliberately not a default worth trusting. Which set a run used is part of the
level of theory — SSSP efficiency and PSLibrary give different numbers for the
same functional — so guessing a filename would put a silent choice into a result
the object claims to record.

### Waiting

```python
mv.dft.status(md, 'runs/')
# {'n_total': 240, 'n_finished': 187, 'n_missing': 53, 'missing': ['12', ...]}
```

Worth running before harvesting a large campaign: a directory that never started
and one that crashed look identical from the object, and the difference decides
whether to resubmit.

### Harvesting

```python
mv.dft.read_outputs(md, 'runs/', level='pbe')

md.obs[['energy_pbe', 'band_gap_pbe', 'converged_pbe', 'dft_error_pbe']]
mv.variants(md)        # ['input', 'relaxed_emt', 'relaxed_pbe']
```

Rows whose run is missing or unconverged get NaN and a **reason** rather than
being dropped. Which candidates failed is a result — a systematically failing
corner of composition space is worth seeing, and a harvest that silently returns
187 rows where you sent 240 hides it.

The result is an ordinary level of theory, so the screen can now be redone
against DFT rather than against the surrogate that chose the shortlist:

```python
mv.thermo.hull(md, level='pbe', source='relaxed_pbe', references=known_phases)
mv.compare_levels(md, 'energy_per_atom')
#      emt    mace-mpa      pbe
# 3  -0.09      -3.71     -3.68
# 7  -0.11      -3.66     -3.70
```

That table is the point of the whole exercise. The surrogate picked the
shortlist, DFT checked it, and both numbers are still in the object with their
provenance intact — so "how well did the cheap model rank these?" is a question
you can answer rather than one you have to remember the answer to.

```python
mv.pl.parity(md, 'energy_per_atom', 'mace-mpa', 'pbe')
```

## What this does not do

- **No job submission.** By design; see above.
- **No lazy object.** Chunked, not out-of-core.
- **VASP and Quantum ESPRESSO only** for input generation, and VASP only for
  output parsing.
- **No model ships.** `mv.calc` is a registration interface, so a fresh install
  screens with EMT until you register something better. Weights are hundreds of
  megabytes with their own licences, and the Matbench Discovery leaders are
  currently separated by less than the spread between seeds — a hardcoded default
  would be stale on arrival and wrong in a way you could not see.

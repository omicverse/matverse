# Installation

## The short version

```bash
pip install matverse
```

That gives you the whole screening pipeline: data loading, structure
preprocessing, composition features, the `emt` calculator, convex hulls,
screening, and the chemical-space toolchain.

## Core dependencies stay small on purpose

```
anndata  numpy  pandas  scipy  pymatgen  ase
```

That is the entire required set. Everything else is an optional extra, imported
**inside** the function that needs it, failing with an error that names the extra
to install.

This is not minimalism for its own sake. The materials Python ecosystem has real
and current version conflicts — AMSET pins Python `>=3.9` while pymatgen requires
`>=3.11`, `reaction-network`'s PyPI release trails its git by two years, and
matminer has not released since April 2024. A library that pulled all of that
into its install requirements would break environments on contact.
`pip install matverse` must never be the thing that breaks yours.

## Optional extras

```bash
pip install "matverse[analysis]"      # scikit-learn, igraph, leidenalg
pip install "matverse[descriptors]"   # dscribe, for mv.feat.soap
pip install "matverse[mp]"            # mp-api, for Materials Project queries
pip install "matverse[mlip]"          # mace-torch
pip install "matverse[matminer]"      # matminer delegation
```

| Extra | Unlocks | Needed for |
|---|---|---|
| `analysis` | `mv.tl.cluster`, non-Euclidean neighbour metrics | Leiden and k-means routes |
| `descriptors` | `mv.feat.soap` | telling polymorphs apart, which composition descriptors cannot do |
| `mp` | `mv.data.from_mp`, `mv.thermo.references_from_mp` | absolute hulls against known phases |
| `mlip` | `mace-mpa`, `mace-omat` levels | screening with something better than EMT |
| `matminer` | `mv.feat.matminer` | reusing an existing matminer featuriser |

`mv.tl.pca` and `mv.tl.neighbors` work without `analysis`: PCA is an exact SVD
(the element axis is at most 118 columns wide, so there is nothing to
approximate) and the neighbour search falls back to an exact brute-force
computation.

## Which calculator should I install?

`emt` ships working and needs nothing extra. It is real, fast, and parameterised
only for Al, Cu, Ag, Au, Ni, Pd, Pt, H, C, N and O — enough to exercise a
pipeline honestly, not enough to screen with.

For real work you want a machine-learned interatomic potential. matverse ships
**no default** beyond `emt`, deliberately: the Matbench Discovery leaders are
currently separated by less than the spread between seeds and the ranking
reorders monthly, so a hardcoded "best model" would be stale on arrival and wrong
in a way you could not see. Ask what your installation can run:

```python
mv.calc.available()
```

and register whatever you have:

```python
from mace.calculators import mace_mp

mv.calc.register_calculator(
    "mace-mpa", lambda: mace_mp(model="medium-mpa-0"),
    kind="mlip", method="MACE-MPA-0",
    reference="PBE+U (MPtrj + sAlex)", surrogate=True, license="MIT")
```

```{warning}
Model weights are not uniformly open. MACE-MP and MACE-MPA are MIT; MACE-OMAT
and MACE-MATPES are ASL and forbid commercial use; UMA's licence excludes
several countries. matverse records the licence on every level and
`mv.calc.check_licenses(md)` reads it back off the object — but recording is
not the same as clearing, so check the terms before you rely on a result.
```

## Development install

```bash
git clone https://github.com/matverse/matverse
cd matverse
pip install -e ".[dev,analysis]"
pytest -q
```

The suite runs against a real calculator rather than a mock, on a six-material
Al–Cu–Ni library small enough to finish in seconds.

## Building the docs

```bash
pip install -r matverse_guide/requirements.txt
cd matverse_guide/docs
make html
```

`api/user.md` is regenerated from the registry on every build, so a newly
decorated function appears without anyone editing a docs file.

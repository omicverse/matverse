"""``mv.datasets`` — real materials to work on.

Every tutorial and example needs something to run on, and a library whose only
examples are hand-built face-centred cubic cells teaches its own test fixtures
rather than the domain. These are real structures with published lattice
parameters and space groups: LiFePO₄ as it is actually reported, Li₁₀GeP₂S₁₂ as
the solid electrolyte people actually study.

Two kinds of source
-------------------
**Bundled** datasets need no network and no API key. They read from the
structures pymatgen already ships — a hard dependency, so always present — which
avoids both re-distributing crystallographic data and depending on a download
during a tutorial or a test.

**Fetched** datasets pull from Materials Project or any OPTIMADE provider, and
cache to disk so the second call is free. Those need network, and
:func:`from_mp` needs a key.

```python
md = mv.datasets.load('battery_cathodes')   # offline, real, immediate
mv.datasets.available()                     # what is bundled, and what each is for
md = mv.datasets.fetch('Li-Fe-P-O', provider='mp')   # network
```

The bundled sets are deliberately small — three to six materials each. They
exist so an example runs in seconds and its output fits on a screen, not to be
a screening campaign. :func:`fetch` is where a real candidate library comes
from.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from anndata import AnnData

from ._registry import register_function

#: Curated sets, each one a coherent screening scenario rather than an arbitrary
#: grab-bag. The comment on each says what it is *for*, because which materials
#: to use is the first question an example has to answer.
BUNDLED = {
    "battery_cathodes": {
        "structures": ["LiFePO4", "NaFePO4", "Li3V2(PO4)3"],
        "description": "Olivine and NASICON cathode materials. Li and Na "
                       "analogues of the same framework sit side by side, so "
                       "an intercalation comparison has something to compare.",
        "use_for": "mv.neb migration barriers, mv.thermo.hull stability",
    },
    "solid_electrolytes": {
        "structures": ["Li10GeP2S12", "Li2O", "Li2O2"],
        "description": "A superionic conductor and two simple lithium oxides. "
                       "Li10GeP2S12 is the canonical fast-ion conductor and is "
                       "58 atoms, so it is also a realistic cost test.",
        "use_for": "mv.md.conductivity, mv.neb.barrier",
    },
    "oxides": {
        "structures": ["SrTiO3", "TiO2", "VO2", "BaNiO3", "SiO2"],
        "description": "Perovskite, rutile, and a metal-insulator transition "
                       "compound. Spans wide-gap insulator to correlated metal.",
        "use_for": "mv.prop.xrd, mv.thermo.hull, band gap screening",
    },
    "semiconductors": {
        "structures": ["Si", "SiO2", "TlBiSe2", "Sn"],
        "description": "Silicon and its oxide, a topological insulator, and "
                       "grey tin. Band gaps from zero to nine electronvolts.",
        "use_for": "mv.dft.read_dos, electronic structure screening",
    },
    "simple": {
        "structures": ["Si", "CsCl", "Li2O", "Graphite"],
        "description": "Four small textbook structures, none above four atoms "
                       "except graphite. For a first look and for doctests.",
        "use_for": "getting started",
    },
}

#: Elemental metals from published room-temperature lattice parameters, for the
#: EMT calculator — which is parameterised only for these plus H, C, N and O, and
#: is the only calculator that ships working. Values in angstrom.
FCC_METALS = {
    "Al": 4.0495, "Cu": 3.6149, "Ni": 3.5240, "Ag": 4.0853,
    "Au": 4.0782, "Pd": 3.8907, "Pt": 3.9242,
}


def _pymatgen_structures_dir() -> Path:
    """Where pymatgen keeps the structures it ships.

    Located from the package directory rather than by reading
    ``pymatgen.util.testing.STRUCTURES_DIR``. That module imports pytest at
    module scope, so asking it for a path made ``mv.datasets.load`` fail with
    ``No module named 'pytest'`` on any installation that does not happen to
    have the test tooling — which is most of them. The constant is only a
    ``Path`` join; there is nothing to gain by importing a test helper to get
    it.
    """
    import pymatgen

    roots = [Path(p) for p in getattr(pymatgen, "__path__", [])]
    for root in roots:
        for candidate in (root / "util" / "structures",
                          root / "util" / "testing" / "structures"):
            if candidate.is_dir():
                return candidate

    # Fall back on the constant for a layout we do not know about, accepting
    # the pytest dependency rather than failing outright.
    try:
        import pymatgen.util.testing as testing
        directory = getattr(testing, "STRUCTURES_DIR", None)
        if directory is not None and Path(directory).is_dir():
            return Path(directory)
    except ImportError:
        pass

    raise FileNotFoundError(
        "cannot find pymatgen's bundled structures; matverse reads them "
        "from there rather than shipping its own copy. Use "
        "mv.datasets.metals() or mv.datasets.fetch() instead.")


@register_function(
    aliases=["available datasets", "list datasets", "which datasets",
             "what data is there", "example data"],
    category="datasets",
    description="List the bundled datasets, what each contains and which "
                "analysis it is meant for.",
    examples=["mv.datasets.available()"],
    related=["mv.datasets.load", "mv.datasets.fetch"],
)
def available() -> dict:
    """Bundled dataset names mapped to what they hold and what they are for."""
    return {name: dict(meta) for name, meta in BUNDLED.items()}


@register_function(
    aliases=["load dataset", "example dataset", "sample data", "load data",
             "real structures", "tutorial data"],
    category="datasets",
    description="Load a bundled dataset of real published structures, with no "
                "network and no API key.",
    produces={"structures": ["input"], "X": ["composition"],
              "obs": ["name", "spacegroup", "dataset"]},
    examples=["md = mv.datasets.load('battery_cathodes')",
              "md = mv.datasets.load('simple')"],
    related=["mv.datasets.available", "mv.datasets.fetch",
             "mv.datasets.metals"],
    notes="Real structures with published lattice parameters, read from the set "
          "pymatgen ships rather than re-distributed here — which keeps the "
          "package small and the provenance clear.\n\n"
          "Deliberately three to six materials each. They exist so an example "
          "runs in seconds and its output fits on a screen; mv.datasets.fetch "
          "is where a real candidate library comes from.",
)
def load(name: str) -> AnnData:
    """A bundled dataset of real structures."""
    import pandas as pd
    from pymatgen.core import Structure

    from .data import from_structures

    if name not in BUNDLED:
        raise KeyError(f"no dataset {name!r}; available: {sorted(BUNDLED)}. "
                       f"mv.datasets.available() says what each is for.")

    directory = _pymatgen_structures_dir()
    wanted = BUNDLED[name]["structures"]
    built, rows = [], []
    for stem in wanted:
        path = directory / f"{stem}.json"
        if not path.exists():
            continue
        structure = Structure.from_file(str(path))
        try:
            spacegroup = structure.get_space_group_info()[0]
        except Exception:
            spacegroup = ""
        built.append(structure)
        rows.append({"name": stem, "spacegroup": spacegroup, "dataset": name,
                     "source": "pymatgen bundled structures"})

    if not built:
        raise FileNotFoundError(
            f"none of {wanted} was found in {directory}; the set pymatgen "
            f"ships has changed. mv.datasets.metals() needs no files.")

    md = from_structures(built, pd.DataFrame(rows))
    md.uns["dataset"] = {"name": name, **BUNDLED[name],
                         "n_requested": len(wanted), "n_loaded": len(built)}
    return md


@register_function(
    aliases=["metals", "elemental metals", "fcc metals", "emt metals",
             "test metals"],
    category="datasets",
    description="Build the face-centred cubic metals from their published "
                "room-temperature lattice parameters — the elements the EMT "
                "calculator can actually run.",
    produces={"structures": ["input"], "X": ["composition"],
              "obs": ["name", "lattice_parameter"]},
    examples=["md = mv.datasets.metals()",
              "md = mv.datasets.metals(['Cu', 'Ag', 'Au'])"],
    related=["mv.datasets.load", "mv.calc.available"],
    notes="EMT is the only calculator matverse ships working, and it is "
          "parameterised for Al, Cu, Ag, Au, Ni, Pd, Pt plus H, C, N and O. "
          "These seven are therefore the materials on which every example can "
          "run end to end without downloading a model — which is why they get "
          "a function of their own rather than being typed out in each "
          "tutorial.",
)
def metals(symbols=None, supercell=None) -> AnnData:
    """Elemental fcc metals, from published lattice parameters."""
    import pandas as pd
    from pymatgen.core import Lattice, Structure

    from .data import from_structures

    wanted = list(symbols) if symbols is not None else list(FCC_METALS)
    unknown = [s for s in wanted if s not in FCC_METALS]
    if unknown:
        raise KeyError(f"no lattice parameter for {unknown}; have "
                       f"{sorted(FCC_METALS)}")

    built, rows = [], []
    for symbol in wanted:
        a = FCC_METALS[symbol]
        structure = Structure(Lattice.cubic(a), [symbol] * 4,
                              [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        if supercell is not None:
            structure.make_supercell(list(supercell))
        built.append(structure)
        rows.append({"name": symbol, "lattice_parameter": a,
                     "source": "published room-temperature lattice parameter"})

    md = from_structures(built, pd.DataFrame(rows))
    md.uns["dataset"] = {"name": "metals", "n_loaded": len(built),
                         "description": "elemental fcc metals EMT can run"}
    return md


def cache_dir() -> Path:
    """Where fetched datasets are kept.

    Follows ``MATVERSE_DATA`` when set, then ``XDG_CACHE_HOME``, then
    ``~/.cache``. On a cluster the first is the one to set — a home directory is
    usually small, NFS-backed and shared, and a downloaded corpus does not
    belong there.
    """
    override = os.environ.get("MATVERSE_DATA")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "matverse"


@register_function(
    aliases=["fetch", "download dataset", "get real data", "query database",
             "fetch structures", "candidate library"],
    category="datasets",
    description="Fetch a real candidate library from Materials Project or any "
                "OPTIMADE provider, caching it so the second call costs "
                "nothing.",
    produces={"structures": ["input"], "X": ["composition"]},
    dispatch="provider='mp' uses the Materials Project client and needs "
             "MP_API_KEY; any other name is looked up as an OPTIMADE provider",
    examples=["md = mv.datasets.fetch('Li-Fe-P-O', provider='mp')",
              "md = mv.datasets.fetch('nelements=2 AND elements HAS \"Ni\"', "
              "provider='alexandria')"],
    related=["mv.data.from_mp", "mv.data.from_optimade",
             "mv.datasets.cached"],
    notes="The query means different things per provider: a chemical system for "
          "Materials Project, an OPTIMADE filter expression otherwise. That is "
          "not a matverse inconsistency — it is what each API takes, and "
          "translating between them would silently change what was asked for.\n"
          "\nCaches to MATVERSE_DATA, XDG_CACHE_HOME, or ~/.cache. Set the "
          "first on a cluster: a home directory is small, NFS-backed and "
          "shared, and a downloaded corpus does not belong there.",
)
def fetch(query: str, provider: str = "mp", max_n: int = 200,
          refresh: bool = False, api_key: str | None = None) -> AnnData:
    """Fetch and cache a dataset. Needs network."""
    import anndata

    from ._core import record

    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stem = _cache_key(query, provider, max_n)
    path = directory / f"{stem}.h5ad"

    if path.exists() and not refresh:
        try:
            md = anndata.read_h5ad(path)
        except Exception as exc:
            # A cache entry written by a different anndata version, or a file
            # truncated by a job that was killed mid-write, must not be fatal:
            # a cache exists to make things faster, and one that can make a
            # working call fail is worse than no cache at all. Re-fetch.
            warnings.warn(
                f"the cached {path.name} could not be read "
                f"({type(exc).__name__}), so it is being fetched again. "
                f"Delete it if this repeats.", stacklevel=2)
        else:
            md.uns.setdefault("dataset", {})["from_cache"] = str(path)
            return md

    from .data import from_mp, from_optimade

    if provider == "mp":
        md = from_mp({"chemsys": query} if "-" in query else {"formula": query},
                     api_key=api_key, max_n=max_n)
    else:
        md = from_optimade(query, provider=provider, max_n=max_n)

    md.uns["dataset"] = {"name": f"{provider}:{query}", "provider": provider,
                         "query": query, "n_loaded": int(md.n_obs),
                         "cached_at": str(path)}
    md.write_h5ad(path)
    record(md, "datasets.fetch", provider=provider, query=query,
           n=int(md.n_obs))
    return md


def _cache_key(query: str, provider: str, max_n: int) -> str:
    """A filename that is stable, readable, and safe on every filesystem."""
    import hashlib

    digest = hashlib.sha1(f"{provider}|{query}|{max_n}".encode()).hexdigest()[:10]
    readable = "".join(c if c.isalnum() else "_" for c in query)[:40]
    return f"{provider}_{readable}_{digest}"


@register_function(
    aliases=["cached datasets", "what is cached", "list cache", "clear cache"],
    category="datasets",
    description="List the datasets already fetched and cached on this machine, "
                "with their sizes.",
    examples=["mv.datasets.cached()"],
    related=["mv.datasets.fetch"],
)
def cached() -> list:
    """What has been fetched already, newest first."""
    directory = cache_dir()
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.h5ad"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        out.append({"path": str(path),
                    "size_mb": round(path.stat().st_size / 1e6, 2)})
    return out


__all__ = ["available", "load", "metals", "fetch", "cached", "cache_dir",
           "BUNDLED", "FCC_METALS"]

"""``mv.data`` — get materials in, and back out.

Entry points for the shapes the ecosystem already emits. None of them invents a
format, and every ``from_*`` has a matching way back, because a one-way adapter
is an adoption dead end.

Constructors are the one place matverse returns instead of depositing: there is
no object yet to deposit into.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import new, record, structures
from ._registry import register_function


@register_function(
    aliases=["from structures", "build dataset", "new dataset",
             "from pymatgen", "load structures"],
    category="data",
    description="Build a matverse dataset from a list of pymatgen Structure "
                "objects, with the composition matrix and element axis built in.",
    produces={"structures": ["input"], "X": ["composition"], "var": ["Z"]},
    examples=["md = mv.data.from_structures([s1, s2, s3])",
              "md = mv.data.from_structures(structs, obs=metadata)"],
    related=["mv.data.from_cif", "mv.data.from_ase", "mv.pp.describe"],
)
def from_structures(structures_: list, obs: pd.DataFrame | None = None,
                    build_X: bool = True) -> AnnData:
    """From a bare list of pymatgen ``Structure`` objects."""
    return new(structures_, obs, source="data.from_structures", build_X=build_X)


@register_function(
    aliases=["from cif", "read cif", "load cif files", "cif directory"],
    category="data",
    description="Build a dataset from CIF files, either a directory of them or "
                "an explicit list of paths, recording each file's name in obs.",
    produces={"structures": ["input"], "obs": ["source_file"],
              "X": ["composition"]},
    examples=["md = mv.data.from_cif('structures/')",
              "md = mv.data.from_cif(['a.cif', 'b.cif'])"],
    related=["mv.data.from_structures", "mv.data.to_cif"],
)
def from_cif(path, pattern: str = "*.cif") -> AnnData:
    """From a directory of CIFs, or a list of paths."""
    from pathlib import Path

    from pymatgen.core import Structure

    if isinstance(path, (str, Path)):
        root = Path(path)
        paths = sorted(root.glob(pattern)) if root.is_dir() else [root]
    else:
        paths = [Path(p) for p in path]
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {path}")

    loaded, names, failed = [], [], []
    for p in paths:
        try:
            loaded.append(Structure.from_file(str(p)))
            names.append(p.name)
        except Exception as exc:
            failed.append(f"{p.name}: {type(exc).__name__}")
    if not loaded:
        raise ValueError(f"no CIF parsed successfully; {len(failed)} failed: "
                         f"{failed[:3]}")

    md = new(loaded, pd.DataFrame({"source_file": names}),
             source="data.from_cif")
    if failed:
        md.uns["read_errors"] = failed
    return md


@register_function(
    aliases=["from matminer", "adopt dataframe", "matminer dataframe",
             "from pandas"],
    category="data",
    description="Adopt a matminer-style DataFrame with a structure column, "
                "turning its numeric columns into a feature block and the rest "
                "into obs.",
    produces={"structures": ["input"], "obsm": ["X_matminer"],
              "X": ["composition"]},
    examples=["md = mv.data.from_matminer(df)",
              "md = mv.data.from_matminer(df, structure_col='final_structure')"],
    related=["mv.data.to_matminer"],
    notes="matminer's convention is already 'rows are materials', which is why "
          "the adoption is lossless in both directions.",
)
def from_matminer(df: pd.DataFrame, structure_col: str = "structure") -> AnnData:
    """Adopt a matminer-style DataFrame."""
    if structure_col not in df.columns:
        raise KeyError(f"no {structure_col!r} column; matminer's convention "
                       f"puts pymatgen Structures there. Columns: "
                       f"{list(df.columns)}")
    meta = [c for c in df.columns if c != structure_col]
    numeric = [c for c in meta if pd.api.types.is_numeric_dtype(df[c])]
    other = [c for c in meta if c not in numeric]

    md = new(list(df[structure_col]), df[other] if other else None,
             source="data.from_matminer")
    if numeric:
        md.obsm["X_matminer"] = df[numeric].to_numpy(dtype=float)
        md.uns["features"]["X_matminer"] = {"names": numeric,
                                            "featurizer": "matminer"}
    return md


@register_function(
    aliases=["to matminer", "to dataframe", "export dataframe"],
    category="data",
    description="Flatten a dataset back to the DataFrame the wider ecosystem "
                "reads, with structures in a column and feature blocks expanded.",
    requires={"structures": ["{variant}"]},
    examples=["df = mv.data.to_matminer(md)",
              "df = mv.data.to_matminer(md, variant='relaxed_emt')"],
    related=["mv.data.from_matminer"],
)
def to_matminer(md: AnnData, variant: str = "input") -> pd.DataFrame:
    """Back to a flat DataFrame."""
    df = md.obs.copy().reset_index(drop=True)
    df.insert(0, "structure", structures(md, variant))
    for block, meta in md.uns.get("features", {}).items():
        if block not in md.obsm:
            continue
        arr = md.obsm[block]
        for i, name in enumerate(meta.get("names", [])):
            if i < arr.shape[1]:
                df[name] = arr[:, i]
    return df


@register_function(
    aliases=["from ase", "adopt ase atoms", "ase atoms", "from atoms"],
    category="data",
    description="Build a dataset from ASE Atoms objects, keeping per-atom "
                "arrays rather than dropping them.",
    produces={"structures": ["input"], "uns": ["sites"], "X": ["composition"]},
    examples=["md = mv.data.from_ase(atoms_list)"],
    related=["mv.data.to_ase"],
    notes="Per-atom arrays go to uns['sites'] as one record per material. That "
          "is honest but not vectorised; a proper site axis is the open design "
          "problem named in the README.",
)
def from_ase(atoms_list: list, obs: pd.DataFrame | None = None) -> AnnData:
    """From ASE ``Atoms``. Per-atom arrays are kept, not silently dropped."""
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    md = new([adaptor.get_structure(a) for a in atoms_list], obs,
             source="data.from_ase")
    md.uns["sites"] = [
        {k: v.tolist() for k, v in a.arrays.items()
         if k not in ("positions", "numbers")}
        for a in atoms_list]
    return md


@register_function(
    aliases=["to ase", "export ase atoms", "as atoms"],
    category="data",
    description="Convert a structure variant to a list of ASE Atoms objects.",
    requires={"structures": ["{variant}"]},
    examples=["atoms = mv.data.to_ase(md)",
              "atoms = mv.data.to_ase(md, variant='relaxed_emt')"],
    related=["mv.data.from_ase", "mv.calc.relax"],
)
def to_ase(md: AnnData, variant: str = "input") -> list:
    """To ASE ``Atoms``."""
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    return [adaptor.get_atoms(s) for s in structures(md, variant)]


@register_function(
    aliases=["to pymatgen", "get structures", "structure list"],
    category="data",
    description="Return one structure variant as a plain list of pymatgen "
                "Structure objects.",
    requires={"structures": ["{variant}"]},
    examples=["structs = mv.data.to_pymatgen(md, variant='relaxed_emt')"],
    related=["mv.data.from_structures"],
)
def to_pymatgen(md: AnnData, variant: str = "input") -> list:
    """One structure variant, as a list."""
    return structures(md, variant)


@register_function(
    aliases=["to cif", "write cif", "export structures"],
    category="data",
    description="Write every structure of one variant to a CIF file in a "
                "directory, named by obs index.",
    requires={"structures": ["{variant}"]},
    produces={"files": ["<directory>/<name>.cif"]},
    examples=["mv.data.to_cif(md, 'out/', variant='relaxed_emt')"],
    related=["mv.data.from_cif"],
)
def to_cif(md: AnnData, directory, variant: str = "input") -> list:
    """Write a variant out as CIFs; returns the paths written."""
    from pathlib import Path

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, s in zip(md.obs_names, structures(md, variant)):
        path = root / f"{name}.cif"
        s.to(filename=str(path))
        written.append(str(path))
    return written


#: OPTIMADE providers matverse knows a base URL for. The protocol is the point —
#: any provider implementing it works by passing base_url directly.
OPTIMADE_PROVIDERS = {
    "mp": "https://optimade.materialsproject.org/v1",
    "oqmd": "https://oqmd.org/optimade/v1",
    "alexandria": "https://alexandria.icams.rub.de/pbe/v1",
    "cod": "https://www.crystallography.net/cod/optimade/v1",
    "mpds": "https://api.mpds.io/v1",
    "nmd": "https://nomad-lab.eu/prod/rae/optimade/v1",
    "jarvis": "https://jarvis.nist.gov/optimade/jarvisdft/v1",
    "odbx": "https://optimade.odbx.science/v1",
}


@register_function(
    aliases=["optimade", "from optimade", "query optimade", "federated query",
             "search databases"],
    category="data",
    description="Query any OPTIMADE-compliant materials database with one "
                "filter expression and build a dataset from the result.",
    produces={"structures": ["input"], "obs": ["optimade_id", "provider"],
              "X": ["composition"], "levels": ["{provider}"]},
    examples=["md = mv.data.from_optimade('elements HAS ALL \"Al\",\"Ni\"', "
              "provider='mp')",
              "md = mv.data.from_optimade('nelements=2', "
              "base_url='https://optimade.example.org/v1')"],
    related=["mv.data.from_mp", "mv.data.optimade_providers"],
    notes="One protocol against roughly twenty providers beats twenty bespoke "
          "API clients, and is the reason this is the primary connector rather "
          "than mv.data.from_mp. A provider's own client is worth reaching for "
          "only when its payload is richer than OPTIMADE exposes — which for "
          "Materials Project it is.",
)
def from_optimade(filter: str, provider: str = "mp",
                  base_url: str | None = None, max_n: int = 100,
                  timeout: float = 60.0) -> AnnData:
    """Query an OPTIMADE endpoint. Needs network access."""
    import json
    import urllib.parse
    import urllib.request

    url = base_url or OPTIMADE_PROVIDERS.get(provider)
    if url is None:
        raise ValueError(
            f"unknown provider {provider!r}; known: "
            f"{sorted(OPTIMADE_PROVIDERS)}. Any OPTIMADE endpoint works — pass "
            f"base_url= directly.")

    query = urllib.parse.urlencode({"filter": filter,
                                    "page_limit": min(int(max_n), 100)})
    request = urllib.request.Request(
        f"{url.rstrip('/')}/structures?{query}",
        headers={"User-Agent": "matverse", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"OPTIMADE query to {url} failed: {type(exc).__name__}: {exc}"
        ) from exc

    return from_optimade_response(payload, provider=provider, max_n=max_n)


@register_function(
    aliases=["parse optimade", "optimade response", "from optimade json",
             "adopt optimade"],
    category="data",
    description="Build a dataset from an already-fetched OPTIMADE response, "
                "for cached payloads and for tests that must not hit a network.",
    produces={"structures": ["input"], "obs": ["optimade_id", "provider"],
              "X": ["composition"], "levels": ["{provider}"]},
    examples=["md = mv.data.from_optimade_response(payload)"],
    related=["mv.data.from_optimade"],
    notes="Separate from the query on purpose. Parsing is deterministic and "
          "testable; fetching is neither, and a function that does both can "
          "only be tested against a live server.",
)
def from_optimade_response(payload: dict, provider: str = "optimade",
                           max_n: int | None = None) -> AnnData:
    """Parse an OPTIMADE ``/structures`` response into a dataset."""
    from ._core import set_level

    entries = payload.get("data", [])
    if max_n is not None:
        entries = entries[:max_n]
    if not entries:
        raise ValueError("the OPTIMADE response contains no structures; check "
                         "the filter")

    built, rows, failed = [], [], []
    for entry in entries:
        attributes = entry.get("attributes", {})
        try:
            built.append(_structure_from_optimade(attributes))
        except Exception as exc:
            failed.append(f"{entry.get('id')}: {type(exc).__name__}: {exc}")
            continue
        rows.append({
            "optimade_id": str(entry.get("id", "")),
            "provider": provider,
            "formula": str(attributes.get("chemical_formula_reduced", "")),
            "nelements": attributes.get("nelements", np.nan),
        })

    if not built:
        raise ValueError(f"no structure parsed; {len(failed)} failed: "
                         f"{failed[:3]}")

    md = new(built, pd.DataFrame(rows), source="data.from_optimade")
    set_level(md, provider, kind="dft", method=f"OPTIMADE provider {provider}",
              reference=None, surrogate=False, license=None, uncertainty=None,
              n_returned=len(entries))
    if failed:
        md.uns["read_errors"] = failed
    return md


def _structure_from_optimade(attributes: dict):
    """A pymatgen Structure from OPTIMADE structure attributes.

    OPTIMADE gives cartesian positions and lattice vectors in angstrom, which is
    what pymatgen wants; the conversion is naming rather than arithmetic. Partial
    occupancy arrives as a species list, and is kept rather than silently
    collapsed to the majority element.
    """
    from pymatgen.core import Lattice, Structure

    lattice = attributes.get("lattice_vectors")
    positions = attributes.get("cartesian_site_positions")
    species_at_sites = attributes.get("species_at_sites")
    if lattice is None or positions is None or species_at_sites is None:
        raise ValueError("response lacks lattice_vectors, "
                         "cartesian_site_positions or species_at_sites; the "
                         "provider may need response_fields set")

    definitions = {s["name"]: s for s in attributes.get("species", [])}
    occupancies = []
    for name in species_at_sites:
        definition = definitions.get(name)
        if definition is None:
            occupancies.append({name: 1.0})
            continue
        elements = definition.get("chemical_symbols", [name])
        fractions = definition.get("concentration", [1.0] * len(elements))
        occupancies.append({e: float(f) for e, f in zip(elements, fractions)
                            if e != "vacancy"})

    return Structure(Lattice(np.asarray(lattice, dtype=float)),
                     occupancies, np.asarray(positions, dtype=float),
                     coords_are_cartesian=True)


@register_function(
    aliases=["optimade providers", "which databases", "list providers",
             "available databases"],
    category="data",
    description="List the OPTIMADE providers matverse knows a base URL for.",
    examples=["mv.data.optimade_providers()"],
    related=["mv.data.from_optimade"],
)
def optimade_providers() -> dict:
    """Known provider names mapped to their OPTIMADE base URLs."""
    return dict(OPTIMADE_PROVIDERS)


@register_function(
    aliases=["from materials project", "from mp", "query materials project",
             "mp api", "download materials"],
    category="data",
    description="Query the Materials Project and build a dataset from the "
                "returned summary documents.",
    produces={"structures": ["input"], "obs": ["material_id", "formula"],
              "levels": ["mp"], "X": ["composition"]},
    examples=["md = mv.data.from_mp({'elements': ['Fe', 'O'], "
              "'num_elements': (2, 2)})"],
    related=["mv.thermo.references_from_mp"],
    notes="Kept thin: mp-api owns the query language. Note that Materials "
          "Project moved to Delta-table-backed products in the 2026.04.13 "
          "release and requires mp-api >= 0.46.2.",
)
def from_mp(criteria: dict, api_key: str | None = None,
            max_n: int = 200) -> AnnData:
    """Query the Materials Project."""
    import os

    try:
        from mp_api.client import MPRester
    except ImportError as exc:                            # pragma: no cover
        raise ImportError("mv.data.from_mp needs `pip install matverse[mp]`"
                          ) from exc
    from ._core import set_level

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("set MP_API_KEY or pass api_key=")
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(**criteria)[:max_n]

    obs = pd.DataFrame({
        "material_id": [str(d.material_id) for d in docs],
        "formula": [d.formula_pretty for d in docs],
        "energy_per_atom_mp": [d.energy_per_atom for d in docs],
        "e_above_hull_mp": [d.energy_above_hull for d in docs],
        "band_gap_mp": [d.band_gap for d in docs],
    })
    md = new([d.structure for d in docs], obs, source="data.from_mp")
    set_level(md, "mp", kind="dft", method="PBE+U (Materials Project)",
              reference=None, surrogate=False, license="CC-BY-4.0",
              uncertainty=None, provider="Materials Project", criteria=criteria)
    return md


__all__ = ["from_structures", "from_cif", "from_matminer", "to_matminer",
           "from_optimade", "from_optimade_response", "optimade_providers",
           "OPTIMADE_PROVIDERS",
           "from_ase", "to_ase", "to_pymatgen", "to_cif", "from_mp"]

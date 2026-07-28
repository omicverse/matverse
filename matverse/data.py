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
    produces={"structures": ["input"], "X": ["composition"]},
    examples=["md = mv.data.from_ase(atoms_list)",
              "sites = mv.multi.sites(mv.data.from_ase(atoms_list))"],
    related=["mv.data.to_ase", "mv.multi.sites"],
    notes="Per-atom arrays — magnetic moments, charges, tags — become site "
          "properties on the structure itself, so they travel with it, "
          "serialise with it, and appear as columns the moment you build a "
          "sites object with mv.multi.sites.\n\n"
          "They are deliberately not put in uns. Per-atom data is aligned to a "
          "material, and uns does not subset with the object: md[mask] would "
          "keep every record while dropping rows, leaving each surviving row "
          "pointing at the wrong atoms. That is the same failure structures "
          "themselves had before v0.1.1.",
)
def from_ase(atoms_list: list, obs: pd.DataFrame | None = None) -> AnnData:
    """From ASE ``Atoms``. Per-atom arrays are kept, not silently dropped."""
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    built = []
    for atoms in atoms_list:
        structure = adaptor.get_structure(atoms)
        for key, values in atoms.arrays.items():
            if key in ("positions", "numbers"):
                continue                     # already the structure's own state
            values = np.asarray(values)
            if len(values) == len(structure):
                structure.add_site_property(key, values.tolist())
        built.append(structure)
    return new(built, obs, source="data.from_ase")


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


@register_function(
    aliases=["from iterable", "streaming", "build in chunks", "large dataset",
             "from generator", "millions of structures"],
    category="data",
    description="Build a dataset from an iterable of structures in blocks, "
                "concatenating as it goes, so a corpus larger than memory can "
                "be read without materialising it.",
    produces={"structures": ["input"], "X": ["composition"]},
    examples=["md = mv.data.from_iterable(read_cifs(), chunk_size=5000)",
              "md = mv.data.from_iterable(gen, chunk_size=1000, max_n=50000)"],
    related=["mv.data.from_structures", "mv.utils.map_chunks"],
    notes="Element axes are unioned across blocks by anndata's concat, so a "
          "block containing an element no earlier block had widens the axis "
          "rather than failing. Alexandria is 5.06M entries and OMat24 is "
          "roughly 110M calculations; a constructor taking a list rules both "
          "out before anything else does.",
)
def from_iterable(source, chunk_size: int = 5000, max_n: int | None = None,
                  obs_source=None) -> AnnData:
    """Build a dataset from a stream of structures, block by block."""
    import anndata

    blocks, buffer, meta, seen = [], [], [], 0
    metadata = iter(obs_source) if obs_source is not None else None

    for structure in source:
        buffer.append(structure)
        if metadata is not None:
            meta.append(next(metadata, {}))
        seen += 1
        if len(buffer) >= chunk_size:
            blocks.append(_block(buffer, meta))
            buffer, meta = [], []
        if max_n is not None and seen >= max_n:
            break
    if buffer:
        blocks.append(_block(buffer, meta))

    if not blocks:
        raise ValueError("the iterable yielded no structures")
    if len(blocks) == 1:
        md = blocks[0]
    else:
        md = anndata.concat(blocks, join="outer", index_unique=None,
                            uns_merge="first", merge="first")
        md.obs_names = [str(i) for i in range(md.n_obs)]
        md.X = md.X.tocsr() if hasattr(md.X, "tocsr") else md.X
        _restore_uns(md, blocks[0])

    # Each block recorded its own construction. The caller made one call, so
    # the history should show one, whether it took one block or fifty.
    md.uns["provenance"] = []
    record(md, "data.from_iterable", chunk_size=chunk_size, n=int(md.n_obs),
           n_blocks=len(blocks))
    return md


def _block(structures_: list, meta: list) -> AnnData:
    frame = pd.DataFrame(meta) if meta else None
    return new(list(structures_), frame, source="data.from_iterable")


def _restore_uns(md: AnnData, template: AnnData) -> None:
    """Concat keeps only the first block's uns; the conventions must survive.

    Provenance is cleared rather than seeded: each block recorded its own
    construction, and the caller wants one entry for the stream, which
    :func:`from_iterable` appends after this returns.
    """
    md.uns.setdefault("features", {})
    md.uns.setdefault("levels", {})
    md.uns["provenance"] = []
    md.uns["X_is"] = template.uns.get("X_is", "composition_atoms_reduced")


@register_function(
    aliases=["from extxyz", "read xyz", "trajectory", "from ase db",
             "training corpus", "read extxyz"],
    category="data",
    description="Read structures from an extended-XYZ file or ASE database, in "
                "blocks, which is the shape machine-learned-potential training "
                "corpora arrive in.",
    produces={"structures": ["input"], "X": ["composition"]},
    examples=["md = mv.data.from_ase_file('mptrj.extxyz', max_n=10000)"],
    related=["mv.data.from_iterable", "mv.data.from_ase"],
    notes="Reads with ASE's own iterator rather than loading the file, so a "
          "hundred-million-frame corpus can be sampled without being read.",
)
def from_ase_file(path, index: str = ":", chunk_size: int = 5000,
                  max_n: int | None = None) -> AnnData:
    """From an extended-XYZ file, ASE database, or anything ASE can read."""
    try:
        from ase.io import iread
    except ImportError as exc:                            # pragma: no cover
        raise ImportError("mv.data.from_ase_file needs ase") from exc
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()

    def stream():
        for atoms in iread(str(path), index=index):
            yield adaptor.get_structure(atoms)

    md = from_iterable(stream(), chunk_size=chunk_size, max_n=max_n)
    md.uns["source_file"] = str(path)
    return md


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
    url = base_url or OPTIMADE_PROVIDERS.get(provider)
    if url is None:
        raise ValueError(
            f"unknown provider {provider!r}; known: "
            f"{sorted(OPTIMADE_PROVIDERS)}. Any OPTIMADE endpoint works — pass "
            f"base_url= directly.")

    params = {"filter": filter, "page_limit": min(int(max_n), 100)}
    endpoint = f"{url.rstrip('/')}/structures"
    headers = {"User-Agent": "matverse", "Accept": "application/json"}
    try:
        payload = _get_json(endpoint, params, headers, timeout)
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
        # "check the filter" is the wrong advice when the provider returned
        # nothing for any query at all — Materials Project's OPTIMADE mirror
        # answers 200 with data_returned=0 to an *empty* filter, and a user
        # sent to debug their filter will not find the problem there.
        available = payload.get("meta", {}).get("data_returned")
        if available == 0:
            raise ValueError(
                "the OPTIMADE endpoint returned no structures and reports "
                "data_returned=0, which means the provider itself is serving "
                "nothing rather than your filter matching nothing. Try another "
                "provider — mv.data.optimade_providers() lists them — or pass "
                "base_url= for an endpoint you trust.")
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


def _get_json(endpoint: str, params: dict, headers: dict, timeout: float):
    """Fetch JSON, preferring requests over urllib.

    Not a style preference. ``urllib`` verifies TLS against the interpreter's
    system certificate bundle, and on a cluster that bundle is frequently
    stale or absent — every HTTPS call then dies with
    ``CERTIFICATE_VERIFY_FAILED`` on a machine whose network is perfectly fine.
    ``requests`` ships ``certifi`` and verifies against that instead, so it
    works where urllib does not. It arrives with pymatgen, so it is present
    wherever matverse is; urllib remains the fallback for the case where it is
    somehow not.
    """
    import json

    try:
        import requests
    except ImportError:                                  # pragma: no cover
        import urllib.parse
        import urllib.request

        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(f"{endpoint}?{query}", headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    response = requests.get(endpoint, params=params, headers=headers,
                            timeout=timeout)
    response.raise_for_status()
    return response.json()


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
           "from_iterable", "from_ase_file",
           "from_optimade", "from_optimade_response", "optimade_providers",
           "OPTIMADE_PROVIDERS",
           "from_ase", "to_ase", "to_pymatgen", "to_cif", "from_mp"]

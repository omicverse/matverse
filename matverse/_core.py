"""Slot convention, the level-of-theory type, and provenance.

matverse stores a materials dataset in an ``AnnData``. Not a subclass and not a
wrapper: the object you get back is an ``AnnData``, writable to ``h5ad`` and
readable by anything that speaks it.

    X                       materials x elements — the composition matrix
    var                     one row per element — the periodic table
    obs                     one row per material
    obsm['structures']      structures, one column per variant
    obsm[...]               descriptors, embeddings
    obsp                    pairwise: similarity
    uns['features']         which featuriser produced which block
    uns['levels']           per-level-of-theory provenance (see below)
    uns['provenance']       operations applied, in order

Structures are in ``obsm``, not ``uns``, and that placement is load-bearing.
``uns`` does not subset with the object, so ``md[mask]`` would keep every
structure while dropping rows, and each surviving row would point at the wrong
one — the exact failure this substrate is supposed to make impossible. ``obsm``
is aligned to the material axis by construction. Serialising each structure to
JSON is what lets it live there, and has the second benefit of making the object
writable to ``h5ad`` without special handling.

Three rules
-----------
**Operations deposit; they do not return.** ``mv.pp.standardize(md)`` writes
``uns['structures']['primitive']`` rather than handing back an object the caller
must find a home for. After any step you can ask the object what is in it, which
is also what makes a run reproducible from the object alone.

**A result carries its level of theory in the slot name.** ``obs['energy_mace']``
and ``obs['energy_pbe']`` are different quantities. ``uns['levels'][level]``
holds what produced each, so comparing a surrogate against DFT requires naming
both instead of silently averaging them.

**Elements are the axis of X.** A composition matrix is materials x elements,
sparse and non-negative — the same shape as a cells x genes matrix, which is
what lets the ordination and differential-enrichment toolchain apply without
being rewritten. See ``mv.feat.composition_matrix``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from anndata import AnnData

CONTAINERS = ("obs", "var", "obsm", "obsp", "layers", "uns")

#: ``uns['levels'][level]`` fields, and what each is for.
LEVEL_FIELDS = {
    "kind": "dft | mlip | classical | experiment | model",
    "method": "human name of the method, e.g. 'MACE' or 'PBE+U'",
    "reference": "what this level reproduces — 'PBE+U (OMat24)', 'r2SCAN'; "
                 "None for a primary method",
    "surrogate": "True when the number approximates a more expensive method",
    "license": "licence of the weights or code; None when not applicable",
    "uncertainty": "how obs['<quantity>_<level>_std'] was produced, or None",
}

#: Licences that forbid commercial use. Recorded so a screen cannot silently
#: produce a commercial result from a non-commercial checkpoint.
NONCOMMERCIAL_LICENSES = {"asl", "cc-by-nc", "cc-by-nc-4.0", "non-commercial"}


def composition_matrix(structures: list, elements: list[str] | None = None):
    """Atom counts of each structure's reduced composition, as a sparse matrix.

    Returns ``(X, element_symbols)``. Counts come from the **reduced** formula so
    that a supercell and its primitive cell occupy the same row of chemical
    space — cell size belongs in ``obs['nsites']``, not in the composition.
    """
    from scipy.sparse import csr_matrix

    amounts = []
    for s in structures:
        comp = _composition_of(s)
        amounts.append({str(el): float(amt) for el, amt
                        in comp.reduced_composition.get_el_amt_dict().items()})
    if elements is None:
        elements = sorted({el for row in amounts for el in row},
                          key=_element_sort_key)
    index = {el: j for j, el in enumerate(elements)}

    indptr, indices, data = [0], [], []
    for row in amounts:
        for el, amt in sorted(row.items(), key=lambda kv: index.get(kv[0], -1)):
            if el in index and amt:
                indices.append(index[el])
                data.append(amt)
        indptr.append(len(indices))
    X = csr_matrix((np.asarray(data, dtype=np.float32),
                    np.asarray(indices, dtype=np.int32),
                    np.asarray(indptr, dtype=np.int32)),
                   shape=(len(structures), len(elements)))
    return X, list(elements)


def _composition_of(structure):
    """The composition of a pymatgen Structure, Composition or formula string."""
    comp = getattr(structure, "composition", None)
    if comp is not None:
        return comp
    from pymatgen.core.composition import Composition
    return Composition(structure)


def _element_sort_key(symbol: str) -> tuple:
    """Order elements by atomic number, so ``var`` reads as the periodic table."""
    try:
        from pymatgen.core.periodic_table import Element
        return (0, int(Element(symbol).Z))
    except Exception:
        return (1, symbol)


def new(structures: list, obs: pd.DataFrame | None = None,
        source: str = "structures", build_X: bool = True) -> AnnData:
    """A dataset holding ``structures`` as the ``input`` variant.

    ``X`` is the materials x elements composition matrix and ``var`` is the
    periodic table restricted to the elements present. It is built here, at
    construction, rather than deposited later by a featuriser: AnnData ties
    ``X``'s width to ``var`` so it cannot be widened in place, and every
    matverse operation writes in place. Composition is intrinsic to a material
    rather than derived from it, so this costs nothing conceptually — derived
    descriptors still go to ``obsm``, whose width is free.

    ``build_X=False`` restores the width-zero ``X`` of earlier versions, for
    datasets whose rows are not single compositions.
    """
    n = len(structures)
    # Name the rows before handing the frame to AnnData rather than after.
    # AnnData coerces an integer index to strings and warns while doing it, and
    # that warning then lands in the output of every notebook cell that builds
    # a dataset — a library's own construction path should not be noisy.
    names = pd.Index([str(i) for i in range(n)], dtype=object)
    obs = (pd.DataFrame(index=names) if obs is None
           else obs.reset_index(drop=True).set_axis(names))

    if build_X and n:
        X, elements = composition_matrix(structures)
        var = _element_frame(elements)
        md = AnnData(X=X, obs=obs, var=var)
    else:
        md = AnnData(X=np.zeros((n, 0), dtype=np.float32), obs=obs)

    md.uns["features"] = {}
    md.uns["levels"] = {}
    md.uns["provenance"] = []
    md.uns["X_is"] = "composition_atoms_reduced" if md.n_vars else "empty"
    deposit_structures(md, "input", structures)
    record(md, source)
    return md


def _element_frame(elements: list[str]) -> pd.DataFrame:
    """``var`` for the element axis; degrades to a bare index if pymatgen's
    periodic table cannot be read."""
    try:
        from .elements import element_frame
        return element_frame(elements)
    except Exception:                                    # pragma: no cover
        return pd.DataFrame(index=pd.Index(elements, name="element"))


#: Structures live in ``obsm[STRUCTURE_KEY]`` as a frame of JSON strings, one
#: column per variant.
#: Set on a bands object so functions can tell the axes apart. Lives here
#: rather than in mv.elec because mv.prop.dispersion builds the same axis for
#: phonons, and a convention two namespaces share is not one namespace's.
AXIS_KEY = "matverse_axis"

STRUCTURE_KEY = "structures"


def variants(md: AnnData) -> list[str]:
    """Structure variants present on this object."""
    frame = md.obsm.get(STRUCTURE_KEY)
    return [] if frame is None else list(frame.columns)


def structures(md: AnnData, variant: str = "input", rows=None) -> list:
    """Fetch a structure variant, with a useful error when it is absent.

    Structures are stored serialised — see :func:`deposit_structures` — and
    decoded here, with the decoded list cached on the object so that a pipeline
    of ten operations pays the cost once rather than ten times. Subsetting or
    copying produces a fresh object without the cache, which is also correct
    cache invalidation.

    ``rows`` decodes only the rows named, and caches nothing. Decoding is what
    costs at scale — five million serialised structures are a few gigabytes of
    strings, and the pymatgen objects they become are several times that — so a
    chunked pass over a large dataset must be able to ask for a window rather
    than for everything.
    """
    have = variants(md)
    if variant not in have:
        raise KeyError(
            f"no structure variant {variant!r}; have {sorted(have)}. "
            f"Operations deposit variants under a name — e.g. mv.pp.standardize "
            f"writes 'primitive', mv.calc.relax writes 'relaxed_<level>'.")

    column = md.obsm[STRUCTURE_KEY][variant]
    if rows is not None:
        index = np.asarray(rows)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        return [_decode(column.iloc[int(i)]) for i in index]

    cache = getattr(md, "_mv_structure_cache", None)
    if cache is None:
        cache = {}
        try:
            object.__setattr__(md, "_mv_structure_cache", cache)
        except Exception:                                # pragma: no cover
            cache = {}
    token = (len(column), variant)
    hit = cache.get(variant)
    if hit is not None and hit[0] == token:
        return list(hit[1])

    decoded = [_decode(s) for s in column]
    cache[variant] = (token, decoded)
    return list(decoded)


def deposit_structures(md: AnnData, variant: str, value: Iterable) -> None:
    """Store a structure variant, aligned to the material axis.

    Two things force this into ``obsm`` rather than ``uns``. Structures in
    ``uns`` do not subset with the object, so ``md[mask]`` would silently keep
    all of them and every row would point at the wrong structure — which defeats
    the reason for being on this substrate at all. And ``uns`` cannot hold a list
    of pymatgen objects in a way ``h5ad`` can write, so the object would not
    survive a save.

    Serialising to JSON solves both: ``obsm`` frames are aligned by construction
    and write without special handling.
    """
    encoded = [_encode(s) for s in value]
    if len(encoded) != md.n_obs:
        raise ValueError(
            f"variant {variant!r} has {len(encoded)} structures but the object "
            f"has {md.n_obs} materials; a structure variant is aligned to the "
            f"material axis")
    frame = md.obsm.get(STRUCTURE_KEY)
    if frame is None:
        frame = pd.DataFrame(index=md.obs_names.copy())
    else:
        frame = frame.copy()
    frame[variant] = pd.Series(encoded, index=md.obs_names, dtype=object)
    md.obsm[STRUCTURE_KEY] = frame
    cache = getattr(md, "_mv_structure_cache", None)
    if isinstance(cache, dict):
        cache.pop(variant, None)


def _encode(structure) -> str:
    """A structure as JSON. ``as_dict`` keeps site properties and oxidation
    states, which a CIF round trip would quietly drop."""
    import json

    if isinstance(structure, str):
        return structure
    return json.dumps(structure.as_dict(), default=_jsonable)


def _jsonable(value):
    """Last resort for values ``json`` does not know.

    pymatgen puts numpy arrays into site properties — an ``Interface`` carries
    its interface-normal vector that way, and so does anything built from a
    calculator that attached per-site data. ``as_dict`` passes them through
    untouched, so encoding failed with a bare TypeError naming only 'ndarray'
    and no hint of which structure or which property.
    """
    import numpy as _np

    if isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, (_np.integer, _np.floating, _np.bool_)):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(
        f"cannot store a site property of type {type(value).__name__} in a "
        f"structure; matverse serialises structures to JSON so the object "
        f"survives write_h5ad. Convert it to a list or a number first.")


def _decode(payload):
    """JSON back to a pymatgen object, periodic or not.

    A ``Molecule`` is a ``Structure`` without a lattice, and the composition
    axis does not care which it got — H2O contributes H:2, O:1 exactly as a
    crystal would. Only the decode has to know, so it dispatches on whether the
    payload carries a lattice rather than assuming periodicity.
    """
    import json

    from pymatgen.core import Molecule, Structure

    if not isinstance(payload, str):
        return payload
    data = json.loads(payload)
    if "lattice" in data:
        return Structure.from_dict(data)
    return Molecule.from_dict(data)


def require(md: AnnData, container: str, key: str, hint: str = "") -> Any:
    """Read a slot, or fail with what would have produced it.

    The error names the operation to run, because the common failure in a
    deposit-style API is calling a step before the one that fills its input.
    """
    holder = md.uns if container == "uns" else getattr(md, container, None)
    if holder is None or key not in holder:
        msg = f"{container}[{key!r}] is not present"
        if not hint:
            producers = _producers_of(key)
            if producers:
                hint = "run " + " or ".join(producers) + " first"
        raise ValueError(f"{msg}; {hint}" if hint else msg)
    return holder[key]


def _producers_of(key: str) -> list[str]:
    """Ask the registry which function writes a slot. Best-effort."""
    try:
        from ._registry import get_registry
        return [e["public_name"] for e in get_registry().producers_of(key)]
    except Exception:                                    # pragma: no cover
        return []


def set_level(md: AnnData, level: str, *, kind: str, method: str,
              reference: str | None = None, surrogate: bool = False,
              license: str | None = None, uncertainty: str | None = None,
              **extra: Any) -> dict:
    """Record what produced everything tagged ``level``.

    ``reference`` matters as much as ``surrogate`` now that surrogates disagree
    with each other: a model trained on OMat24 reproduces PBE+U, one trained on
    MatPES reproduces r2SCAN, and mixing those is the same class of error as
    mixing PBE with HSE06 one level up.

    ``license`` is recorded because model weights are not uniformly open —
    MACE-MP and MACE-MPA are MIT, MACE-OMAT and MACE-MATPES are ASL
    (non-commercial), UMA's licence excludes several countries. A screening
    result carries the licence of whatever produced it.
    """
    entry = {"kind": kind, "method": method, "reference": reference,
             "surrogate": bool(surrogate), "license": license,
             "uncertainty": uncertainty, **extra}
    md.uns.setdefault("levels", {})[level] = entry
    return entry


def level_info(md: AnnData, level: str) -> dict:
    levels = md.uns.get("levels", {})
    if level not in levels:
        raise KeyError(f"no level {level!r}; have {sorted(levels)}. A level is "
                       f"recorded by the operation that computes at it, e.g. "
                       f"mv.calc.energy(md, level={level!r}).")
    return levels[level]


def levels_used(md: AnnData) -> list[str]:
    return sorted(md.uns.get("levels", {}))


def check_commercial_use(md: AnnData) -> list[str]:
    """Levels in this object whose licence forbids commercial use.

    Not a legal opinion — a reminder that the object knows something the user
    may not, and that a screen mixing levels inherits the strictest of them.
    """
    out = []
    for level, info in md.uns.get("levels", {}).items():
        lic = (info or {}).get("license")
        if lic and str(lic).strip().lower() in NONCOMMERCIAL_LICENSES:
            out.append(level)
    return sorted(out)


def compare_levels(md: AnnData, quantity: str,
                   levels: list[str] | None = None) -> pd.DataFrame:
    """Line up one quantity across every level that computed it.

    The naming convention made usable: ``compare_levels(md, 'energy_per_atom')``
    returns a frame whose columns are levels, with each level's record attached,
    so surrogate-versus-DFT is a table rather than an act of memory.
    """
    known = levels or levels_used(md)
    cols = {lv: f"{quantity}_{lv}" for lv in known}
    present = {lv: c for lv, c in cols.items() if c in md.obs}
    if not present:
        raise ValueError(
            f"no obs column of the form '{quantity}_<level>' for levels "
            f"{sorted(known)}; obs has {list(md.obs.columns)}")
    df = pd.DataFrame({lv: md.obs[c].to_numpy(dtype=float)
                       for lv, c in present.items()},
                      index=list(md.obs_names))
    df.attrs["levels"] = {lv: md.uns.get("levels", {}).get(lv, {})
                          for lv in present}
    return df


def deposit_grid(md: AnnData, quantity: str, level: str, values,
                 grid, unit: str = "", **meta: Any) -> str:
    """Store a grid-shaped result — a spectrum, a density of states, a pattern.

    Grid data is ``n_materials x n_grid_points``: an XRD pattern over 2-theta, a
    DOS over energy, a phonon spectrum over frequency. It goes into ``obsm``
    under ``'<quantity>_<level>'``, with the shared grid axis recorded once in
    ``uns['grids'][quantity]``.

    That placement decides something the design had left open. Array-shaped
    results were going to use AnnData ``layers`` to hold the level of theory,
    with scalars using a name suffix — two conventions, split by container.
    Putting grids in ``obsm`` instead means the suffix works everywhere:
    ``obs['energy_pbe']`` and ``obsm['xrd_pbe']`` read the same way, one rule
    covers both, and a measured pattern is ``obsm['xrd_experiment']`` rather
    than a separate kind of thing.

    Returns the ``obsm`` key it wrote.
    """
    import numpy as _np

    values = _np.asarray(values, dtype=float)
    grid = _np.asarray(grid, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{quantity!r} values must be 2-D "
                         f"(materials x grid), got shape {values.shape}")
    if values.shape[0] != md.n_obs:
        raise ValueError(
            f"{quantity!r} has {values.shape[0]} rows but the object has "
            f"{md.n_obs} materials; a grid result is aligned to the material axis")
    if values.shape[1] != len(grid):
        raise ValueError(
            f"{quantity!r} has {values.shape[1]} columns but the grid has "
            f"{len(grid)} points")

    key = f"{quantity}_{level}"
    md.obsm[key] = values
    grids = md.uns.setdefault("grids", {})
    known = grids.get(quantity)
    if known is not None and not _np.array_equal(
            _np.asarray(known.get("values"), dtype=float), grid):
        raise ValueError(
            f"{quantity!r} already has a grid of {len(known.get('values', []))} "
            f"points on this object and the new one differs. Levels of the same "
            f"quantity must share a grid, or they cannot be compared; pass the "
            f"same range and step, or use a different quantity name.")
    grids[quantity] = {"values": grid, "unit": unit, **meta}
    return key


def grid_of(md: AnnData, quantity: str):
    """The shared axis a grid-shaped result is defined on."""
    grids = md.uns.get("grids", {})
    if quantity not in grids:
        raise KeyError(f"no grid for {quantity!r}; have {sorted(grids)}")
    return np.asarray(grids[quantity]["values"], dtype=float)


def append_record(holder: dict, key: str, entry: dict) -> None:
    """Append a record to an ordered, ``h5ad``-writable list in ``uns``.

    A plain list of dicts cannot be written: anndata turns it into an object
    array and h5py refuses it. A dict keyed by a zero-padded index writes as a
    nested group and still reads back in order, so the same information survives
    a save without the caller having to know why.
    """
    store = holder.setdefault(key, {})
    if isinstance(store, list):                  # tolerate an older object
        store = {f"{i:04d}": v for i, v in enumerate(store)}
        holder[key] = store
    store[f"{len(store):04d}"] = entry


def records(holder: dict, key: str) -> list:
    """Read back what :func:`append_record` stored, in order."""
    store = holder.get(key, {})
    if isinstance(store, list):
        return list(store)
    return [store[k] for k in sorted(store)]


def record(md: AnnData, op: str, **params: Any) -> None:
    """Append an operation to ``uns['provenance']``.

    Parameters are recorded with the call so the list replays as code rather
    than reading as a list of verbs.
    """
    if params:
        args = ", ".join(f"{k}={v!r}" for k, v in params.items())
        op = f"{op}({args})"

    # h5ad stores a list of strings and reads it back as a numpy array, which
    # has no .append — so every operation on a saved-and-reloaded object used
    # to fail on its provenance write. Normalising here rather than at read
    # time means it holds however the object arrived: from disk, from a cache,
    # from anndata.concat, or from someone else's pipeline.
    existing = md.uns.get("provenance")
    if existing is None:
        md.uns["provenance"] = [op]
        return
    if not isinstance(existing, list):
        existing = [str(x) for x in existing]
        md.uns["provenance"] = existing
    existing.append(op)


def provenance(md: AnnData) -> list[str]:
    return list(md.uns.get("provenance", []))


__all__ = ["new", "structures", "deposit_structures", "variants", "require",
           "append_record", "records",
           "record", "provenance", "set_level", "level_info", "levels_used",
           "compare_levels", "check_commercial_use", "composition_matrix",
           "deposit_grid", "grid_of", "AXIS_KEY", "CONTAINERS", "LEVEL_FIELDS",
           "NONCOMMERCIAL_LICENSES", "STRUCTURE_KEY"]

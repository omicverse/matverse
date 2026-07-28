"""``mv.utils`` — units, checkpointing, and getting work onto a cluster.

Units
-----
Materials software mixes eV, kJ/mol, Rydberg and Hartree, and a number that
arrives in the wrong one is not detectably wrong — it is merely off by 96, or by
13.6, or by 27.2. matverse's own operations are all in eV and angstrom, so the
risk is at the boundary: a column pasted in from a spreadsheet, a property read
out of a database that reports kJ/mol.

:func:`set_units` records what a column is in and :func:`convert` changes it,
depositing the converted column under its own name rather than overwriting.
Recording is the cheap half and catches most of it.

Checkpointing
-------------
A screen over ten thousand candidates outlives a walltime limit. :func:`resume`
skips the rows an operation has already filled, so re-running the same script
after a job is killed continues rather than restarts.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import append_record, record
from ._registry import register_function

#: Conversion factors into matverse's internal units: eV and angstrom.
TO_EV = {
    "ev": 1.0,
    "mev": 1.0e-3,
    "kj/mol": 0.010364269656262175,
    "kcal/mol": 0.04336414,
    "ry": 13.605693122994,
    "rydberg": 13.605693122994,
    "ha": 27.211386245988,
    "hartree": 27.211386245988,
    "j": 6.241509074e18,
}

TO_ANGSTROM = {
    "angstrom": 1.0, "a": 1.0, "ang": 1.0,
    "nm": 10.0, "pm": 0.01, "bohr": 0.529177210903,
}

#: What matverse's own operations produce, by column-name prefix.
INTERNAL_UNITS = {
    "energy": "eV",
    "energy_per_atom": "eV/atom",
    "e_above_hull": "eV/atom",
    "formation_energy": "eV/atom",
    "max_force": "eV/angstrom",
    "force_magnitude": "eV/angstrom",
    "volume": "angstrom^3",
    "density": "g/cm^3",
    "min_distance": "angstrom",
    "bulk_modulus": "GPa",
    "shear_modulus": "GPa",
}


@register_function(
    aliases=["set units", "declare units", "record units", "units"],
    category="utils",
    description="Record the physical unit a column is in, so a later "
                "conversion or comparison can check rather than assume.",
    requires={"obs": ["{column}"]},
    produces={"uns": ["units"]},
    examples=["mv.utils.set_units(md, 'formation_energy_expt', 'kJ/mol')"],
    related=["mv.utils.convert", "mv.utils.check_units"],
    notes="matverse's own operations work in eV and angstrom throughout. This "
          "exists for the boundary — a column pasted in from a spreadsheet or "
          "read out of a database that reports kJ/mol.",
)
def set_units(md: AnnData, column: str, unit: str) -> None:
    """Declare the unit of an ``obs`` column."""
    if column not in md.obs:
        raise ValueError(f"obs[{column!r}] absent; available: "
                         f"{list(md.obs.columns)}")
    md.uns.setdefault("units", {})[column] = str(unit)
    record(md, "utils.set_units", column=column, unit=unit)


@register_function(
    aliases=["convert units", "unit conversion", "to ev", "change units"],
    category="utils",
    description="Convert a column into matverse's internal units — eV for "
                "energies, angstrom for lengths — depositing the result under "
                "its own name.",
    requires={"obs": ["{column}"]},
    produces={"obs": ["{key_added}"], "uns": ["units"]},
    examples=["mv.utils.convert(md, 'formation_energy_expt', 'kJ/mol')"],
    related=["mv.utils.set_units"],
    notes="Deposits rather than overwrites. A converted column beside the "
          "original is auditable; a silently rewritten one is the bug this "
          "function exists to prevent, one step later.",
)
def convert(md: AnnData, column: str, unit: str | None = None,
            kind: str = "energy", key_added: str | None = None) -> None:
    """Convert an ``obs`` column into eV or angstrom."""
    if column not in md.obs:
        raise ValueError(f"obs[{column!r}] absent")
    source = (unit or md.uns.get("units", {}).get(column))
    if source is None:
        raise ValueError(
            f"no unit known for {column!r}; pass unit= or declare it with "
            f"mv.utils.set_units first")

    table = {"energy": TO_EV, "length": TO_ANGSTROM}.get(kind)
    if table is None:
        raise ValueError(f"kind must be 'energy' or 'length', got {kind!r}")
    key = str(source).strip().lower()
    if key not in table:
        raise ValueError(f"unknown {kind} unit {source!r}; known: "
                         f"{sorted(table)}")

    target = "eV" if kind == "energy" else "angstrom"
    name = key_added or f"{column}_{target.lower()}"
    md.obs[name] = md.obs[column].to_numpy(dtype=float) * table[key]
    md.uns.setdefault("units", {})[name] = target
    md.uns["units"].setdefault(column, str(source))
    record(md, "utils.convert", column=column, unit=source, key_added=name)


@register_function(
    aliases=["check units", "unit check", "which units", "audit units"],
    category="utils",
    description="Report the unit of every numeric column, filling in what "
                "matverse itself produced and naming the ones nobody has "
                "declared.",
    examples=["mv.utils.check_units(md)"],
    related=["mv.utils.set_units", "mv.utils.convert"],
)
def check_units(md: AnnData) -> dict:
    """Column to unit, with ``None`` where nothing is known."""
    import pandas as pd

    declared = dict(md.uns.get("units", {}))
    out: dict[str, str | None] = {}
    for column in md.obs.columns:
        if not pd.api.types.is_numeric_dtype(md.obs[column]):
            continue
        if column in declared:
            out[column] = declared[column]
            continue
        out[column] = _internal_unit(column)
    return out


def _internal_unit(column: str) -> str | None:
    """The unit matverse would have produced this column in, if it did."""
    for prefix in sorted(INTERNAL_UNITS, key=len, reverse=True):
        if column == prefix or column.startswith(prefix + "_"):
            return INTERNAL_UNITS[prefix]
    return None


@register_function(
    aliases=["resume", "skip done", "continue run", "which rows remain",
             "unfinished rows"],
    category="utils",
    description="Report which materials an operation has not filled yet, so a "
                "screen killed by a walltime limit continues rather than "
                "restarts.",
    requires={"obs": ["{column}"]},
    examples=["todo = mv.utils.resume(md, 'energy_mace-mpa')"],
    related=["mv.calc.energy", "mv.utils.checkpoint"],
    notes="Returns a boolean mask rather than mutating anything, because what "
          "to do about a half-finished column is the caller's decision — "
          "recompute the failures, or skip them and record how many.",
)
def resume(md: AnnData, column: str) -> np.ndarray:
    """Which rows still need computing: absent column, or NaN in it."""
    if column not in md.obs:
        return np.ones(md.n_obs, dtype=bool)
    values = md.obs[column].to_numpy(dtype=float)
    return ~np.isfinite(values)


@register_function(
    aliases=["checkpoint", "save progress", "write checkpoint",
             "periodic save"],
    category="utils",
    description="Write the object to disk and record that it was checkpointed, "
                "so a long screen has a recoverable state.",
    produces={"files": ["<path>"], "uns": ["checkpoints"]},
    examples=["mv.utils.checkpoint(md, 'run.h5ad')"],
    related=["mv.utils.resume"],
)
def checkpoint(md: AnnData, path, note: str = "") -> str:
    """Write to ``path`` and record the write in the object."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    append_record(md.uns, "checkpoints",
                  {"path": str(target), "n_obs": int(md.n_obs),
                   "n_operations": len(md.uns.get("provenance", [])),
                   "note": note})
    md.write_h5ad(target)
    return str(target)


@register_function(
    aliases=["chunks", "batches", "iterate in chunks", "split into batches",
             "process in blocks"],
    category="utils",
    description="Iterate over the dataset in row blocks, so an operation that "
                "would not fit in memory at full size runs a block at a time.",
    examples=["for block in mv.utils.chunks(md, 500): mv.calc.relax(block)"],
    related=["mv.utils.map_chunks", "mv.utils.checkpoint"],
    notes="Each block is a copy, so writes land on the copy rather than on the "
          "parent. mv.utils.map_chunks is the version that puts the results "
          "back.",
)
def chunks(md: AnnData, size: int = 1000):
    """Yield ``(start, block)`` for each row block of the dataset."""
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    for start in range(0, md.n_obs, int(size)):
        stop = min(start + int(size), md.n_obs)
        yield start, md[start:stop].copy()


@register_function(
    aliases=["map chunks", "apply in chunks", "chunked", "batch apply",
             "run in blocks", "out of memory", "too big for memory"],
    category="utils",
    description="Apply an expensive operation block by block and merge the "
                "results back onto the parent, optionally checkpointing after "
                "each block so a job killed by a walltime limit resumes.",
    produces={"uns": ["chunked"]},
    examples=["mv.utils.map_chunks(md, lambda b: mv.calc.relax(b, "
              "level='mace-mpa'), size=500, checkpoint_to='run.h5ad')"],
    related=["mv.utils.chunks", "mv.utils.resume", "mv.utils.checkpoint"],
    notes="Merges back the obs columns, obsm blocks and structure variants each "
          "block produced. It cannot merge uns, because a per-block uns entry "
          "is a statement about that block and quietly keeping the last one "
          "would be wrong — a screen's criteria and a hull's reference count "
          "both mean something different per block than they do overall.",
)
def map_chunks(md: AnnData, operation, size: int = 1000,
               checkpoint_to=None, skip_if: str | None = None) -> dict:
    """Run ``operation(block)`` over row blocks and merge what it wrote back.

    ``skip_if`` names an ``obs`` column: blocks where it is already finite are
    skipped, which is what makes a re-run after a killed job continue rather
    than restart.
    """
    import pandas as pd

    n_done, n_skipped, errors = 0, 0, []
    for start, block in chunks(md, size):
        stop = start + block.n_obs
        if skip_if is not None and skip_if in md.obs:
            todo = resume(md, skip_if)[start:stop]
            if not todo.any():
                n_skipped += block.n_obs
                continue
        before = _snapshot(block)
        try:
            operation(block)
        except Exception as exc:
            errors.append(f"rows {start}:{stop}: {type(exc).__name__}: {exc}")
            continue
        _merge_back(md, block, start, stop, before)
        n_done += block.n_obs
        if checkpoint_to is not None:
            checkpoint(md, checkpoint_to,
                       note=f"after rows {start}:{stop}")

    md.uns["chunked"] = {"size": int(size), "n_processed": n_done,
                         "n_skipped": n_skipped, "errors": errors}
    record(md, "utils.map_chunks", size=size, n_processed=n_done,
           n_skipped=n_skipped, n_errors=len(errors))
    return md.uns["chunked"]


def _snapshot(block: AnnData) -> dict:
    """What a block held before the operation, so new slots can be told apart."""
    from ._core import STRUCTURE_KEY, variants

    return {"obs": set(block.obs.columns),
            "obsm": set(k for k in block.obsm if k != STRUCTURE_KEY),
            "variants": set(variants(block))}


def _merge_back(md: AnnData, block: AnnData, start: int, stop: int,
                before: dict) -> None:
    """Write a block's new columns, blocks and variants into the parent rows."""
    import pandas as pd

    from ._core import STRUCTURE_KEY

    for column in block.obs.columns:
        values = block.obs[column].to_numpy()
        if column not in md.obs:
            md.obs[column] = _empty_like(values, md.n_obs)
        target = md.obs[column].to_numpy(copy=True)
        try:
            target[start:stop] = values
        except (ValueError, TypeError):
            target = target.astype(object)
            target[start:stop] = values
        md.obs[column] = target

    for key in block.obsm:
        if key == STRUCTURE_KEY:
            continue
        arr = np.asarray(block.obsm[key])
        if key not in md.obsm:
            md.obsm[key] = np.full((md.n_obs, arr.shape[1]), np.nan,
                                   dtype=float)
        md.obsm[key][start:stop] = arr

    frame = block.obsm.get(STRUCTURE_KEY)
    if frame is None:
        return
    parent = md.obsm[STRUCTURE_KEY]
    for variant in frame.columns:
        if variant not in parent.columns:
            parent[variant] = pd.Series([None] * md.n_obs,
                                        index=md.obs_names, dtype=object)
        parent.iloc[start:stop,
                    parent.columns.get_loc(variant)] = frame[variant].to_numpy()
    md.obsm[STRUCTURE_KEY] = parent
    cache = getattr(md, "_mv_structure_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _empty_like(values: np.ndarray, n: int) -> np.ndarray:
    """A full-length column to write a block's values into."""
    if values.dtype.kind == "f":
        return np.full(n, np.nan)
    if values.dtype.kind in "iu":
        return np.full(n, np.nan)          # widened; a partial int column has holes
    if values.dtype.kind == "b":
        return np.zeros(n, dtype=bool)
    return np.array([""] * n, dtype=object)


@register_function(
    aliases=["submit", "slurm", "batch job", "run on cluster", "sbatch",
             "hpc submission"],
    category="utils",
    description="Write a Slurm batch script that runs a matverse script over "
                "this dataset, with resources sized for the level of theory "
                "being used.",
    produces={"files": ["<path>"]},
    examples=["mv.utils.slurm_script('screen.py', 'job.sbatch', "
              "partition='normal', hours=4)"],
    related=["mv.utils.checkpoint"],
    notes="Writes the script rather than submitting it. Submitting is a "
          "side effect on a shared machine, and a script you can read before "
          "running is worth more than one command less to type.",
)
def slurm_script(script: str, path: str, partition: str = "normal",
                 hours: int = 2, cpus: int = 4, memory: str = "16GB",
                 gpus: int = 0, job_name: str = "matverse",
                 setup: str = "", python: str = "python") -> str:
    """Write a Slurm batch script. Returns the path."""
    from pathlib import Path

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={int(hours):02d}:00:00",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --mem={memory}",
    ]
    if gpus:
        lines.append(f"#SBATCH --gpus={int(gpus)}")
    lines += [
        "#SBATCH --output=%x-%j.out",
        "",
        "set -euo pipefail",
        "",
        "# Caches default to $HOME, which is small and shared. Point them at",
        "# scratch so a model download does not fill a home directory.",
        'export HF_HOME="${SCRATCH:-$PWD}/hf"',
        'export XDG_CACHE_HOME="${SCRATCH:-$PWD}/cache"',
        'export PIP_CACHE_DIR="${SCRATCH:-$PWD}/pip"',
        "",
    ]
    if setup:
        lines += [setup, ""]
    lines += [f"{python} {script}", ""]

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return str(target)


@register_function(
    aliases=["summary", "describe object", "what is in here", "overview",
             "inspect"],
    category="utils",
    description="Summarise what an object contains — its axes, structure "
                "variants, levels of theory, screens and provenance — in one "
                "readable block.",
    examples=["print(mv.utils.summary(md))"],
    related=["mv.provenance", "mv.levels_used"],
    notes="What an agent reads first. The object is meant to answer 'what has "
          "been done to me' without anyone consulting a notebook, and this is "
          "that answer rendered.",
)
def summary(md: AnnData) -> str:
    """A readable description of the object's contents."""
    from ._core import levels_used, provenance, variants

    lines = [f"matverse dataset: {md.n_obs} materials x {md.n_vars} elements"]
    if md.n_vars:
        elements = list(map(str, md.var_names))
        shown = ", ".join(elements[:12]) + (" ..." if len(elements) > 12 else "")
        lines.append(f"  elements   {shown}")

    struct_variants = variants(md)
    if struct_variants:
        lines.append(f"  structures {', '.join(struct_variants)}")

    levels = levels_used(md)
    if levels:
        lines.append("  levels")
        for level in levels:
            info = md.uns["levels"][level]
            bits = [str(info.get("method", level))]
            if info.get("reference"):
                bits.append(f"-> {info['reference']}")
            if info.get("license"):
                bits.append(f"[{info['license']}]")
            lines.append(f"    {level:16s} {' '.join(bits)}")

    grids = md.uns.get("grids", {})
    if grids:
        lines.append("  grids")
        for name, meta in grids.items():
            lines.append(f"    {name:16s} {len(meta.get('values', []))} points "
                         f"{meta.get('unit', '')}")

    blocks = [k for k in md.obsm if k != "structures"]
    if blocks:
        lines.append(f"  obsm       {', '.join(blocks)}")

    screens = md.uns.get("screens", {})
    for name, info in screens.items():
        lines.append(f"  screen     {name}: {info['n_pass']}/{info['n_total']} "
                     f"passed {info['criteria']}")

    noncommercial = _noncommercial(md)
    if noncommercial:
        lines.append(f"  ! levels forbidding commercial use: "
                     f"{', '.join(noncommercial)}")
    if md.uns.get("phase_diagram", {}).get("closed_system"):
        lines.append("  ! the hull is closed over this dataset — "
                     "e_above_hull is relative")

    steps = provenance(md)
    if steps:
        lines.append(f"  provenance {len(steps)} operations")
        for step in steps:
            lines.append(f"    {step}")
    return "\n".join(lines)


def _noncommercial(md: AnnData) -> list[str]:
    try:
        from ._core import check_commercial_use
        return check_commercial_use(md)
    except Exception:                                     # pragma: no cover
        return []


__all__ = ["set_units", "convert", "check_units", "resume", "checkpoint",
           "chunks", "map_chunks",
           "slurm_script", "summary", "TO_EV", "TO_ANGSTROM", "INTERNAL_UNITS"]

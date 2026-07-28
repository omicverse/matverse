"""``mv.dft`` — first-principles inputs out, results back in.

matverse runs no DFT and submits no jobs. Workflow management is a solved
problem with three good answers — atomate2 with jobflow-remote, quacc, AiiDA —
and a fourth would be a maintenance liability with no upside.

What is *not* solved is the boundary. A screen lives in one object, DFT lives in
a directory tree, and the correspondence between the two is normally maintained
by a naming convention and someone's memory. This module writes one directory per
material with the row's identity recorded inside it, and reads the finished
directories back onto the rows they came from.

    mv.dft.write_inputs(md, 'runs/', preset='relax')   # one directory per row
    # ... sbatch, atomate2, quacc, a week ...
    mv.dft.read_outputs(md, 'runs/', level='pbe')      # back onto the same rows

The parsed results land as an ordinary level of theory, so a DFT energy and a
machine-learned one sit side by side and `mv.thermo.hull` refuses to mix them
unless told the references agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from anndata import AnnData

from ._core import deposit_structures, record, set_level, structures
from ._registry import register_function

#: Input presets, mapped to the pymatgen VASP input set that implements each.
#: Named for the job rather than the class so the caller need not know pymatgen's
#: naming, and so a preset can be repointed when Materials Project revises a set.
VASP_PRESETS = {
    "relax": ("pymatgen.io.vasp.sets", "MPRelaxSet",
              "Materials Project relaxation — PBE(+U), the settings MP's own "
              "entries use, so results are comparable with the hull."),
    "static": ("pymatgen.io.vasp.sets", "MPStaticSet",
               "Single point on a fixed geometry, denser k-mesh than relax."),
    "bands": ("pymatgen.io.vasp.sets", "MPNonSCFSet",
              "Non-self-consistent run along a k-path, for a band structure."),
    "scan": ("pymatgen.io.vasp.sets", "MPScanRelaxSet",
             "r2SCAN relaxation. A different level of theory from PBE — tag it "
             "as one."),
    "hse": ("pymatgen.io.vasp.sets", "MPHSERelaxSet",
            "HSE06 hybrid. Expensive; screen with something cheaper first."),
}

#: What each preset reproduces, recorded on the level so a hull can check.
PRESET_REFERENCE = {
    "relax": "PBE+U", "static": "PBE+U", "bands": "PBE+U",
    "scan": "r2SCAN", "hse": "HSE06",
}

#: The file matverse writes into each directory to keep the row identity.
MANIFEST = "matverse.json"


@register_function(
    aliases=["write inputs", "vasp inputs", "dft inputs", "prepare dft",
             "generate incar", "write poscar", "set up calculations"],
    category="dft",
    description="Write one first-principles input directory per material, "
                "each carrying a manifest that records which row it came from.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["dft_directory"], "files": ["<root>/<name>/"],
              "uns": ["dft"]},
    dispatch="preset= selects the input set: relax, static, bands, scan or hse. "
             "scan and hse are different levels of theory, not settings.",
    examples=["mv.dft.write_inputs(md, 'runs/')",
              "mv.dft.write_inputs(md, 'runs/', preset='static', "
              "source='relaxed_emt')"],
    related=["mv.dft.read_outputs", "mv.dft.presets", "mv.utils.slurm_script"],
    notes="Writes inputs and stops. Submission belongs to atomate2, quacc or "
          "AiiDA, and a fourth workflow manager would be a liability. The "
          "manifest is what makes the round trip work without a naming "
          "convention: read_outputs finds each row by identity rather than by "
          "matching a directory name to an index.",
)
def write_inputs(md: AnnData, root, preset: str = "relax",
                 source: str = "input", code: str = "vasp",
                 overrides: dict | None = None,
                 potcar_spec: bool = True,
                 pseudopotentials: dict | None = None) -> list:
    """Write one input directory per material. Returns the paths written."""
    if code == "espresso":
        return _write_espresso(md, root, preset, source, overrides,
                               pseudopotentials)
    if code != "vasp":
        raise ValueError(f"code must be 'vasp' or 'espresso', got {code!r}")
    if preset not in VASP_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; use {sorted(VASP_PRESETS)}")

    module_name, class_name, _ = VASP_PRESETS[preset]
    try:
        module = __import__(module_name, fromlist=[class_name])
        input_set = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:          # pragma: no cover
        raise ImportError(
            f"pymatgen does not expose {class_name}; the input sets were "
            f"reorganised in recent releases. Check pymatgen.io.vasp.sets."
        ) from exc

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    written, failed = [], []

    for name, structure in zip(md.obs_names, structures(md, source)):
        directory = root / str(name)
        try:
            settings = input_set(structure, user_incar_settings=overrides or {})
            settings.write_input(str(directory),
                                 potcar_spec=potcar_spec)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
            written.append("")
            continue
        (directory / MANIFEST).write_text(json.dumps({
            "obs_name": str(name),
            "preset": preset,
            "source_variant": source,
            "reference": PRESET_REFERENCE.get(preset),
            "matverse": "input directory; the obs_name is how read_outputs "
                        "finds the row this belongs to",
        }, indent=2), encoding="utf-8")
        written.append(str(directory))

    md.obs["dft_directory"] = written
    md.uns["dft"] = {
        "root": str(root), "preset": preset, "code": code, "source": source,
        "reference": PRESET_REFERENCE.get(preset),
        "n_written": int(sum(1 for w in written if w)),
        "potcar_spec": bool(potcar_spec),
        "errors": failed,
    }
    if potcar_spec:
        md.uns["dft"]["note"] = (
            "POTCARs were written as a specification, not as files. VASP "
            "pseudopotentials are licensed and cannot be redistributed; point "
            "pymatgen at your own with PMG_VASP_PSP_DIR and set "
            "potcar_spec=False.")
    record(md, "dft.write_inputs", preset=preset, source=source,
           n_written=int(sum(1 for w in written if w)))
    return written


#: Quantum ESPRESSO calculation types, by matverse preset name.
QE_CALCULATIONS = {"relax": "vc-relax", "static": "scf", "bands": "bands",
                   "scan": "vc-relax", "hse": "scf"}


def _write_espresso(md: AnnData, root, preset: str, source: str,
                    overrides: dict | None,
                    pseudopotentials: dict | None) -> list:
    """Quantum ESPRESSO input, one directory per material.

    Pseudopotentials are named, not shipped. Which set a run used is part of the
    level of theory — SSSP efficiency and PSLibrary give different numbers for
    the same functional — so guessing a filename would put a silent choice into
    a result the object claims to record.
    """
    from pymatgen.io.pwscf import PWInput

    if preset not in QE_CALCULATIONS:
        raise ValueError(f"unknown preset {preset!r}; use "
                         f"{sorted(QE_CALCULATIONS)}")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    written, failed = [], []

    for name, structure in zip(md.obs_names, structures(md, source)):
        directory = root / str(name)
        directory.mkdir(parents=True, exist_ok=True)
        symbols = sorted({str(el) for el in structure.composition.elements})
        pseudo = {s: (pseudopotentials or {}).get(s, f"{s}.UPF")
                  for s in symbols}
        try:
            PWInput(
                structure, pseudo=pseudo,
                control={"calculation": QE_CALCULATIONS[preset],
                         "prefix": str(name), "tprnfor": True, "tstress": True},
                system={"ecutwfc": 60, "ecutrho": 480,
                        **(overrides or {}).get("system", {})},
                electrons={"conv_thr": 1e-8},
                kpoints_grid=(6, 6, 6),
            ).write_file(str(directory / "pw.in"))
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
            written.append("")
            continue
        (directory / MANIFEST).write_text(json.dumps({
            "obs_name": str(name), "preset": preset, "code": "espresso",
            "source_variant": source,
            "reference": PRESET_REFERENCE.get(preset),
        }, indent=2), encoding="utf-8")
        written.append(str(directory))

    md.obs["dft_directory"] = written
    md.uns["dft"] = {
        "root": str(root), "preset": preset, "code": "espresso",
        "source": source, "reference": PRESET_REFERENCE.get(preset),
        "n_written": int(sum(1 for w in written if w)), "errors": failed,
        "note": "pseudopotentials are named, not shipped. Which set a run used "
                "is part of the level of theory — SSSP and PSLibrary disagree "
                "for the same functional — so pass pseudopotentials= rather "
                "than accepting the placeholder filenames.",
    }
    record(md, "dft.write_inputs", preset=preset, source=source, code="espresso",
           n_written=int(sum(1 for w in written if w)))
    return written


@register_function(
    aliases=["read outputs", "parse vasprun", "read dft", "collect results",
             "parse dft outputs", "harvest calculations"],
    category="dft",
    description="Parse finished first-principles runs back onto the rows they "
                "came from, depositing the energies, the relaxed geometry and "
                "the band gap as one level of theory.",
    produces={"obs": ["energy_{level}", "energy_per_atom_{level}",
                      "band_gap_{level}", "converged_{level}"],
              "structures": ["relaxed_{level}"], "levels": ["{level}"],
              "uns": ["dft"]},
    prerequisites=["mv.dft.write_inputs"],
    examples=["mv.dft.read_outputs(md, 'runs/', level='pbe')"],
    related=["mv.dft.write_inputs", "mv.thermo.hull"],
    notes="Rows whose run is missing or unconverged get NaN and a reason rather "
          "than being dropped, because which candidates failed is a result — a "
          "systematically failing corner of composition space is worth seeing.",
)
def read_outputs(md: AnnData, root, level: str = "pbe",
                 filename: str = "vasprun.xml",
                 require_converged: bool = True) -> None:
    """Parse completed runs into a level of theory."""
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except ImportError as exc:                            # pragma: no cover
        raise ImportError("mv.dft.read_outputs needs pymatgen's VASP "
                          "parsers") from exc

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    energies, per_atom, gaps, converged, reasons = [], [], [], [], []
    relaxed = list(structures(md, "input"))
    n_read = 0

    for i, name in enumerate(md.obs_names):
        path = _find_output(root, str(name), filename)
        if path is None:
            _append_missing(energies, per_atom, gaps, converged, reasons,
                            "no output found")
            continue
        try:
            run = Vasprun(str(path), parse_dos=False, parse_eigen=False)
        except Exception as exc:
            _append_missing(energies, per_atom, gaps, converged, reasons,
                            f"{type(exc).__name__}: {exc}")
            continue

        ok = bool(getattr(run, "converged", False))
        if require_converged and not ok:
            _append_missing(energies, per_atom, gaps, converged, reasons,
                            "did not converge")
            converged[-1] = False
            continue

        structure = run.final_structure
        energy = float(run.final_energy)
        energies.append(energy)
        per_atom.append(energy / len(structure))
        gaps.append(_band_gap(run))
        converged.append(ok)
        reasons.append("")
        relaxed[i] = structure
        n_read += 1

    md.obs[f"energy_{level}"] = energies
    md.obs[f"energy_per_atom_{level}"] = per_atom
    md.obs[f"band_gap_{level}"] = gaps
    md.obs[f"converged_{level}"] = converged
    md.obs[f"dft_error_{level}"] = reasons
    deposit_structures(md, f"relaxed_{level}", relaxed)

    declared = md.uns.get("dft", {})
    set_level(md, level, kind="dft",
              method=declared.get("preset", "VASP"),
              reference=declared.get("reference"),
              surrogate=False, license=None, uncertainty=None,
              code="VASP", root=str(root), n_read=n_read,
              n_missing=int(md.n_obs - n_read))
    record(md, "dft.read_outputs", level=level, root=str(root), n_read=n_read)


def _append_missing(energies, per_atom, gaps, converged, reasons,
                    why: str) -> None:
    energies.append(np.nan)
    per_atom.append(np.nan)
    gaps.append(np.nan)
    converged.append(False)
    reasons.append(why)


def _find_output(root: Path, name: str, filename: str):
    """The output file for one row, by manifest first and directory name after.

    Looking at the manifest first means a directory renamed by a workflow
    manager still resolves, which is the usual reason a hand-rolled harvest
    silently attaches results to the wrong row.
    """
    direct = root / name / filename
    if direct.exists():
        return direct
    for manifest in root.glob(f"*/{MANIFEST}"):
        try:
            if json.loads(manifest.read_text()).get("obs_name") == name:
                candidate = manifest.parent / filename
                return candidate if candidate.exists() else None
        except Exception:
            continue
    return None


def _band_gap(run) -> float:
    try:
        gap = run.eigenvalue_band_properties[0]
        return float(gap)
    except Exception:
        return float("nan")


@register_function(
    aliases=["presets", "dft presets", "which input sets", "list presets"],
    category="dft",
    description="List the first-principles input presets, what each is for, and "
                "which level of theory it reproduces.",
    examples=["mv.dft.presets()"],
    related=["mv.dft.write_inputs"],
)
def presets() -> dict:
    """Preset name to its description and the method it reproduces."""
    return {name: {"input_set": f"{module}.{cls}",
                   "reference": PRESET_REFERENCE.get(name),
                   "description": description}
            for name, (module, cls, description) in VASP_PRESETS.items()}


@register_function(
    aliases=["dft status", "which runs finished", "check runs",
             "calculation status", "run report"],
    category="dft",
    description="Report how many first-principles runs are finished, missing or "
                "unconverged, and which rows they belong to.",
    requires={"obs": ["dft_directory"]},
    prerequisites=["mv.dft.write_inputs"],
    examples=["mv.dft.status(md, 'runs/')"],
    related=["mv.dft.read_outputs", "mv.utils.resume"],
    notes="Worth running before read_outputs on a large campaign: a directory "
          "that never started and one that crashed look identical from the "
          "object, and the difference decides whether to resubmit.",
)
def status(md: AnnData, root, filename: str = "vasprun.xml") -> dict:
    """How many runs have produced output, and which rows have not."""
    root = Path(root)
    finished, missing = [], []
    for name in md.obs_names:
        path = _find_output(root, str(name), filename)
        (finished if path is not None else missing).append(str(name))
    return {"root": str(root), "n_total": int(md.n_obs),
            "n_finished": len(finished), "n_missing": len(missing),
            "missing": missing[:50],
            "truncated": len(missing) > 50}


__all__ = ["write_inputs", "read_outputs", "presets", "status",
           "VASP_PRESETS", "PRESET_REFERENCE"]

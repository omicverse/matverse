"""How much of pymatgen matverse reaches, and where the rest went.

matverse is not a pymatgen wrapper — it is a substrate that keeps pymatgen's
results attached to the materials they belong to. But "how much of pymatgen does
it cover" is a fair question with a wrong answer available, and this module
exists because the wrong answer was given twice: a figure of 15/48 was quoted
against a denominator that counted only the top level of ``pymatgen.analysis``,
where the real tree has 110 leaf modules.

Every public pymatgen module is classified here into exactly one bucket, and
``tests/test_pymatgen_coverage.py`` fails on any module that is not. That makes
the number reproducible and makes a gap something you have to write down rather
than something you can leave unmentioned.

The buckets
-----------
``WRAPPED``
    Imported by matverse and reachable through a named matverse function.

``NATIVE``
    The capability exists in matverse, implemented without importing pymatgen's
    module — usually because the computation runs through ASE or because the
    result had to land on an AnnData axis anyway. Covered as a capability;
    honest to distinguish from WRAPPED, because a bug in pymatgen's version is
    not a bug in ours and vice versa.

``INTERNAL``
    Plumbing rather than API: helpers, constants, plotting for pymatgen's own
    figures, base classes. Nothing to cover.

``NOT_A_GOAL``
    Deliberately outside matverse, with the reason recorded. A wrapper here
    would add a name rather than a capability, or would be a maintenance
    liability with no user.

``TODO``
    A real gap. Anything not named in the other four lands here by default, so
    the list cannot quietly shrink.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

# --------------------------------------------------------------- WRAPPED
#: pymatgen module -> the matverse functions that reach it.
WRAPPED: Dict[str, List[str]] = {
    "analysis.adsorption": ["mv.surf.adsorption_sites"],
    "analysis.bond_valence": ["mv.transform.oxidation_states"],
    "analysis.chemenv.coordination_environments.chemenv_strategies":
        ["mv.env.chemenv"],
    "analysis.chemenv.coordination_environments.coordination_geometry_finder":
        ["mv.env.chemenv"],
    "analysis.chemenv.coordination_environments.structure_environments":
        ["mv.env.chemenv"],
    "analysis.diffraction.xrd": ["mv.prop.xrd"],
    "analysis.dimensionality": ["mv.prop.dimensionality"],
    "analysis.eos": ["mv.prop.eos"],
    "analysis.graphs": ["mv.prop.dimensionality", "mv.env.bonds"],
    "analysis.interface_reactions": ["mv.iface.reactivity"],
    "analysis.interfaces.coherent_interfaces": ["mv.iface.build"],
    "analysis.interfaces.substrate_analyzer": ["mv.iface.match"],
    "analysis.local_env": ["mv.env.coordination", "mv.env.bonds",
                           "mv.prop.dimensionality"],
    "analysis.magnetism.analyzer": ["mv.mag.orderings", "mv.mag.describe"],
    "analysis.nmr": ["mv.prop.nmr", "mv.prop.efg"],
    "analysis.phase_diagram": ["mv.thermo.hull", "mv.thermo.chempot_limits"],
    "analysis.piezo": ["mv.prop.piezoelectric"],
    "analysis.pourbaix_diagram": ["mv.thermo.pourbaix"],
    "analysis.reaction_calculator": ["mv.thermo.reaction"],
    "analysis.solar.slme": ["mv.prop.slme"],
    "analysis.structure_matcher": ["mv.pp.dedup", "mv.gen.validate"],
    "analysis.wulff": ["mv.surf.wulff"],
    "core.composition": ["mv.data.from_structures"],
    "core.periodic_table": ["mv.multi.sites", "mv.pl.periodic_table"],
    "core.surface": ["mv.surf.slabs"],
    "electronic_structure.core": ["mv.elec.bands"],
    "core.entries": ["mv.thermo.hull", "mv.thermo.reaction"],
    "analysis.compatibility": ["mv.thermo.corrections"],
    # pymatgen moved these into analysis/ and core/ in 2026.5; the older
    # layout matverse also supports still has them here. Both names are
    # listed so the map holds on either, and only the one that exists in
    # the installed tree is ever counted.
    "entries.compatibility": ["mv.thermo.corrections"],
    "entries.computed_entries": ["mv.thermo.hull", "mv.thermo.reaction"],
    "core.lattice": ["mv.data.from_structures"],
    "core.structure": ["mv.data.from_structures", "mv.mol.from_molecules"],
    "io.ase": ["mv.data.from_ase", "mv.data.to_ase"],
    "io.cif": ["mv.data.from_cif", "mv.data.to_cif"],
    "io.lobster.outputs": ["mv.elec.cohp"],
    "io.pwscf": ["mv.dft.write_inputs"],
    "io.vasp.outputs": ["mv.dft.read_outputs", "mv.dft.read_dos"],
    "io.vasp.sets": ["mv.dft.write_inputs"],
    "io.xyz": ["mv.mol.from_molecules"],
    "symmetry.analyzer": ["mv.pp.standardize", "mv.pp.describe",
                          "mv.mol.point_group"],
    "symmetry.bandstructure": ["mv.elec.kpath"],
    "transformations.advanced_transformations": ["mv.transform.apply"],
    "transformations.standard_transformations": ["mv.transform.apply"],
    "transformations.site_transformations": ["mv.transform.apply"],
}

#: Names that mean the same capability in different pymatgen layouts. matverse
#: supports two pymatgen versions and they disagree about where several modules
#: live; a claim is backed if any member of its group is reached.
EQUIVALENT = [
    frozenset({"entries.computed_entries", "core.entries",
               "analysis.compatibility.computed_entries"}),
    frozenset({"entries.compatibility", "analysis.compatibility"}),
    frozenset({"entries.entry_tools", "analysis.compatibility.entry_tools"}),
]


def equivalents(module: str) -> frozenset:
    """Every name the installed pymatgen might file this capability under."""
    for group in EQUIVALENT:
        if module in group:
            return group
    return frozenset({module})


# ---------------------------------------------------------------- NATIVE
#: Capability present in matverse, implemented without pymatgen's module.
NATIVE: Dict[str, str] = {
    "analysis.elasticity.elastic":
        "mv.prop.elastic computes the stiffness tensor by finite strain "
        "through an ASE calculator, so the deformations and the stresses are "
        "the calculator's rather than pymatgen's",
    "analysis.elasticity.strain": "see analysis.elasticity.elastic",
    "analysis.elasticity.stress": "see analysis.elasticity.elastic",
    "analysis.transition_state":
        "mv.neb.barrier runs the band through ASE's NEB, which is what the "
        "calculators in mv.calc are written against",
    "analysis.diffusion.neb.pathfinder":
        "mv.neb.hop_endpoints builds the endpoints from the periodic image "
        "convention directly; see the warning in the defects tutorial for why "
        "the image choice is the whole calculation",
    "analysis.diffusion.analyzer":
        "mv.md.run computes the mean squared displacement over its own "
        "trajectory rather than parsing one back",
    "analysis.optics":
        "mv.prop.dielectric derives the absorption coefficient from epsilon "
        "so that epsilon stays recoverable",
    "analysis.ewald":
        "reached through pymatgen's own transformations in mv.disorder."
        "orderings rather than called directly",
    "analysis.structure_analyzer":
        "mv.pp.qc and mv.env cover the parts a screen needs — minimum "
        "distances, coordination, valence",
    "electronic_structure.bandstructure":
        "mv.elec.bands takes a BandStructure the caller already has and gives "
        "it an axis; the class is consumed, never constructed, so matverse "
        "never imports it",
    "electronic_structure.dos":
        "a density of states arrives through mv.dft.read_dos as an array on "
        "the grid convention rather than as pymatgen's Dos object",
    "analysis.xas.spectrum":
        "an XAS spectrum is a curve on an energy grid, which uns['grids'] and "
        "an obsm block already are: mv.exp.attach stores one and "
        "mv.prop.compare_grids compares it against a computed edge",
}

# ------------------------------------------------------------- NOT_A_GOAL
NOT_A_GOAL: Dict[str, str] = {
    "cli": "pymatgen's command line, not a library capability",
    "apps": "pymatgen's own applications (borg, battery)",
    "command_line": "shells out to external binaries pymatgen wraps",
    "vis": "3D visualisation; mv.pl.structure covers the quick look and "
           "VESTA and Crystal Toolkit cover real inspection",
    "util": "internal helpers, testing utilities and provenance",
    "dao": "internal data access object",
    "optimization": "compiled numerical kernels",
    "ext": "third-party web APIs; matverse reaches MP and OPTIMADE through "
           "mv.data and mv.datasets",
    "alchemy": "pymatgen's own transformation-history container; "
               "uns['provenance'] is matverse's answer to the same question",
}

#: Whole ``io`` format families matverse deliberately hands back to pymatgen.
#: The doors matverse opens are in WRAPPED; the rest read far more formats than
#: matverse should try to track, and each is a parser with its own release
#: cycle.
IO_NOT_A_GOAL = {
    "abinit", "adf", "atat", "babel", "common", "core", "cp2k", "cssr",
    "exciting", "feff", "fiesta", "gaussian", "icet", "jarvis", "jdftx",
    "lammps", "lmto", "multiwfn", "nwchem", "openff", "packmol", "phonopy",
    "prismatic", "pwmat", "qchem", "registry", "res", "shengbte", "template",
    "wannier90", "xcrysden", "xr", "xtb", "zeopp",
}

#: ``io.optimade`` is reached through mv.data.from_optimade, which speaks the
#: protocol directly rather than through pymatgen's client.
IO_NATIVE = {"optimade"}

#: Path fragments that mark a module as plumbing rather than API.
INTERNAL_MARKERS = (
    ".utils.", ".utils", ".plotting.", ".plotting", ".constants",
    ".core.core", "._",
)

#: Individual modules that are plumbing despite not matching a marker.
INTERNAL = {
    "analysis.compatibility.compatibility",
    "analysis.diffraction.core",
    "analysis.alloys.rgb",
    "analysis.defects.constants",
    "analysis.chemenv.utils.chemenv_config",
    "analysis.chemenv.utils.chemenv_errors",
    "analysis.molecule_structure_comparator",
    "entries.entry_tools",
    "io.vasp.help",
    "io.common",
}


_STUB = re.compile(r"from\s+(pymatgen[\w.]*)\s+import\s+\*")


def _root(root: str | None = None) -> str:
    if root is not None:
        return root
    import pymatgen
    return list(pymatgen.__path__)[0]


def aliases(root: str | None = None) -> Dict[str, str]:
    """Re-export shims, mapped to the module that actually holds the code.

    pymatgen moves modules between subpackages and leaves a stub behind —
    ``pymatgen.analysis.local_env`` is now three lines re-exporting
    ``pymatgen.core.local_env``, and twenty-one others moved with it in the same
    release. Counting both names doubles the denominator and puts the real
    module in the gap list while the stub reads as covered, which is precisely
    the failure this file exists to prevent. Detected by reading the source
    rather than hard-coded, so the next reorganisation is absorbed rather than
    mismeasured.
    """
    root = _root(root)
    found: Dict[str, str] = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if not d.startswith("_") and d != "__pycache__"]
        for name in sorted(files):
            if not name.endswith(".py") or name.startswith("_"):
                continue
            path = os.path.join(dirpath, name)
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            if len(text) > 2000:
                continue                      # too much code to be a shim
            match = _STUB.search(text)
            if match:
                rel = os.path.relpath(path, root)[:-3].replace(os.sep, ".")
                found[rel] = match.group(1).replace("pymatgen.", "")
    return found


def reached_modules() -> Dict[str, List[str]]:
    """pymatgen modules matverse actually reaches, resolved by import.

    Textual matching cannot do this. ``from pymatgen.io.lobster import
    Icohplist`` names a package, and the class it pulls out lives in
    ``pymatgen.io.lobster.outputs``; ``from pymatgen.analysis import local_env``
    never writes the leaf at all. So each imported name is resolved and asked
    where it was defined, which is the only answer that survives pymatgen
    moving things.

    Returns module -> the names matverse imports from it.
    """
    import glob
    import importlib

    here = os.path.dirname(os.path.abspath(__file__))
    source = ""
    for path in sorted(glob.glob(os.path.join(here, "*.py"))):
        if os.path.basename(path) == "_coverage.py":
            continue
        source += open(path, encoding="utf-8").read() + "\n"
    source = re.sub(r"\\\s*\n\s*", "", source)

    out: Dict[str, List[str]] = {}
    # Two shapes: `import a, b` to end of line, and `import (a, b, ...)` which
    # may run over several lines.
    pattern = re.compile(
        r"from\s+(pymatgen[\w.]*)\s+import\s+(?:\(([^)]*)\)|([^\n(]+))")
    for package, bracketed, plain in pattern.findall(source):
        try:
            module = importlib.import_module(package)
        except Exception:
            continue
        for raw in re.split(r"[,\s]+", (bracketed or plain).strip()):
            name = raw.split(" as ")[0].strip()
            if not name or name.startswith("#"):
                continue
            target = getattr(module, name, None)
            if target is None:
                # `from pymatgen.analysis import local_env` does not bind the
                # submodule on the parent until it is imported.
                try:
                    target = importlib.import_module(f"{package}.{name}")
                except Exception:
                    continue
            origin = getattr(target, "__module__", None) or getattr(
                target, "__name__", None)
            if not origin or not str(origin).startswith("pymatgen"):
                origin = package
            key = str(origin).replace("pymatgen.", "", 1)
            out.setdefault(key, []).append(name)
    # plain `import pymatgen.x.y` forms
    for dotted in re.findall(r"(?<!from )\bimport\s+(pymatgen[\w.]*)", source):
        out.setdefault(dotted.replace("pymatgen.", "", 1), [])
    # module paths held as strings and imported dynamically — mv.dft names its
    # VASP input sets that way, so the reference is real but never a statement.
    for quoted in set(re.findall(r"[\"']（?(pymatgen[\w.]+)[\"']", source)):
        try:
            importlib.import_module(quoted)
        except Exception:
            continue
        out.setdefault(quoted.replace("pymatgen.", "", 1), [])
    return {k: sorted(set(v)) for k, v in sorted(out.items())}


def canonical(module: str, alias_map: Dict[str, str] | None = None) -> str:
    """Follow re-export shims to the module that holds the implementation."""
    alias_map = aliases() if alias_map is None else alias_map
    seen = set()
    while module in alias_map and module not in seen:
        seen.add(module)
        module = alias_map[module]
    return module


def public_modules(root: str | None = None) -> List[str]:
    """Every public pymatgen module that holds code, shims resolved away."""
    root = _root(root)
    alias_map = aliases(root)
    out = set()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if not d.startswith("_") and d != "__pycache__"]
        # A package __init__ can hold real API rather than only re-exports:
        # pymatgen's correction schemes live in analysis/compatibility/
        # __init__.py, and skipping it made the classes matverse calls
        # invisible to the count.
        init = os.path.join(dirpath, "__init__.py")
        if dirpath != root and os.path.exists(init):
            try:
                text = open(init, encoding="utf-8").read()
            except OSError:
                text = ""
            if re.search(r"^\s*(class|def)\s", text, re.M):
                package = os.path.relpath(dirpath, root).replace(os.sep, ".")
                if package not in alias_map:
                    out.add(package)
        for name in sorted(files):
            if name.endswith(".py") and not name.startswith("_"):
                rel = os.path.relpath(os.path.join(dirpath, name), root)[:-3]
                module = rel.replace(os.sep, ".")
                if module in alias_map:
                    continue                  # counted as its target
                out.add(module)
    return sorted(out)


def classify(module: str, alias_map: Dict[str, str] | None = None) -> str:
    """Which bucket a pymatgen module falls in."""
    alias_map = aliases() if alias_map is None else alias_map
    module = canonical(module, alias_map)
    top = module.split(".")[0]
    if top in NOT_A_GOAL:
        return "NOT_A_GOAL"
    if module in INTERNAL or any(m in module for m in INTERNAL_MARKERS):
        return "INTERNAL"
    # The maps may name either a shim or the module it points at.
    for name, bucket in ((WRAPPED, "WRAPPED"), (NATIVE, "NATIVE")):
        if module in name:
            return bucket
        if any(canonical(k, alias_map) == module for k in name):
            return bucket
    if top == "io":
        family = module.split(".")[1] if "." in module else ""
        if family in IO_NATIVE:
            return "NATIVE"
        if family in IO_NOT_A_GOAL:
            return "NOT_A_GOAL"
    return "TODO"


def report(root: str | None = None) -> dict:
    """Counts per bucket, and the modules still in TODO."""
    alias_map = aliases(root)
    buckets: Dict[str, List[str]] = {}
    for module in public_modules(root):
        buckets.setdefault(classify(module, alias_map), []).append(module)
    total = sum(len(v) for v in buckets.values())
    reachable = (len(buckets.get("WRAPPED", []))
                 + len(buckets.get("NATIVE", []))
                 + len(buckets.get("TODO", [])))
    covered = len(buckets.get("WRAPPED", [])) + len(buckets.get("NATIVE", []))
    return {
        "total": total,
        "buckets": {k: len(v) for k, v in sorted(buckets.items())},
        "in_scope": reachable,
        "covered": covered,
        "fraction": covered / reachable if reachable else 0.0,
        "todo": sorted(buckets.get("TODO", [])),
    }


def summary(root: str | None = None) -> str:
    data = report(root)
    lines = [f"pymatgen modules: {data['total']}"]
    for bucket, count in data["buckets"].items():
        lines.append(f"  {bucket:12s} {count:4d}")
    lines.append("")
    lines.append(f"in scope: {data['in_scope']}; covered: {data['covered']} "
                 f"= {data['fraction']:.1%}")
    if data["todo"]:
        lines.append("")
        lines.append("still to do:")
        lines += [f"  {m}" for m in data["todo"]]
    return "\n".join(lines)


__all__ = ["WRAPPED", "NATIVE", "NOT_A_GOAL", "INTERNAL", "EQUIVALENT",
           "equivalents", "public_modules",
           "aliases", "canonical", "classify", "report", "summary"]

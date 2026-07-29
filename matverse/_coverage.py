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
    "analysis.eos": ["mv.prop.eos", "mv.prop.quasiharmonic"],
    "analysis.cost": ["mv.prop.cost"],
    "analysis.prototypes": ["mv.pp.prototype"],
    "symmetry.groups": ["mv.pp.symmetry"],
    "analysis.defects.core":
        "the generators mv.pp.defects calls yield Defect objects from here - "
        "an antisite comes back as a defects.core.Substitution - and it is "
        "those objects the supercells are built from",
    "analysis.defects.supercells":
        "Defect.get_supercell_structure, which mv.pp.defects calls for every "
        "interstitial and antisite, picks its cell in this module",
    "electronic_structure.cohp":
        "Icohplist.icohpcollection, which mv.elec.cohp reads, is an "
        "IcohpCollection defined here",
    "core.ion":
        "PourbaixDiagram represents its aqueous species as Ion objects, so "
        "mv.thermo.pourbaix rests on this module even though the diagram "
        "itself needs Materials Project entries to build",
    "analysis.chemenv.coordination_environments.coordination_geometries":
        "the model polyhedra ChemEnv fits are defined here, so the O:6 and T:4 "
        "that mv.env.chemenv reports are this module's data - "
        "LocalGeometryFinder imports it",
    "analysis.chemenv.connectivity.environment_nodes":
        "StructureConnectivity builds its graph out of these nodes, so "
        "mv.env.connectivity walks them",
    "symmetry.structure": ["mv.pp.symmetry"],
    "symmetry.site_symmetries": ["mv.pp.symmetry"],
    "symmetry.settings": ["mv.transform.setting"],
    "core.molecule_matcher": ["mv.mol.match"],
    "analysis.molecule_matcher": ["mv.mol.match"],
    "core.molecule_structure_comparator": ["mv.mol.match"],
    "analysis.molecule_structure_comparator": ["mv.mol.match"],
    "analysis.interfaces.zsl": ["mv.iface.match"],
    "symmetry.kpath": ["mv.elec.kpath"],
    "core.tensors": ["mv.prop.piezoelectric"],
    "core.bonds": ["mv.mol.bond_lengths"],
    "core.molecular_orbitals": ["mv.prop.frontier_orbitals"],
    "core.energy_models": ["mv.prop.electrostatic"],
    "analysis.energy_models": ["mv.prop.electrostatic"],
    "analysis.defects.core": ["mv.pp.defects"],
    "analysis.defects.supercells": ["mv.pp.defects"],
    "electronic_structure.cohp": ["mv.elec.cohp"],
    "analysis.chemenv.coordination_environments.coordination_geometries":
        ["mv.env.chemenv"],
    "analysis.chemenv.connectivity.environment_nodes":
        ["mv.env.connectivity"],
    "core.ion": ["mv.thermo.pourbaix"],
    "core.interface": ["mv.iface.build"],
    "analysis.chemenv.connectivity.connectivity_finder":
        ["mv.env.connectivity"],
    "analysis.chemenv.connectivity.structure_connectivity":
        ["mv.env.connectivity"],
    "analysis.chemenv.connectivity.connected_components":
        ["mv.env.connectivity"],
    "analysis.defects.generators": ["mv.pp.defects"],
    "analysis.structure_prediction.substitution_probability":
        ["mv.gen.predict_substitutions"],
    "analysis.structure_prediction.dopant_predictor":
        ["mv.gen.predict_dopants"],
    "analysis.structure_prediction.substitutor": ["mv.gen.predict_hosts"],
    "core.structure_prediction.substitutor": ["mv.gen.predict_hosts"],
    "analysis.structure_prediction.volume_predictor":
        ["mv.pp.predict_volume"],
    "analysis.hhi": ["mv.prop.supply_risk"],
    "analysis.quasiharmonic": ["mv.prop.quasiharmonic"],
    "analysis.diffraction.neutron": ["mv.prop.neutron"],
    "analysis.diffraction.tem": ["mv.prop.tem"],
    "analysis.diffusion.aimd.rdf": ["mv.md.rdf"],
    "analysis.graphs": ["mv.prop.dimensionality", "mv.env.bonds"],
    "analysis.interface_reactions": ["mv.iface.reactivity"],
    "analysis.interfaces.coherent_interfaces": ["mv.iface.build"],
    "analysis.interfaces.substrate_analyzer": ["mv.iface.match"],
    "analysis.local_env": ["mv.env.coordination", "mv.env.bonds",
                           "mv.prop.dimensionality"],
    "analysis.magnetism.analyzer": ["mv.mag.orderings", "mv.mag.describe"],
    "analysis.magnetism.jahnteller": ["mv.mag.jahn_teller"],
    "analysis.nmr": ["mv.prop.nmr", "mv.prop.efg"],
    "analysis.phase_diagram": ["mv.thermo.hull", "mv.thermo.chempot_limits"],
    "analysis.piezo": ["mv.prop.piezoelectric"],
    "analysis.pourbaix_diagram": ["mv.thermo.pourbaix"],
    "analysis.reaction_calculator": ["mv.thermo.reaction"],
    "analysis.solar.slme": ["mv.prop.slme"],
    "analysis.diffusion.neb.io": ["mv.dft.write_inputs"],
    "analysis.diffusion.neb.periodic_dijkstra": ["mv.neb.percolation"],
    "analysis.alloys.core": ["mv.gen.alloy_pairs"],
    "analysis.quasirrho": ["mv.mol.quasirrho"],
    "analysis.xps": ["mv.elec.xps"],
    "analysis.defects.recombination": ["mv.prop.capture"],
    "analysis.defects.corrections.freysoldt":
        ["mv.thermo.defect_formation"],
    "analysis.diffusion.aimd.clustering": ["mv.md.sites"],
    "analysis.structure_matcher": ["mv.pp.dedup", "mv.gen.validate"],
    "analysis.surface_analysis": ["mv.surf.surface_energy_chempot"],
    "analysis.wulff": ["mv.surf.wulff"],
    "core.composition": ["mv.data.from_structures"],
    "core.periodic_table": ["mv.multi.sites", "mv.pl.periodic_table"],
    "core.surface": ["mv.surf.slabs"],
    "electronic_structure.core": ["mv.elec.bands"],
    "core.entries": ["mv.thermo.hull", "mv.thermo.reaction"],
    "analysis.compatibility": ["mv.thermo.corrections"],
    "analysis.chempot_diagram": ["mv.thermo.chempot_diagram"],
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

#: Modules matverse reaches through an object another module handed it, rather
#: than by importing them. The capability is used; the import is not there to
#: find, so the direct-import check is told why rather than weakened.
TRANSITIVE: Dict[str, str] = {
    "analysis.interfaces.zsl":
        "SubstrateAnalyzer, which mv.iface.match imports, runs the "
        "Zur-McGill lattice search out of this module",
    "symmetry.kpath":
        "HighSymmKpath, which mv.elec.kpath imports, is the front door to the "
        "path constructions here",
    "core.tensors":
        "PiezoTensor subclasses Tensor, so mv.prop.piezoelectric's symmetry "
        "check and IEEE conversion are this module's code",
    "analysis.defects.core":
        "the generators mv.pp.defects calls yield Defect objects from here - "
        "an antisite comes back as a defects.core.Substitution - and it is "
        "those objects the supercells are built from",
    "analysis.defects.supercells":
        "Defect.get_supercell_structure, which mv.pp.defects calls for every "
        "interstitial and antisite, picks its cell in this module",
    "electronic_structure.cohp":
        "Icohplist.icohpcollection, which mv.elec.cohp reads, is an "
        "IcohpCollection defined here",
    "core.ion":
        "PourbaixDiagram represents its aqueous species as Ion objects, so "
        "mv.thermo.pourbaix rests on this module even though the diagram "
        "itself needs Materials Project entries to build",
    "analysis.chemenv.coordination_environments.coordination_geometries":
        "the model polyhedra ChemEnv fits are defined here, so the O:6 and T:4 "
        "that mv.env.chemenv reports are this module's data - "
        "LocalGeometryFinder imports it",
    "analysis.chemenv.connectivity.environment_nodes":
        "StructureConnectivity builds its graph out of these nodes, so "
        "mv.env.connectivity walks them",
    "symmetry.structure":
        "SpacegroupAnalyzer.get_symmetrized_structure returns a "
        "SymmetrizedStructure, and mv.pp.symmetry reads wyckoff_symbols off "
        "it",
    "core.interface":
        "CoherentInterfaceBuilder.get_interfaces yields Interface objects, "
        "which mv.iface.build stores - the numpy site properties they carry "
        "are what broke structure serialisation in v0.1.15",
    "analysis.chemenv.connectivity.structure_connectivity":
        "ConnectivityFinder.get_structure_connectivity returns a "
        "StructureConnectivity, and mv.env.connectivity calls "
        "get_connected_components on it",
    "analysis.chemenv.connectivity.connected_components":
        "the components that call returns are ConnectedComponent objects, and "
        "mv.env.connectivity reads periodicity_vectors off each one",
}


#: Names that mean the same capability in different pymatgen layouts. matverse
#: supports two pymatgen versions and they disagree about where several modules
#: live; a claim is backed if any member of its group is reached.
EQUIVALENT = [
    frozenset({"entries.computed_entries", "core.entries",
               "analysis.compatibility.computed_entries"}),
    frozenset({"entries.compatibility", "analysis.compatibility"}),
    frozenset({"entries.entry_tools", "analysis.compatibility.entry_tools"}),
    frozenset({"analysis.molecule_matcher", "core.molecule_matcher"}),
    frozenset({"analysis.energy_models", "core.energy_models"}),
    frozenset({"analysis.structure_prediction.substitutor",
               "core.structure_prediction.substitutor"}),
    frozenset({"analysis.molecule_structure_comparator",
               "core.molecule_structure_comparator"}),
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
    "analysis.diffusion.neb.full_path_mapper":
        "mv.neb.hops enumerates the symmetry-distinct hops directly, which "
        "is what MigrationGraph is used for - knowing which barriers are "
        "worth computing. Not a wrapper: that module calls "
        "StructureGraph.with_local_env_strategy, renamed upstream to "
        "from_local_env_strategy, so every entry point raises AttributeError "
        "against current pymatgen. Checked on fcc (one hop at a/sqrt2, "
        "multiplicity 24), spinel 8a (one at a*sqrt3/4, multiplicity 16) and "
        "a tetragonal distortion that splits the twelve neighbours into four "
        "in-plane and eight out",
    "analysis.diffusion.aimd.pathway":
        "mv.md.occupancy computes the probability density from the "
        "definition - a histogram of the mobile ions' fractional positions "
        "over the run. Not a wrapper: ProbabilityDensityAnalysis sits beside "
        "generate_stable_sites, which raises on a structure with one stable "
        "site because the condensed distance matrix is empty and scipy's "
        "linkage refuses it, and one well-localised site is the commonest "
        "case there is. The sites that module exists to find are "
        "mv.md.sites' job instead",
    "analysis.compatibility.exp_entries":
        "mv.exp.formation_hull builds the hull from measured formation "
        "enthalpies directly. Not a wrapper: ExpEntry hands ThermoData.value "
        "to PDEntry as an eV energy while every thermochemical table quotes "
        "kJ/mol, and ThermoData carries no unit to check against, so it is "
        "wrong by 96.485 and silent about it. It also rejects any phase "
        "marked gas or liquid, so an oxide hull cannot hold its O2 corner. "
        "matverse takes the unit as a required argument and adds the "
        "elemental references itself - on NIST-JANAF Fe-O it puts hematite "
        "and magnetite on the hull and wustite 0.039 eV/atom above it, "
        "which is the known metallurgy",
    "analysis.thermochemistry":
        "ThermoData is a container for a measured value plus its provenance, "
        "which on this substrate is a column: mv.exp.measure attaches the "
        "value and tags it as an experimental level with its instrument, and "
        "mv.exp.formation_hull consumes it with the unit named. What is not "
        "covered is reading a thermochemical table off disk, and pymatgen "
        "ships no table to read",
    "analysis.diffusion.aimd.van_hove":
        "mv.md.van_hove computes both parts from the definition off a "
        "trajectory. Not a wrapper, deliberately: VanHoveAnalysis exposes "
        "get_1d_plot and get_3d_plot and no data accessor, so wrapping it "
        "would mean reading private attributes, and matverse deposits data "
        "rather than pictures. The normalisation is pinned by two "
        "identities rather than by agreement with anything: the self part "
        "integrates to one, and integrating the distinct part against the "
        "shell volume returns the neighbour count at three radii",
    "analysis.disorder":
        "mv.disorder.sro computes Warren-Cowley parameters from the "
        "definition, alpha_AB = 1 - P(B|A)/c_B, off matverse's own "
        "neighbour lists. Not a wrapper, deliberately: pymatgen's "
        "get_warren_cowley_parameters returns the same value for every "
        "pair, and on B2 - where every nearest neighbour is unlike - gives "
        "-1 for the like pairs as well as the unlike ones, where the "
        "definition requires +1 and -1. mv.disorder.sro gives +1 and -1, "
        "and inverts on the second shell as bcc geometry requires",
    "analysis.compatibility.entry_tools":
        "EntrySet's two useful answers are already matverse's. "
        "get_subset_in_chemsys is a composition filter, which "
        "mv.screen.filter does on obs; ground_states returns the entries on "
        "the hull, which is exactly the rows where mv.thermo.hull has "
        "deposited e_above_hull == 0.",
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
    "analysis.compatibility.mixing_scheme":
        "mv.pp.harmonize solves the same problem by matverse's own route - "
        "fit a per-element offset for each source against a reference using "
        "the compositions they have in common - and works on any batch key "
        "rather than only on a GGA/r2SCAN pair. pymatgen's scheme also could "
        "not be made to emit a single entry here: get_mixing_state_data builds "
        "the comparison table correctly from a set of entries that "
        "process_entries then reports as zero GGA and zero r2SCAN",
    "entries.mixing_scheme": "see analysis.compatibility.mixing_scheme",
    "analysis.defects.thermo":
        "mv.thermo.defect_formation computes the same quantity directly. "
        "FormationEnergyDiagram needs DefectEntry objects with "
        "supercell and bulk ComputedStructureEntries, phase-diagram entries, a "
        "VBM and a gap. Built from what matverse can compute offline, its "
        "formation energies reduce to E_defect - E_bulk + chempot, which is "
        "exactly what mv.thermo.defect_formation already does; the half that "
        "would differ is the image-charge correction, which mv.thermo.defect_formation "
        "now applies from dielectric= alone. Only the potential-alignment "
        "half still needs a LOCPOT",
    "phonon.bandstructure":
        "mv.prop.phonon builds the dynamical matrix from ASE displacements and "
        "deposits frequencies on the grid convention, so pymatgen's phonon "
        "containers are never constructed",
    "phonon.dos": "see phonon.bandstructure",
    "phonon.gruneisen":
        "mv.prop.quasiharmonic computes the Gruneisen parameter from an "
        "equation of state instead",
    "phonon.ir_spectra": "see phonon.bandstructure",
    "phonon.thermal_displacements": "see phonon.bandstructure",
    "core.trajectory":
        "mv.md.run keeps its own trajectory statistics rather than "
        "materialising a Trajectory object",
    "core.units":
        "mv.utils.set_units and mv.utils.convert record units on the object, "
        "which is where a screen needs them",
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

#: Individual ``io`` modules matverse hands back even though it uses a sibling.
#: mv.elec.cohp reads ICOHPLIST through io.lobster.outputs and mv.dft writes
#: inputs through io.vasp.sets; the rest of those two families are more file
#: formats, input writers and a parallel "future" API, which is the same
#: parsing surface the other thirty-three io families are exempted for. Listing
#: a family as covered because one module in it is used would be the mistake
#: this file exists to prevent, and so would leaving the other fifteen in the
#: gap list as though matverse intended to wrap them.
IO_MODULE_NOT_A_GOAL_PREFIXES = ("io.lobster.", "io.vasp.inputs",
                                 "io.vasp.optics")

#: pymatgen's own plotting. ``vis`` is already exempt for 3D visualisation and
#: these are its 2D counterpart; mv.pl is matverse's answer to both.
PLOTTING_NOT_A_GOAL = {"electronic_structure.plotter", "phonon.plotter",
                       "analysis.chemenv.utils.chemenv_config"}

#: Path fragments that mark a module as plumbing rather than API.
INTERNAL_MARKERS = (
    ".utils.", ".utils", ".plotting.", ".plotting", ".constants",
    ".core.core", "._",
)

#: Real gaps that cannot be closed here, and what blocks each. These stay
#: **in scope and uncovered** — they are things matverse should do and does not
#: — but they are not "nobody got to it", and the distinction is the difference
#: between a backlog and a wish.
BLOCKED: Dict[str, str] = {
    "analysis.lobster_env":
        "matverse reads external first-principles output as a matter of "
        "course - mv.dft.read_outputs, mv.dft.read_dos, mv.elec.read_bands "
        "and mv.elec.cohp all take a directory of runs - so this is not "
        "blocked by what matverse is. It is blocked by verification: "
        "LobsterNeighbors needs ICOHPLIST and CHARGE files from a LOBSTER "
        "run. mv.elec.cohp already reads ICOHPLIST from a directory, so the "
        "route in exists; what is missing is a real LOBSTER output "
        "directory to verify against.",
    "analysis.ferroelectricity.polarization":
        "matverse reads external first-principles output as a matter of "
        "course - mv.dft.read_outputs, mv.dft.read_dos, mv.elec.read_bands "
        "and mv.elec.cohp all take a directory of runs - so this is not "
        "blocked by what matverse is. It is blocked by verification: "
        "Polarization.from_outcars_and_structures wants the Berry-phase "
        "output of a sequence of VASP runs along a distortion path. Reading "
        "them is mv.dft.read_outputs' job; the missing piece is such a "
        "sequence to test on.",
    "analysis.piezo_sensitivity":
        "matverse reads external first-principles output as a matter of "
        "course - mv.dft.read_outputs, mv.dft.read_dos, mv.elec.read_bands "
        "and mv.elec.cohp all take a directory of runs - so this is not "
        "blocked by what matverse is. It is blocked by verification: it "
        "needs Born effective charges and force constants from a DFPT run, "
        "which is a vasprun matverse could parse - but not one that exists "
        "here.",
    "analysis.topological.spillage":
        "matverse reads external first-principles output as a matter of "
        "course - mv.dft.read_outputs, mv.dft.read_dos, mv.elec.read_bands "
        "and mv.elec.cohp all take a directory of runs - so this is not "
        "blocked by what matverse is. It is blocked by verification: "
        "SOCSpillage compares two WAVECARs, with and without spin-orbit "
        "coupling. Those are large binary files from two real runs and "
        "there are none here.",
    "analysis.defects.corrections.kumagai":
        "matverse reads external first-principles output as a matter of "
        "course - mv.dft.read_outputs, mv.dft.read_dos, mv.elec.read_bands "
        "and mv.elec.cohp all take a directory of runs - so this is not "
        "blocked by what matverse is. It is blocked by verification: it "
        "needs the atomic site potentials from the OUTCARs of the defective "
        "and pristine supercells. None here.",
    "analysis.defects.ccd":
        "matverse reads external first-principles output as a matter of "
        "course - mv.dft.read_outputs, mv.dft.read_dos, mv.elec.read_bands "
        "and mv.elec.cohp all take a directory of runs - so this is not "
        "blocked by what matverse is. It is blocked by verification: a "
        "configuration-coordinate diagram needs the potential energy "
        "surfaces of two charge states along a distortion, from real runs.",
    "analysis.excitation":
        "matverse reads external first-principles output as a matter of "
        "course - mv.dft.read_outputs, mv.dft.read_dos, mv.elec.read_bands "
        "and mv.elec.cohp all take a directory of runs - so this is not "
        "blocked by what matverse is. It is blocked by verification: an "
        "excitation spectrum comes from a TD-DFT or BSE run; the parser is "
        "the easy half and the run is the missing one.",
    "analysis.bond_dissociation":
        "needs openbabel, and specifically its pybel bindings, which "
        "pymatgen imports as `from openbabel import openbabel, pybel`. "
        "openbabel-wheel 3.1.1.23 does now ship a cp312 manylinux wheel - "
        "the older note here said there was none, which was out of date - "
        "but installing it is not enough: several format plugins need "
        "libXrender.so.1, absent on this system, and pybel builds its "
        "format table by parsing GetSupportedInputFormat() with no "
        "tolerance for a plugin that failed to load, so it dies on a "
        "ValueError and takes BabelMolAdaptor with it. Verified: openbabel "
        "imports, pybel does not.",
    "analysis.fragmenter":
        "needs openbabel's pybel bindings - see analysis.bond_dissociation "
        "for what actually fails and why installing the wheel does not fix "
        "it",
    "analysis.functional_groups":
        "needs openbabel's pybel bindings - see analysis.bond_dissociation. "
        "FunctionalGroupExtractor itself is pure structure analysis and "
        "would be the cheapest of the three to wrap if pybel ever imports "
        "here",
    "analysis.defects.finder":
        "DefectSiteFinder needs dscribe, and dscribe needs numba - not "
        "directly, but through `sparse`, which it imports at module scope. "
        "Verified rather than assumed: with dscribe 2.1.2 on the path, "
        "`from dscribe.descriptors import SOAP` raises ModuleNotFoundError: "
        "No module named 'numba'. The numba version pin is what makes this "
        "expensive; installing it for one module would move the whole "
        "environment.",
    "analysis.chemenv.coordination_environments.voronoi":
        "the Voronoi construction ChemEnv runs before it fits a "
        "polyhedron. Confirmed by runtime check to be loaded and executed "
        "during a mv.env.chemenv call, but pymatgen imports it lazily "
        "inside the finder, so there is no import chain from matverse to "
        "it and no entry point to wrap - only compute_structure_environments",
    "analysis.magnetism.heisenberg":
        "fitting exchange couplings needs spin-polarised energies, and no "
        "calculator matverse ships is spin-polarised, so nothing here can "
        "verify the result.",
    "symmetry.maggroups":
        "a lookup table with no derivation. The database holds all 1651 "
        "magnetic space groups and MagneticSpaceGroup takes a BNS or OG "
        "label, but pymatgen ships no analyser that determines a "
        "structure's magnetic space group from its moments - there is no "
        "MagneticSpaceGroupAnalyzer beside SpacegroupAnalyzer, so there is "
        "no route from a structure to a label.",
    "analysis.compatibility.correction_calculator":
        "fits a correction scheme rather than applying one. "
        "compute_corrections needs experimental formation enthalpies paired "
        "with calculated entries for the same compounds, and pymatgen ships "
        "no such data file - the module references none. Applying the "
        "result is mv.thermo.corrections, wrapped.",
    "electronic_structure.boltztrap": "needs the BoltzTraP binary",
    "electronic_structure.boltztrap2":
        "BoltzTraP2 links against netCDF and does not build here; "
        "mv.elec.transport already reports the install command",
}


#: Individual modules that are plumbing despite not matching a marker.
INTERNAL = {
    # A convenience namespace, not an API. Its own docstring says it
    # "imports the key classes from both vasp_input and vasp_output ... to
    # retain backwards compatibility"; matverse imports io.vasp.outputs and
    # io.vasp.sets directly, which executes this __init__ on the way in.
    "io.vasp",
    # pymatgen's data types and enums rather than capabilities. matverse uses
    # Sites and SymmOps constantly - every Structure is made of them - but
    # there is nothing here to wrap, in the same way transformation_abc has
    # nothing. Kept deliberately short: core.bonds, core.molecular_orbitals,
    # symmetry.groups and symmetry.maggroups all have real API behind them and
    # belong in the gap list, not here.
    "core.sites", "core.operations", "core.spectrum",
    "core.libxcfunc", "core.xcfunc", "core",
    "analysis.compatibility.compatibility",
    "analysis.diffraction.core",
    "analysis.alloys.rgb",
    "analysis.defects.constants",
    "analysis.chemenv.utils.chemenv_config",
    "analysis.chemenv.utils.chemenv_errors",
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
    if module in PLOTTING_NOT_A_GOAL or module.endswith(".plotter"):
        return "NOT_A_GOAL"
    if module == "transformations.transformation_abc":
        return "INTERNAL"
    if any(module == p or module.startswith(p)
           for p in IO_MODULE_NOT_A_GOAL_PREFIXES):
        return "NOT_A_GOAL"
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
    todo = buckets.get("TODO", [])
    return {
        "total": total,
        "buckets": {k: len(v) for k, v in sorted(buckets.items())},
        "blocked": sorted(m for m in todo if m in BLOCKED),
        "open": sorted(m for m in todo if m not in BLOCKED),
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


__all__ = ["WRAPPED", "NATIVE", "NOT_A_GOAL", "INTERNAL", "BLOCKED",
           "EQUIVALENT",
           "TRANSITIVE",
           "equivalents", "public_modules",
           "aliases", "canonical", "classify", "report", "summary"]

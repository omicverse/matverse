"""Probe cases for the namespaces ``test_contracts.py`` does not reach.

``test_contracts.py`` grew alongside the first six namespaces and probes them
well. The library then grew to twenty-nine, and the probe battery did not follow
— which left two thirds of the registry's contract claims asserted rather than
verified, the exact state the audit metric was introduced to expose. This module
holds the rest of the battery, and ``test_contract_coverage.py`` fails if an
entry makes a claim that nothing probes.

Not a test module: the leading underscore keeps pytest from collecting it, so
the case table can be imported by more than one test.
"""

from __future__ import annotations

import numpy as np

import matverse as mv

# Functions whose claims ``test_contracts.py`` already probes. Kept here so the
# coverage check can see the whole picture, and asserted against that module's
# own report so the two cannot drift apart.
NAMES_PROBED_IN_TEST_CONTRACTS = frozenset({
    "mv.pp.standardize", "mv.pp.describe", "mv.pp.qc",
    "mv.pp.normalize_composition", "mv.pp.dedup", "mv.pp.supercell",
    "mv.pp.rattle", "mv.pp.strain", "mv.pp.harmonize",
    "mv.feat.element_stats", "mv.feat.similarity",
    "mv.calc.energy", "mv.calc.relax", "mv.thermo.hull",
    "mv.thermo.chempot_limits", "mv.screen.rank",
    "mv.tl.pca", "mv.tl.neighbors", "mv.tl.cluster",
    "mv.tl.rank_elements_groups",
    "mv.prop.xrd", "mv.prop.rdf", "mv.prop.compare_grids", "mv.prop.elastic",
    "mv.prop.phonon", "mv.prop.free_energy", "mv.prop.thermal_conductivity",
    "mv.exp.measure", "mv.exp.match_xrd", "mv.gen.validate",
    "mv.model.split", "mv.model.fit", "mv.model.cross_validate",
    "mv.opt.start", "mv.opt.suggest", "mv.opt.observe",
    "mv.utils.set_units", "mv.utils.convert", "mv.mag.describe",
})

# Entries that make claims no offline probe can decide, and why. Anything not
# listed here must be probed; the coverage test enforces it.
UNPROBEABLE: dict[str, str] = {
    "mv.data.from_mp": "queries the Materials Project over the network",
    "mv.data.from_optimade": "queries an OPTIMADE provider over the network",
    "mv.datasets.fetch": "downloads from a provider over the network",
    "mv.thermo.references_from_mp": "queries the Materials Project",
    "mv.thermo.pourbaix": "needs Materials Project aqueous entries",
    "mv.dft.read_outputs": "parses vasprun.xml from a real VASP run",
    "mv.dft.read_dos": "parses vasprun.xml from a real VASP run",
    "mv.elec.read_bands": "parses vasprun.xml from a real VASP run",
    "mv.elec.cohp": "parses ICOHPLIST.lobster from a real LOBSTER run",
    "mv.feat.matminer": "matminer is not installed in this environment",
    "mv.feat.soap": "dscribe is not installed in this environment",
    "mv.feat.embed": "needs a registered third-party embedding model",
    "mv.utils.submit": "would submit a real job to the scheduler",
    "mv.disorder.sqs": "needs ATAT's mcsqs on PATH, which is not installed",
    "mv.disorder.dope": "needs enumlib on PATH, which is not installed",
}


# ---------------------------------------------------------------- structures
def _fcc(symbol: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def _mixed(occupancies, a=3.7):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [dict(occupancies)] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def _water():
    from pymatgen.core import Molecule
    return Molecule(["O", "H", "H"],
                    [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


def _methane():
    from pymatgen.core import Molecule
    return Molecule(["C", "H", "H", "H", "H"],
                    [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                     [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])


def _ethanol():
    from pymatgen.core import Molecule
    return Molecule(
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [[-1.2, 0.2, 0], [0.0, -0.6, 0], [1.1, 0.3, 0],
         [-1.2, 0.9, 0.9], [-1.2, 0.9, -0.9], [-2.1, -0.4, 0],
         [0.0, -1.2, 0.9], [0.0, -1.2, -0.9], [1.9, -0.2, 0]])


STRUCTURES = [_fcc("Al", 4.05), _fcc("Cu", 3.61), _fcc("Ni", 3.52)]


# ------------------------------------------------------------------ datasets
def crystals():
    return mv.data.from_structures([s.copy() for s in STRUCTURES])


def described():
    md = crystals()
    mv.pp.describe(md)
    return md


def energised():
    md = crystals()
    mv.calc.energy(md, level="emt")
    return md


def molecules():
    md = mv.mol.from_molecules([_water(), _methane(), _ethanol()])
    mv.pp.describe(md)
    return md


def ethanol():
    md = mv.mol.from_molecules([_ethanol()])
    mv.pp.describe(md)
    return md


def mixture():
    return mv.data.from_structures([
        _mixed({"Cu": 0.5, "Au": 0.5}, a=3.8),
        _mixed({"Cu": 0.2, "Ni": 0.2, "Co": 0.2, "Fe": 0.2, "Mn": 0.2}),
    ])


def olivine():
    md = mv.datasets.load("battery_cathodes")[:1].copy()
    mv.pp.describe(md)
    return md


def one_metal():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    mv.calc.energy(md, level="emt")
    return md


def two_metals():
    md = mv.datasets.metals(["Cu", "Al"])
    mv.pp.describe(md)
    mv.calc.energy(md, level="emt")
    return md


def relaxed_metal():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    mv.calc.relax(md, level="emt", fmax=0.01)
    return md


def oxides():
    """An oxide MP applies +U to, one it does not, and a metal."""
    from pymatgen.core import Lattice, Structure
    def cell(symbols):
        return Structure(Lattice.cubic(5.0), symbols,
                         [[0, 0, 0], [.5, .5, .5], [.25, .25, .25],
                          [.5, 0, 0], [0, .5, 0]][:len(symbols)])
    md = mv.data.from_structures([
        cell(["Fe", "Fe", "O", "O", "O"]),
        cell(["Al", "Al", "O", "O", "O"]),
        cell(["Cu", "Cu"])])
    mv.pp.describe(md)
    md.obs["energy_pbe"] = [-50.0, -60.0, -10.0]
    return md


def oxidized():
    """A real cathode with oxidation states, which the prediction model needs."""
    md = mv.datasets.load("battery_cathodes")[:1].copy()
    mv.pp.describe(md)
    mv.transform.oxidation_states(md)
    return md


def perovskites():
    """LaMnO3 (Mn3+, the textbook Jahn-Teller ion) against SrTiO3 (d0)."""
    from pymatgen.core import Lattice, Structure

    def perovskite(a_site, b_site, a):
        return Structure(Lattice.cubic(a), [a_site, b_site, "O", "O", "O"],
                         [[0, 0, 0], [.5, .5, .5], [.5, .5, 0],
                          [.5, 0, .5], [0, .5, .5]])
    md = mv.data.from_structures([perovskite("La", "Mn", 3.9),
                                  perovskite("Sr", "Ti", 3.905)])
    mv.pp.describe(md)
    return md


def alni():
    """B2 AlNi: two sites, two species — the smallest cell with an antisite."""
    from pymatgen.core import Lattice, Structure
    md = mv.data.from_structures([Structure(
        Lattice.cubic(2.89), ["Al", "Ni"], [[0, 0, 0], [.5, .5, .5]])])
    mv.pp.describe(md)
    return md


def quartz():
    """alpha-quartz, the textbook piezoelectric and a non-centrosymmetric one."""
    from pymatgen.core import Lattice, Structure
    structure = Structure.from_spacegroup(
        "P3121", Lattice.hexagonal(4.913, 5.405), ["Si", "O"],
        [[0.4697, 0.0, 0.0], [0.4135, 0.2669, 0.1191]])
    md = mv.data.from_structures([structure])
    mv.pp.describe(md)
    return md


#: alpha-quartz in Voigt form, pC/N, from the measured d11 and d14.
QUARTZ_PIEZO = np.array([[[2.3, -2.3, 0.0, -0.67, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.67, -4.6],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

#: A model absorber: constant real part, a step edge in the imaginary part.
OPTICS_GRID = np.linspace(0.3, 4.0, 400)


def absorbing():
    """Three model absorbers with gaps either side of the SQ maximum."""
    md = mv.datasets.metals(["Cu", "Al", "Ni"])
    mv.pp.describe(md)
    gaps = [1.34, 1.10, 2.00]
    eps1 = np.tile(4.0, (md.n_obs, OPTICS_GRID.size))
    eps2 = np.stack([np.where(OPTICS_GRID >= g, 6.0, 0.0) for g in gaps])
    mv.prop.dielectric(md, OPTICS_GRID, eps1, eps2, level="pbe")
    md.obs["band_gap_pbe"] = gaps
    return md


def sited_metal():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    return md, mv.multi.sites(md)


def shielding_tensors(n):
    """Principal values 10, 20, 60 — hand-checkable in every convention."""
    return np.tile(np.diag([10.0, 20.0, 60.0]), (n, 1, 1))


def gradient_tensors(n):
    return np.tile(np.diag([-1.0, -2.0, 3.0]), (n, 1, 1))


def magnetic():
    md = mv.datasets.metals(["Ni"])
    mv.pp.describe(md)
    return md


def qc_flagged():
    md = described()
    mv.pp.qc(md)
    return md


def with_alni():
    from pymatgen.core import Lattice, Structure
    alni = Structure(Lattice.cubic(2.89), ["Al", "Ni"],
                     [[0, 0, 0], [.5, .5, .5]])
    md = mv.data.from_structures([s.copy() for s in STRUCTURES] + [alni])
    mv.calc.energy(md, level="emt")
    return md


def submitted():
    """A submission record written by hand — nothing goes near the scheduler."""
    md = described()
    md.uns["submissions"] = {
        "0000": {"job_id": "999999", "script": "screen.sbatch",
                 "submitted": "2026-01-01T00:00:00", "state": "UNKNOWN"}}
    return md


def with_dos():
    from matverse._core import deposit_grid
    md = described()
    energies = np.linspace(-5.0, 5.0, 64)
    values = np.exp(-energies ** 2)[None, :].repeat(md.n_obs, axis=0)
    deposit_grid(md, "dos", "tb", values, energies, unit="states/eV")
    return md


def hopped():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    mv.neb.hop_endpoints(md, "Cu", supercell=(1, 1, 1), key_added="hop")
    return md


def dynamic():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    mv.md.run(md, level="emt", steps=30, equilibration=10, sample_every=5)
    return md


def defective():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    out = mv.pp.defects(md, supercell=(1, 1, 1), kinds=("vacancy",))
    mv.calc.energy(out, level="emt")
    return out


def patterned():
    md = described()
    mv.prop.xrd(md, two_theta=(10, 40), step=3.0)
    return md


def two_levels():
    """A second level, so a committee and a parity plot have two to compare."""
    md = described()
    mv.calc.energy(md, level="emt")
    md.obs["energy_emt2"] = md.obs["energy_emt"] * 1.01
    md.obs["energy_per_atom_emt2"] = md.obs["energy_per_atom_emt"] * 1.01
    md.uns["levels"]["emt2"] = dict(md.uns["levels"]["emt"])
    return md


def element_annotated():
    md = described()
    md.var["n_materials"] = np.asarray((md.X > 0).sum(axis=0)).ravel()
    return md


def hulled():
    md = energised()
    mv.thermo.hull(md, level="emt")
    return md


def pareto_ready():
    md = described()
    mv.screen.pareto(md, {"volume": "min", "density": "max"})
    return md


def embedded():
    md = described()
    mv.pp.normalize_composition(md)
    mv.tl.pca(md, n_comps=2)
    return md


def ranked():
    md = described()
    md.obs["group"] = ["a", "b", "a"]
    mv.tl.rank_elements_groups(md, "group")
    return md


def band_structures(metals):
    """Nearest-neighbour tight binding on each structure's own k-path.

    A real calculation on a real path rather than a fixture, so the bands the
    probe reads have the shape the functions expect.
    """
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.symmetry.bandstructure import HighSymmKpath

    out = []
    for structure in mv.structures(metals):
        path = HighSymmKpath(structure)
        kpoints, _ = path.get_kpoints(line_density=6,
                                      coords_are_cartesian=False)
        k = np.asarray(kpoints, dtype=float)
        dispersion = -2.0 * (np.cos(2 * np.pi * k[:, 0])
                             + np.cos(2 * np.pi * k[:, 1])
                             + np.cos(2 * np.pi * k[:, 2]))
        rows = [dispersion + 3.0 * b - 3.0 for b in range(4)]
        out.append(BandStructureSymmLine(
            kpoints, {Spin.up: np.vstack(rows)},
            structure.lattice.reciprocal_lattice, efermi=0.0,
            labels_dict=dict(path.kpath["kpoints"]), structure=structure))
    return out


# ---------------------------------------------------------------- case table
def cases(tmp):
    """``(function, factory, args, kwargs)`` for everything probed here.

    ``tmp`` is a directory the cases may write into.
    """
    import matplotlib
    matplotlib.use("Agg")

    from pathlib import Path
    tmp = Path(tmp)

    def cif_dir():
        d = tmp / "cifs"
        if not d.exists():
            d.mkdir(parents=True)
            for i, s in enumerate(STRUCTURES):
                s.to(filename=str(d / f"s{i}.cif"))
        return d

    def extxyz():
        p = tmp / "structures.extxyz"
        if not p.exists():
            from ase.io import write
            write(str(p), [s.to_ase_atoms() for s in STRUCTURES])
        return p

    def matminer_frame():
        import pandas as pd
        return pd.DataFrame({"structure": [s.copy() for s in STRUCTURES],
                             "band_gap": [0.0, 0.0, 0.0],
                             "formation_energy": [-0.1, 0.0, 0.2]})

    def written():
        md = described()
        mv.dft.write_inputs(md, str(tmp / "runs2"))
        return md

    # Derived objects built once and copied per probe, because building a slab
    # set or a band structure costs more than the probe does.
    olivine_md = olivine()
    olivine_sites = mv.multi.sites(olivine_md)
    mv.env.coordination(olivine_md, olivine_sites)

    bulk = one_metal()
    facets = mv.surf.slabs(bulk, max_index=1)
    mv.calc.energy(facets, level="emt")

    def wulffable():
        out = facets.copy()
        mv.surf.surface_energy(out, bulk, level="emt")
        return out

    configs = mv.surf.adsorption_sites(facets[:1].copy(), "H", max_sites=2)
    mv.calc.energy(configs, level="emt")

    metals = two_metals()
    pairs = mv.iface.match(metals, max_area=120.0)
    bands = mv.elec.bands(metals, band_structures(metals), level="tb",
                          n_points=60)

    orderings = mv.mag.orderings(magnetic(), max_orderings=2)
    mv.calc.energy(orderings, level="emt")

    nmr_md, nmr_sites = sited_metal()

    return [
        # mv.data — a constructor deposits on the object it returns
        (mv.data.from_structures, lambda: [s.copy() for s in STRUCTURES],
         (), {"returns": "new"}),
        (mv.data.from_ase, lambda: [s.to_ase_atoms() for s in STRUCTURES],
         (), {"returns": "new"}),
        (mv.data.from_iterable, lambda: iter([s.copy() for s in STRUCTURES]),
         (), {"returns": "new"}),
        (mv.data.from_cif, cif_dir, (), {"returns": "new"}),
        (mv.data.from_ase_file, extxyz, (), {"returns": "new"}),
        (mv.data.from_matminer, matminer_frame, (), {"returns": "new"}),
        (mv.data.to_ase, crystals, (), {}),
        (mv.data.to_pymatgen, crystals, (), {}),
        (mv.data.to_matminer, crystals, (), {}),
        (mv.data.to_cif, crystals, (tmp / "out",), {}),
        (mv.data.from_optimade_response, _optimade_payload, (),
         {"returns": "new"}),
        (mv.data.optimade_providers, lambda: None, (), {}),

        # mv.datasets
        (mv.datasets.load, lambda: "battery_cathodes", (), {"returns": "new"}),
        (mv.datasets.metals, lambda: ["Cu", "Al"], (), {"returns": "new"}),
        (mv.datasets.available, lambda: None, (), {}),
        (mv.datasets.cached, lambda: None, (), {}),

        # mv.transform
        (mv.transform.available, lambda: None, (), {}),
        (mv.transform.apply, described, ("PrimitiveCellTransformation",),
         {"key_added": "primitive"}),
        (mv.transform.chain, described,
         ([("PrimitiveCellTransformation", {}),
           ("PerturbStructureTransformation", {"distance": 0.05})],), {}),
        (mv.transform.oxidation_states, olivine, (), {}),
        (mv.transform.expand, mixture,
         ("OrderDisorderedStructureTransformation",),
         {"n": 2, "no_oxi_states": True, "returns": "new"}),

        # mv.mol
        (mv.mol.from_molecules, lambda: [_water(), _methane()], (),
         {"returns": "new"}),
        (mv.mol.point_group, molecules, (), {}),
        (mv.mol.descriptors, molecules, (), {}),
        (mv.mol.match, molecules, (), {}),
        (mv.mol.bond_lengths, molecules, (), {}),
        (mv.mol.fragments, ethanol, (), {"returns": "new"}),
        (mv.mol.bonds, molecules, (mv.multi.sites(molecules()),), {}),

        # mv.disorder
        (mv.disorder.describe, mixture, (), {}),
        (mv.disorder.orderings, mixture, (), {"n": 2, "returns": "new"}),

        # mv.multi and mv.env — the two-object namespaces
        (mv.multi.sites, described, (), {"returns": "new"}),
        (mv.multi.aggregate, olivine_sites.copy, (olivine_md,),
         {"column": "coordination_number", "how": "max",
          "key_added": "max_cn"}),
        (mv.multi.to_mudata, described, (), {}),
        (mv.env.coordination, olivine, (mv.multi.sites(olivine()),), {}),
        (mv.env.bonds, olivine, (mv.multi.sites(olivine()),), {}),
        (mv.env.chemenv, olivine, (mv.multi.sites(olivine()),), {}),
        (mv.env.connectivity, perovskites, (), {}),
        (mv.env.summarise, olivine_sites.copy, (olivine_md,), {}),

        # mv.tl, mv.gen, mv.pp
        (mv.tl.novelty, described, (mv.datasets.metals(["Cu"]),), {}),
        (mv.gen.substitute, described, ({"Al": ["Ga"]},), {"returns": "new"}),
        (mv.pp.defects, described, (),
         {"kinds": ("vacancy",), "returns": "new"}),
        (mv.pp.defects, alni, (),
         {"kinds": ("antisite",), "min_atoms": 8, "max_atoms": 200,
          "returns": "new"}),
        (mv.pp.filter_elements, described, (), {"returns": "new"}),
        (mv.pp.predict_volume, described, (), {}),
        (mv.pp.prototype, described, (), {}),
        (mv.pp.symmetry, described, (), {}),
        (mv.gen.predict_dopants, oxidized, (),
         {"source": "oxidized", "n": 2}),
        (mv.gen.predict_substitutions, oxidized, (),
         {"source": "oxidized", "n": 3, "returns": "new"}),
        (mv.pp.filter_materials, qc_flagged, (), {"returns": "new"}),

        # mv.calc
        (mv.calc.forces, described, (mv.multi.sites(described()),),
         {"level": "emt"}),
        (mv.calc.committee, two_levels, (["emt", "emt2"],),
         {"key": "ensemble"}),
        (mv.calc.check_licenses, energised, (), {}),
        (mv.calc.available, lambda: None, (), {}),

        # mv.screen
        (mv.screen.filter, energised, (), {"energy_emt__lt": 0.0}),
        (mv.screen.pareto, described, ({"volume": "min", "density": "max"},),
         {}),

        # mv.prop
        (mv.prop.eos, relaxed_metal, (),
         {"level": "emt", "source": "relaxed_emt",
          "scales": [0.96, 0.98, 1.0, 1.02, 1.04]}),
        (mv.prop.dimensionality, described, (), {}),
        (mv.prop.cost, described, (), {}),
        (mv.prop.supply_risk, described, (), {}),
        (mv.prop.neutron, described, (),
         {"two_theta": (20.0, 60.0), "step": 1.0}),
        (mv.prop.tem, described, (), {"r_max": 1.0, "step": 0.05}),
        (mv.prop.quasiharmonic, relaxed_metal, (),
         {"level": "emt", "source": "relaxed_emt", "t_max": 500.0,
          "scales": [0.96, 0.98, 1.0, 1.02, 1.04]}),
        (mv.thermo.corrections, oxides, (), {"level": "pbe",
                                             "scheme": "mp2020"}),
        (mv.prop.piezoelectric, quartz, (QUARTZ_PIEZO,), {"level": "exp"}),
        (mv.prop.dielectric, described,
         (OPTICS_GRID, np.tile(4.0, (3, OPTICS_GRID.size)),
          np.tile(1.0, (3, OPTICS_GRID.size))), {"level": "pbe"}),
        (mv.prop.slme, absorbing, (), {"level": "pbe", "thickness": 5e-7}),
        (mv.prop.nmr, lambda: nmr_md, (nmr_sites, shielding_tensors(
            nmr_sites.n_obs)), {"level": "pbe"}),
        (mv.prop.efg, lambda: nmr_md, (nmr_sites, gradient_tensors(
            nmr_sites.n_obs)), {"level": "pbe"}),

        # mv.thermo
        (mv.thermo.reaction, with_alni, (["Al", "Ni"], ["AlNi"]),
         {"level": "emt"}),
        (mv.thermo.defect_formation, defective, (),
         {"host": one_metal(), "level": "emt", "chempot": {"Cu": -3.7},
          "band_gap": 1.2}),

        # mv.surf
        (mv.surf.slabs, one_metal, (), {"max_index": 1, "returns": "new"}),
        (mv.surf.surface_energy, facets.copy, (bulk,), {}),
        (mv.surf.wulff, wulffable, (bulk,), {}),
        (mv.surf.adsorption_sites, lambda: facets[:1].copy(), ("H",),
         {"returns": "new"}),
        (mv.surf.adsorption_energy, configs.copy,
         (facets[:1].copy(), -3.2), {}),

        # mv.iface
        (mv.iface.match, two_metals, (), {"max_area": 120.0, "returns": "new"}),
        (mv.iface.build, two_metals, (1, 0), {"returns": "new"}),
        (mv.iface.reactivity, pairs.copy, (metals,), {"level": "emt"}),

        # mv.elec
        (mv.elec.kpath, two_metals, (), {"line_density": 8}),
        (mv.elec.bands, two_metals, (band_structures(metals),),
         {"level": "tb", "n_points": 60, "returns": "new"}),
        (mv.elec.band_features, bands.copy, (metals,), {"level": "tb"}),
        (mv.elec.transport, metals.copy, (bands,), {"level": "tb"}),
        (mv.elec.dos_fingerprint, with_dos, (), {"level": "tb"}),

        # mv.mag
        (mv.mag.jahn_teller, perovskites, (), {}),
        (mv.mag.orderings, magnetic, (),
         {"max_orderings": 2, "returns": "new"}),
        (mv.mag.ground_state, orderings.copy, (magnetic(),), {"level": "emt"}),

        # mv.neb
        (mv.neb.hop_endpoints, one_metal, ("Cu",), {"supercell": (1, 1, 1)}),
        (mv.neb.barrier, hopped, ("hop_initial", "hop_final"),
         {"level": "emt", "n_images": 3, "steps": 5}),

        # mv.md
        (mv.md.run, one_metal, (),
         {"level": "emt", "steps": 30, "equilibration": 10,
          "sample_every": 5}),
        (mv.md.sweep, one_metal, (),
         {"level": "emt", "temperatures": (300.0, 600.0), "steps": 30,
          "equilibration": 10, "sample_every": 5}),
        (mv.md.conductivity, dynamic, ("Cu",), {"level": "emt"}),
        (mv.md.melt_quench, one_metal, (),
         {"level": "emt", "melt_steps": 20, "quench_steps": 20,
          "equilibrate_steps": 10, "supercell": (1, 1, 1)}),
        (mv.md.batched_available, lambda: None, (), {}),

        # mv.dft
        (mv.dft.write_inputs, described, (str(tmp / "runs"),), {}),
        (mv.dft.status, written, (str(tmp / "runs2"),), {}),
        (mv.dft.presets, lambda: None, (), {}),

        # mv.exp
        (mv.exp.attach, patterned,
         ("xrd", np.zeros((3, 10)), np.linspace(10, 40, 10)), {}),

        # mv.model, mv.opt
        (mv.model.available, lambda: None, (), {}),
        (mv.opt.history, _campaigning, (), {}),

        # mv.utils
        (mv.utils.checkpoint, described, (str(tmp / "ckpt.h5ad"),), {}),
        (mv.utils.resume, energised, ("energy_emt",), {}),
        (mv.utils.map_chunks, described, (lambda block: mv.pp.qc(block),),
         {"size": 2}),
        (mv.utils.slurm_script, lambda: "screen.py",
         (str(tmp / "job.sbatch"),), {}),
        (mv.utils.job_status, submitted, (), {}),
        (mv.utils.check_units, described, (), {}),
        (mv.utils.summary, described, (), {}),
        (mv.utils.chunks, described, (), {}),

        # mv.pl — every plotting claim is a requires
        (mv.pl.set_style, lambda: None, (), {}),
        (mv.pl.structure, described, (0,), {}),
        (mv.pl.periodic_table, element_annotated, (),
         {"color": "n_materials"}),
        (mv.pl.hull, hulled, (), {"level": "emt"}),
        (mv.pl.parity, two_levels, ("energy_per_atom", "emt", "emt2"), {}),
        (mv.pl.pareto, pareto_ready, ("volume", "density"), {}),
        (mv.pl.embedding, embedded, (), {"use_rep": "X_pca"}),
        (mv.pl.spectra, patterned, ("xrd",), {}),
        (mv.pl.provenance, described, (), {}),
        (mv.pl.rank_elements_groups, ranked, (), {}),
    ]


def _campaigning():
    md = described()
    mv.feat.element_stats(md)
    mv.opt.start(md, objective="volume")
    return md


def _optimade_payload():
    """One OPTIMADE structure, in the shape a provider returns it."""
    return {"data": [{
        "id": "mv-1", "type": "structures",
        "attributes": {
            "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.8, 1.8, 1.8]],
            "species_at_sites": ["Al", "Ni"],
            "lattice_vectors": [[3.6, 0.0, 0.0], [0.0, 3.6, 0.0],
                                [0.0, 0.0, 3.6]],
            "species": [{"name": "Al", "chemical_symbols": ["Al"],
                         "concentration": [1.0]},
                        {"name": "Ni", "chemical_symbols": ["Ni"],
                         "concentration": [1.0]}],
            "dimension_types": [1, 1, 1], "nperiodic_dimensions": 3,
        }}]}

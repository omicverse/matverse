"""First-principles I/O, phase equilibria beyond the hull, and lattice dynamics.

The phonon tests check against physics rather than against a stored number: the
zero-point energy of copper, and the classical limit of the heat capacity. A
regression test that only compares to last week's output cannot tell a
refactoring from a sign error.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

import matverse as mv

#: Boltzmann's constant in eV/K, for the Dulong-Petit check.
K_B = 8.617333262e-5


@pytest.fixture(scope="module")
def metals():
    from pymatgen.core import Lattice, Structure

    def fcc(symbol, a):
        return Structure(Lattice.cubic(a), [symbol] * 4,
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    return [fcc("Cu", 3.61), fcc("Al", 4.05)]


class TestDftInputs:
    def test_one_directory_per_material(self, md, tmp_path):
        written = mv.dft.write_inputs(md, tmp_path / "runs")
        assert len([p for p in written if p]) == md.n_obs
        assert md.uns["dft"]["n_written"] == md.n_obs

    def test_a_real_input_set_is_written(self, md, tmp_path):
        written = mv.dft.write_inputs(md, tmp_path / "runs")
        from pathlib import Path
        files = {f.name for f in Path(written[0]).iterdir()}
        assert {"INCAR", "KPOINTS", "POSCAR"} <= files

    def test_the_manifest_records_the_row(self, md, tmp_path):
        written = mv.dft.write_inputs(md, tmp_path / "runs")
        from pathlib import Path
        manifest = json.loads((Path(written[0]) / "matverse.json").read_text())
        assert manifest["obs_name"] == str(md.obs_names[0])
        assert manifest["reference"] == "PBE+U"

    def test_the_preset_decides_the_level_reproduced(self, md, tmp_path):
        mv.dft.write_inputs(md, tmp_path / "a", preset="relax")
        assert md.uns["dft"]["reference"] == "PBE+U"
        mv.dft.write_inputs(md, tmp_path / "b", preset="scan")
        assert md.uns["dft"]["reference"] == "r2SCAN"

    def test_potcars_are_a_specification_by_default(self, md, tmp_path):
        mv.dft.write_inputs(md, tmp_path / "runs")
        assert "licensed" in md.uns["dft"]["note"]

    def test_an_unknown_preset_lists_the_options(self, md, tmp_path):
        with pytest.raises(ValueError, match="unknown preset"):
            mv.dft.write_inputs(md, tmp_path / "runs", preset="magic")

    def test_the_neb_endpoint_preset_fixes_the_cell_and_drops_symmetry(
            self, md, tmp_path):
        """Two endpoints of a hop are only comparable if neither cell moved,
        and a vacancy hop is only *there* if symmetry is off — VASP with
        ISYM on will symmetrise the displaced atom back. Both are settings a
        user gets wrong by hand, which is the reason to have a preset."""
        pytest.importorskip("pymatgen.analysis.diffusion.neb.io")
        from pathlib import Path
        mv.neb.hop_endpoints(md, species="Cu", supercell=(2, 2, 2))
        written = mv.dft.write_inputs(md, tmp_path / "neb",
                                      preset="neb-endpoint",
                                      source="hop_initial")
        incar = (Path(written[0]) / "INCAR").read_text()
        settings = dict(
            line.split("=", 1) for line in incar.splitlines() if "=" in line)
        settings = {k.strip(): v.strip() for k, v in settings.items()}
        assert settings["ISIF"] == "2"
        assert settings["ISYM"] == "0"
        assert md.uns["dft"]["reference"] == "PBE+U"

    def test_the_neb_preset_is_offered_alongside_the_others(self):
        assert "neb-endpoint" in mv.dft.presets()
        assert "diffusion" in mv.dft.presets()["neb-endpoint"]["description"]

    def test_an_unknown_code_is_refused(self, md, tmp_path):
        with pytest.raises(ValueError, match="'vasp' or 'espresso'"):
            mv.dft.write_inputs(md, tmp_path / "runs", code="castep")

    def test_quantum_espresso_input_is_written(self, md, tmp_path):
        from pathlib import Path
        written = mv.dft.write_inputs(md, tmp_path / "qe", code="espresso")
        assert md.uns["dft"]["code"] == "espresso"
        text = (Path(written[0]) / "pw.in").read_text()
        assert "calculation" in text and "vc-relax" in text

    def test_pseudopotentials_are_named_not_shipped(self, md, tmp_path):
        """Which pseudopotential set a run used is part of the level of theory
        — SSSP and PSLibrary disagree for the same functional — so guessing a
        filename would put a silent choice into a recorded result."""
        mv.dft.write_inputs(md, tmp_path / "qe", code="espresso")
        assert "not shipped" in md.uns["dft"]["note"]

    def test_a_named_pseudopotential_is_used(self, md, tmp_path):
        from pathlib import Path
        written = mv.dft.write_inputs(
            md, tmp_path / "qe", code="espresso",
            pseudopotentials={"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"})
        text = (Path(written[0]) / "pw.in").read_text()
        assert "Al.pbe-n-kjpaw_psl.1.0.0.UPF" in text

    def test_presets_describe_themselves(self):
        table = mv.dft.presets()
        assert table["hse"]["reference"] == "HSE06"
        assert "expensive" in table["hse"]["description"].lower()


class TestDftStatus:
    def test_nothing_finished_yet(self, md, tmp_path):
        mv.dft.write_inputs(md, tmp_path / "runs")
        report = mv.dft.status(md, tmp_path / "runs")
        assert report["n_finished"] == 0
        assert report["n_missing"] == md.n_obs

    def test_a_finished_run_is_counted(self, md, tmp_path):
        written = mv.dft.write_inputs(md, tmp_path / "runs")
        from pathlib import Path
        (Path(written[0]) / "vasprun.xml").write_text("<modeling/>")
        report = mv.dft.status(md, tmp_path / "runs")
        assert report["n_finished"] == 1

    def test_a_renamed_directory_still_resolves(self, md, tmp_path):
        """The manifest is what makes the round trip survive a workflow manager
        renaming directories — the usual reason a hand-rolled harvest attaches
        results to the wrong row."""
        written = mv.dft.write_inputs(md, tmp_path / "runs")
        from pathlib import Path
        original = Path(written[0])
        renamed = original.parent / "job-000041"
        original.rename(renamed)
        (renamed / "vasprun.xml").write_text("<modeling/>")

        report = mv.dft.status(md, tmp_path / "runs")
        assert report["n_finished"] == 1
        assert str(md.obs_names[0]) not in report["missing"]


class TestDftOutputs:
    def test_missing_runs_become_nan_with_a_reason(self, md, tmp_path):
        (tmp_path / "runs").mkdir()
        mv.dft.read_outputs(md, tmp_path / "runs", level="pbe")
        assert np.isnan(md.obs["energy_pbe"].to_numpy(dtype=float)).all()
        assert (md.obs["dft_error_pbe"] == "no output found").all()

    def test_a_failed_run_is_recorded_not_dropped(self, md, tmp_path):
        written = mv.dft.write_inputs(md, tmp_path / "runs")
        from pathlib import Path
        (Path(written[0]) / "vasprun.xml").write_text("not xml at all")
        mv.dft.read_outputs(md, tmp_path / "runs", level="pbe")
        assert md.n_obs == 6                      # nothing was dropped
        assert md.obs["dft_error_pbe"].iloc[0] != ""

    def test_the_level_carries_what_the_preset_reproduces(self, md, tmp_path):
        mv.dft.write_inputs(md, tmp_path / "runs", preset="scan")
        mv.dft.read_outputs(md, tmp_path / "runs", level="r2scan")
        assert mv.level_info(md, "r2scan")["reference"] == "r2SCAN"
        assert mv.level_info(md, "r2scan")["surrogate"] is False

    def test_a_missing_root_is_an_error(self, md, tmp_path):
        with pytest.raises(FileNotFoundError):
            mv.dft.read_outputs(md, tmp_path / "nowhere", level="pbe")


class TestDefects:
    def test_vacancies_are_enumerated(self, md):
        defective = mv.pp.defects(md, supercell=(2, 1, 1))
        assert defective.n_obs > md.n_obs
        assert set(defective.obs["defect"]) == {"vacancy"}
        assert set(defective.obs["parent"]) <= set(md.obs_names)

    def test_symmetry_equivalent_sites_are_not_repeated(self):
        """A 32-atom supercell of an elemental fcc metal has one distinct
        vacancy, not 32. Enumerating all of them wastes a calculator on 31."""
        from pymatgen.core import Lattice, Structure
        fcc = Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        md = mv.data.from_structures([fcc])
        defective = mv.pp.defects(md, supercell=(2, 2, 2))
        assert defective.n_obs == 1

    def test_substitutions_widen_the_element_axis(self, md):
        defective = mv.pp.defects(md, supercell=(1, 1, 1),
                                  kinds=("substitution",),
                                  substitutions={"Al": ["Mg"]})
        assert "Mg" in set(defective.var_names)
        assert set(defective.obs["added"]) == {"Mg"}

    def test_substitution_without_a_mapping_is_refused(self, md):
        with pytest.raises(ValueError, match="no substitutions"):
            mv.pp.defects(md, kinds=("substitution",))

    def test_an_unknown_kind_is_named(self, md):
        """'interstitial' was an unknown kind until v0.1.24 and is now one of
        four, so the check needs a kind that really does not exist."""
        with pytest.raises(ValueError, match="unknown defect kind"):
            mv.pp.defects(md, kinds=("frenkel",))

    def test_a_defect_dataset_is_an_ordinary_dataset(self, md):
        defective = mv.pp.defects(md, supercell=(1, 1, 1))
        mv.pp.describe(defective)
        assert "formula" in defective.obs
        assert any(step.startswith("pp.defects")
                   for step in mv.provenance(defective))


class TestEmbedders:
    def test_a_registered_embedder_deposits_a_block(self, md):
        def fake(structures_):
            return np.array([[len(s), float(s.volume)] for s in structures_])

        mv.feat.register_embedder("fake", fake, method="FakeNet",
                                  license="MIT")
        mv.feat.embed(md, model="fake")
        assert md.obsm["X_fake"].shape == (md.n_obs, 2)
        assert md.uns["features"]["X_fake"]["license"] == "MIT"

    def test_an_unregistered_model_says_none_ship(self, md):
        with pytest.raises(KeyError, match="matverse ships none"):
            mv.feat.embed(md, model="nothing-here")

    def test_a_wrong_shape_is_caught(self, md):
        mv.feat.register_embedder("wrong", lambda s: np.zeros((2, 3)))
        with pytest.raises(ValueError, match="one row per material"):
            mv.feat.embed(md, model="wrong")


class TestReactions:
    @pytest.fixture
    def alloys(self):
        from pymatgen.core import Lattice, Structure

        def fcc(symbol, a):
            return Structure(Lattice.cubic(a), [symbol] * 4,
                             [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

        md = mv.data.from_structures([
            fcc("Al", 4.05), fcc("Ni", 3.52),
            Structure(Lattice.cubic(2.89), ["Al", "Ni"],
                      [[0, 0, 0], [.5, .5, .5]]),
        ])
        mv.calc.relax(md, level="emt", fmax=0.05)
        return md

    def test_a_reaction_balances_and_evaluates(self, alloys):
        result = mv.thermo.reaction(alloys, ["Al", "Ni"], ["AlNi"], level="emt")
        assert "energy" in result and np.isfinite(result["energy"])
        assert "Al" in result["equation"] and "Ni" in result["equation"]

    def test_the_result_is_recorded_on_the_object(self, alloys):
        mv.thermo.reaction(alloys, ["Al", "Ni"], ["AlNi"], level="emt",
                           name="formation")
        assert "formation" in alloys.uns["reactions"]
        assert alloys.uns["reactions"]["formation"]["level"] == "emt"

    def test_a_missing_formula_says_what_is_available(self, alloys):
        with pytest.raises(ValueError, match="this dataset has"):
            mv.thermo.reaction(alloys, ["Al", "Fe"], ["AlFe"], level="emt")

    def test_it_needs_energies_first(self, md):
        with pytest.raises(ValueError, match="mv.calc.energy"):
            mv.thermo.reaction(md, ["Al"], ["Al"], level="emt")

    def test_chempot_limits_cover_the_stable_phases(self, alloys):
        mv.thermo.chempot_limits(alloys, level="emt")
        limits = alloys.uns["chempot_limits"]
        assert limits["closed_system"] is True
        assert "bounded by this dataset" in limits["note"]
        assert len(limits["limits"]) >= 1


class TestDefectThermodynamics:
    @pytest.fixture
    def defective(self):
        from pymatgen.core import Lattice, Structure
        fcc = Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        host = mv.data.from_structures([fcc])
        mv.calc.energy(host, level="emt")
        defects = mv.pp.defects(host, supercell=(2, 2, 2))
        mv.calc.energy(defects, level="emt")
        return defects, host

    def test_formation_energy_is_a_line_not_a_number(self, defective):
        """It depends on where the Fermi level sits, so it goes on the grid
        axis rather than into a single column."""
        defects, host = defective
        mv.thermo.defect_formation(defects, host=host, level="emt",
                                   chempot={"Cu": -3.5}, band_gap=1.5)
        curve = defects.obsm["formation_vs_fermi_emt"]
        assert curve.shape == (defects.n_obs, 200)
        grid = mv.grid_of(defects, "formation_vs_fermi")
        assert grid[0] == 0.0 and grid[-1] == pytest.approx(1.5)

    def test_the_stable_charge_is_recorded(self, defective):
        defects, host = defective
        mv.thermo.defect_formation(defects, host=host, level="emt",
                                   chempot={"Cu": -3.5}, band_gap=1.5)
        assert defects.obs["stable_charge_emt"].iloc[0] != ""

    def test_a_missing_chemical_potential_is_refused_loudly(self, defective):
        """A defect creates or destroys atoms, and what they cost is not
        derivable from the defective cell alone."""
        defects, host = defective
        with pytest.warns(UserWarning, match="chemical potential"):
            mv.thermo.defect_formation(defects, host=host, level="emt",
                                       chempot={}, band_gap=1.5)
        assert np.isnan(
            defects.obs["defect_formation_energy_emt"].to_numpy(
                dtype=float)).all()

    def test_the_missing_correction_is_declared(self, defective):
        """A charged defect in a periodic cell interacts with its own images.
        Not correcting is defensible; not saying so is not."""
        defects, host = defective
        mv.thermo.defect_formation(defects, host=host, level="emt",
                                   chempot={"Cu": -3.5})
        recorded = defects.uns["defect_thermodynamics"]
        assert recorded["image_charge_correction"] is False
        assert recorded["potential_alignment"] is False
        # And it says how to get the correction rather than only that it is
        # absent, which is the difference between a caveat and a next step.
        assert "dielectric=" in recorded["note"]
        assert "locpots=" in recorded["note"]

    def test_it_needs_energies_on_both_objects(self, defective):
        defects, host = defective
        bare = mv.data.from_structures(mv.structures(host))
        with pytest.raises(ValueError, match="mv.calc.relax"):
            mv.thermo.defect_formation(defects, host=bare, level="emt")

    def test_it_refuses_a_dataset_that_is_not_defects(self, defective):
        defects, host = defective
        with pytest.raises(ValueError, match="mv.pp.defects"):
            mv.thermo.defect_formation(host, host=host, level="emt")


class TestElectronicStructure:
    def test_missing_runs_give_nan_not_zero(self, md, tmp_path):
        """A material with no output has no band gap. Reporting zero would
        make every unfinished run look like a metal."""
        (tmp_path / "runs").mkdir()
        mv.dft.read_dos(md, tmp_path / "runs", level="pbe")
        assert np.isnan(md.obs["band_gap_pbe"].to_numpy(dtype=float)).all()
        assert not md.obs["is_metal_pbe"].any()

    def test_the_dos_lands_on_a_fermi_referenced_grid(self, md, tmp_path):
        (tmp_path / "runs").mkdir()
        mv.dft.read_dos(md, tmp_path / "runs", level="pbe",
                        energy_range=(-5.0, 5.0), n_points=100)
        grid = mv.grid_of(md, "dos")
        assert md.obsm["dos_pbe"].shape == (md.n_obs, 100)
        assert grid[0] == -5.0 and grid[-1] == 5.0
        assert "Fermi" in md.uns["grids"]["dos"]["unit"]

    def test_the_functional_caveat_is_recorded(self, md, tmp_path):
        """A PBE gap is roughly half the experimental one, which is why
        is_metal is trustworthy and a small gap is not."""
        (tmp_path / "runs").mkdir()
        mv.dft.read_dos(md, tmp_path / "runs", level="pbe")
        assert "half the experimental" in mv.level_info(md, "pbe")["note"]

    def test_a_missing_root_is_an_error(self, md, tmp_path):
        with pytest.raises(FileNotFoundError):
            mv.dft.read_dos(md, tmp_path / "nowhere", level="pbe")


class TestPhonons:
    @pytest.fixture(scope="class")
    def vibrating(self, metals):
        md = mv.data.from_structures(list(metals))
        mv.pp.describe(md)
        mv.calc.relax(md, level="emt", fmax=0.005)
        mv.prop.phonon(md, level="emt", source="relaxed_emt",
                       supercell=(1, 1, 1))
        return md

    def test_a_relaxed_metal_has_no_imaginary_modes(self, vibrating):
        assert (vibrating.obs["n_imaginary_modes_emt"] == 0).all()
        assert vibrating.obs["dynamically_stable_emt"].all()

    def test_the_dos_lands_on_a_shared_grid(self, vibrating):
        grid = mv.grid_of(vibrating, "phonon_dos")
        assert vibrating.obsm["phonon_dos_emt"].shape == (2, len(grid))
        assert vibrating.uns["grids"]["phonon_dos"]["unit"] == "THz"

    def test_the_dos_is_normalised(self, vibrating):
        grid = mv.grid_of(vibrating, "phonon_dos")
        for row in vibrating.obsm["phonon_dos_emt"]:
            area = np.trapezoid(row, grid) if hasattr(np, "trapezoid") \
                else np.trapz(row, grid)
            assert area == pytest.approx(1.0, rel=1e-6)

    def test_zero_point_energy_matches_copper(self, vibrating):
        """Literature ZPE for fcc Cu is about 0.03 eV/atom."""
        by_formula = dict(zip(vibrating.obs["formula"],
                              vibrating.obs["zero_point_energy_emt"]))
        assert by_formula["Cu"] == pytest.approx(0.03, abs=0.015)

    def test_bcc_copper_is_dynamically_unstable(self):
        """The check a hull cannot make.

        Copper's stable phase is fcc; bcc copper is dynamically unstable and
        shows imaginary modes even though its composition is identical. A
        composition can sit on the convex hull and still be a structure that
        will not hold together, which is why generated candidates need this
        check and not only a hull distance.
        """
        from pymatgen.core import Lattice, Structure
        bcc = Structure(Lattice.cubic(2.9), ["Cu", "Cu"],
                        [[0, 0, 0], [.5, .5, .5]])
        md = mv.data.from_structures([bcc])
        mv.prop.phonon(md, level="emt", supercell=(2, 2, 2))
        assert md.obs["n_imaginary_modes_emt"].iloc[0] > 0
        assert not md.obs["dynamically_stable_emt"].iloc[0]

    def test_a_large_supercell_is_refused_with_a_reason(self, metals):
        md = mv.data.from_structures([metals[0]])
        mv.prop.phonon(md, level="emt", supercell=(3, 3, 3))
        # The failure is recorded rather than raised, one row at a time.
        assert md.obs["n_imaginary_modes_emt"].iloc[0] == -1
        assert mv.level_info(md, "emt")["n_failed"] == 1


class TestThermalConductivity:
    @pytest.fixture(scope="class")
    def metals4(self):
        """Four fcc metals whose Debye temperatures span a factor of two."""
        from pymatgen.core import Lattice, Structure

        def fcc(symbol, a):
            return Structure(Lattice.cubic(a), [symbol] * 4,
                             [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

        md = mv.data.from_structures([fcc("Cu", 3.61), fcc("Al", 4.05),
                                      fcc("Au", 4.08), fcc("Ag", 4.09)])
        mv.pp.describe(md)
        mv.calc.relax(md, level="emt", fmax=0.01)
        mv.prop.elastic(md, level="emt", source="relaxed_emt")
        mv.prop.phonon(md, level="emt", source="relaxed_emt",
                       supercell=(1, 1, 1))
        mv.prop.thermal_conductivity(md, level="emt", temperature=300.0)
        return md

    def test_debye_temperatures_land_near_the_literature(self, metals4):
        """Literature: Cu 343, Al 428, Au 165, Ag 225 K."""
        theta = dict(zip(metals4.obs["formula"],
                         metals4.obs["debye_temperature_emt"]))
        assert theta["Au"] == pytest.approx(165.0, rel=0.35)
        assert theta["Ag"] == pytest.approx(225.0, rel=0.35)
        assert theta["Cu"] == pytest.approx(343.0, rel=0.35)

    def test_the_heavy_soft_metals_come_out_lowest(self, metals4):
        theta = dict(zip(metals4.obs["formula"],
                         metals4.obs["debye_temperature_emt"]))
        assert theta["Au"] < theta["Ag"] < theta["Cu"]

    def test_conductivity_is_positive_and_ordered(self, metals4):
        """Gold conducts heat through its lattice least of the four, which is
        what a Debye temperature of 165 K buys."""
        kappa = dict(zip(metals4.obs["formula"],
                         metals4.obs["thermal_conductivity_emt"]))
        assert all(v > 0 for v in kappa.values())
        assert kappa["Au"] < kappa["Ag"] < kappa["Cu"]

    def test_conductivity_falls_with_temperature(self, metals4):
        """Slack's expression is inversely proportional to T, which is the
        expected behaviour above the Debye temperature."""
        hot = metals4.copy()
        mv.prop.thermal_conductivity(hot, level="emt", temperature=900.0)
        assert (hot.obs["thermal_conductivity_emt"].to_numpy(dtype=float) <
                metals4.obs["thermal_conductivity_emt"].to_numpy(
                    dtype=float)).all()

    def test_the_gruneisen_source_is_recorded(self, metals4):
        """It comes from the Poisson ratio when mv.prop.elastic has run and
        from a default otherwise, and which one matters to the answer."""
        assert metals4.uns["thermal_conductivity"]["emt"]["gruneisen_source"] \
            == "Poisson ratio"

    def test_an_explicit_gruneisen_overrides_everything(self, metals4):
        forced = metals4.copy()
        mv.prop.thermal_conductivity(forced, level="emt", gruneisen=2.0)
        assert (forced.obs["gruneisen_emt"] == 2.0).all()
        assert forced.uns["thermal_conductivity"]["emt"]["gruneisen_source"] \
            == "explicit"

    def test_sound_velocities_are_physical(self, metals4):
        """A few thousand metres per second for a metal."""
        v = metals4.obs["sound_velocity_emt"].to_numpy(dtype=float)
        assert ((v > 500) & (v < 10000)).all()

    def test_it_needs_phonons_first(self, md):
        with pytest.raises(ValueError, match="mv.prop.phonon"):
            mv.prop.thermal_conductivity(md, level="emt")

    def test_the_model_names_itself_as_approximate(self, metals4):
        recorded = metals4.uns["thermal_conductivity"]["emt"]
        assert recorded["model"] == "Slack"
        assert "phono3py" in recorded["note"]


class TestVibrationalThermodynamics:
    @pytest.fixture(scope="class")
    def vibrating(self, metals):
        md = mv.data.from_structures(list(metals))
        mv.pp.describe(md)
        mv.calc.relax(md, level="emt", fmax=0.005)
        mv.prop.phonon(md, level="emt", source="relaxed_emt",
                       supercell=(1, 1, 1))
        return md

    def test_heat_capacity_approaches_the_classical_limit(self, vibrating):
        """Dulong-Petit: well above the Debye temperature, Cv -> 3 k_B per atom.

        This is the check worth running on a new calculator, and it is the
        reason these quantities are reported per atom rather than per mode.
        """
        mv.prop.free_energy(vibrating, level="emt", temperature=1500.0)
        capacity = vibrating.obs["heat_capacity_emt"].to_numpy(dtype=float)
        assert capacity == pytest.approx(3.0 * K_B, rel=0.05)

    def test_heat_capacity_falls_at_low_temperature(self, vibrating):
        mv.prop.free_energy(vibrating, level="emt", temperature=1500.0)
        hot = vibrating.obs["heat_capacity_emt"].to_numpy(dtype=float).copy()
        mv.prop.free_energy(vibrating, level="emt", temperature=50.0)
        cold = vibrating.obs["heat_capacity_emt"].to_numpy(dtype=float)
        assert (cold < hot).all()

    def test_entropy_is_positive_and_grows_with_temperature(self, vibrating):
        mv.prop.free_energy(vibrating, level="emt", temperature=100.0)
        low = vibrating.obs["vibrational_entropy_emt"].to_numpy(dtype=float).copy()
        mv.prop.free_energy(vibrating, level="emt", temperature=800.0)
        high = vibrating.obs["vibrational_entropy_emt"].to_numpy(dtype=float)
        assert (low > 0).all() and (high > low).all()

    def test_free_energy_falls_with_temperature(self, vibrating):
        mv.prop.free_energy(vibrating, level="emt", temperature=100.0)
        low = vibrating.obs["vibrational_free_energy_emt"].to_numpy(dtype=float).copy()
        mv.prop.free_energy(vibrating, level="emt", temperature=800.0)
        high = vibrating.obs["vibrational_free_energy_emt"].to_numpy(dtype=float)
        assert (high < low).all()

    def test_it_needs_a_phonon_dos(self, md):
        with pytest.raises(ValueError, match="mv.prop.phonon"):
            mv.prop.free_energy(md, level="emt")

    def test_the_temperature_is_recorded(self, vibrating):
        mv.prop.free_energy(vibrating, level="emt", temperature=456.0)
        assert vibrating.uns["thermal"]["emt"]["temperature"] == 456.0


def _has_defects_addon() -> bool:
    try:
        import pymatgen.analysis.defects.corrections.freysoldt  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_defects_addon(),
                    reason="pymatgen-analysis-defects is an optional extra")
class TestImageChargeCorrection:
    """The electrostatic half of the Freysoldt correction.

    It needs only the cell, the charge and the dielectric constant — no LOCPOT
    — so its three scalings are exact and can be asserted rather than eyeballed:
    q squared, one over epsilon, one over the cell length. The half that does
    need a LOCPOT is the potential alignment; see TestPotentialAlignment.
    """

    @staticmethod
    def _system(repeat):
        from pymatgen.core import Lattice, Structure
        fcc = Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        fcc.make_supercell(list(repeat))
        host = mv.data.from_structures([fcc])
        mv.calc.energy(host, level="emt")
        defects = mv.pp.defects(host, kinds=("vacancy",))
        mv.calc.energy(defects, level="emt")
        return defects, host

    @staticmethod
    def _curve(defects, host, dielectric, gap=2.0):
        out = defects.copy()
        mv.thermo.defect_formation(out, host=host, level="emt",
                                   chempot={"Cu": -3.5}, band_gap=gap,
                                   dielectric=dielectric)
        return out, out.obsm["formation_vs_fermi_emt"][0]

    def test_without_a_dielectric_nothing_changes(self):
        """The default has to stay exactly what it was, or every existing
        number silently moves."""
        defects, host = self._system((2, 2, 2))
        plain, before = self._curve(defects, host, None)
        assert plain.uns["defect_thermodynamics"]["image_charge_correction"] \
            is False
        assert plain.uns["defect_thermodynamics"]["dielectric"] is None
        assert np.isfinite(before).all()

    def test_the_correction_halves_when_epsilon_doubles(self):
        """1/epsilon, asserted at the top of the gap where the most charged
        state is the stable one and the envelope is that state's line."""
        defects, host = self._system((2, 2, 2))
        _, plain = self._curve(defects, host, None)
        _, ten = self._curve(defects, host, 10.0)
        _, twenty = self._curve(defects, host, 20.0)
        # rel=1e-4, not tighter: perform_es_corr converges the Madelung sum
        # numerically with mad_tol=1e-4, so the scaling is exact in the
        # algebra and carries about a part in a million in the arithmetic.
        assert (ten[-1] - plain[-1]) == \
            pytest.approx(2.0 * (twenty[-1] - plain[-1]), rel=1e-4)

    def test_the_correction_is_positive(self):
        """It removes a spurious stabilisation, so it can only raise the
        formation energy."""
        defects, host = self._system((2, 2, 2))
        _, plain = self._curve(defects, host, None)
        _, corrected = self._curve(defects, host, 10.0)
        assert (corrected >= plain - 1e-9).all()
        assert corrected[-1] > plain[-1]

    def test_a_bigger_cell_needs_less_correcting(self):
        """1/L. The image interaction is the reason small supercells lie, and
        the correction has to shrink as the cell grows or it is not that."""
        small, host_s = self._system((2, 2, 2))
        big, host_b = self._system((3, 3, 3))
        _, s_plain = self._curve(small, host_s, None)
        _, s_corr = self._curve(small, host_s, 10.0)
        _, b_plain = self._curve(big, host_b, None)
        _, b_corr = self._curve(big, host_b, 10.0)
        shift_small = s_corr[-1] - s_plain[-1]
        shift_big = b_corr[-1] - b_plain[-1]
        assert shift_big < shift_small
        # cells are 2a and 3a, so the ratio of the shifts is 3/2
        assert shift_small / shift_big == pytest.approx(1.5, rel=0.02)

    def test_the_neutral_state_is_never_corrected(self):
        """q = 0 has no image charge, so at the valence band maximum — where
        every charge state costs the same and the neutral one therefore wins —
        the corrected and uncorrected envelopes must coincide."""
        defects, host = self._system((2, 2, 2))
        _, plain = self._curve(defects, host, None)
        _, corrected = self._curve(defects, host, 10.0)
        assert corrected[0] == pytest.approx(plain[0], abs=1e-9)

    def test_it_says_which_half_it_did(self):
        """With dielectric= alone only the electrostatic half is applied, and
        the run must name the half it left out rather than leaving the reader
        to discover it."""
        defects, host = self._system((2, 2, 2))
        out, _ = self._curve(defects, host, 10.0)
        record = out.uns["defect_thermodynamics"]
        assert record["image_charge_correction"] is True
        assert record["potential_alignment"] is False
        assert record["dielectric"] == 10.0
        assert "image-charge) only" in record["correction_terms"]
        assert "locpots=" in record["correction_terms"]
        assert record["correction_error"] is None


@pytest.mark.skipif(not _has_defects_addon(),
                    reason="pymatgen-analysis-defects is an optional extra")
class TestCaptureCoefficient:
    """Checked against exact scaling laws rather than a reference number.

    The Shockley-Read-Hall coefficient is quadratic in the electron-phonon
    matrix element and thermally activated. Both are properties of the
    expression, not of any particular defect, so they hold whatever the inputs.
    """

    BASE = dict(dQ=1.0, dE=1.0, omega_i=0.02, omega_f=0.02)

    @staticmethod
    def _cell(n=1):
        from pymatgen.core import Lattice, Structure
        st = Structure(Lattice.cubic(10.0), ["Ga", "N"],
                       [[0, 0, 0], [.5, .5, .5]])
        return mv.data.from_structures([st] * n)

    def test_it_is_quadratic_in_the_coupling(self):
        md = self._cell(2)
        mv.prop.capture(md, coupling=[1e-3, 2e-3], **self.BASE)
        c = md.obs["capture_coefficient_srh"].to_numpy()
        assert c[1] / c[0] == pytest.approx(4.0, rel=1e-6)

    def test_no_coupling_means_no_capture(self):
        md = self._cell()
        mv.prop.capture(md, coupling=0.0, **self.BASE)
        assert float(md.obs["capture_coefficient_srh"].iloc[0]) == \
            pytest.approx(0.0, abs=1e-30)

    def test_it_is_thermally_activated(self):
        md = self._cell()
        mv.prop.capture(md, coupling=1e-3, temperature=300.0, key_added="cold",
                        **self.BASE)
        mv.prop.capture(md, coupling=1e-3, temperature=600.0, key_added="hot",
                        **self.BASE)
        assert float(md.obs["capture_coefficient_hot"].iloc[0]) > \
            float(md.obs["capture_coefficient_cold"].iloc[0])

    def test_the_temperature_is_recorded_with_the_number(self):
        """A capture coefficient without its temperature is not a quantity."""
        md = self._cell()
        mv.prop.capture(md, coupling=1e-3, temperature=450.0, **self.BASE)
        assert md.uns["capture"]["srh"]["temperature"] == 450.0
        assert md.uns["capture"]["srh"]["unit"] == "cm^3/s"

    def test_one_value_or_one_per_row(self):
        md = self._cell(3)
        mv.prop.capture(md, coupling=1e-3, **self.BASE)
        assert np.isfinite(md.obs["capture_coefficient_srh"]).all()
        with pytest.raises(ValueError, match="one or one per row"):
            mv.prop.capture(md, coupling=[1e-3, 2e-3], **self.BASE)

    def test_the_radiative_channel_needs_a_photon_energy(self):
        md = self._cell()
        with pytest.raises(ValueError, match="omega_photon"):
            mv.prop.capture(md, coupling=1e-3, kind="radiative", **self.BASE)

    def test_the_radiative_channel_computes(self):
        md = self._cell()
        mv.prop.capture(md, coupling=1e-3, kind="radiative",
                        omega_photon=0.8, **self.BASE)
        assert np.isfinite(
            float(md.obs["capture_coefficient_radiative"].iloc[0]))

    def test_an_impossible_photon_is_refused_not_returned_as_nan(self):
        """dE is 1.0 eV here, so a 1.5 eV photon carries away more than the
        transition provides. get_Rad_coef returns NaN for that without
        complaint, which would arrive as a silently blank column."""
        md = self._cell()
        with pytest.raises(ValueError, match="exceeds dE"):
            mv.prop.capture(md, coupling=1e-3, kind="radiative",
                            omega_photon=1.5, **self.BASE)

    def test_an_unknown_channel_is_refused(self):
        md = self._cell()
        with pytest.raises(ValueError, match="'srh' or 'radiative'"):
            mv.prop.capture(md, coupling=1e-3, kind="auger", **self.BASE)


class TestConfigurationCoordinate:
    """A harmonic curve has a frequency that can be written down, so the fit
    is checked against the closed form and, separately, against pymatgen's own
    HarmonicDefect.omega_eV — two routes to the same number."""

    #: hbar * sqrt(1 eV / (amu angstrom^2)), in eV.
    K = 0.0646541380

    @staticmethod
    def _curve(curvature, n=21, span=2.0, shift=0.0):
        from pymatgen.core import Lattice, Structure
        st = Structure(Lattice.cubic(4.0), ["Ga", "N"],
                       [[0, 0, 0], [.5, .5, .5]])
        Q = np.linspace(-span, span, n)
        md = mv.data.from_structures([st] * n)
        md.obs["Q"] = Q
        md.obs["energy_pbe"] = 0.5 * curvature * (Q - shift) ** 2
        return md

    def test_the_frequency_is_the_closed_form(self):
        md = self._curve(0.6)
        mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")
        assert float(md.obs["cc_frequency_pbe"].iloc[0]) == \
            pytest.approx(self.K * np.sqrt(0.6), rel=1e-9)

    def test_it_agrees_with_pymatgens_own_conversion(self):
        """HarmonicDefect.omega_eV applies the same constant to the same
        curvature. Two independent routes, one answer."""
        pytest.importorskip("pymatgen.analysis.defects.ccd")
        from pymatgen.analysis.defects.ccd import HarmonicDefect, _get_omega
        for curvature in (0.01, 0.25, 0.6):
            md = self._curve(curvature)
            mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")
            Q = md.obs["Q"].to_numpy()
            E = md.obs["energy_pbe"].to_numpy()
            reference = HarmonicDefect(omega=_get_omega(Q, E, 0.0, 0.0),
                                       charge_state=0, ispin=1)
            assert float(md.obs["cc_frequency_pbe"].iloc[0]) == \
                pytest.approx(reference.omega_eV, abs=1e-7)

    def test_the_frequency_goes_as_the_square_root_of_the_curvature(self):
        soft = self._curve(0.15)
        stiff = self._curve(0.60)
        mv.prop.configuration_coordinate(soft, coordinate="Q", level="pbe")
        mv.prop.configuration_coordinate(stiff, coordinate="Q", level="pbe")
        assert float(stiff.obs["cc_frequency_pbe"].iloc[0]) == \
            pytest.approx(2.0 * float(soft.obs["cc_frequency_pbe"].iloc[0]),
                          rel=1e-9)

    def test_a_centred_curve_has_no_relaxation_energy(self):
        md = self._curve(0.6, shift=0.0)
        mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")
        assert float(md.obs["cc_relaxation_pbe"].iloc[0]) == \
            pytest.approx(0.0, abs=1e-9)

    def test_an_offset_minimum_gives_the_franck_condon_shift(self):
        """The relaxation energy is the curve's height at Q = 0 above its own
        minimum, which for a parabola offset by dQ is exactly (1/2) c dQ^2."""
        curvature, offset = 0.6, 1.5
        md = self._curve(curvature, shift=offset)
        mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")
        assert float(md.obs["cc_relaxation_pbe"].iloc[0]) == \
            pytest.approx(0.5 * curvature * offset ** 2, rel=1e-6)

    def test_huang_rhys_is_the_relaxation_in_phonons(self):
        md = self._curve(0.6, shift=1.5)
        mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")
        relaxation = float(md.obs["cc_relaxation_pbe"].iloc[0])
        frequency = float(md.obs["cc_frequency_pbe"].iloc[0])
        assert float(md.obs["cc_huang_rhys_pbe"].iloc[0]) == \
            pytest.approx(relaxation / frequency, rel=1e-9)

    def test_an_inverted_curve_is_refused(self):
        """Negative curvature is a saddle, and a saddle has no harmonic
        frequency — returning an imaginary one as NaN would hide that the
        geometry was wrong."""
        md = self._curve(-0.6)
        with pytest.raises(ValueError, match="not a minimum"):
            mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")

    def test_three_points_are_needed(self):
        md = self._curve(0.6, n=2)
        with pytest.raises(ValueError, match="needs three points"):
            mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")

    def test_a_missing_coordinate_says_what_it_wants(self):
        md = self._curve(0.6)
        with pytest.raises(ValueError, match="mass-weighted"):
            mv.prop.configuration_coordinate(md, coordinate="nope",
                                             level="pbe")

    def test_the_frequency_can_feed_a_capture_coefficient(self):
        """The reason this exists: mv.prop.capture needs omega and nothing
        produced one."""
        pytest.importorskip("pymatgen.analysis.defects.recombination")
        md = self._curve(0.6, shift=1.0)
        mv.prop.configuration_coordinate(md, coordinate="Q", level="pbe")
        omega = float(md.obs["cc_frequency_pbe"].iloc[0])
        mv.prop.capture(md, dQ=1.0, dE=1.0, omega_i=omega, omega_f=omega,
                        coupling=1e-3)
        assert np.isfinite(
            float(md.obs["capture_coefficient_srh"].iloc[0]))


_has_phonopy = importlib.util.find_spec("phonopy") is not None
_has_seekpath = importlib.util.find_spec("seekpath") is not None


@pytest.fixture(scope="module")
def fcc_copper():
    from pymatgen.core import Lattice, Structure
    a = 3.61
    return Structure(Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2],
                              [a / 2, a / 2, 0]]), ["Cu"], [[0, 0, 0]])


@pytest.mark.skipif(not _has_phonopy, reason="needs phonopy")
class TestPhonopyMethod:
    """The interpolated route, checked against things that are true of any
    correct implementation rather than against my own numbers."""

    @pytest.fixture(scope="class")
    def both(self):
        from pymatgen.core import Lattice, Structure
        a = 3.61
        fcc = Structure(Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2],
                                 [a / 2, a / 2, 0]]), ["Cu"], [[0, 0, 0]])
        out = {}
        for method in ("commensurate", "phonopy"):
            md = mv.data.from_structures([fcc])
            mv.prop.phonon(md, level="emt", supercell=(3, 3, 3),
                           method=method)
            mv.prop.free_energy(md, level="emt", temperature=300.0)
            out[method] = md
        return out

    def test_a_missing_phonopy_raises_rather_than_returning_nan(
            self, fcc_copper, monkeypatch):
        """The per-structure loop turns any exception into a NaN row and a
        failure count, which is right for a structure that would not converge
        and wrong for an absent dependency — it produced a full column of NaN
        with no mention of phonopy. Caught by running the suite with phonopy
        hidden, which is the only way this was ever going to show up."""
        import builtins
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] == "phonopy":
                raise ImportError("No module named 'phonopy'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        md = mv.data.from_structures([fcc_copper])
        with pytest.raises(ImportError, match="matverse\\[phonons\\]"):
            mv.prop.phonon(md, level="emt", method="phonopy")

    def test_an_unknown_method_is_refused(self, fcc_copper):
        md = mv.data.from_structures([fcc_copper])
        with pytest.raises(ValueError, match="commensurate"):
            mv.prop.phonon(md, level="emt", method="phono3py")

    def test_both_methods_agree_that_fcc_copper_is_stable(self, both):
        for md in both.values():
            assert bool(md.obs["dynamically_stable_emt"].iloc[0])
            assert int(md.obs["n_imaginary_modes_emt"].iloc[0]) == 0

    def test_the_two_methods_land_within_a_few_meV(self, both):
        """Different sampling of the same force constants, so they may not
        agree exactly — but a disagreement beyond ~10 meV/atom would mean one
        of them is wrong, not merely coarser."""
        f = [float(md.obs["vibrational_free_energy_emt"].iloc[0])
             for md in both.values()]
        assert abs(f[0] - f[1]) < 0.010

    def test_the_interpolated_free_energy_is_the_lower_one(self, both):
        """The commensurate route misses the low-frequency weight that a
        converged mesh picks up near Gamma, and missing low-frequency weight
        raises the free energy. This is the whole reason the method exists; if
        it ever inverts, the q-point weights have been dropped again."""
        assert (float(both["phonopy"].obs["vibrational_free_energy_emt"].iloc[0])
                < float(both["commensurate"].obs[
                    "vibrational_free_energy_emt"].iloc[0]))

    def test_the_heat_capacity_reaches_the_dulong_petit_limit(self, fcc_copper):
        """3 k_B per atom well above the Debye temperature — the check that
        does not depend on the calculator being any good."""
        md = mv.data.from_structures([fcc_copper])
        mv.prop.phonon(md, level="emt", supercell=(3, 3, 3), method="phonopy")
        mv.prop.free_energy(md, level="emt", temperature=2000.0)
        c = float(md.obs["heat_capacity_emt"].iloc[0])
        assert c == pytest.approx(3 * 8.617333262e-5, rel=0.02)

    def test_the_zero_point_energy_does_not_depend_on_the_cell(self):
        """Per-atom means per-atom: the same crystal described by its
        primitive or its conventional cell must give the same number. This is
        where a wrong divisor hides, because a single cell choice looks
        plausible on its own."""
        from pymatgen.core import Lattice, Structure
        a = 3.61
        primitive = Structure(Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2],
                                       [a / 2, a / 2, 0]]), ["Cu"], [[0, 0, 0]])
        conventional = Structure(
            Lattice.cubic(a), ["Cu"] * 4,
            [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        zpe = []
        for cell in (primitive, conventional):
            md = mv.data.from_structures([cell])
            mv.prop.phonon(md, level="emt", supercell=(3, 3, 3),
                           method="phonopy", mesh=(21, 21, 21))
            zpe.append(float(md.obs["zero_point_energy_emt"].iloc[0]))
        assert zpe[0] == pytest.approx(zpe[1], abs=1e-4)
        # And it is copper's, roughly 30 meV/atom, not an order out.
        assert 0.02 < zpe[0] < 0.05

    def test_the_method_is_recorded_with_the_grid(self, both):
        grids = both["phonopy"].uns["grids"]["phonon_dos"]
        assert grids["method"] == "phonopy"
        assert grids["mesh"] == [13, 13, 13]
        assert both["commensurate"].uns["grids"]["phonon_dos"]["mesh"] is None

    def test_the_mesh_is_actually_sampled(self, fcc_copper):
        """A denser mesh must move the answer, or the mesh argument is being
        ignored — which is exactly the bug a smoke test would not see."""
        free = []
        for mesh in ((3, 3, 3), (21, 21, 21)):
            md = mv.data.from_structures([fcc_copper])
            mv.prop.phonon(md, level="emt", supercell=(2, 2, 2),
                           method="phonopy", mesh=mesh)
            mv.prop.free_energy(md, level="emt", temperature=300.0)
            free.append(float(md.obs["vibrational_free_energy_emt"].iloc[0]))
        assert abs(free[0] - free[1]) > 1e-4


@pytest.mark.skipif(not (_has_phonopy and _has_seekpath),
                    reason="needs phonopy and seekpath")
class TestPhononDispersion:
    @pytest.fixture(scope="class")
    def dispersed(self):
        from pymatgen.core import Lattice, Structure
        a = 3.61
        fcc = Structure(Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2],
                                 [a / 2, a / 2, 0]]), ["Cu"], [[0, 0, 0]])
        bcc = Structure(Lattice.cubic(2.9), ["Cu", "Cu"],
                        [[0, 0, 0], [0.5, 0.5, 0.5]])
        md = mv.data.from_structures([fcc, bcc])
        md.obs_names = ["fcc", "bcc"]
        return mv.prop.dispersion(md, level="emt", supercell=(3, 3, 3))

    def test_it_returns_a_bands_axis(self, dispersed):
        from matverse._core import AXIS_KEY
        assert dispersed.uns[AXIS_KEY] == "bands"
        assert dispersed.uns["unit"] == "THz"
        assert {"material", "branch_index", "is_acoustic"} <= set(
            dispersed.obs.columns)

    def test_three_acoustic_branches_per_material(self, dispersed):
        counts = dispersed.obs.groupby("material", observed=True)[
            "is_acoustic"].sum()
        assert (counts == 3).all()

    def test_the_acoustic_branches_vanish_at_gamma(self, dispersed):
        """The path from seekpath starts at Gamma, and the three acoustic
        branches are zero there by the translational sum rule. Any correct
        implementation satisfies this; a wrong mass or a wrong cell does not."""
        X = np.asarray(dispersed.X)
        acoustic = X[np.asarray(dispersed.obs["is_acoustic"])]
        assert np.abs(acoustic[:, 0]).max() < 0.05

    def test_bcc_copper_is_unstable_and_fcc_is_not(self, dispersed):
        """bcc Cu is dynamically unstable — this is the instability a
        Gamma-only or coarsely commensurate calculation can miss entirely."""
        by_material = dispersed.obs.groupby("material", observed=True)[
            "is_imaginary"].any()
        assert bool(by_material["bcc"])
        assert not bool(by_material["fcc"])

    def test_the_frequencies_are_physical_for_copper(self, dispersed):
        """Copper's phonons top out near 7-8 THz. An order of magnitude out
        means the units are wrong, which is the classic silent failure."""
        assert 3.0 < float(np.asarray(dispersed.X).max()) < 15.0

    def test_it_records_what_produced_it(self, dispersed):
        assert dispersed.uns["dispersion"]["supercell"] == [3, 3, 3]
        assert dispersed.uns["dispersion"]["n_failed"] == 0
        assert "emt" in dispersed.uns["levels"]

    def test_the_path_ticks_start_and_end_the_path(self, dispersed):
        for name in ("fcc", "bcc"):
            ticks = dispersed.uns["path_labels"][name]
            assert min(ticks) == 0.0 and max(ticks) == 1.0
            assert "Gamma" in ticks[0.0]        # every path starts at Gamma

    def test_a_discontinuous_path_joins_both_endpoints(self, dispersed):
        """Where the path jumps, the line stops at one point and resumes at
        another, and both belong at that abscissa. Copper's conventional paths
        contain such a jump in both phases, so a tick reading only one of them
        would be mislabelling the plot."""
        for name in ("fcc", "bcc"):
            assert any("|" in text
                       for text in dispersed.uns["path_labels"][name].values())

    def test_the_ticks_land_where_the_acoustic_branches_return_to_zero(self):
        """Gamma appears twice on this path and the acoustic branches are zero
        at both, which ties the tick positions to the physics rather than to
        my arithmetic on segment lengths.

        Sampled at the native path length: phonopy takes 51 q-points per
        segment, and resampling onto fewer than that interpolates across the
        sharp V at an interior Gamma so it no longer reaches zero. That is a
        documented artefact of sharing one abscissa between materials, not a
        wrong tick, and 306 = 6 segments x 51 avoids it."""
        from pymatgen.core import Lattice, Structure
        a = 3.61
        md = mv.data.from_structures([Structure(
            Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
            ["Cu"], [[0, 0, 0]])])
        md.obs_names = ["fcc"]
        ph = mv.prop.dispersion(md, level="emt", supercell=(3, 3, 3),
                                n_points=306)
        fraction = ph.var["path_fraction"].to_numpy(dtype=float)
        acoustic = np.asarray(ph.X)[np.asarray(ph.obs["is_acoustic"])]
        gammas = [f for f, text in ph.uns["path_labels"]["fcc"].items()
                  if "Gamma" in text]
        assert len(gammas) >= 2
        for position in gammas:
            column = int(np.argmin(np.abs(fraction - position)))
            assert np.abs(acoustic[:, column]).max() < 0.01

    def test_the_resampling_artefact_is_the_documented_size(self, dispersed):
        """The notes claim ~0.08 THz at the default n_points and exactness at
        the native length. A claim with a number in it should fail if the
        number changes."""
        fraction = dispersed.var["path_fraction"].to_numpy(dtype=float)
        rows = np.asarray(dispersed.obs["material"]).astype(str) == "fcc"
        acoustic = np.asarray(dispersed.X)[
            rows & np.asarray(dispersed.obs["is_acoustic"])]
        interior = [f for f, text in dispersed.uns["path_labels"]["fcc"].items()
                    if "Gamma" in text and f > 0.0]
        worst = max(
            float(np.abs(acoustic[:, int(np.argmin(np.abs(fraction - g)))]
                         ).max()) for g in interior)
        assert worst < 0.2, "the artefact grew beyond what the notes describe"
        # And the endpoint of the path is exact regardless of n_points.
        assert np.abs(acoustic[:, 0]).max() < 0.01

    def test_mv_pl_bands_plots_it_unchanged(self, dispersed):
        """The claim in the registry notes, checked rather than asserted — it
        was false when first written, because the object was missing the
        var['path_fraction'] column mv.pl.bands reads."""
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        ax = mv.pl.bands(dispersed, materials=["fcc", "bcc"])
        assert ax._matverse_n_bands == dispersed.n_obs
        # And the ordinate says frequency, not electron energy.
        assert "THz" in ax.get_ylabel()

    def test_a_band_gap_is_refused_on_a_phonon_spectrum(self, dispersed):
        """Same axis, different physics. Without the guard this returns a
        plausible number in THz labelled as an electronic gap."""
        md = mv.datasets.metals(["Cu"])
        with pytest.raises(ValueError, match="phonon dispersion"):
            mv.elec.band_features(dispersed, md, level="emt")

    def test_failures_are_reported_not_silently_dropped(self, fcc_copper):
        from pymatgen.core import Lattice, Structure
        md = mv.data.from_structures([fcc_copper])
        md.obs_names = ["only"]
        with pytest.raises(ValueError, match="no dispersion"):
            # Helium has no EMT parameters, so every force call raises.
            bad = mv.data.from_structures(
                [Structure(Lattice.cubic(3.0), ["He"], [[0, 0, 0]])])
            mv.prop.dispersion(bad, level="emt")


@pytest.mark.skipif(not _has_defects_addon(),
                    reason="pymatgen-analysis-defects is an optional extra")
class TestPotentialAlignment:
    """The other half of Freysoldt, the one that needs a LOCPOT.

    There is no VASP output to test against, but the term does not need one to
    be checked: it enters the formation energy as q * dV, so a rigid shift of
    either potential must move every charge state by exactly q times that
    shift, and must not move the neutral one at all. That is a statement about
    the physics rather than about this implementation, and it is what a wrong
    sign, a double-counted electrostatic term or a dropped charge factor would
    all break.
    """

    LENGTH = 10.0
    GRID = 32

    @classmethod
    def _system(cls):
        from pymatgen.core import Lattice, Structure
        bulk = Structure(Lattice.cubic(cls.LENGTH), ["Si"] * 4,
                         [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        vacancy = Structure(Lattice.cubic(cls.LENGTH), ["Si"] * 3,
                            [[0, 0, 0], [.5, .5, 0], [.5, 0, .5]])
        host = mv.data.from_structures([bulk])
        host.obs_names = ["bulk"]
        host.obs["energy_emt"] = [-20.0]
        defects = mv.data.from_structures([vacancy])
        defects.obs_names = ["vac"]
        defects.obs["parent"] = "bulk"
        defects.obs["defect"] = "vacancy"
        defects.obs["removed"] = "Si"
        defects.obs["added"] = ""
        defects.obs["energy_emt"] = [-15.0]
        return defects, host, bulk, vacancy

    @classmethod
    def _potentials(cls, bulk, vacancy, shift):
        """Built through Poscar rather than from a bare Structure.

        pymatgen 2026 widened Locpot to accept either; 2025 and earlier take a
        Poscar only, and passing a Structure fails with a confusing
        'Structure has no attribute structure'. Poscar works on both, and the
        3.10 leg of CI runs the older one.
        """
        from pymatgen.io.vasp.inputs import Poscar
        from pymatgen.io.vasp.outputs import Locpot
        flat = np.zeros((cls.GRID,) * 3)
        return {"vac": Locpot(Poscar(vacancy), {"total": flat + shift}),
                "bulk": Locpot(Poscar(bulk), {"total": flat})}

    @classmethod
    def _energy(cls, charge, shift):
        defects, host, bulk, vacancy = cls._system()
        mv.thermo.defect_formation(
            defects, host=host, level="emt", chempot={"Si": -5.0},
            band_gap=1.1, dielectric=10.0, charges=(charge,),
            locpots=cls._potentials(bulk, vacancy, shift))
        return defects, float(defects.obsm["formation_vs_fermi_emt"][0][0])

    @pytest.mark.parametrize("charge", [-2, -1, 1, 2])
    def test_a_rigid_shift_moves_the_energy_by_exactly_q_times_it(self, charge):
        before = self._energy(charge, 0.0)[1]
        after = self._energy(charge, 0.5)[1]
        assert after - before == pytest.approx(charge * 0.5, abs=1e-4)

    def test_the_neutral_state_is_untouched(self):
        """q * dV is zero at q = 0. If the alignment were applied as a bare
        potential difference rather than a charge times one, this is the test
        that would catch it."""
        assert self._energy(0, 0.0)[1] == pytest.approx(
            self._energy(0, 0.75)[1], abs=1e-9)

    def test_the_electrostatic_term_is_not_counted_twice(self):
        """get_freysoldt_correction returns *both* terms, and the
        electrostatic one is already applied from dielectric= alone. The shift
        tests above cannot see a double count, because a constant offset in
        the correction cancels out of a difference — so this one pins the
        magnitude against pymatgen computed independently."""
        from pymatgen.analysis.defects.corrections.freysoldt import (
            get_freysoldt_correction, perform_es_corr)
        from pymatgen.analysis.defects.utils import QModel

        charge, shift = 1, 0.3
        defects, host, bulk, vacancy = self._system()
        potentials = self._potentials(bulk, vacancy, shift)
        expected = (
            float(get_freysoldt_correction(
                q=charge, dielectric=10.0,
                defect_locpot=potentials["vac"],
                bulk_locpot=potentials["bulk"],
                defect_frac_coords=[0.0, 0.5, 0.5]).correction_energy)
            - float(perform_es_corr(vacancy.lattice, q=charge,
                                    dielectric=10.0, q_model=QModel())))

        with_locpots = self._energy(charge, shift)[1]
        plain, _, _, _ = self._system()
        mv.thermo.defect_formation(plain, host=host, level="emt",
                                   chempot={"Si": -5.0}, band_gap=1.1,
                                   dielectric=10.0, charges=(charge,))
        without = float(plain.obsm["formation_vs_fermi_emt"][0][0])

        assert with_locpots - without == pytest.approx(expected, abs=1e-6)

    def test_it_says_which_terms_it_applied(self):
        aligned = self._energy(1, 0.0)[0].uns["defect_thermodynamics"]
        assert aligned["potential_alignment"] is True
        assert "potential alignment" in aligned["correction_terms"]
        assert aligned["correction_error"] is None

        defects, host, _, _ = self._system()
        mv.thermo.defect_formation(defects, host=host, level="emt",
                                   chempot={"Si": -5.0}, band_gap=1.1,
                                   dielectric=10.0)
        plain = defects.uns["defect_thermodynamics"]
        assert plain["potential_alignment"] is False
        assert "image-charge) only" in plain["correction_terms"]

    def test_alignment_without_a_dielectric_is_refused(self):
        """Applying the alignment alone would be worse than applying neither,
        so it is an error rather than a half-correction."""
        defects, host, bulk, vacancy = self._system()
        with pytest.raises(ValueError, match="half of Freysoldt"):
            mv.thermo.defect_formation(
                defects, host=host, level="emt", chempot={"Si": -5.0},
                locpots=self._potentials(bulk, vacancy, 0.0))

    def test_a_missing_locpot_is_reported_not_silently_skipped(self):
        defects, host, bulk, vacancy = self._system()
        potentials = self._potentials(bulk, vacancy, 0.0)
        del potentials["bulk"]
        mv.thermo.defect_formation(
            defects, host=host, level="emt", chempot={"Si": -5.0},
            band_gap=1.1, dielectric=10.0, locpots=potentials)
        recorded = defects.uns["defect_thermodynamics"]
        assert recorded["potential_alignment"] is False
        assert "no LOCPOT" in recorded["correction_error"]


class TestDefectSiteFromTwoCells:
    """Locating the defect without dscribe, which cannot import on numpy 2.5."""

    @staticmethod
    def _cells():
        from pymatgen.core import Lattice, Structure
        lattice = Lattice.cubic(8.0)
        bulk = Structure(lattice, ["Cu"] * 4,
                         [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        return lattice, bulk

    def test_a_vacancy_is_the_site_that_went_missing(self):
        from matverse.thermo import _locate
        from pymatgen.core import Structure
        lattice, bulk = self._cells()
        vacancy = Structure(lattice, ["Cu"] * 3,
                            [[0, 0, 0], [.5, .5, 0], [.5, 0, .5]])
        assert _locate(vacancy, bulk) == pytest.approx([0.0, 0.5, 0.5])

    def test_an_interstitial_is_the_site_that_appeared(self):
        from matverse.thermo import _locate
        from pymatgen.core import Structure
        lattice, bulk = self._cells()
        interstitial = Structure(
            lattice, ["Cu"] * 5,
            [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
             [.25, .25, .25]])
        assert _locate(interstitial, bulk) == pytest.approx([.25, .25, .25])

    def test_a_substitution_is_the_site_whose_species_changed(self):
        from matverse.thermo import _locate
        from pymatgen.core import Structure
        lattice, bulk = self._cells()
        substituted = Structure(lattice, ["Cu", "Cu", "Al", "Cu"],
                                [[0, 0, 0], [.5, .5, 0], [.5, 0, .5],
                                 [0, .5, .5]])
        assert _locate(substituted, bulk) == pytest.approx([0.5, 0.0, 0.5])

    def test_no_defect_and_too_many_defects_both_give_nothing(self):
        """Guessing would be worse than declining. A cell that differs by two
        sites is not a point defect, and one that differs by none has no
        defect to locate."""
        from matverse.thermo import _locate
        from pymatgen.core import Structure
        lattice, bulk = self._cells()
        assert _locate(bulk, bulk) is None
        assert _locate(Structure(lattice, ["Cu"] * 2,
                                 [[0, 0, 0], [.5, .5, 0]]), bulk) is None


_has_hiphive = importlib.util.find_spec("hiphive") is not None


@pytest.mark.skipif(not _has_hiphive, reason="needs hiphive")
class TestSelfConsistentPhonons:
    """Phonons at temperature, which is a different question from phonons.

    obs['dynamically_stable_{level}'] is a statement about 0 K, and a great
    many real materials fail it and exist anyway. These tests use the three
    copper lattices because they give the three different answers this can
    return, and because which one each gives is known independently.
    """

    @staticmethod
    def _lattice(kind, a):
        from ase.build import bulk
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor().get_structure(bulk("Cu", kind, a=a,
                                                    cubic=False))

    @pytest.fixture(scope="class")
    def sampled(self):
        md = mv.data.from_structures([self._lattice("bcc", 2.9),
                                      self._lattice("fcc", 3.61),
                                      self._lattice("sc", 2.4)])
        md.obs_names = ["bcc", "fcc", "sc"]
        # Deliberately past the defaults. This is a stochastic fixed-point
        # iteration and the tests below assert a specific physical outcome, so
        # they have to be run where it has actually converged - at 8
        # iterations CI produced 22 imaginary modes at 300 K where there
        # should be none, on a solve that the drift diagnostic flagged.
        # 5 K is safely below the transition and 300 K safely above it. 50 K
        # is not: bcc copper stabilises between 25 K and 100 K here, and
        # exactly where moves with the convergence - 8 iterations put it at
        # 100 K and 30 put it at 50 K. Asserting a point that sits on the
        # boundary is asserting the resolution limit, not the physics.
        mv.prop.phonon_at_temperature(
            md, level="emt", temperatures=(5., 25., 300., 600.),
            supercell=(5, 5, 5), cutoff=5.0, n_structures=30,
            n_iterations=30)
        return md

    @staticmethod
    def _drift(md):
        return np.asarray(md.uns["self_consistent_phonons"]["emt"][
            "convergence_drift"], dtype=float)

    def test_the_fixture_actually_converged(self, sampled):
        """Run first so that a convergence failure is named as one rather than
        showing up as the physics being wrong."""
        assert (self._drift(sampled) < 0.05).all()

    def test_bcc_copper_is_unstable_cold_and_stable_warm(self, sampled):
        """The whole point. bcc metals are unstable in the harmonic
        approximation and stabilised by anharmonicity at temperature; a screen
        that discarded them on the 0 K answer would discard the answer."""
        assert (self._drift(sampled)[0] < 0.05).all(), "did not converge"
        counts = sampled.obsm["imaginary_modes_vs_temperature_emt"][0]
        assert (counts[:2] > 0).all()          # 5 K and 25 K: unstable
        assert (counts[2:] == 0).all()         # 300 K and 600 K: stable
        assert float(
            sampled.obs["stabilisation_temperature_emt"].iloc[0]) == 300.0

    def test_fcc_copper_never_needed_stabilising(self, sampled):
        """NaN here means 'already stable at the lowest temperature scanned',
        which is a different thing from 'never stabilised' and must not be
        read as the same."""
        assert (sampled.obsm["imaginary_modes_vs_temperature_emt"][1] == 0).all()
        assert np.isnan(
            float(sampled.obs["stabilisation_temperature_emt"].iloc[1]))
        assert "already stable" in \
            sampled.uns["self_consistent_phonons"]["emt"]["verdict"][1]

    def test_simple_cubic_copper_is_stabilised_by_nothing(self, sampled):
        """The other NaN. Same column, opposite meaning."""
        assert (sampled.obsm["imaginary_modes_vs_temperature_emt"][2] > 0).all()
        assert np.isnan(
            float(sampled.obs["stabilisation_temperature_emt"].iloc[2]))
        assert "unstable at every" in \
            sampled.uns["self_consistent_phonons"]["emt"]["verdict"][2]

    def test_the_two_nans_are_told_apart(self, sampled):
        """A NaN that means two opposite things is worse than no column."""
        verdicts = sampled.uns["self_consistent_phonons"]["emt"]["verdict"]
        assert verdicts[1] != verdicts[2]

    def test_the_answer_does_not_depend_on_the_iteration_count(self):
        """A fixed-point iteration run for a fixed number of steps does not
        report its own convergence. If this ever fails, the stabilisation
        temperature is wherever the walk stopped."""
        results = []
        for iterations in (20, 40):
            md = mv.data.from_structures([self._lattice("bcc", 2.9)])
            mv.prop.phonon_at_temperature(
                md, level="emt", temperatures=(5., 300., 600.),
                supercell=(5, 5, 5), cutoff=5.0, n_structures=30,
                n_iterations=iterations)
            results.append(
                md.obsm["imaginary_modes_vs_temperature_emt"][0] > 0)
        assert (results[0] == results[1]).all()

    def test_a_cutoff_that_wraps_around_the_cell_is_refused(self):
        """A force constant must not reach an atom and its own periodic image
        at once. Checked here rather than left to a confusing failure deep
        inside the fit."""
        md = mv.data.from_structures([self._lattice("bcc", 2.9)])
        with pytest.raises(ValueError, match="half the shortest"):
            mv.prop.phonon_at_temperature(md, level="emt", supercell=(2, 2, 2),
                                          cutoff=5.0,
                                          temperatures=(300.,))

    def test_a_cluster_space_too_small_to_mean_anything_is_flagged(self):
        md = mv.data.from_structures([self._lattice("bcc", 2.9)])
        with pytest.warns(RuntimeWarning, match="free parameters"):
            mv.prop.phonon_at_temperature(
                md, level="emt", temperatures=(300.,), supercell=(5, 5, 5),
                cutoff=4.0, n_structures=10, n_iterations=5)
        assert md.uns["self_consistent_phonons"]["emt"][
            "n_free_parameters"][0] < 6

    def test_it_records_how_far_from_converged_each_solve_was(self, sampled):
        drift = np.asarray(sampled.uns["self_consistent_phonons"]["emt"][
            "convergence_drift"], dtype=float)
        assert drift.shape == (3, 4)
        assert np.isfinite(drift).all()
        assert (drift >= 0).all()

    def test_it_does_not_print_over_the_caller(self, sampled, capsys):
        """hiphive reports at length on stdout, and a library should not write
        into the middle of somebody else's loop."""
        md = mv.data.from_structures([self._lattice("fcc", 3.61)])
        capsys.readouterr()
        mv.prop.phonon_at_temperature(
            md, level="emt", temperatures=(300.,), supercell=(5, 5, 5),
            cutoff=5.0, n_structures=10, n_iterations=5)
        assert capsys.readouterr().out == ""


def _pbsn_database():
    """pycalphad's own Pb-Sn assessment, which ships with its tests.

    matverse ships no database — assessed ones are years of work and mostly
    licensed — so the only one available to test against is pycalphad's, and
    this returns None rather than guessing if the layout changes.
    """
    import importlib.util
    import pathlib
    if importlib.util.find_spec("pycalphad") is None:
        return None
    import pycalphad
    found = list(pathlib.Path(pycalphad.__file__).parent.rglob("pbsn.tdb"))
    return str(found[0]) if found else None


@pytest.mark.skipif(_pbsn_database() is None,
                    reason="needs pycalphad and its Pb-Sn assessment")
class TestCalphad:
    """Checked against the Pb-Sn eutectic, which is a measured number.

    A CALPHAD database is fitted to measured phase boundaries, so the thing to
    verify is that the fit is being read correctly — not the thermodynamics,
    which is somebody else's assessment.
    """

    ETUECTIC = "Pb0.261Sn0.739"

    def test_it_brackets_the_measured_eutectic(self):
        """Two solids below, one liquid above. The measured eutectic is 456 K,
        and 450/455 straddle it."""
        below = mv.data.from_compositions([self.ETUECTIC])
        mv.thermo.calphad(below, _pbsn_database(), temperature=450.0)
        assert below.obs["calphad_n_phases"].iloc[0] == 2
        assert "LIQUID" not in below.obs["calphad_phases"].iloc[0]

        above = mv.data.from_compositions([self.ETUECTIC])
        mv.thermo.calphad(above, _pbsn_database(), temperature=455.0)
        assert above.obs["calphad_phases"].iloc[0] == "LIQUID"

    def test_the_phase_fractions_are_a_partition(self):
        md = mv.data.from_compositions([self.ETUECTIC, "Pb0.9Sn0.1"])
        mv.thermo.calphad(md, _pbsn_database(), temperature=450.0)
        # The major fraction cannot exceed the whole, and a single-phase
        # material is entirely that phase.
        assert (md.obs["calphad_major_fraction"] <= 1.0 + 1e-9).all()
        assert float(md.obs["calphad_major_fraction"].iloc[1]) == \
            pytest.approx(1.0, abs=1e-6)
        assert md.obs["calphad_n_phases"].iloc[1] == 1

    def test_an_unassessed_element_is_skipped_not_projected(self):
        """Dropping copper and renormalising would answer a question about a
        different material, so the row is left empty and counted."""
        md = mv.data.from_compositions(["PbSnCu"])
        with pytest.warns(RuntimeWarning, match="no equilibrium"):
            mv.thermo.calphad(md, _pbsn_database(), temperature=450.0)
        assert md.obs["calphad_phases"].iloc[0] == ""
        assert md.uns["calphad"]["n_skipped"] == 1
        assert "CU" in md.uns["calphad"]["errors"][0]

    def test_asking_for_an_unassessed_element_is_refused_up_front(self):
        md = mv.data.from_compositions(["PbSn"])
        with pytest.raises(ValueError, match="does not assess"):
            mv.thermo.calphad(md, _pbsn_database(), temperature=450.0,
                              elements=["PB", "SN", "CU"])

    def test_it_records_the_database_it_read(self):
        """Two assessments of the same system disagree, so which one produced
        a number is part of the number."""
        md = mv.data.from_compositions([self.ETUECTIC])
        mv.thermo.calphad(md, _pbsn_database(), temperature=450.0)
        recorded = md.uns["calphad"]
        assert "pbsn" in recorded["database"].lower()
        assert recorded["temperature"] == 450.0
        assert set(recorded["assessed_elements"]) == {"PB", "SN"}
        assert "error bar" in recorded["note"]

    def test_it_needs_a_formula_column(self):
        from pymatgen.core import Lattice, Structure
        md = mv.data.from_structures(
            [Structure(Lattice.cubic(4.0), ["Pb"], [[0, 0, 0]])])
        with pytest.raises(ValueError, match="from_compositions"):
            mv.thermo.calphad(md, _pbsn_database(), temperature=450.0)


def _olivine_pair():
    """LiFePO4 and its delithiated framework, plus a lithium reference.

    The energies are supplied directly rather than computed, so the test says
    something about the voltage arithmetic rather than about a potential. The
    values are what CHGNet gives for a positions-only relaxation, which
    reproduces the measured plateau.
    """
    from pymatgen.core import Lattice, Structure

    lifepo4 = mv.datasets.load("battery_cathodes")[:1].copy()
    lithiated = mv.structures(lifepo4, "input")[0]
    delithiated = lithiated.copy()
    delithiated.remove_species(["Li"])

    cathode = mv.data.from_structures([lithiated, delithiated])
    cathode.obs_names = ["LiFePO4", "FePO4"]
    cathode.obs["energy_ref"] = [-210.8473, -189.3663]
    cathode.obs["relax_converged_ref"] = [True, True]

    metal = mv.data.from_structures([Structure(
        Lattice.cubic(3.51), ["Li", "Li"], [[0, 0, 0], [.5, .5, .5]])])
    metal.obs["energy_ref"] = [-3.7474]
    metal.obs["relax_converged_ref"] = [True]
    return cathode, metal


class TestIntercalationVoltage:
    """Checked against a measured plateau, not against another code.

    LiFePO4 discharges at 3.4-3.5 V with a theoretical capacity of 170 mAh/g.
    Those are the numbers to hit.
    """

    def test_it_reproduces_the_lifepo4_plateau(self):
        cathode, metal = _olivine_pair()
        mv.thermo.voltage(cathode, working_ion="Li", level="ref",
                          reference=metal)
        assert float(cathode.obs["voltage_ref"].iloc[0]) == \
            pytest.approx(3.5, abs=0.15)
        assert float(cathode.obs["capacity_gravimetric_ref"].iloc[0]) == \
            pytest.approx(170.0, rel=0.02)

    def test_every_row_of_a_framework_carries_its_voltage(self):
        """The voltage belongs to the pair, so both rows report it — a screen
        sorting on the column should not have to know which row is which."""
        cathode, metal = _olivine_pair()
        mv.thermo.voltage(cathode, working_ion="Li", level="ref",
                          reference=metal)
        values = cathode.obs["voltage_ref"].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert values[0] == pytest.approx(values[1])

    def test_an_unconverged_row_is_excluded_not_averaged_in(self):
        """The bug this was written after: reading energy_{level} without
        checking relax_converged_{level} gave 78 V for LiFePO4, from a cell a
        foundation potential had collapsed to 2 cubic angstrom per atom. The
        diagnostic was False the whole time."""
        cathode, metal = _olivine_pair()
        cathode.obs["relax_converged_ref"] = [True, False]
        with pytest.warns(RuntimeWarning, match="did not converge"):
            mv.thermo.voltage(cathode, working_ion="Li", level="ref",
                              reference=metal)
        assert np.isnan(float(cathode.obs["voltage_ref"].iloc[0]))
        assert "FePO4" in cathode.uns["electrode"]["ref"]["unconverged"]

    def test_an_unconverged_reference_is_refused_outright(self):
        """A bad reference shifts every voltage by the same amount, so no
        comparison between cathodes reveals it. That makes it worse than a bad
        cathode, and it raises rather than warns."""
        cathode, metal = _olivine_pair()
        metal.obs["relax_converged_ref"] = [False]
        with pytest.raises(ValueError, match="reference did not converge"):
            mv.thermo.voltage(cathode, working_ion="Li", level="ref",
                              reference=metal)

    def test_a_missing_reference_is_refused_rather_than_guessed(self):
        cathode, _ = _olivine_pair()
        with pytest.raises(ValueError, match="reference= is required"):
            mv.thermo.voltage(cathode, working_ion="Li", level="ref")

    def test_two_frameworks_are_not_averaged_together(self):
        """Li and Na analogues of the same framework are different electrodes.
        Grouping by what is left when the working ion is removed keeps them
        apart; without it they become one meaningless average."""
        cathode, metal = _olivine_pair()
        extra = mv.datasets.load("battery_cathodes")[1:2].copy()
        merged = mv.data.from_structures(
            list(mv.structures(cathode, "input"))
            + list(mv.structures(extra, "input")))
        merged.obs_names = ["LiFePO4", "FePO4", "NaFePO4"]
        merged.obs["energy_ref"] = [-210.8473, -189.3663, -205.0]
        merged.obs["relax_converged_ref"] = [True, True, True]
        with pytest.warns(RuntimeWarning, match="no voltage"):
            mv.thermo.voltage(merged, working_ion="Li", level="ref",
                              reference=metal)
        # The sodium analogue has no lithium pair, so it gets no voltage.
        assert np.isnan(float(merged.obs["voltage_ref"].iloc[2]))
        assert np.isfinite(float(merged.obs["voltage_ref"].iloc[0]))

    def test_it_needs_energies(self):
        cathode, metal = _olivine_pair()
        del cathode.obs["energy_ref"]
        with pytest.raises(ValueError, match="mv.calc.relax"):
            mv.thermo.voltage(cathode, working_ion="Li", level="ref",
                              reference=metal)


class TestScattering:
    """S(q) and G(r) are definitions, so they can be checked against closed
    forms rather than against another code."""

    SIGMA = 3.0
    DENSITY = 0.05

    @classmethod
    def _hard_sphere(cls, step=0.005, r_max=20.0):
        from matverse._core import deposit_grid
        from pymatgen.core import Lattice, Structure

        side = (400 / cls.DENSITY) ** (1 / 3)
        rng = np.random.default_rng(0)
        cell = Structure(Lattice.cubic(side), ["Ar"] * 400, rng.random((400, 3)))
        md = mv.data.from_structures([cell])
        r = np.arange(step, r_max, step)
        g = np.where(r < cls.SIGMA, 0.0, 1.0)
        deposit_grid(md, "rdf", "test", g[None, :], r, unit="angstrom")
        return md, r

    @classmethod
    def _analytic(cls, q):
        """S(q) for a step g(r): 1 - 4 pi rho / q^3 [sin(qd) - qd cos(qd)]."""
        d = cls.SIGMA
        return 1.0 - 4 * np.pi * cls.DENSITY / q ** 3 * (
            np.sin(q * d) - q * d * np.cos(q * d))

    def test_an_ideal_gas_has_a_structure_factor_of_one(self):
        """g = 1 everywhere means no structure, and S(q) says so exactly."""
        from matverse._core import deposit_grid, grid_of
        from pymatgen.core import Lattice, Structure

        cell = Structure(Lattice.cubic(20.0), ["Ar"], [[0, 0, 0]])
        md = mv.data.from_structures([cell])
        r = np.arange(0.01, 20.0, 0.005)
        deposit_grid(md, "rdf", "ideal", np.ones((1, r.size)), r,
                     unit="angstrom")
        mv.prop.scattering(md, level="ideal", q_max=12.0, n_points=40)
        factor = md.obsm["structure_factor_ideal"][0]
        assert factor == pytest.approx(np.ones_like(factor), abs=1e-6)

    def test_it_reproduces_the_hard_sphere_transform(self):
        from matverse._core import grid_of

        md, _ = self._hard_sphere()
        mv.prop.scattering(md, level="test", q_max=12.0, n_points=60)
        q = grid_of(md, "structure_factor")
        computed = md.obsm["structure_factor_test"][0]
        assert np.abs(computed - self._analytic(q)).max() < 0.03

    def test_the_error_follows_the_r_grid_it_was_given(self):
        """The integral is quadrature on whatever mv.prop.rdf produced, so a
        coarser step is a worse S(q) — linearly. Stated in the notes as 0.24,
        0.12, 0.048 and 0.012 for steps of 0.10, 0.05, 0.02 and 0.005, and
        asserted here as monotone so the claim cannot rot."""
        from matverse._core import grid_of

        errors = []
        for step in (0.10, 0.02):
            md, _ = self._hard_sphere(step=step)
            mv.prop.scattering(md, level="test", q_max=12.0, n_points=60)
            q = grid_of(md, "structure_factor")
            errors.append(float(np.abs(
                md.obsm["structure_factor_test"][0] - self._analytic(q)).max()))
        assert errors[0] > errors[1] * 2

    def test_the_reduced_pdf_is_the_definition(self):
        """G(r) = 4 pi r rho [g(r) - 1], with nothing fitted."""
        md, r = self._hard_sphere()
        mv.prop.scattering(md, level="test", q_max=12.0, n_points=20)
        reduced = md.obsm["pdf_test"][0]
        inside = r < self.SIGMA
        assert reduced[inside] == pytest.approx(
            -4 * np.pi * r[inside] * self.DENSITY, rel=1e-6)
        assert reduced[r > self.SIGMA] == pytest.approx(0.0, abs=1e-12)

    def test_it_names_the_missing_step(self):
        md = mv.data.from_compositions(["Ar"])
        with pytest.raises(ValueError, match="mv.prop.rdf"):
            mv.prop.scattering(md, level="nothing")


class TestSuperconductivity:
    """Half computed, half supplied, and the tests keep the halves apart."""

    @staticmethod
    def _with_dos(frequencies, weights=None, f_max=20.0, n=4000):
        from matverse._core import deposit_grid
        from pymatgen.core import Lattice, Structure

        grid = np.linspace(0.1, f_max, n)
        density = np.zeros_like(grid)
        for j, centre in enumerate(np.atleast_1d(frequencies)):
            height = 1.0 if weights is None else weights[j]
            density += height * np.exp(-0.5 * ((grid - centre) / 0.02) ** 2)
        md = mv.data.from_structures(
            [Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]])])
        deposit_grid(md, "phonon_dos", "test", density[None, :], grid,
                     unit="THz")
        return md

    def test_a_single_frequency_returns_itself(self):
        """omega_log of a delta at w0 is w0, by construction."""
        for centre in (3.0, 8.0):
            md = self._with_dos(centre)
            mv.prop.superconductivity(md, level="test", coupling=1.0)
            assert float(md.obs["omega_log_test"].iloc[0]) == \
                pytest.approx(centre, rel=1e-3)

    def test_the_soft_end_is_weighted_more(self):
        """The 1/omega weighting puts the logarithmic average of 4 and 16 THz
        below their geometric mean of 8."""
        md = self._with_dos([4.0, 16.0])
        mv.prop.superconductivity(md, level="test", coupling=1.0)
        value = float(md.obs["omega_log_test"].iloc[0])
        assert value < 8.0
        assert value == pytest.approx(5.28, rel=0.02)

    def test_coupling_has_no_default(self):
        """matverse cannot compute lambda, and a guessed one produces a
        temperature that reads as a prediction."""
        md = self._with_dos(6.0)
        with pytest.raises(ValueError, match="coupling="):
            mv.prop.superconductivity(md, level="test")

    def test_tc_vanishes_at_the_formula_s_own_limit(self):
        """As lambda approaches mu*(1 + 0.62 lambda) the exponent diverges.
        That is where the formula stops having a solution, not where
        superconductivity stops, and it returns exactly zero rather than an
        overflow."""
        from matverse.prop import _allen_dynes

        assert _allen_dynes(0.10, 232.0, 252.0, 0.13) == 0.0
        assert _allen_dynes(0.05, 232.0, 252.0, 0.13) == 0.0
        assert _allen_dynes(1.00, 232.0, 252.0, 0.13) > 1.0

    def test_the_strong_coupling_factors_are_included(self):
        """Allen-Dynes with f1 and f2, not bare McMillan. They agree within a
        few per cent at lambda = 0.5 and differ by a third at lambda = 3;
        leaving them out is the usual reason a strong-coupling estimate comes
        out low."""
        from matverse.prop import _allen_dynes

        def mcmillan(lam, w_log, mu=0.13):
            return w_log / 1.2 * np.exp(
                -1.04 * (1 + lam) / (lam - mu * (1 + 0.62 * lam)))

        weak = _allen_dynes(0.5, 100.0, 150.0, 0.13) / mcmillan(0.5, 100.0)
        strong = _allen_dynes(3.0, 100.0, 150.0, 0.13) / mcmillan(3.0, 100.0)
        assert weak == pytest.approx(1.0, abs=0.05)
        assert strong > 1.25

    def test_it_records_that_the_dos_stood_in_for_alpha2F(self):
        md = self._with_dos(6.0)
        mv.prop.superconductivity(md, level="test", coupling=1.0)
        recorded = md.uns["superconductivity"]["test"]
        assert "alpha^2 F" in recorded["spectral_function"]
        assert "not computed here" in recorded["coupling_source"]


class TestCatalysisScaling:
    """A scaling relation is a straight line, so a synthetic one with a known
    slope says whether the fit is a fit."""

    @staticmethod
    def _surfaces(slope=0.5, intercept=-0.10, noise=0.02, n=8):
        rng = np.random.default_rng(0)
        oxygen = np.linspace(-2.5, 0.5, n)
        md = mv.data.from_compositions([f"Pt{i + 1}" for i in range(n)])
        md.obs["E_O"] = oxygen
        md.obs["E_OH"] = slope * oxygen + intercept + rng.normal(0, noise, n)
        return md

    def test_it_recovers_a_known_slope(self):
        md = self._surfaces()
        mv.surf.scaling(md, x="E_O", y="E_OH")
        fit = md.uns["scaling"]["fits"]["all"]
        assert fit["slope"] == pytest.approx(0.5, abs=0.02)
        assert fit["intercept_eV"] == pytest.approx(-0.10, abs=0.03)
        assert fit["r_squared"] > 0.99

    def test_the_residual_is_the_departure_from_the_line(self):
        """The interesting column: a surface that beats the scaling ceiling
        has to break the relation, so the residual is the candidate signal
        and the fit is the background."""
        md = self._surfaces()
        mv.surf.scaling(md, x="E_O", y="E_OH")
        residual = md.obs["scaling_residual"].to_numpy(dtype=float)
        predicted = (md.uns["scaling"]["fits"]["all"]["slope"]
                     * md.obs["E_O"].to_numpy(dtype=float)
                     + md.uns["scaling"]["fits"]["all"]["intercept_eV"])
        assert residual == pytest.approx(
            md.obs["E_OH"].to_numpy(dtype=float) - predicted, abs=1e-9)

    def test_two_points_are_not_a_scaling_relation(self):
        md = self._surfaces(n=2)
        with pytest.raises(ValueError, match="not a scaling relation"):
            mv.surf.scaling(md, x="E_O", y="E_OH")

    def test_a_missing_column_lists_what_is_there(self):
        md = self._surfaces()
        with pytest.raises(ValueError, match="mv.surf.adsorption_energy"):
            mv.surf.scaling(md, x="E_O", y="E_nothing")


class TestVolcano:
    @staticmethod
    def _surfaces():
        md = mv.data.from_compositions([f"Pt{i + 1}" for i in range(8)])
        md.obs["E_O"] = np.linspace(-2.5, 0.5, 8)
        return md

    def test_the_peak_sits_at_the_optimum(self):
        md = self._surfaces()
        mv.surf.volcano(md, descriptor="E_O", optimum=-1.6)
        best = int(np.argmax(md.obs["volcano_activity"].to_numpy(dtype=float)))
        distances = np.abs(md.obs["distance_from_optimum"].to_numpy(float))
        assert best == int(np.argmin(distances))

    def test_the_distance_is_signed_so_the_side_is_readable(self):
        """Binding too weakly and too strongly are different problems, and a
        magnitude cannot tell them apart."""
        md = self._surfaces()
        mv.surf.volcano(md, descriptor="E_O", optimum=-1.6)
        offset = md.obs["distance_from_optimum"].to_numpy(dtype=float)
        assert offset.min() < 0 < offset.max()

    def test_slopes_of_the_same_sign_are_refused(self):
        """Two limbs with the same sign is a straight line dressed as a
        peak."""
        md = self._surfaces()
        with pytest.raises(ValueError, match="opposite"):
            mv.surf.volcano(md, descriptor="E_O", optimum=-1.6,
                            slopes=(1.0, 1.0))

    def test_it_records_that_the_optimum_came_from_outside(self):
        md = self._surfaces()
        mv.surf.volcano(md, descriptor="E_O", optimum=-1.6)
        assert "not derived here" in md.uns["volcano"]["optimum_source"]


class TestFigureOfMerit:
    """zT needs a relaxation time nobody has. The ceiling does not, and that
    is the point of computing it separately."""

    @staticmethod
    def _transport(seebeck=200e-6, sigma_over_tau=1e18, lattice=None):
        md = mv.data.from_compositions(["Bi2Te3"])
        md.obs["seebeck_x"] = [seebeck]
        md.obs["sigma_over_tau_x"] = [sigma_over_tau]
        if lattice is not None:
            md.obs["thermal_conductivity_x"] = [lattice]
        return md

    def test_the_ceiling_needs_no_relaxation_time(self):
        """tau cancels between sigma and the electronic thermal conductivity,
        so S^2 / L survives not knowing it. At 200 microvolts per kelvin that
        is 1.639."""
        md = self._transport(lattice=1.0)
        mv.prop.zt(md, level="x", relaxation_time=1e-14, temperature=700.0)
        assert float(md.obs["zt_ceiling_x"].iloc[0]) == \
            pytest.approx(1.639, rel=0.01)

    def test_zt_approaches_the_ceiling_as_the_lattice_stops_mattering(self):
        """Checked across four decades of conductivity: 0.027, 0.24, 1.03,
        1.55 and then 1.6384 against an analytic 1.6393."""
        values = []
        for sigma in (1e17, 1e18, 1e19, 1e22):
            md = self._transport(sigma_over_tau=sigma, lattice=1.0)
            mv.prop.zt(md, level="x", relaxation_time=1e-14, temperature=700.0)
            values.append(float(md.obs["zt_x"].iloc[0]))
        assert values == sorted(values), "zT must rise with conductivity"
        ceiling = float(md.obs["zt_ceiling_x"].iloc[0])
        assert values[-1] == pytest.approx(ceiling, rel=0.01)
        assert values[-1] < ceiling, "the ceiling is approached, not crossed"

    def test_the_relaxation_time_has_no_default(self):
        md = self._transport(lattice=1.0)
        with pytest.raises(ValueError, match="relaxation_time="):
            mv.prop.zt(md, level="x")

    def test_a_missing_lattice_conductivity_is_warned_about(self):
        """Counting only the electronic part flatters every material, so the
        omission is said out loud rather than absorbed."""
        md = self._transport()
        with pytest.warns(RuntimeWarning, match="overestimate"):
            mv.prop.zt(md, level="x", relaxation_time=1e-14)

    def test_it_needs_transport_first(self):
        md = mv.data.from_compositions(["Bi2Te3"])
        with pytest.raises(ValueError, match="mv.elec.transport"):
            mv.prop.zt(md, level="x", relaxation_time=1e-14)

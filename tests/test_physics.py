"""First-principles I/O, phase equilibria beyond the hull, and lattice dynamics.

The phonon tests check against physics rather than against a stored number: the
zero-point energy of copper, and the classical limit of the heat capacity. A
regression test that only compares to last week's output cannot tell a
refactoring from a sign error.
"""

from __future__ import annotations

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
        assert "pydefect" in recorded["note"]

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
    need a LOCPOT is the potential alignment, and it is absent by design.
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
        """The potential-alignment term is missing and must be named, not
        left for the reader to discover."""
        defects, host = self._system((2, 2, 2))
        out, _ = self._curve(defects, host, 10.0)
        record = out.uns["defect_thermodynamics"]
        assert record["image_charge_correction"] is True
        assert record["dielectric"] == 10.0
        assert "LOCPOT" in record["correction_terms"]
        assert record["correction_error"] is None

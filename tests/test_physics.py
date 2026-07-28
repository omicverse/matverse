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

    def test_only_vasp_is_claimed(self, md, tmp_path):
        with pytest.raises(ValueError, match="only code='vasp'"):
            mv.dft.write_inputs(md, tmp_path / "runs", code="espresso")

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

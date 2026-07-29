"""Units, checkpointing, elastic constants and federated ingestion."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


class TestUnits:
    def test_conversion_is_correct(self, md):
        md.obs["expt"] = np.full(md.n_obs, 100.0)
        mv.utils.convert(md, "expt", "kJ/mol")
        assert md.obs["expt_ev"].iloc[0] == pytest.approx(1.0364, abs=1e-4)

    def test_rydberg_and_hartree(self, md):
        md.obs["ry"] = np.ones(md.n_obs)
        md.obs["ha"] = np.ones(md.n_obs)
        mv.utils.convert(md, "ry", "Ry")
        mv.utils.convert(md, "ha", "Hartree")
        assert md.obs["ry_ev"].iloc[0] == pytest.approx(13.6057, abs=1e-3)
        assert md.obs["ha_ev"].iloc[0] == pytest.approx(27.2114, abs=1e-3)

    def test_conversion_deposits_rather_than_overwrites(self, md):
        md.obs["expt"] = np.full(md.n_obs, 100.0)
        mv.utils.convert(md, "expt", "kJ/mol")
        assert md.obs["expt"].iloc[0] == 100.0        # the original survives

    def test_lengths_convert_too(self, md):
        md.obs["bond"] = np.full(md.n_obs, 1.0)
        mv.utils.convert(md, "bond", "bohr", kind="length")
        assert md.obs["bond_angstrom"].iloc[0] == pytest.approx(0.5292,
                                                                abs=1e-3)

    def test_declared_units_are_remembered(self, md):
        md.obs["expt"] = np.zeros(md.n_obs)
        mv.utils.set_units(md, "expt", "kJ/mol")
        mv.utils.convert(md, "expt")                  # no unit= needed
        assert md.uns["units"]["expt_ev"] == "eV"

    def test_an_undeclared_column_says_so(self, md):
        md.obs["mystery"] = np.zeros(md.n_obs)
        with pytest.raises(ValueError, match="no unit known"):
            mv.utils.convert(md, "mystery")

    def test_an_unknown_unit_lists_the_known_ones(self, md):
        md.obs["expt"] = np.zeros(md.n_obs)
        with pytest.raises(ValueError, match="unknown energy unit"):
            mv.utils.convert(md, "expt", "furlongs")

    def test_matverse_own_columns_are_inferred(self, md):
        mv.pp.describe(md)
        mv.calc.energy(md, level="emt")
        units = mv.utils.check_units(md)
        assert units["energy_per_atom_emt"] == "eV/atom"
        assert units["volume"] == "angstrom^3"

    def test_an_unknown_column_reports_none(self, md):
        md.obs["mystery"] = np.zeros(md.n_obs)
        assert mv.utils.check_units(md)["mystery"] is None


class TestCheckpointing:
    def test_resume_reports_unfinished_rows(self, md):
        assert mv.utils.resume(md, "energy_emt").all()      # nothing done yet
        mv.calc.energy(md, level="emt")
        assert not mv.utils.resume(md, "energy_emt").any()

    def test_resume_spots_a_partial_column(self, md):
        mv.calc.energy(md, level="emt")
        values = md.obs["energy_emt"].to_numpy(dtype=float).copy()
        values[2] = np.nan
        md.obs["energy_emt"] = values
        todo = mv.utils.resume(md, "energy_emt")
        assert todo.sum() == 1 and todo[2]

    def test_checkpoint_writes_and_records(self, md, tmp_path):
        from matverse._core import records

        mv.pp.describe(md)
        path = mv.utils.checkpoint(md, tmp_path / "run.h5ad", note="after qc")
        assert records(md.uns, "checkpoints")[-1]["note"] == "after qc"

        import anndata
        back = anndata.read_h5ad(path)
        assert back.n_obs == md.n_obs

    def test_a_checkpointed_object_reloads_with_its_history(self, md, tmp_path):
        """A record list in uns must survive h5ad, which a list of dicts does
        not — anndata turns one into an object array and h5py refuses it."""
        from matverse._core import records

        mv.pp.describe(md)
        mv.utils.checkpoint(md, tmp_path / "a.h5ad", note="first")
        path = mv.utils.checkpoint(md, tmp_path / "b.h5ad", note="second")

        import anndata
        back = anndata.read_h5ad(path)
        notes = [r["note"] for r in records(back.uns, "checkpoints")]
        assert notes == ["first", "second"]

    def test_slurm_script_is_written_not_submitted(self, tmp_path):
        path = mv.utils.slurm_script("screen.py", tmp_path / "job.sbatch",
                                     partition="normal", hours=4, gpus=1)
        text = open(path).read()
        assert "#SBATCH --partition=normal" in text
        assert "#SBATCH --time=04:00:00" in text
        assert "#SBATCH --gpus=1" in text
        assert "python screen.py" in text
        # Caches redirected off a small shared home directory.
        assert "HF_HOME" in text and "SCRATCH" in text


class TestSummary:
    def test_it_reports_the_shape_and_the_levels(self, md):
        mv.pp.describe(md)
        mv.calc.energy(md, level="emt")
        text = mv.utils.summary(md)
        assert "6 materials x 3 elements" in text
        assert "emt" in text and "EMT" in text
        assert "input" in text

    def test_it_warns_about_a_closed_hull(self, md):
        mv.calc.energy(md, level="emt")
        with pytest.warns(UserWarning):
            mv.thermo.hull(md, level="emt")
        assert "hull is closed" in mv.utils.summary(md)

    def test_it_warns_about_a_noncommercial_level(self, md):
        from ase.calculators.emt import EMT
        mv.calc.register_calculator("restricted", EMT, method="Restricted",
                                    license="ASL")
        mv.calc.energy(md, level="restricted")
        assert "forbidding commercial use" in mv.utils.summary(md)


class TestElastic:
    @pytest.fixture(scope="class")
    def relaxed(self):
        from pymatgen.core import Lattice, Structure

        def fcc(symbol, a):
            return Structure(Lattice.cubic(a), [symbol] * 4,
                             [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

        md = mv.data.from_structures([fcc("Cu", 3.61), fcc("Al", 4.05),
                                      fcc("Ni", 3.52)])
        mv.pp.describe(md)
        mv.calc.relax(md, level="emt", fmax=0.01)
        mv.prop.elastic(md, level="emt", source="relaxed_emt")
        return md

    def test_moduli_are_deposited(self, relaxed):
        for column in ("bulk_modulus_emt", "shear_modulus_emt",
                       "youngs_modulus_emt", "poisson_ratio_emt"):
            assert column in relaxed.obs
        assert relaxed.obsm["elastic_tensor_emt"].shape == (3, 36)

    def test_the_tensor_is_symmetric(self, relaxed):
        for row in relaxed.obsm["elastic_tensor_emt"]:
            C = row.reshape(6, 6)
            assert np.allclose(C, C.T, atol=1e-6)

    def test_relaxed_metals_are_born_stable(self, relaxed):
        assert relaxed.obs["elastic_stable_emt"].all()

    def test_moduli_are_physically_ordered(self, relaxed):
        """EMT gets magnitudes only roughly right; the ordering it gets right.

        Literature bulk moduli: Ni 180 > Cu 140 > Al 76 GPa.
        """
        by_formula = dict(zip(relaxed.obs["formula"],
                              relaxed.obs["bulk_modulus_emt"]))
        assert by_formula["Ni"] > by_formula["Cu"] > by_formula["Al"]

    def test_poisson_ratio_is_in_range(self, relaxed):
        nu = relaxed.obs["poisson_ratio_emt"].to_numpy(dtype=float)
        assert ((nu > -1.0) & (nu < 0.5)).all()

    def test_moduli_are_positive(self, relaxed):
        assert (relaxed.obs["bulk_modulus_emt"] > 0).all()
        assert (relaxed.obs["shear_modulus_emt"] > 0).all()


class TestOptimade:
    @staticmethod
    def _payload():
        """A minimal but real-shaped OPTIMADE /structures response."""
        return {"data": [
            {"id": "mp-134", "type": "structures", "attributes": {
                "lattice_vectors": [[0.0, 2.02, 2.02], [2.02, 0.0, 2.02],
                                    [2.02, 2.02, 0.0]],
                "cartesian_site_positions": [[0.0, 0.0, 0.0]],
                "species_at_sites": ["Al"],
                "species": [{"name": "Al", "chemical_symbols": ["Al"],
                             "concentration": [1.0]}],
                "chemical_formula_reduced": "Al", "nelements": 1}},
            {"id": "mp-2", "type": "structures", "attributes": {
                "lattice_vectors": [[0.0, 1.8, 1.8], [1.8, 0.0, 1.8],
                                    [1.8, 1.8, 0.0]],
                "cartesian_site_positions": [[0.0, 0.0, 0.0]],
                "species_at_sites": ["Cu"],
                "species": [{"name": "Cu", "chemical_symbols": ["Cu"],
                             "concentration": [1.0]}],
                "chemical_formula_reduced": "Cu", "nelements": 1}},
        ]}

    def test_a_response_becomes_a_dataset(self):
        md = mv.data.from_optimade_response(self._payload(), provider="mp")
        assert md.shape == (2, 2)
        assert list(md.obs["optimade_id"]) == ["mp-134", "mp-2"]
        assert set(md.var_names) == {"Al", "Cu"}

    def test_the_provider_becomes_a_level(self):
        md = mv.data.from_optimade_response(self._payload(), provider="oqmd")
        assert mv.level_info(md, "oqmd")["kind"] == "dft"

    def test_partial_occupancy_is_kept(self):
        payload = self._payload()
        payload["data"] = payload["data"][:1]
        payload["data"][0]["attributes"]["species"] = [
            {"name": "Al", "chemical_symbols": ["Al", "Cu"],
             "concentration": [0.7, 0.3]}]
        md = mv.data.from_optimade_response(payload)
        from matverse._core import structures
        assert not structures(md)[0].is_ordered

    def test_a_broken_entry_is_reported_not_swallowed(self):
        payload = self._payload()
        payload["data"][1]["attributes"].pop("lattice_vectors")
        md = mv.data.from_optimade_response(payload)
        assert md.n_obs == 1
        assert len(md.uns["read_errors"]) == 1

    def test_an_empty_response_says_so(self):
        with pytest.raises(ValueError, match="no structures"):
            mv.data.from_optimade_response({"data": []})

    def test_all_entries_failing_is_an_error(self):
        payload = self._payload()
        for entry in payload["data"]:
            entry["attributes"].pop("species_at_sites")
        with pytest.raises(ValueError, match="no structure parsed"):
            mv.data.from_optimade_response(payload)

    def test_providers_are_listed(self):
        providers = mv.data.optimade_providers()
        assert "mp" in providers and "oqmd" in providers
        assert providers["mp"].startswith("https://")

    def test_an_unknown_provider_names_the_alternative(self):
        with pytest.raises(ValueError, match="pass base_url"):
            mv.data.from_optimade("nelements=2", provider="nowhere")


class TestSubmission:
    """Submitting is a small step past writing a script, and the useful part
    is the link back: which job is computing this dataset."""

    def _script(self, md, tmp_path):
        return mv.utils.slurm_script(
            "echo matverse", path=tmp_path / "job.sbatch",
            partition="normal", hours=1, cpus=1, memory="2GB")

    def test_a_dry_run_returns_the_command_without_running_it(self, md,
                                                              tmp_path):
        """What you want on a login node, and in a test."""
        script = self._script(md, tmp_path)
        entry = mv.utils.submit(md, script, dry_run=True)
        assert entry["state"] == "dry run"
        assert entry["job_id"] is None
        assert entry["command"].startswith("sbatch ")

    def test_the_job_is_recorded_on_the_object(self, md, tmp_path):
        script = self._script(md, tmp_path)
        mv.utils.submit(md, script, dry_run=True)
        jobs = mv.records(md.uns["submissions"], "jobs")
        assert len(jobs) == 1
        assert jobs[0]["script"] == str(script)

    def test_submissions_accumulate_rather_than_replace(self, md, tmp_path):
        script = self._script(md, tmp_path)
        mv.utils.submit(md, script, dry_run=True)
        mv.utils.submit(md, script, dry_run=True)
        assert len(mv.records(md.uns["submissions"], "jobs")) == 2

    def test_a_missing_script_says_what_writes_one(self, md, tmp_path):
        with pytest.raises(FileNotFoundError, match="slurm_script"):
            mv.utils.submit(md, tmp_path / "nothing.sbatch")

    def test_status_of_a_dry_run_says_nothing_was_submitted(self, md,
                                                            tmp_path):
        script = self._script(md, tmp_path)
        mv.utils.submit(md, script, dry_run=True)
        report = mv.utils.job_status(md)
        assert report["n_jobs"] == 1
        assert "dry runs" in report["note"]

    def test_the_record_survives_a_round_trip(self, md, tmp_path):
        """Submissions go through append_record, so they are h5ad-writable —
        a list of dicts in uns is not."""
        import anndata

        script = self._script(md, tmp_path)
        mv.utils.submit(md, script, dry_run=True)
        path = tmp_path / "with_jobs.h5ad"
        md.write_h5ad(path)
        back = anndata.read_h5ad(path)
        assert len(mv.records(back.uns["submissions"], "jobs")) == 1

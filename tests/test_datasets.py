"""Bundled datasets, and that they are real.

The point of this module is that examples stop being built out of the library's
own test fixtures. So the tests check the data is what it claims — published
structures with the space groups they are reported in — rather than only that
something loaded.
"""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


class TestBundled:
    def test_every_advertised_dataset_loads(self):
        for name in mv.datasets.available():
            md = mv.datasets.load(name)
            assert md.n_obs > 0, name
            assert md.n_vars > 0, name

    def test_each_one_says_what_it_is_for(self):
        """Which materials to use is the first question an example has to
        answer, so the description is part of the dataset."""
        for name, meta in mv.datasets.available().items():
            assert meta["use_for"], name
            assert meta["description"], name

    def test_the_structures_are_the_published_ones(self):
        """Space groups from the literature, not from a cubic cell someone
        typed in."""
        md = mv.datasets.load("battery_cathodes")
        by_name = dict(zip(md.obs["name"], md.obs["spacegroup"]))
        assert by_name["LiFePO4"] == "P2_1/c"
        assert by_name["NaFePO4"] == "Pnma"

    def test_a_real_superionic_conductor_is_available(self):
        md = mv.datasets.load("solid_electrolytes")
        names = list(md.obs["name"])
        assert "Li10GeP2S12" in names
        sizes = dict(zip(md.obs["name"],
                         [len(s) for s in mv.structures(md)]))
        assert sizes["Li10GeP2S12"] > 40      # a realistic cost, not a toy

    def test_the_composition_axis_spans_the_chemistry(self):
        md = mv.datasets.load("oxides")
        assert "O" in set(md.var_names)
        assert md.n_vars >= 5

    def test_provenance_is_recorded(self):
        md = mv.datasets.load("simple")
        assert md.uns["dataset"]["name"] == "simple"
        assert md.obs["source"].iloc[0]

    def test_an_unknown_name_lists_the_options(self):
        with pytest.raises(KeyError, match="available"):
            mv.datasets.load("not-a-dataset")

    def test_a_loaded_dataset_is_an_ordinary_object(self):
        md = mv.datasets.load("simple")
        mv.pp.describe(md)
        mv.prop.xrd(md, two_theta=(10, 40), step=0.1)
        assert "formula" in md.obs
        assert md.obsm["xrd_calc"].shape[0] == md.n_obs


class TestMetals:
    def test_they_are_the_elements_emt_can_run(self):
        """EMT is the only calculator that ships working, so these are the
        materials on which every example runs end to end."""
        md = mv.datasets.metals()
        mv.calc.energy(md, level="emt")
        assert np.isfinite(
            md.obs["energy_per_atom_emt"].to_numpy(dtype=float)).all()

    def test_lattice_parameters_are_the_published_ones(self):
        md = mv.datasets.metals(["Cu", "Al", "Au"])
        by_name = dict(zip(md.obs["name"], md.obs["lattice_parameter"]))
        assert by_name["Cu"] == pytest.approx(3.615, abs=0.01)
        assert by_name["Al"] == pytest.approx(4.050, abs=0.01)
        assert by_name["Au"] == pytest.approx(4.078, abs=0.01)

    def test_a_subset_can_be_asked_for(self):
        md = mv.datasets.metals(["Ni", "Pd"])
        assert list(md.obs["name"]) == ["Ni", "Pd"]

    def test_a_supercell_can_be_asked_for(self):
        small = mv.datasets.metals(["Cu"])
        big = mv.datasets.metals(["Cu"], supercell=(2, 2, 2))
        assert len(mv.structures(big)[0]) == 8 * len(mv.structures(small)[0])

    def test_an_element_emt_cannot_run_is_refused(self):
        with pytest.raises(KeyError, match="no lattice parameter"):
            mv.datasets.metals(["Fe"])


class TestCache:
    def test_the_cache_directory_respects_the_override(self, tmp_path,
                                                       monkeypatch):
        """A home directory on a cluster is small, NFS-backed and shared, and a
        downloaded corpus does not belong there."""
        monkeypatch.setenv("MATVERSE_DATA", str(tmp_path / "somewhere"))
        assert mv.datasets.cache_dir() == tmp_path / "somewhere"

    def test_listing_an_empty_cache_is_not_an_error(self, tmp_path,
                                                    monkeypatch):
        monkeypatch.setenv("MATVERSE_DATA", str(tmp_path / "empty"))
        assert mv.datasets.cached() == []

    def test_a_cached_dataset_is_listed_and_reused(self, tmp_path,
                                                   monkeypatch):
        monkeypatch.setenv("MATVERSE_DATA", str(tmp_path / "cache"))
        directory = mv.datasets.cache_dir()
        directory.mkdir(parents=True)

        md = mv.datasets.load("simple")
        md.write_h5ad(directory / "mp_Li_Fe_abcdef1234.h5ad")

        listed = mv.datasets.cached()
        assert len(listed) == 1
        assert listed[0]["size_mb"] > 0

"""The substrate: composition matrix, element axis, levels, provenance."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


class TestCompositionMatrix:
    def test_x_is_materials_by_elements(self, md):
        assert md.shape == (6, 3)
        assert set(md.var_names) == {"Al", "Cu", "Ni"}

    def test_var_names_ordered_by_atomic_number(self, md):
        assert list(md.var_names) == ["Al", "Ni", "Cu"]

    def test_counts_come_from_the_reduced_formula(self, md):
        """A supercell and its primitive cell are the same chemistry."""
        X = md.X.toarray()
        al = list(md.var_names).index("Al")
        # fcc Al has four sites but reduces to Al.
        assert X[0, al] == pytest.approx(1.0)
        assert md.uns["X_is"] == "composition_atoms_reduced"

    def test_alloy_row_carries_both_elements(self, md):
        X = md.X.toarray()
        cols = {str(e): i for i, e in enumerate(md.var_names)}
        assert X[3, cols["Cu"]] == pytest.approx(1.0)   # CuAl3
        assert X[3, cols["Al"]] == pytest.approx(3.0)

    def test_var_is_the_periodic_table(self, md):
        assert md.var.loc["Cu", "Z"] == pytest.approx(29.0)
        assert md.var.loc["Al", "electronegativity"] == pytest.approx(1.61)
        assert bool(md.var.loc["Ni", "is_transition_metal"])

    def test_subsetting_keeps_everything_aligned(self, md):
        sub = md[[0, 3]].copy()
        assert sub.shape == (2, 3)
        assert sub.X.toarray()[1, list(sub.var_names).index("Al")] == 3.0

    def test_subsetting_carries_the_structures(self, md):
        """The reason for being on this substrate at all.

        Structures used to live in uns, which does not subset, so md[mask] kept
        every structure while dropping rows and each surviving row pointed at
        the wrong one.
        """
        formulas = [s.composition.reduced_formula for s in mv.structures(md)]
        sub = md[[1, 4]].copy()
        assert len(mv.structures(sub)) == 2
        assert [s.composition.reduced_formula for s in mv.structures(sub)] == \
            [formulas[1], formulas[4]]

    def test_a_variant_must_be_aligned_to_the_material_axis(self, md):
        from matverse._core import deposit_structures
        with pytest.raises(ValueError, match="aligned to the material axis"):
            deposit_structures(md, "wrong", mv.structures(md)[:2])

    def test_build_X_false_restores_the_v01_layout(self, structures):
        md = mv.data.from_structures(structures, build_X=False)
        assert md.n_vars == 0
        assert md.uns["X_is"] == "empty"


class TestLevels:
    def test_set_and_read_back(self, md):
        from matverse._core import set_level
        set_level(md, "pbe", kind="dft", method="PBE", surrogate=False,
                  license="n/a")
        info = mv.level_info(md, "pbe")
        assert info["kind"] == "dft"
        assert info["surrogate"] is False
        assert mv.levels_used(md) == ["pbe"]

    def test_missing_level_names_how_to_create_one(self, md):
        with pytest.raises(KeyError, match="mv.calc.energy"):
            mv.level_info(md, "nope")

    def test_noncommercial_licences_are_reported(self, md):
        from matverse._core import set_level
        set_level(md, "mace-omat", kind="mlip", method="MACE-OMAT-0",
                  license="ASL", surrogate=True)
        set_level(md, "mace-mpa", kind="mlip", method="MACE-MPA-0",
                  license="MIT", surrogate=True)
        assert mv.check_commercial_use(md) == ["mace-omat"]

    def test_compare_levels_lines_up_one_quantity(self, md):
        from matverse._core import set_level
        md.obs["energy_per_atom_a"] = np.arange(6, dtype=float)
        md.obs["energy_per_atom_b"] = np.arange(6, dtype=float) + 0.5
        set_level(md, "a", kind="mlip", method="A", reference="PBE")
        set_level(md, "b", kind="dft", method="B")
        df = mv.compare_levels(md, "energy_per_atom")
        assert list(df.columns) == ["a", "b"]
        assert df.attrs["levels"]["a"]["reference"] == "PBE"

    def test_compare_levels_says_what_is_missing(self, md):
        with pytest.raises(ValueError, match="band_gap_<level>"):
            mv.compare_levels(md, "band_gap", levels=["pbe"])


class TestProvenance:
    def test_construction_is_recorded(self, md):
        assert mv.provenance(md) == ["data.from_structures"]

    def test_operations_record_their_parameters(self, md):
        mv.pp.describe(md)
        assert mv.provenance(md)[-1] == "pp.describe(source='input')"

    def test_provenance_accumulates_in_order(self, md):
        mv.pp.describe(md)
        mv.pp.qc(md)
        ops = [p.split("(")[0] for p in mv.provenance(md)]
        assert ops == ["data.from_structures", "pp.describe", "pp.qc"]


class TestErrorMessages:
    def test_missing_variant_lists_what_exists(self, md):
        with pytest.raises(KeyError, match="have \\['input'\\]"):
            mv.structures(md, "relaxed_pbe")

    def test_require_names_the_producing_function(self, md):
        from matverse._core import require
        with pytest.raises(ValueError, match="mv.pp.standardize"):
            require(md, "obs", "spacegroup")


class TestRoundTrip:
    def test_h5ad_survives_a_round_trip(self, md, tmp_path):
        mv.pp.describe(md)
        mv.feat.element_stats(md)
        path = tmp_path / "md.h5ad"
        md.write_h5ad(path)

        import anndata
        back = anndata.read_h5ad(path)
        assert back.shape == md.shape
        assert list(back.var_names) == list(md.var_names)
        assert "X_element_stats" in back.obsm
        assert back.uns["provenance"][0] == "data.from_structures"

    def test_work_continues_on_a_reloaded_object(self, md, tmp_path):
        """h5ad stores a list of strings and reads it back as a numpy array,
        which has no .append — so every operation on a saved-and-reloaded
        object failed on its own provenance write. Saving is only useful if
        you can pick the object back up.
        """
        import anndata

        mv.pp.describe(md)
        path = tmp_path / "md.h5ad"
        md.write_h5ad(path)
        back = anndata.read_h5ad(path)

        mv.pp.qc(back)                       # would raise AttributeError
        mv.prop.xrd(back, two_theta=(10, 40), step=0.5)

        history = mv.provenance(back)
        assert history[0] == "data.from_structures"
        assert any(step.startswith("pp.qc") for step in history)
        assert any(step.startswith("prop.xrd") for step in history)

    def test_ase_per_atom_arrays_stay_aligned(self):
        """Per-atom data belongs to a material, so it cannot live in uns.

        uns does not subset: md[mask] would keep every record while dropping
        rows, leaving each surviving row pointing at the wrong atoms. Putting
        the arrays on the structure's own site properties makes alignment
        automatic — the same lesson structures themselves taught in v0.1.1.
        """
        from ase.build import bulk

        atoms = [bulk("Cu", "fcc", a=3.61, cubic=True),
                 bulk("Al", "fcc", a=4.05, cubic=True)]
        for i, a in enumerate(atoms):
            a.arrays["magmom"] = np.full(len(a), float(i) + 1)

        md = mv.data.from_ase(atoms)
        assert "sites" not in md.uns
        assert mv.structures(md)[0].site_properties["magmom"][0] == 1.0

        subset = md[[1]].copy()
        assert mv.structures(subset)[0].site_properties["magmom"][0] == 2.0

    def test_ase_per_atom_arrays_survive_h5ad(self, tmp_path):
        from ase.build import bulk

        atoms = [bulk("Cu", "fcc", a=3.61, cubic=True)]
        atoms[0].arrays["magmom"] = np.arange(len(atoms[0]), dtype=float)

        md = mv.data.from_ase(atoms)
        path = tmp_path / "ase.h5ad"
        md.write_h5ad(path)

        import anndata
        back = anndata.read_h5ad(path)
        assert mv.structures(back)[0].site_properties["magmom"] == [0.0, 1.0,
                                                                    2.0, 3.0]

    def test_ase_per_atom_arrays_reach_the_sites_axis(self):
        """The payoff: they need no separate plumbing to become columns."""
        from ase.build import bulk

        atoms = [bulk("Cu", "fcc", a=3.61, cubic=True)]
        atoms[0].arrays["magmom"] = np.arange(len(atoms[0]), dtype=float)

        sites = mv.multi.sites(mv.data.from_ase(atoms))
        assert "site_magmom" in sites.obs
        assert list(sites.obs["site_magmom"]) == [0.0, 1.0, 2.0, 3.0]

    def test_matminer_dataframe_round_trip(self, md):
        mv.pp.describe(md)
        df = mv.data.to_matminer(md)
        assert "structure" in df.columns and "formula" in df.columns
        again = mv.data.from_matminer(df)
        assert again.n_obs == md.n_obs
        assert set(again.var_names) == set(md.var_names)

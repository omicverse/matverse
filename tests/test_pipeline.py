"""The screening pipeline, end to end, on a real calculator."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


class TestPreprocess:
    def test_standardize_deposits_variants_and_symmetry(self, md):
        mv.pp.standardize(md)
        assert {"primitive", "conventional"} <= set(md.obsm["structures"].columns)
        assert md.obs["spacegroup_number"].iloc[0] > 0
        assert md.obs["crystal_system"].iloc[0] == "cubic"

    def test_describe_fills_the_usual_columns(self, md):
        mv.pp.describe(md)
        for col in ("formula", "nsites", "volume", "density", "n_elements"):
            assert col in md.obs
        assert md.obs["formula"].iloc[0] == "Al"
        assert md.obs["n_elements"].iloc[3] == 2

    def test_qc_flags_a_broken_structure(self, structures):
        from pymatgen.core import Lattice, Structure
        broken = Structure(Lattice.cubic(4.0), ["Cu", "Cu"],
                           [[0, 0, 0], [0.02, 0, 0]])       # 0.08 A apart
        md = mv.data.from_structures(structures + [broken])
        mv.pp.qc(md)
        assert bool(md.obs["is_valid"].iloc[0])
        assert not bool(md.obs["is_valid"].iloc[-1])
        assert "min_distance" in md.obs["qc_reason"].iloc[-1]

    def test_filter_materials_drops_and_records(self, structures):
        from pymatgen.core import Lattice, Structure
        broken = Structure(Lattice.cubic(4.0), ["Cu", "Cu"],
                           [[0, 0, 0], [0.02, 0, 0]])
        md = mv.data.from_structures(structures + [broken])
        mv.pp.qc(md)
        clean = mv.pp.filter_materials(md)
        assert clean.n_obs == md.n_obs - 1
        assert "n_dropped=1" in mv.provenance(clean)[-1]

    def test_filter_elements_prunes_the_element_axis(self, md):
        sub = md[[0, 1]].copy()                             # Al and Cu only
        pruned = mv.pp.filter_elements(sub, min_materials=1)
        assert set(pruned.var_names) == {"Al", "Cu"}
        assert list(pruned.var["n_materials"]) == [1, 1]

    def test_normalize_composition_makes_rows_sum_to_one(self, md):
        mv.pp.normalize_composition(md)
        totals = np.asarray(md.layers["fraction"].sum(axis=1)).ravel()
        assert np.allclose(totals, 1.0)

    def test_dedup_finds_a_repeated_structure(self, structures):
        md = mv.data.from_structures(structures + [structures[0].copy()])
        mv.pp.dedup(md)
        assert md.uns["dedup"]["n_duplicates"] == 1
        assert bool(md.obs["is_duplicate"].iloc[-1])
        assert md.obs["duplicate_of"].iloc[-1] == "0"

    def test_supercell_and_rattle_deposit_variants(self, md):
        mv.pp.supercell(md, [2, 1, 1])
        mv.pp.rattle(md, stdev=0.02, seed=0)
        assert "supercell_2x1x1" in md.obsm["structures"].columns
        assert "rattled" in md.obsm["structures"].columns
        big = mv.structures(md, "supercell_2x1x1")
        assert len(big[0]) == 2 * len(mv.structures(md, "input")[0])

    def test_a_supercell_does_not_move_in_chemical_space(self, md):
        """Composition comes from the reduced formula, so it is cell-independent."""
        mv.pp.supercell(md, [2, 2, 2])
        again = mv.data.from_structures(mv.structures(md, "supercell_2x2x2"))
        assert np.allclose(again.X.toarray(), md.X.toarray())


class TestFeatures:
    def test_element_stats_is_a_product_of_X_and_var(self, md):
        mv.feat.element_stats(md, properties=["electronegativity"],
                              statistics=("mean", "min", "max"))
        block = md.obsm["X_element_stats"]
        assert block.shape == (6, 3)
        names = md.uns["features"]["X_element_stats"]["names"]
        assert names == ["electronegativity_mean", "electronegativity_min",
                         "electronegativity_max"]
        # Pure Al: every statistic is Al's own value.
        assert block[0, 0] == pytest.approx(1.61, abs=1e-6)
        assert block[0, 1] == pytest.approx(block[0, 2])

    def test_weighted_mean_respects_stoichiometry(self, md):
        mv.feat.element_stats(md, properties=["electronegativity"],
                              statistics=("mean",))
        al, cu = 1.61, 1.90
        # Row 3 is CuAl3: three parts Al to one part Cu.
        assert md.obsm["X_element_stats"][3, 0] == pytest.approx(
            0.75 * al + 0.25 * cu, abs=1e-6)

    def test_block_width_matches_its_recorded_names(self, md):
        mv.feat.element_stats(md)
        names = md.uns["features"]["X_element_stats"]["names"]
        block = md.obsm["X_element_stats"]
        assert block.shape[1] == len(names)
        assert np.isfinite(block).any()

    def test_similarity_is_symmetric_with_unit_diagonal(self, md):
        mv.feat.element_stats(md)
        mv.feat.similarity(md)
        S = md.obsp["similarity_X_element_stats"]
        assert np.allclose(np.diag(S), 1.0)
        assert np.allclose(S, S.T)


class TestCalc:
    def test_energy_deposits_a_level(self, md):
        mv.calc.energy(md, level="emt")
        assert np.isfinite(md.obs["energy_emt"]).all()
        info = mv.level_info(md, "emt")
        assert info["kind"] == "classical" and info["surrogate"] is True
        assert info["n_failed"] == 0

    def test_relax_lowers_the_energy_and_names_the_variant(self, md):
        mv.calc.energy(md, level="emt")
        before = md.obs["energy_emt"].to_numpy(dtype=float).copy()
        mv.calc.relax(md, level="emt", fmax=0.1, steps=50)
        after = md.obs["energy_emt"].to_numpy(dtype=float)
        assert "relaxed_emt" in md.obsm["structures"].columns
        assert (after <= before + 1e-6).all()
        assert md.obs["relax_converged_emt"].all()

    def test_unknown_level_says_what_is_available(self, md):
        with pytest.raises(KeyError, match="register_calculator"):
            mv.calc.energy(md, level="not-a-model")

    def test_available_reports_why_a_backend_is_missing(self):
        table = mv.calc.available()
        assert table["emt"]["method"] == "EMT"
        assert "unavailable" in table["mace-omat"] or \
            table["mace-omat"]["license"] == "ASL"

    def test_registering_a_calculator_records_its_metadata(self, md):
        from ase.calculators.emt import EMT
        mv.calc.register_calculator("house", EMT, kind="mlip",
                                    method="HouseFF", reference="r2SCAN",
                                    license="ASL")
        mv.calc.energy(md, level="house")
        assert mv.level_info(md, "house")["reference"] == "r2SCAN"
        assert mv.calc.check_licenses(md) == ["house"]

    def test_committee_reports_a_spread(self, md):
        from ase.calculators.emt import EMT
        mv.calc.register_calculator("emt2", EMT, method="EMT again")
        mv.calc.energy(md, level="emt")
        mv.calc.energy(md, level="emt2")
        mv.calc.committee(md, ["emt", "emt2"], key="ens")
        assert np.allclose(md.obs["energy_per_atom_ens_std"], 0.0)
        assert mv.level_info(md, "ens")["uncertainty"].startswith("committee")


class TestThermo:
    def test_closed_hull_warns_that_it_is_relative(self, md):
        mv.calc.energy(md, level="emt")
        with pytest.warns(UserWarning, match="closed_system"):
            mv.thermo.hull(md, level="emt")
        assert md.uns["phase_diagram"]["closed_system"] is True
        assert md.uns["phase_diagram"]["built"] is True

    def test_hull_reports_distance_and_decomposition(self, md):
        mv.calc.relax(md, level="emt", fmax=0.1, steps=50)
        with pytest.warns(UserWarning):
            mv.thermo.hull(md, level="emt", source="relaxed_emt")
        above = md.obs["e_above_hull_emt"].to_numpy(dtype=float)
        assert np.isfinite(above).all()
        assert (above >= -1e-6).all()
        assert md.obs["is_stable_emt"].any()
        assert md.obs["decomposes_to_emt"].str.len().gt(0).any()

    def test_elementals_anchor_the_hull(self, md):
        """With Al, Cu and Ni present, formation energies are well defined."""
        mv.calc.energy(md, level="emt")
        with pytest.warns(UserWarning):
            mv.thermo.hull(md, level="emt")
        assert md.uns["phase_diagram"]["has_elemental_references"] is True
        assert np.isfinite(md.obs["formation_energy_emt"]).all()

    def test_hull_needs_energies_first(self, md):
        with pytest.raises(ValueError, match="mv.calc.energy"):
            mv.thermo.hull(md, level="emt")

    def test_mixing_levels_is_refused(self, md, make_md):
        from matverse.thermo import LevelMismatch
        mv.calc.energy(md, level="emt")
        md.uns["levels"]["emt"]["reference"] = "PBE"

        ref = make_md()
        mv.calc.energy(ref, level="emt")
        ref.uns["levels"]["emt"]["reference"] = "r2SCAN"

        with pytest.raises(LevelMismatch, match="not a hull of anything"):
            mv.thermo.hull(md, level="emt", references=ref)

    def test_references_make_the_hull_absolute(self, md, make_md):
        mv.calc.energy(md, level="emt")
        ref = make_md()
        mv.calc.energy(ref, level="emt")
        mv.thermo.hull(md, level="emt", references=ref)
        assert md.uns["phase_diagram"]["closed_system"] is False
        assert md.uns["phase_diagram"]["n_references"] == 6


class TestScreen:
    def test_filter_deposits_criteria_not_a_shorter_list(self, md):
        mv.pp.describe(md)
        mv.screen.filter(md, n_elements__le=1, name="elemental")
        assert md.n_obs == 6
        assert md.uns["screens"]["elemental"]["n_pass"] == 3
        assert md.uns["screens"]["elemental"]["criteria"] == {"n_elements__le": 1}

    def test_nan_never_passes(self, md):
        md.obs["value"] = [1.0, np.nan, 1.0, np.nan, 1.0, 1.0]
        mv.screen.filter(md, value__lt=2.0)
        assert md.obs["passes"].sum() == 4

    def test_rank_puts_the_best_first(self, md):
        md.obs["value"] = [5.0, 1.0, 3.0, 2.0, 4.0, np.nan]
        mv.screen.rank(md, by="value")
        assert md.obs["rank"].iloc[1] == 1.0
        assert np.isnan(md.obs["rank"].iloc[5])

    def test_pareto_finds_the_non_dominated_set(self, md):
        md.obs["a"] = [1.0, 2.0, 3.0, 1.0, 5.0, 9.0]
        md.obs["b"] = [3.0, 2.0, 1.0, 1.0, 5.0, 9.0]
        mv.screen.pareto(md, {"a": "min", "b": "min"})
        assert bool(md.obs["pareto"].iloc[3])       # (1, 1) dominates everything
        assert not bool(md.obs["pareto"].iloc[5])   # (9, 9) is dominated
        assert md.uns["pareto"]["pareto"]["n_optimal"] == 1

    def test_pareto_rejects_a_bad_sense(self, md):
        md.obs["a"] = np.ones(6)
        with pytest.raises(ValueError, match="'min' or 'max'"):
            mv.screen.pareto(md, {"a": "smallest"})


class TestTools:
    def test_pca_of_chemical_space(self, md):
        mv.pp.normalize_composition(md)
        mv.tl.pca(md, n_comps=2)
        assert md.obsm["X_pca"].shape == (6, 2)
        assert md.uns["pca"]["variance_ratio"].sum() <= 1.0 + 1e-9

    def test_neighbors_and_kmeans(self, md):
        mv.pp.normalize_composition(md)
        mv.tl.pca(md, n_comps=2)
        mv.tl.neighbors(md, n_neighbors=3)
        assert md.obsp["connectivities"].shape == (6, 6)
        mv.tl.cluster(md, method="kmeans", n_clusters=2)
        assert md.obs["cluster"].nunique() == 2

    def test_rank_elements_groups_recovers_the_obvious_chemistry(self, md):
        """Group the Al-containing materials and ask what distinguishes them.

        The answer must be Al. If this fails, making X the composition matrix
        bought nothing and the design should fall back to an empty X.
        """
        has_al = np.asarray(md[:, "Al"].X.todense()).ravel() > 0
        md.obs["has_al"] = ["yes" if v else "no" for v in has_al]
        mv.tl.rank_elements_groups(md, "has_al")

        top = md.uns["rank_elements_groups"]["yes"].iloc[0]
        assert top["element"] == "Al"
        assert top["frac_in_group"] == 1.0
        assert top["pval"] < 0.1

    def test_rank_elements_groups_by_amount(self, md):
        md.obs["group"] = ["a", "b", "b", "a", "a", "b"]
        mv.tl.rank_elements_groups(md, "group", method="fraction")
        frame = md.uns["rank_elements_groups"]["a"]
        assert {"mean_frac_in_group", "zscore", "qval"} <= set(frame.columns)
        assert len(frame) == md.n_vars

    def test_rank_elements_groups_needs_a_real_column(self, md):
        with pytest.raises(ValueError, match="absent"):
            mv.tl.rank_elements_groups(md, "not_a_column")

    def test_novelty_scores_zero_against_itself(self, md, make_md):
        mv.tl.novelty(md, reference=make_md())
        assert np.allclose(md.obs["novelty_distance"], 0.0)

    def test_novelty_is_positive_for_unseen_chemistry(self, md):
        from pymatgen.core import Lattice, Structure
        new = Structure(Lattice.cubic(3.9), ["Pd"] * 4,
                        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        candidates = mv.data.from_structures([new])
        mv.tl.novelty(candidates, reference=md)
        assert candidates.obs["novelty_distance"].iloc[0] > 0.5


class TestFullRun:
    def test_screen_end_to_end(self, md):
        mv.pp.standardize(md)
        mv.pp.describe(md)
        mv.pp.qc(md)
        md = mv.pp.filter_materials(md)
        mv.feat.element_stats(md)
        mv.calc.relax(md, level="emt", fmax=0.1, steps=50)
        with pytest.warns(UserWarning):
            mv.thermo.hull(md, level="emt", source="relaxed_emt")
        mv.screen.filter(md, e_above_hull_emt__lt=0.05, n_elements__le=2)
        mv.tl.rank_elements_groups(md, "passes")

        assert md.uns["screens"]["passes"]["n_total"] == 6
        assert md.n_obs == 6                        # a screen keeps its rejects
        shortlist = md[md.obs["passes"]]
        assert shortlist.n_obs == md.uns["screens"]["passes"]["n_pass"]

        ops = [p.split("(")[0] for p in mv.provenance(md)]
        assert ops == ["data.from_structures", "pp.standardize", "pp.describe",
                       "pp.qc", "pp.filter_materials", "feat.element_stats",
                       "calc.relax", "thermo.hull", "screen.filter",
                       "tl.rank_elements_groups"]

    def test_the_whole_object_still_saves(self, md, tmp_path):
        mv.pp.standardize(md)
        mv.pp.describe(md)
        mv.feat.element_stats(md)
        mv.calc.relax(md, level="emt", fmax=0.2, steps=20)
        mv.pp.normalize_composition(md)
        mv.tl.pca(md, n_comps=2)
        mv.screen.filter(md, energy_emt__lt=100.0)

        path = tmp_path / "run.h5ad"
        md.write_h5ad(path)

        import anndata
        back = anndata.read_h5ad(path)
        assert "relaxed_emt" in back.obsm["structures"].columns
        assert back.uns["levels"]["emt"]["method"] == "EMT"
        assert len(mv.structures(back, "relaxed_emt")) == 6

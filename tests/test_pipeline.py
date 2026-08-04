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


def _spec(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


class TestLevelsOfTheory:
    """What each level claims about itself.

    Not backend-free, though the first version of this assumed it was: the
    metadata is built inside _builtin alongside the import, so a level whose
    backend is absent appears as {'unavailable': ...} rather than with its
    fields. Every check here therefore skips the levels this environment
    cannot construct, and asserts that it checked something.
    """

    @staticmethod
    def _present(names):
        levels = mv.calc.available()
        found = {n: levels[n] for n in names
                 if n in levels and "unavailable" not in levels[n]}
        if not found:
            pytest.skip(f"no backend installed for any of {names}")
        return found

    def test_the_dft_levels_are_not_surrogates(self):
        """The whole point of the kind/surrogate fields. GPAW solves the
        Kohn-Sham equations; everything else here reproduces something that
        did, and a screen that cannot tell them apart cannot report what
        produced a number."""
        for entry in self._present(("gpaw-pbe", "gpaw-pbe-fast")).values():
            assert entry["kind"] == "dft"
            assert entry["surrogate"] is False
            assert entry["reference"] == "PBE"

    def test_the_surrogates_say_so(self):
        found = self._present(("m3gnet", "tensornet", "orb", "mace-mpa",
                               "chgnet"))
        for entry in found.values():
            assert entry["surrogate"] is True
            assert entry["kind"] == "mlip"

    def test_the_reference_functional_is_recorded_and_differs(self):
        """Mixing a PBE surrogate with an r2SCAN one is the same class of
        error as mixing PBE with HSE06. On silicon the two M3GNet checkpoints
        differ by 6 eV for the same cell, so this is not a nicety."""
        found = self._present(("m3gnet", "m3gnet-r2scan"))
        if len(found) < 2:
            pytest.skip("both matgl checkpoints are needed to compare them")
        assert "PBE" in found["m3gnet"]["reference"]
        assert "r2SCAN" in found["m3gnet-r2scan"]["reference"]
        assert found["m3gnet"]["reference"] != \
            found["m3gnet-r2scan"]["reference"]

    def test_the_dft_levels_record_what_decides_convergence(self):
        """A DFT number without its cutoff and k-mesh is not reproducible."""
        for entry in self._present(("gpaw-pbe", "gpaw-pbe-fast")).values():
            assert entry["plane_wave_cutoff_eV"] > 0
            assert len(entry["kpoint_mesh"]) == 3

    def test_the_gpaw_mesh_is_fixed_not_a_density(self):
        """A density-based mesh changes discretely with cell size, which puts
        a step in E(V). On silicon that gave a bulk modulus of -879 GPa at
        density 2.0 and 256 GPa at 2.5, against 88.7 at a fixed mesh."""
        for entry in self._present(("gpaw-pbe", "gpaw-pbe-fast")).values():
            assert "kpoint_density_per_inv_angstrom" not in entry
            assert all(isinstance(k, int) for k in entry["kpoint_mesh"])


@pytest.mark.skipif(not _spec("matgl"), reason="matgl is an optional extra")
class TestMatglStressUnit:
    def test_the_stress_is_in_ase_units_not_gigapascals(self):
        """matgl returns stress in GPa by default while ASE's contract is
        eV/A^3. Left alone it makes every stress-derived quantity exactly
        160.2x too large, with no error raised anywhere — the factor is the
        GPa-to-eV/A^3 conversion, which is what makes it recognisable."""
        from ase.build import bulk
        from matverse.calc import _builtin
        factory, _ = _builtin("tensornet")
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.calc = factory()
        stress = abs(np.asarray(atoms.get_stress())).max()
        # Silicon near equilibrium: hundredths of eV/A^3, not units of it.
        assert stress < 0.5, f"stress {stress} looks like GPa, not eV/A^3"


@pytest.mark.skipif(not _spec("gpaw"), reason="GPAW is an optional extra")
class TestRealDFT:
    """One genuine Kohn-Sham calculation, small enough for CI.

    The crystal validations are too slow to run here — a silicon equation of
    state takes about eight minutes — so those numbers live in the notes as
    measured values: a relaxed lattice constant of 5.479 A against a PBE
    literature 5.47, and a bulk modulus of 88.7 GPa against 88-89.
    """

    def test_it_actually_solves_something(self):
        from ase import Atoms
        from gpaw import GPAW
        molecule = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                         cell=[6, 6, 6], pbc=False)
        molecule.calc = GPAW(mode="lcao", basis="sz(dzp)", xc="PBE",
                             txt=None, h=0.25)
        energy = molecule.get_potential_energy()
        assert np.isfinite(energy)
        assert -12.0 < energy < 0.0          # bound, and not absurd
        assert np.isfinite(molecule.get_forces()).all()

    def test_the_level_reaches_matverse_as_dft(self):
        """Constructed through mv.calc's own registry rather than directly,
        so a broken factory shows up here."""
        from matverse.calc import _builtin
        factory, meta = _builtin("gpaw-pbe-fast")
        assert meta["kind"] == "dft" and meta["surrogate"] is False
        assert factory() is not None


@pytest.mark.skipif(not _spec("matplotlib"), reason="needs matplotlib")
class TestSpacegroupPlot:
    def test_it_groups_by_crystal_system(self):
        import matplotlib
        matplotlib.use("Agg")
        from matverse.pl import _crystal_system
        # Boundaries of the international convention, which is where an
        # off-by-one would sit.
        assert _crystal_system(1) == "triclinic"
        assert _crystal_system(2) == "triclinic"
        assert _crystal_system(3) == "monoclinic"
        assert _crystal_system(15) == "monoclinic"
        assert _crystal_system(16) == "orthorhombic"
        assert _crystal_system(195) == "cubic"
        assert _crystal_system(230) == "cubic"

    def test_it_plots_a_distribution(self):
        import matplotlib
        matplotlib.use("Agg")
        md = mv.datasets.metals(["Cu", "Al", "Ni"])
        mv.pp.symmetry(md)
        ax = mv.pl.spacegroups(md)
        assert ax._matverse_n_groups >= 1

    def test_it_says_what_it_left_out(self):
        import matplotlib
        matplotlib.use("Agg")
        md = mv.datasets.metals(["Cu", "Al", "Ni", "Ag", "Au"])
        mv.pp.symmetry(md)
        ax = mv.pl.spacegroups(md, top=1)
        assert ax._matverse_n_groups == 1
        assert ax._matverse_dropped >= 0

    def test_a_missing_column_names_the_fix(self):
        md = mv.datasets.metals(["Cu"])
        with pytest.raises(ValueError, match="mv.pp.symmetry"):
            mv.pl.spacegroups(md)

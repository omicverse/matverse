"""Proposing candidates from what has been made before.

Nothing here computes an energy. These functions rank substitutions, dopants and
volumes from the statistics of known compounds, which is what you use to decide
what is worth relaxing — the step before every other namespace.

The assertions are chemistry rather than stored output: fluorine is the dopant
LiFePO4 is actually doped with, LiVPO4 and LiTiPO4 are real olivines, and a
volume predicted from bond lengths lands within a few percent of the measured
cell.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def cathodes():
    md = mv.datasets.load("battery_cathodes")
    mv.pp.describe(md)
    mv.transform.oxidation_states(md)
    return md


class TestVolumePrediction:
    def test_it_lands_close_to_the_measured_cell(self, cathodes):
        """Within a few percent on three published cathodes — close enough to
        start a relaxation, nowhere near close enough to report."""
        md = cathodes.copy()
        mv.pp.predict_volume(md)
        ratio = md.obs["volume_scale"].to_numpy(dtype=float)
        assert ((ratio > 0.93) & (ratio < 1.07)).all(), ratio

    def test_the_rescaled_variant_has_the_predicted_volume(self, cathodes):
        md = cathodes.copy()
        mv.pp.predict_volume(md)
        rescaled = [s.volume for s in mv.structures(md, "rescaled")]
        assert rescaled == pytest.approx(
            md.obs["predicted_volume"].to_numpy(dtype=float), rel=1e-6)

    def test_the_input_is_left_alone(self, cathodes):
        """It deposits a variant; the cell you gave it is still there."""
        md = cathodes.copy()
        before = [s.volume for s in mv.structures(md, "input")]
        mv.pp.predict_volume(md)
        after = [s.volume for s in mv.structures(md, "input")]
        assert after == pytest.approx(before)


class TestDopantPrediction:
    @pytest.fixture(scope="class")
    def doped(self, cathodes):
        md = cathodes.copy()
        mv.gen.predict_dopants(md, source="oxidized", n=5)
        return md

    def test_fluorine_is_the_n_type_choice_for_a_phosphate(self, doped):
        """F on the oxygen site is the doping strategy actually used on
        LiFePO4, and it comes out top without being told."""
        by_name = dict(zip(doped.obs["name"], doped.obs["n_type_dopant"]))
        assert by_name["LiFePO4"] == "F-"

    def test_both_polarities_are_reported(self, doped):
        assert (doped.obs["n_type_probability"] > 0).all()
        assert (doped.obs["p_type_probability"] > 0).all()

    def test_the_full_ranking_survives_in_uns(self, doped):
        """The second and third choices are usually the interesting ones, so
        obs carries the top one and uns keeps the list."""
        ranked = doped.uns["dopants"]["ranked"]
        first = ranked[str(doped.obs_names[0])]
        assert len(first["n_type"]) > 1
        probabilities = [c["probability"] for c in first["n_type"]]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_it_needs_oxidation_states(self, cathodes):
        """n-type versus p-type is arithmetic on charge, so a neutral structure
        cannot be classified at all."""
        md = cathodes.copy()
        mv.gen.predict_dopants(md, source="input")
        assert md.uns["dopants"]["n_failed"] == md.n_obs


class TestSubstitutionPrediction:
    @pytest.fixture(scope="class")
    def candidates(self, cathodes):
        out = mv.gen.predict_substitutions(cathodes[:1].copy(),
                                           source="oxidized", n=8)
        mv.pp.describe(out)
        return out

    def test_it_proposes_real_olivines(self, candidates):
        """LiVPO4 and LiTiPO4 are known olivine analogues of LiFePO4, and the
        model reaches them from the ICSD statistics alone."""
        formulas = set(candidates.obs["formula"])
        assert {"LiVPO4", "LiTiPO4"} & formulas, formulas

    def test_the_identity_substitution_is_not_a_candidate(self, candidates):
        assert all(swap.strip() for swap in candidates.obs["substitution"])
        assert not any("Fe2+->Fe2+" in s for s in candidates.obs["substitution"])

    def test_candidates_are_ranked(self, candidates):
        probabilities = candidates.obs["substitution_probability"].to_numpy(
            dtype=float)
        assert (probabilities > 0).all()

    def test_rows_point_back_at_their_parent(self, candidates, cathodes):
        assert set(candidates.obs["parent"]) <= {str(x)
                                                 for x in cathodes.obs_names}

    def test_the_result_is_an_ordinary_dataset(self, candidates):
        """A candidate list is a materials object, so the rest of matverse
        works on it unchanged."""
        mv.pp.qc(candidates)
        assert "is_valid" in candidates.obs
        assert candidates.X.shape[0] == candidates.n_obs

    def test_it_refuses_without_oxidation_states(self, cathodes):
        md = cathodes.copy()
        with pytest.raises(ValueError, match="oxidation states"):
            mv.gen.predict_substitutions(md, source="input")


class TestJahnTeller:
    """Textbook ions, textbook answers."""

    @staticmethod
    def _perovskite(a_site, b_site, a):
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(a), [a_site, b_site, "O", "O", "O"],
                         [[0, 0, 0], [.5, .5, .5], [.5, .5, 0],
                          [.5, 0, .5], [0, .5, .5]])

    @pytest.fixture(scope="class")
    def perovskites(self):
        md = mv.data.from_structures([self._perovskite("La", "Mn", 3.9),
                                      self._perovskite("Sr", "Ti", 3.905),
                                      self._perovskite("La", "Ni", 3.85)])
        mv.pp.describe(md)
        mv.mag.jahn_teller(md)
        return md

    def test_manganite_is_active_and_strong(self, perovskites):
        """Mn3+ is d4 high spin: one electron in a doubly degenerate e_g level,
        which is why LaMnO3 is orthorhombic rather than cubic."""
        by_formula = dict(zip(perovskites.obs["formula"],
                              perovskites.obs["jahn_teller_active"]))
        strength = dict(zip(perovskites.obs["formula"],
                            perovskites.obs["jahn_teller_strength"]))
        assert bool(by_formula["LaMnO3"])
        assert strength["LaMnO3"] == "strong"

    def test_a_d0_ion_is_not_active(self, perovskites):
        """Ti4+ has no d electrons, so there is no degeneracy to lift."""
        by_formula = dict(zip(perovskites.obs["formula"],
                              perovskites.obs["jahn_teller_active"]))
        assert not bool(by_formula["SrTiO3"])

    def test_the_responsible_ion_is_named(self, perovskites):
        """Knowing a material is active is not useful without knowing which
        site is doing it."""
        species = dict(zip(perovskites.obs["formula"],
                           perovskites.obs["jahn_teller_species"]))
        assert species["LaMnO3"] == "Mn3+"
        assert species["LaNiO3"] == "Ni3+"
        assert species["SrTiO3"] == ""

    def test_the_distortion_itself_is_kept(self, perovskites):
        """obs says whether and how strongly; uns keeps the ligand bond
        lengths, which are the distortion rather than a label for it."""
        detail = perovskites.uns["jahn_teller"]["per_material"]
        first = detail[str(perovskites.obs_names[0])]
        assert first["sites"]
        assert "ligand_bond_lengths" in first["sites"][0]

    def test_a_screen_can_use_it(self, perovskites):
        mv.screen.filter(perovskites, jahn_teller_active__eq=True,
                         name="distorting")
        assert list(perovskites.obs["distorting"]) == [True, False, True]


class TestInterstitialsAndAntisites:
    """Two defect kinds that are not a site the input already has.

    A vacancy or a substitution is a site you can point at. An interstitial has
    to be located — the Voronoi construction finds the holes — and an antisite
    is the cross product of the species present, which for a quaternary is more
    combinations than anyone enumerates by hand.
    """

    @pytest.fixture(scope="class")
    def host(self):
        """B2 AlNi: two sites, two species — the smallest cell that has an
        antisite at all, so the mechanics are tested without paying for a
        168-site supercell four times."""
        from pymatgen.core import Lattice, Structure
        md = mv.data.from_structures([Structure(
            Lattice.cubic(2.89), ["Al", "Ni"], [[0, 0, 0], [.5, .5, .5]])])
        mv.pp.describe(md)
        return md

    @pytest.fixture(scope="class")
    def cathode(self):
        md = mv.datasets.load("battery_cathodes")[:1].copy()
        mv.pp.describe(md)
        return md

    def test_antisites_cover_the_species_cross_product(self, cathode):
        """LiFePO4 has four species, so every ordered pair is a candidate
        swap — and Fe on the Li site is the defect that blocks its
        one-dimensional lithium channel."""
        out = mv.pp.defects(cathode, kinds=("antisite",), min_atoms=60,
                            max_atoms=260)
        assert out.n_obs > 10
        swaps = {(r, a) for r, a in zip(out.obs["removed"], out.obs["added"])}
        assert ("Fe", "Li") in swaps
        assert all(r != a for r, a in swaps)

    def test_interstitials_add_an_atom(self, host):
        """The defining property: one more atom than the host supercell."""
        out = mv.pp.defects(host, kinds=("interstitial",),
                            interstitial_species=["Al"], min_atoms=8,
                            max_atoms=200)
        mv.pp.describe(out)
        assert out.n_obs > 0
        assert set(out.obs["added"]) == {"Al"}
        assert all(r == "" for r in out.obs["removed"])

    def test_a_vacancy_still_needs_no_extra_package(self, cathode):
        """The kinds matverse builds itself keep working, and honour the
        supercell you asked for."""
        out = mv.pp.defects(cathode, kinds=("vacancy",), supercell=(1, 1, 1))
        mv.pp.describe(out)
        parent = len(mv.structures(cathode)[0])
        assert set(out.obs["nsites"]) == {parent - 1}

    def test_the_defect_kind_is_recorded(self, host):
        out = mv.pp.defects(host, kinds=("antisite",), min_atoms=8,
                            max_atoms=200)
        assert set(out.obs["defect"]) == {"antisite"}
        assert set(out.obs["parent"]) == {str(host.obs_names[0])}

    def test_an_unknown_kind_is_refused(self, host):
        with pytest.raises(ValueError, match="unknown defect kind"):
            mv.pp.defects(host, kinds=("frenkel",))

    def test_the_result_is_an_ordinary_dataset(self, host):
        out = mv.pp.defects(host, kinds=("interstitial",),
                            interstitial_species=["Al"], min_atoms=8,
                            max_atoms=200)
        mv.pp.describe(out)
        mv.pp.qc(out)
        assert "is_valid" in out.obs


    def test_an_impossible_supercell_window_says_so(self):
        """It used to report "no defect was generated", which blames the
        chemistry for what is an argument problem."""
        md = mv.datasets.load("battery_cathodes")[:1].copy()
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="none produced a supercell"):
            mv.pp.defects(md, kinds=("antisite",), min_atoms=2, max_atoms=5)


class TestPrototype:
    """A space group says which symmetries a structure has; a prototype says
    which structure it *is*. Fm-3m covers rocksalt, fcc and half-Heusler alike.
    """

    @pytest.fixture(scope="class")
    def named(self):
        from pymatgen.core import Lattice, Structure
        md = mv.data.from_structures([
            Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64),
                                      ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]]),
            Structure.from_spacegroup("Fd-3m", Lattice.cubic(3.567), ["C"],
                                      [[0, 0, 0]]),
            Structure.from_spacegroup("F-43m", Lattice.cubic(5.65),
                                      ["Ga", "As"],
                                      [[0, 0, 0], [.25, .25, .25]]),
            Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
        ])
        mv.pp.describe(md)
        mv.pp.prototype(md)
        return md

    def test_the_strukturbericht_symbols_are_the_textbook_ones(self, named):
        assert list(named.obs["strukturbericht"]) == ["B1", "A4", "B3", "A1"]

    def test_the_minerals_are_named(self, named):
        minerals = list(named.obs["prototype_mineral"])
        assert "Halite" in minerals[0]
        assert minerals[1] == "diamond"
        assert "Zincblende" in minerals[2]

    def test_a_space_group_does_not_determine_the_prototype(self, named):
        """Rocksalt and fcc copper are both Fm-3m and are not the same
        structure — which is the reason this function exists."""
        by_symbol = dict(zip(named.obs["formula"],
                             named.obs["strukturbericht"]))
        assert by_symbol["NaCl"] != by_symbol["Cu"]

    def test_everything_matched_is_counted(self, named):
        assert named.uns["prototype"]["n_matched"] == named.n_obs
        assert named.uns["prototype"]["n_unmatched"] == 0

    def test_an_unmatched_structure_gets_no_guess(self):
        """'not in AFLOW' is a fact worth keeping distinct from a wrong label,
        so a miss is an empty string."""
        from pymatgen.core import Lattice, Structure
        odd = Structure(Lattice.cubic(9.0), ["Cu", "Al", "Ni", "Ag", "Au"],
                        [[0, 0, 0], [.13, .27, .41], [.55, .61, .07],
                         [.29, .83, .66], [.71, .19, .92]])
        md = mv.data.from_structures([odd])
        mv.pp.prototype(md)
        assert list(md.obs["prototype"]) == [""]
        assert md.uns["prototype"]["n_unmatched"] == 1


class TestSymmetry:
    """What a space group symbol implies, from the coordinates."""

    @staticmethod
    def _fcc():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    @staticmethod
    def _perovskite():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"],
                         [[0, 0, 0], [.5, .5, .5], [.5, .5, 0],
                          [.5, 0, .5], [0, .5, .5]])

    @pytest.fixture(scope="class")
    def analysed(self):
        md = mv.data.from_structures([self._fcc(), self._perovskite()])
        mv.pp.describe(md)
        mv.pp.symmetry(md)
        return md

    def test_the_crystal_system_and_point_group(self, analysed):
        assert list(analysed.obs["crystal_system"]) == ["cubic", "cubic"]
        assert list(analysed.obs["point_group"]) == ["m-3m", "m-3m"]

    def test_the_perovskite_wyckoff_set_is_the_textbook_one(self, analysed):
        """Sr at 1a, Ti at 1b, O at 3c — arrived at from the coordinates
        rather than from the label."""
        assert analysed.obs["wyckoff"].iloc[1] == "1a, 1b, 3c"
        assert analysed.obs["n_wyckoff"].iloc[1] == 3

    def test_a_close_packed_metal_has_one_distinct_site(self, analysed):
        assert analysed.obs["n_wyckoff"].iloc[0] == 1
        assert analysed.obs["wyckoff"].iloc[0] == "4a"

    def test_the_operation_count_matches_the_space_group(self, analysed):
        """Fm-3m has 192 operations in the conventional cell and Pm-3m 48."""
        assert analysed.obs["n_symmetry_operations"].iloc[0] == 192
        assert analysed.obs["n_symmetry_operations"].iloc[1] == 48

    def test_site_symmetry_separates_special_from_general(self, analysed):
        """In the perovskite Sr and Ti sit on the most symmetric sites and the
        oxygens do not; in fcc copper every site is equivalent."""
        assert analysed.obs["min_site_symmetry"].iloc[0] == \
            analysed.obs["max_site_symmetry"].iloc[0]
        assert analysed.obs["min_site_symmetry"].iloc[1] < \
            analysed.obs["max_site_symmetry"].iloc[1]

    def test_wyckoff_count_predicts_the_defect_count(self):
        """The reason to care: distinct sites are what mv.pp.defects
        enumerates, so n_wyckoff says how many vacancies to expect."""
        md = mv.data.from_structures([self._perovskite()])
        mv.pp.describe(md)
        mv.pp.symmetry(md)
        defective = mv.pp.defects(md, kinds=("vacancy",), supercell=(1, 1, 1))
        assert defective.n_obs == int(md.obs["n_wyckoff"].iloc[0])

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

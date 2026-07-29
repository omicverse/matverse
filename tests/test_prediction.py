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

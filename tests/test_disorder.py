"""Partial occupancy, ordered approximants, doping.

The configurational entropy has a closed form for an equiatomic mixture —
``k_B ln m`` per mixed site — so these tests check arithmetic against theory
rather than against stored output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")


def _mixed(occupancies, a=3.7):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [dict(occupancies)] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


@pytest.fixture(scope="module")
def mixture():
    binary = _mixed({"Cu": 0.5, "Au": 0.5}, a=3.8)
    quinary = _mixed({"Cu": 0.2, "Ni": 0.2, "Co": 0.2, "Fe": 0.2, "Mn": 0.2})
    ordered = mv.structures(mv.datasets.metals(["Cu"]))[0]
    md = mv.data.from_structures([binary, quinary, ordered])
    mv.disorder.describe(md, temperature=1500.0)
    return md


class TestDescribe:
    def test_ordered_and_disordered_are_told_apart(self, mixture):
        assert list(mixture.obs["is_ordered"]) == [False, False, True]
        assert list(mixture.obs["n_disordered_sites"]) == [4, 4, 0]

    def test_equiatomic_entropy_is_k_ln_m(self, mixture):
        """The closed form: an equiatomic mixture of m species on every site
        has ideal configurational entropy k_B ln(m) per atom."""
        entropy = mixture.obs["configurational_entropy"].to_numpy(dtype=float)
        assert entropy[0] == pytest.approx(mv.disorder.KB_EV * np.log(2),
                                           rel=1e-9)
        assert entropy[1] == pytest.approx(mv.disorder.KB_EV * np.log(5),
                                           rel=1e-9)
        assert entropy[2] == 0.0

    def test_more_components_means_more_entropy(self, mixture):
        entropy = mixture.obs["configurational_entropy"].to_numpy(dtype=float)
        assert entropy[1] > entropy[0] > entropy[2]

    def test_the_entropy_term_is_what_a_hull_leaves_out(self, mixture):
        """At a synthesis temperature the term is larger than the distance a
        screen usually calls 'close to the hull', which is why a material can
        sit above the line at 0 K and still be the phase that forms."""
        term = mixture.obs["entropy_term_300K"].to_numpy(dtype=float)
        assert term[1] == pytest.approx(-1500.0 * mv.disorder.KB_EV
                                        * np.log(5), rel=1e-9)
        assert abs(term[1]) > 0.15          # eV/atom, bigger than a 50 meV cut

    def test_site_disorder_measures_the_split(self, mixture):
        worst = mixture.obs["max_site_disorder"].to_numpy(dtype=float)
        assert worst[0] == pytest.approx(0.5)     # evenly split two ways
        assert worst[1] == pytest.approx(0.8)     # evenly split five ways
        assert worst[2] == pytest.approx(0.0)


class TestOrderings:
    @pytest.fixture(scope="class")
    def ordered(self, mixture):
        out = mv.disorder.orderings(mixture, n=3)
        mv.pp.describe(out)
        return out

    def test_the_result_is_ordinary_and_ordered(self, ordered):
        """A DFT code cannot take a fractionally occupied cell, which is the
        whole point of the function."""
        assert all(s.is_ordered for s in mv.structures(ordered))
        assert "formula" in ordered.obs

    def test_rows_point_back_at_their_parent(self, ordered, mixture):
        labels = {str(x) for x in mixture.obs_names}
        assert set(ordered.obs["parent"]) <= labels

    def test_an_already_ordered_input_passes_through_once(self, ordered):
        counts = ordered.obs["parent"].value_counts()
        assert counts["2"] == 1

    def test_the_composition_is_preserved(self, ordered):
        """An ordered approximant of Cu0.5Au0.5 is CuAu, not something else."""
        formulas = set(ordered.obs.loc[ordered.obs["parent"] == "0", "formula"])
        assert formulas == {"CuAu"}

    def test_arbitrary_ranking_is_admitted_rather_than_hidden(self, ordered):
        """Without oxidation states every Ewald energy is zero, so the order is
        arbitrary — honest for an alloy, and the object says so."""
        assert "arbitrary" in ordered.uns["orderings"]["ranking"]
        energies = ordered.obs["ewald_energy"].to_numpy(dtype=float)
        assert np.nan_to_num(energies).tolist() == [0.0] * len(energies)

    def test_it_says_it_returned_a_subset(self, ordered):
        assert "combinatorially" in ordered.uns["orderings"]["note"]
        assert ordered.uns["orderings"]["n_requested"] == 3


class TestSqs:
    def test_a_missing_backend_names_the_download(self, mixture):
        """An SQS is not the ordered ground state, so silently substituting
        mv.disorder.orderings would be the wrong answer under the right name."""
        try:
            from pymatgen.command_line.mcsqs_caller import run_mcsqs  # noqa
            import shutil
            if shutil.which("mcsqs"):
                pytest.skip("ATAT is installed here")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="mcsqs"):
            mv.disorder.sqs(mixture, scaling=1, search_time=0.05)


class TestDope:
    def test_an_empty_result_is_reported_as_a_missing_install(self):
        """pymatgen's DopingTransformation enumerates with enumlib and returns
        an empty list rather than raising when it is absent, so a doping study
        would silently produce nothing."""
        md = mv.datasets.load("oxides")[:1].copy()
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="enumlib"):
            mv.disorder.dope(md, "Nb5+", min_length=5.0)

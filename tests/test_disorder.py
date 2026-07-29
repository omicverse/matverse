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


class TestShortRangeOrder:
    """Checked against B2, where the answer is fixed by the geometry.

    In CsCl-ordered brass every nearest neighbour of a copper is a zinc, so the
    Warren-Cowley parameter for the unlike pair is exactly -1 and for the like
    pair exactly +1. The second and third shells are the other way round,
    because those sites are like. None of this is approximate — it is what the
    definition gives on that structure — which is what makes it worth testing
    against rather than against stored output.
    """

    A = 2.95

    @staticmethod
    def _b2(a):
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(a), ["Cu", "Zn"],
                         [[0, 0, 0], [.5, .5, .5]])

    @pytest.fixture
    def b2(self):
        md = mv.data.from_structures([self._b2(self.A)])
        md.obs_names = ["B2"]
        return md

    @pytest.fixture
    def solution(self):
        """A random 50/50 arrangement on the same lattice, 128 sites."""
        from pymatgen.core import Structure
        base = self._b2(self.A)
        base.make_supercell([4, 4, 4])
        rng = np.random.RandomState(0)
        pick = rng.permutation(len(base))
        half = len(base) // 2
        md = mv.data.from_structures([Structure(
            base.lattice,
            ["Cu" if pick[i] < half else "Zn" for i in range(len(base))],
            base.frac_coords)])
        md.obs_names = ["random"]
        return md

    @staticmethod
    def _frame(md, key):
        return dict(zip(md.uns["sro"][key]["pairs"], md.obsm[f"sro_{key}"][0]))

    def test_b2_first_shell_is_the_textbook_answer(self, b2):
        mv.disorder.sro(b2, key_added="s1")
        alpha = self._frame(b2, "s1")
        assert alpha["Cu-Zn"] == pytest.approx(-1.0)
        assert alpha["Zn-Cu"] == pytest.approx(-1.0)
        assert alpha["Cu-Cu"] == pytest.approx(1.0)
        assert alpha["Zn-Zn"] == pytest.approx(1.0)

    def test_the_second_shell_inverts(self, b2):
        """bcc's second shell is the six <100> sites, which in B2 are the same
        species. A cumulative radius would pool them with the eight unlike
        first neighbours and report +0.14 instead of -1."""
        mv.disorder.sro(b2, shell=2, key_added="s2")
        alpha = self._frame(b2, "s2")
        assert alpha["Cu-Cu"] == pytest.approx(-1.0)
        assert alpha["Cu-Zn"] == pytest.approx(1.0)

    def test_a_random_solution_is_random(self, solution):
        mv.disorder.sro(solution, key_added="s1")
        alpha = np.array(list(self._frame(solution, "s1").values()))
        assert np.abs(alpha).max() < 0.15, \
            "a random arrangement should give alpha near zero"

    def test_order_and_disorder_are_told_apart(self, b2, solution):
        """The single number a screen would sort on."""
        mv.disorder.sro(b2, key_added="s1")
        mv.disorder.sro(solution, key_added="s1")
        assert float(b2.obs["sro_rms_s1"].iloc[0]) > \
            10 * float(solution.obs["sro_rms_s1"].iloc[0])

    def test_the_sum_rule_holds(self, b2):
        """Summing alpha_AB over B, weighted by each B's concentration, must
        give zero for every A: the neighbours of an A atom are *some* species,
        whatever the ordering. It is the one identity that catches a
        normalisation error, and it is not something the B2 values alone
        would reveal."""
        mv.disorder.sro(b2, key_added="s1")
        alpha = self._frame(b2, "s1")
        for a in ("Cu", "Zn"):
            total = sum(0.5 * alpha[f"{a}-{b}"] for b in ("Cu", "Zn"))
            assert total == pytest.approx(0.0, abs=1e-9)

    def test_a_pure_element_has_no_order_to_report(self):
        """Every neighbour of a copper is a copper because there is nothing
        else, so P(Cu|Cu)/c_Cu is 1/1 and alpha is zero. Not NaN: the
        structure is perfectly 'random' for the one species it has."""
        from pymatgen.core import Lattice, Structure
        pure = Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        md = mv.data.from_structures([pure])
        mv.disorder.sro(md, key_added="s1")
        assert self._frame(md, "s1")["Cu-Cu"] == pytest.approx(0.0)

    def test_an_explicit_cutoff_overrides_the_shell(self, b2):
        """A cutoff past the second shell pools both, which is the +0.14 the
        shell logic exists to avoid — asserted so the override is known to
        actually override."""
        mv.disorder.sro(b2, cutoff=3.1, key_added="wide")
        alpha = self._frame(b2, "wide")
        assert alpha["Cu-Cu"] == pytest.approx(1.0 / 7.0, abs=1e-6)

    def test_the_definition_is_recorded_beside_the_numbers(self, b2):
        mv.disorder.sro(b2, key_added="s1")
        assert "1 - P(B|A)" in b2.uns["sro"]["s1"]["definition"]
        assert b2.uns["sro"]["s1"]["shell"] == 1

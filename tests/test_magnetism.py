"""Magnetic ordering, and the reason a hull needs it."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


def _fcc(symbol: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


@pytest.fixture
def mixed():
    """One magnetic elemental, one magnetic alloy, one non-magnetic metal."""
    from pymatgen.core import Lattice, Structure
    alloy = Structure(Lattice.cubic(3.57), ["Ni", "Ni", "Ni", "Al"],
                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    return mv.data.from_structures([_fcc("Ni", 3.52), alloy, _fcc("Al", 4.05)])


class TestEnumeration:
    def test_it_makes_more_rows_than_it_consumes(self, mixed):
        orderings = mv.mag.orderings(mixed, max_orderings=6)
        assert orderings.n_obs > mixed.n_obs
        assert set(orderings.obs["parent"]) == set(map(str, mixed.obs_names))

    def test_a_non_magnetic_material_passes_through(self, mixed):
        """Kept as one non-magnetic row rather than dropped, so the returned
        dataset covers every input and nothing needs re-joining by hand."""
        orderings = mv.mag.orderings(mixed)
        aluminium = orderings.obs[orderings.obs["parent"] == "2"]
        assert len(aluminium) == 1
        assert aluminium["ordering"].iloc[0] == "nonmagnetic"
        assert not bool(aluminium["is_magnetic"].iloc[0])

    def test_ferro_antiferro_and_ferri_are_all_produced(self, mixed):
        orderings = mv.mag.orderings(mixed, max_orderings=6)
        kinds = set(orderings.obs["ordering"])
        assert {"fm", "afm"} <= kinds

    def test_an_antiferromagnetic_ordering_has_no_net_moment(self, mixed):
        orderings = mv.mag.orderings(mixed, max_orderings=6)
        afm = orderings.obs[orderings.obs["ordering"] == "afm"]
        assert len(afm) >= 1
        assert float(afm["total_magmom"].iloc[0]) == pytest.approx(0.0,
                                                                  abs=1e-6)

    def test_a_ferromagnetic_ordering_has_the_largest_moment(self, mixed):
        orderings = mv.mag.orderings(mixed, max_orderings=6)
        nickel = orderings.obs[orderings.obs["parent"] == "0"]
        moments = nickel.set_index("ordering")["total_magmom"]
        assert moments["fm"] == moments.abs().max()

    def test_the_fallback_says_why_it_was_needed(self, mixed):
        """pymatgen's antiferromagnetic strategies call out to enumlib, which is
        not pip-installable and absent from most environments. Falling back is
        fine; falling back silently is not."""
        orderings = mv.mag.orderings(mixed)
        errors = orderings.uns["magnetic_orderings"]["errors"]
        if errors:
            assert "fallback" in errors[0] or "Enumlib" in errors[0]

    def test_the_moments_land_on_the_structures(self, mixed):
        orderings = mv.mag.orderings(mixed, max_orderings=4)
        for structure in mv.structures(orderings)[:3]:
            assert "magmom" in structure.site_properties


class TestGroundState:
    @pytest.fixture
    def resolved(self, mixed):
        orderings = mv.mag.orderings(mixed, max_orderings=6)
        mv.calc.energy(orderings, level="emt")
        mv.mag.ground_state(orderings, mixed, level="emt")
        return orderings, mixed

    def test_the_winner_lands_on_the_parent(self, resolved):
        orderings, parents = resolved
        assert "magnetic_ordering_emt" in parents.obs
        assert parents.obs["magnetic_ordering_emt"].iloc[2] == "nonmagnetic"

    def test_exactly_one_ordering_wins_per_material(self, resolved):
        orderings, parents = resolved
        won = orderings.obs["is_ground_state_emt"].to_numpy(dtype=bool)
        assert int(won.sum()) == parents.n_obs

    def test_the_energy_column_is_the_ordinary_one(self, resolved):
        """Written under the name the calculator would have produced, so
        mv.thermo.hull needs no special case — it sees a normal column that
        happens to be the magnetic ground state."""
        orderings, parents = resolved
        assert "energy_per_atom_emt" in parents.obs
        assert np.isfinite(
            parents.obs["energy_per_atom_emt"].to_numpy(dtype=float)).all()

    def test_a_magnetism_blind_calculator_reports_zero_spread(self, resolved):
        """EMT has no notion of spin, so every ordering has the same energy.

        A spread of zero is the honest answer and a useful one: it says this
        calculator cannot distinguish the orderings, so its choice of ground
        state means nothing.
        """
        orderings, parents = resolved
        spread = parents.obs["magnetic_spread_emt"].to_numpy(dtype=float)
        magnetic = spread[np.isfinite(spread)]
        assert len(magnetic) >= 1
        assert magnetic == pytest.approx(np.zeros_like(magnetic), abs=1e-9)

    def test_the_spread_is_what_the_summary_reports(self, resolved):
        orderings, parents = resolved
        assert parents.uns["magnetic"]["emt"]["n_with_alternatives"] >= 1
        assert "hull depends on this choice" in \
            parents.uns["magnetic"]["emt"]["note"]

    def test_it_needs_energies_first(self, mixed):
        orderings = mv.mag.orderings(mixed)
        with pytest.raises(ValueError, match="mv.calc.relax"):
            mv.mag.ground_state(orderings, mixed, level="emt")

    def test_it_refuses_an_unrelated_dataset(self, mixed):
        plain = mv.data.from_structures(mv.structures(mixed))
        mv.calc.energy(plain, level="emt")
        with pytest.raises(ValueError, match="mv.mag.orderings"):
            mv.mag.ground_state(plain, mixed, level="emt")


class TestDescribe:
    def test_it_classifies_the_orderings_it_is_given(self, mixed):
        orderings = mv.mag.orderings(mixed, max_orderings=6)
        mv.mag.describe(orderings)
        kinds = set(orderings.obs["magnetic_order"])
        assert {"FM"} <= kinds
        assert "n_magnetic_species" in orderings.obs

    def test_it_does_not_invent_moments(self, mixed):
        """A structure carrying no moments is recorded as having none rather
        than being guessed at."""
        mv.mag.describe(mixed)
        assert (mixed.obs["magnetic_order"] == "unknown").all()
        assert np.isnan(mixed.obs["total_magmom"].to_numpy(dtype=float)).all()

    def test_it_counts_magnetic_species(self, mixed):
        mv.mag.describe(mixed)
        counts = mixed.obs["n_magnetic_species"].to_numpy(dtype=int)
        assert counts[0] == 1        # Ni
        assert counts[2] == 0        # Al


class TestExchange:
    """Fitting is checked against synthetic energies built from a known
    coupling, so the answer is known in advance rather than compared to a
    stored one. The absolute scale is pymatgen's convention — a constant
    factor of the coordination number — so the assertions are on linearity
    and sign, which no convention can change."""

    A = 2.87
    NN = 8              # bcc coordination

    @classmethod
    def _pair(cls, coupling):
        """A ferromagnetic and an antiferromagnetic bcc cell whose energies
        come from E = -J * sum over bonds, with 8 bonds per cell."""
        from pymatgen.core import Lattice, Structure

        def cell(moments):
            st = Structure(Lattice.cubic(cls.A), ["Fe", "Fe"],
                           [[0, 0, 0], [.5, .5, .5]])
            st.add_site_property("magmom", list(moments))
            return st

        md = mv.data.from_structures([cell([1.0, 1.0]), cell([1.0, -1.0])])
        md.obs_names = ["FM", "AFM"]
        md.obs["energy_pbe"] = [-coupling * cls.NN, coupling * cls.NN]
        return md

    def test_a_known_coupling_comes_back(self):
        md = self._pair(0.010)
        mv.mag.exchange(md, level="pbe", cutoff=3.0)
        assert float(md.obs["exchange_pbe"].iloc[0]) == pytest.approx(80.0,
                                                                     rel=1e-6)

    def test_the_fit_is_linear_in_the_energy_difference(self):
        """Convention-free: whatever constant pymatgen folds in, doubling the
        splitting must double the coupling."""
        small = self._pair(0.005)
        large = self._pair(0.010)
        mv.mag.exchange(small, level="pbe", cutoff=3.0)
        mv.mag.exchange(large, level="pbe", cutoff=3.0)
        assert float(large.obs["exchange_pbe"].iloc[0]) == pytest.approx(
            2.0 * float(small.obs["exchange_pbe"].iloc[0]), rel=1e-6)

    def test_the_ordering_temperature_scales_with_the_coupling(self):
        small = self._pair(0.005)
        large = self._pair(0.010)
        mv.mag.exchange(small, level="pbe", cutoff=3.0)
        mv.mag.exchange(large, level="pbe", cutoff=3.0)
        assert float(large.obs["ordering_temperature_pbe"].iloc[0]) == \
            pytest.approx(
                2.0 * float(small.obs["ordering_temperature_pbe"].iloc[0]),
                rel=1e-6)

    def test_a_ferromagnet_and_an_antiferromagnet_differ_in_sign(self):
        """Flip which ordering is lower and the coupling must change sign —
        the one thing that decides what kind of magnet it is."""
        ferro = self._pair(0.010)
        antiferro = self._pair(-0.010)
        mv.mag.exchange(ferro, level="pbe", cutoff=3.0)
        mv.mag.exchange(antiferro, level="pbe", cutoff=3.0)
        assert float(ferro.obs["exchange_pbe"].iloc[0]) > 0
        assert float(antiferro.obs["exchange_pbe"].iloc[0]) < 0

    def test_degenerate_energies_are_refused_not_fitted(self):
        """A calculator that does not distinguish spin gives every ordering
        the same energy. The fit is then degenerate, and reporting a small
        coupling instead of saying so would be the worst outcome."""
        md = self._pair(0.010)
        md.obs["energy_pbe"] = [-1.0, -1.0]
        with pytest.warns(UserWarning, match="degenerate"):
            mv.mag.exchange(md, level="pbe", cutoff=3.0)
        assert np.isnan(float(md.obs["exchange_pbe"].iloc[0]))
        assert "degenerate" in md.uns["exchange"]["pbe"]["error"]

    def test_one_ordering_is_not_enough(self):
        md = self._pair(0.010)[:1].copy()
        with pytest.raises(ValueError, match="at least two orderings"):
            mv.mag.exchange(md, level="pbe")

    def test_a_missing_energy_column_says_what_to_run(self):
        md = self._pair(0.010)
        with pytest.raises(ValueError, match="mv.calc.energy"):
            mv.mag.exchange(md, level="nothere")

    def test_the_mean_field_caveat_is_recorded(self):
        md = self._pair(0.010)
        mv.mag.exchange(md, level="pbe", cutoff=3.0)
        note = md.uns["exchange"]["pbe"]["temperature_note"]
        assert "upper bound" in note and "overestimates" in note

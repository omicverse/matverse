"""mv.mech — relaxed-ion stiffness, cleavage cuts, derivative checks, folds."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv

ase = pytest.importorskip("ase")
from ase.build import bulk  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402


def _equilibrium_fcc_cu():
    """EMT Cu at its own lattice constant (minimum of E(a))."""
    a_grid = np.linspace(3.55, 3.65, 11)
    e = []
    for a in a_grid:
        at = bulk("Cu", "fcc", a=a, cubic=True)
        at.calc = EMT()
        e.append(at.get_potential_energy())
    c = np.polyfit(a_grid, e, 2)
    a0 = -c[1] / (2 * c[0])
    at = bulk("Cu", "fcc", a=a0, cubic=True)
    at.calc = EMT()
    return at


class TestElasticConstants:
    def test_cubic_form_and_bravais_relaxation_is_a_noop(self):
        at = _equilibrium_fcc_cu()
        res = mv.mech.elastic_constants(at, EMT(), also_clamped=True)
        assert res["C"].shape == (6, 6)
        assert np.allclose(res["C"], res["C"].T)
        assert res["C11"] > res["C12"] > 0 and res["C44"] > 0
        assert res["max_deviation_from_cubic"] < 0.02 * res["C11"]
        assert np.abs(res["residual_stress"]).max() < 0.5        # GPa
        # one atom per primitive cell: nothing to relax internally
        assert res["C44"] == pytest.approx(res["clamped"]["C44"], rel=1e-2)

    def test_multilattice_relaxed_is_softer_than_clamped(self):
        # hcp has two atoms per primitive cell and no inversion centre at the
        # atoms, so some strains drive an inner displacement
        at = bulk("Cu", "hcp", a=2.55, c=4.16)
        res = mv.mech.elastic_constants(at, EMT(), also_clamped=True,
                                        fmax=1e-6)
        diff = res["clamped"]["C"] - res["C"]
        eig = np.linalg.eigvalsh(0.5 * (diff + diff.T))
        assert eig.min() > -0.5                     # relaxed never stiffer
        assert eig.max() > 0.5                      # and genuinely softer
        assert res["all_inner_relaxations_converged"]

    def test_needs_a_calculator(self):
        at = bulk("Cu", "fcc", a=3.6, cubic=True)
        with pytest.raises(ValueError):
            mv.mech.elastic_constants(at)


class TestCleavage:
    def test_fcc_111_single_cut_positive_and_relaxation_lowers(self):
        at = _equilibrium_fcc_cu()
        cuts = mv.mech.cleavage_energies(at, (1, 1, 1), EMT(),
                                         min_thickness=12.0, vacuum=10.0)
        # every (111) gap in fcc is the same: one inequivalent cut
        assert len(cuts) == 1
        c = cuts[0]
        assert 0.3 < c["gamma_unrelaxed"] < 3.0
        assert c["gamma_relaxed"] <= c["gamma_unrelaxed"] + 1e-6
        assert len(c["slab"]) == c["n_atoms"]

    def test_diamond_111_has_two_cuts_and_the_wide_gap_is_lower(self):
        manybody = pytest.importorskip("matscipy.calculators.manybody")
        from matscipy.calculators.manybody.explicit_forms.stillinger_weber \
            import StillingerWeber, Stillinger_Weber_PRB_31_5262_Si
        calc = manybody.Manybody(**StillingerWeber(
            Stillinger_Weber_PRB_31_5262_Si))
        si = bulk("Si", "diamond", a=5.431, cubic=True)
        cuts = mv.mech.cleavage_energies(si, (1, 1, 1), calc,
                                         min_thickness=15.0, relax=False)
        assert len(cuts) == 2
        assert cuts[0]["gap"] > cuts[1]["gap"]
        assert cuts[0]["gamma_unrelaxed"] < cuts[1]["gamma_unrelaxed"]


class TestCheckGradient:
    def test_correct_gradient_passes_and_wrong_one_fails(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(5, 5)); A = A @ A.T
        x = rng.normal(size=5)
        E = lambda v: 0.5 * v @ A @ v + np.sin(v).sum()   # noqa: E731
        good = A @ x + np.cos(x)
        ok = mv.mech.check_gradient(E, x, analytic=good, n_samples=5, h=1e-5)
        assert ok["max_rel_err"] < 1e-6
        bad = mv.mech.check_gradient(E, x, analytic=good * 1.05, n_samples=5)
        assert bad["max_rel_err"] > 1e-2

    def test_force_sign(self):
        x = np.array([[0.3, -0.2, 0.1]])
        E = lambda v: float((v ** 2).sum())                # noqa: E731
        forces = -2 * x
        rep = mv.mech.check_gradient(E, x, analytic=forces, sign=-1.0,
                                     n_samples=3)
        assert rep["max_rel_err"] < 1e-6


class TestBracketFold:
    """Cusp model: equilibria of E = x^4/4 - x^2/2 - k x, folds at
    k = +-2/(3*sqrt(3))."""

    K_FOLD = 2.0 / (3.0 * np.sqrt(3.0))

    @staticmethod
    def solve(k, x):
        """Local minimisation from x: Newton where the curvature is positive,
        a descent step otherwise (so it can only reach a minimum)."""
        x = float(np.asarray(x).ravel()[0])
        for _ in range(500):
            g, h = x ** 3 - x - k, 3 * x ** 2 - 1
            x -= g / h if h > 1e-3 else 0.2 * g
            if abs(g) < 1e-13:
                break
        return np.array([x]), bool(abs(x ** 3 - x - k) < 1e-8 and 3 * x ** 2 > 1)

    def test_upper_and_lower_fold_of_one_branch(self):
        up = mv.mech.bracket_fold(self.solve, 0.0, np.array([-1.0]), step=0.05,
                                  jump=0.3, resolution=1e-3, direction=+1)
        assert up["found"] and up["width"] <= 1e-3
        assert up["trapped_load"] < self.K_FOLD < up["detrapped_load"]
        assert up["fold"] == pytest.approx(self.K_FOLD, abs=2e-3)
        assert up["detrapped_state"][0] > 0.5         # snapped to other branch

        down = mv.mech.bracket_fold(self.solve, 0.0, np.array([1.0]),
                                    step=0.05, jump=0.3, resolution=1e-3,
                                    direction=-1)
        assert down["fold"] == pytest.approx(-self.K_FOLD, abs=2e-3)

    def test_max_load_stops_without_a_fold(self):
        res = mv.mech.bracket_fold(self.solve, 0.0, np.array([-1.0]),
                                   step=0.05, jump=0.3, max_load=0.2)
        assert not res["found"] and res["fold"] is None


def test_registered():
    for name in ("elastic_constants", "cleavage_energies", "check_gradient",
                 "bracket_fold"):
        assert mv.registry.get(f"mv.mech.{name}") is not None

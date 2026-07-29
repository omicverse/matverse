"""The equation of state, and the relaxation it depends on.

Two independent routes to the bulk modulus live on the same object: a curvature
in volume (`mv.prop.eos`) and a curvature in strain (`mv.prop.elastic`). They
are the same quantity, so they have to agree — and making them agree is what
found the bug these tests now pin, which is that `mv.calc.relax` never moved the
lattice.

The numbers asserted here are EMT's, not nature's. EMT reproduces the metals it
was fitted to and does not reproduce aluminium, and the tests say so rather than
quietly choosing a tolerance wide enough to cover both.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")

#: Bulk moduli in GPa, CRC Handbook, room temperature.
EXPERIMENT = {"Cu": 140.0, "Ni": 180.0, "Ag": 100.0, "Al": 76.0}


@pytest.fixture(scope="module")
def relaxed():
    md = mv.datasets.metals(["Cu", "Ni", "Ag", "Al"])
    mv.pp.describe(md)
    mv.calc.relax(md, level="emt", fmax=0.005)
    return md


@pytest.fixture(scope="module")
def fitted(relaxed):
    md = relaxed.copy()
    mv.prop.eos(md, level="emt", source="relaxed_emt")
    mv.prop.elastic(md, level="emt", source="relaxed_emt")
    return md


class TestRelaxationMovesTheCell:
    """Until v0.1.17 it did not, and nothing in the suite noticed."""

    def test_the_volume_actually_changes(self, relaxed):
        """The forces on an fcc metal vanish by symmetry, so a positions-only
        optimiser converges instantly without moving anything and still calls
        the result 'relaxed'. Only the cell has anywhere to go."""
        before = [s.volume for s in mv.structures(relaxed, "input")]
        after = [s.volume for s in mv.structures(relaxed, "relaxed_emt")]
        assert all(abs(a - b) > 1e-6 for a, b in zip(after, before)), (
            "relaxation left every volume exactly unchanged, which is what a "
            "positions-only relaxation does to a high-symmetry cell")

    def test_relaxing_lowers_the_energy(self, relaxed):
        md = relaxed.copy()
        mv.calc.energy(md, level="emt", source="input")
        at_input = md.obs["energy_per_atom_emt"].to_numpy(dtype=float)
        relaxed_energy = relaxed.obs["energy_per_atom_emt"].to_numpy(dtype=float)
        assert (relaxed_energy <= at_input + 1e-9).all()

    def test_cell_false_holds_the_lattice(self):
        """A slab or a fixed-volume comparison needs the cell held, so the
        old behaviour stays reachable — by asking for it."""
        md = mv.datasets.metals(["Cu"])
        mv.pp.describe(md)
        mv.calc.relax(md, level="emt", fmax=0.005, cell=False)
        before = mv.structures(md, "input")[0].volume
        after = mv.structures(md, "relaxed_emt")[0].volume
        assert after == pytest.approx(before)

    def test_what_was_done_is_recorded(self):
        md = mv.datasets.metals(["Cu"])
        mv.pp.describe(md)
        mv.calc.relax(md, level="emt", fmax=0.01, cell=False)
        steps = [s for s in mv.provenance(md) if s.startswith("calc.relax")]
        assert steps and "cell=False" in steps[0]


class TestEquationOfState:
    def test_the_fit_is_good(self, fitted):
        """A residual far above a meV per atom is a fit that should not be
        read, so the number is reported rather than assumed."""
        residual = fitted.obs["eos_residual_emt"].to_numpy(dtype=float)
        assert (residual < 1e-3).all(), f"poor EOS fits: {residual}"

    def test_the_equilibrium_volume_is_where_the_relaxation_landed(self, fitted):
        """Two ways of finding the same minimum: an optimiser, and the vertex
        of a fitted curve."""
        fit_v0 = fitted.obs["equilibrium_volume_emt"].to_numpy(dtype=float)
        relaxed_v = np.array([s.volume
                              for s in mv.structures(fitted, "relaxed_emt")])
        assert fit_v0 == pytest.approx(relaxed_v, rel=0.01)

    def test_the_pressure_derivative_is_physical(self, fitted):
        """B0' is close to 4 for most solids — it is the leading correction in
        the Birch-Murnaghan expansion, not a free parameter.

        Checked for the three metals EMT reproduces. Aluminium comes out at
        1.95 against a measured 4.4, and that is asserted separately rather
        than absorbed by a wider bound here."""
        b1 = dict(zip(fitted.obs["formula"],
                      fitted.obs["bulk_modulus_derivative_emt"]))
        for symbol in ("Cu", "Ni", "Ag"):
            assert 2.0 < b1[symbol] < 7.0, (symbol, b1[symbol])

    def test_the_curve_is_a_minimum_at_unit_scale(self, fitted):
        """The energy-volume curve of a relaxed structure has its lowest point
        at the volume it was relaxed to."""
        grid = mv.grid_of(fitted, "eos")
        curves = np.asarray(fitted.obsm["eos_emt"], dtype=float)
        assert grid[np.argmin(curves, axis=1)] == pytest.approx(1.0, abs=0.021)

    def test_the_grid_is_a_scale_factor_not_a_volume(self, fitted):
        """Materials of different size have no common volume axis; the strain
        series they were computed on is common by construction."""
        assert mv.grid_of(fitted, "eos").min() < 1.0 < \
            mv.grid_of(fitted, "eos").max()
        assert fitted.uns["grids"]["eos"]["grid_unit"] == "V/V_input"

    def test_too_few_points_to_fit_is_refused(self, relaxed):
        """Four parameters cannot be fitted to three points, and returning NaN
        for every material would look like a calculator failure."""
        with pytest.raises(ValueError, match="at least four"):
            mv.prop.eos(relaxed.copy(), level="emt", scales=[0.98, 1.0, 1.02])


class TestTwoRoutesAgree:
    """The argument for keeping both on one object."""

    def test_eos_and_elastic_give_the_same_bulk_modulus(self, fitted):
        """A curvature in volume and a curvature in strain are the same
        quantity. Before the cell was relaxed they differed by 9-12%, always
        one-signed, because the stiffness was being taken about a geometry
        under residual tensile stress."""
        eos_b = fitted.obs["bulk_modulus_eos_emt"].to_numpy(dtype=float)
        elastic_b = fitted.obs["bulk_modulus_emt"].to_numpy(dtype=float)
        assert eos_b == pytest.approx(elastic_b, rel=0.03)

    def test_emt_reproduces_the_metals_it_was_fitted_to(self, fitted):
        """Cu, Ni and Ag come out within 15% of experiment."""
        by_formula = dict(zip(fitted.obs["formula"],
                              fitted.obs["bulk_modulus_eos_emt"]))
        for symbol in ("Cu", "Ni", "Ag"):
            assert by_formula[symbol] == pytest.approx(
                EXPERIMENT[symbol], rel=0.15), symbol

    def test_and_does_not_reproduce_aluminium(self, fitted):
        """Half the experimental value, and pinned deliberately.

        A tolerance loose enough to pass Al would pass almost anything, and the
        failure is the point: the level of theory is in the slot name because
        emt and pbe are different quantities, and this is what that looks like
        when it matters.

        The whole curve is wrong, not just its curvature at the minimum — B0'
        comes out at 1.95 against a measured 4.4 — so this is EMT having the
        wrong shape for aluminium rather than a fit that missed."""
        by_formula = dict(zip(fitted.obs["formula"],
                              fitted.obs["bulk_modulus_eos_emt"]))
        derivative = dict(zip(fitted.obs["formula"],
                              fitted.obs["bulk_modulus_derivative_emt"]))
        assert by_formula["Al"] < 0.7 * EXPERIMENT["Al"]
        assert derivative["Al"] < 2.5

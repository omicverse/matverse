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


class TestQuasiharmonic:
    """Thermal expansion, and the identity that had to replace the model.

    pymatgen's QuasiharmonicDebyeApprox reports a Gruneisen parameter that
    matches experiment and a set of optimum volumes that do not: the volume
    minimum it finds moves twelve times too little with temperature. The
    Gruneisen parameter is right and the bulk modulus is right, so the
    thermodynamic identity built from them is the one to trust.
    """

    #: Volumetric thermal expansion at 300 K, /K, CRC Handbook. Aluminium is
    #: listed but checked loosely: EMT gets its Gruneisen parameter wrong by a
    #: factor of two, and every quantity built from it inherits that.
    MEASURED_ALPHA = {"Cu": 5.0e-5, "Ag": 5.7e-5}
    MEASURED_ALPHA_AL = 6.9e-5
    #: Gruneisen parameters, room temperature.
    MEASURED_GAMMA = {"Cu": 1.96, "Ag": 2.4}

    @pytest.fixture(scope="module")
    def qha(self):
        md = mv.datasets.metals(["Cu", "Ag", "Al"])
        mv.pp.describe(md)
        mv.calc.relax(md, level="emt", fmax=0.005)
        mv.prop.quasiharmonic(md, level="emt", source="relaxed_emt",
                              t_max=900.0, poisson=0.34)
        return md

    def test_thermal_expansion_matches_experiment(self, qha):
        """For the metals EMT reproduces, within 25%."""
        by_formula = dict(zip(qha.obs["formula"],
                              qha.obs["thermal_expansion_qha_emt"]))
        for symbol, measured in self.MEASURED_ALPHA.items():
            assert by_formula[symbol] == pytest.approx(measured, rel=0.25), (
                symbol, by_formula[symbol], measured)

    def test_aluminium_is_the_usual_outlier(self, qha):
        """Held to a factor of two rather than 25%, and deliberately.

        EMT gives aluminium a Gruneisen parameter of 0.83 against a measured
        2.2, so an expansion built from it cannot be trusted to the precision
        the noble metals reach — the same aluminium that comes out at half the
        right bulk modulus and a B0' of 1.95 against 4.4. A tolerance wide
        enough to pass it everywhere would stop the other metals from
        testing anything."""
        by_formula = dict(zip(qha.obs["formula"],
                              qha.obs["thermal_expansion_qha_emt"]))
        assert 0.5 * self.MEASURED_ALPHA_AL < by_formula["Al"] < \
            2.0 * self.MEASURED_ALPHA_AL

    def test_the_gruneisen_parameter_is_right_for_the_noble_metals(self, qha):
        """It is the term the model gets right, and the reason the identity
        works at all."""
        by_formula = dict(zip(qha.obs["formula"], qha.obs["gruneisen_emt"]))
        for symbol, measured in self.MEASURED_GAMMA.items():
            assert by_formula[symbol] == pytest.approx(measured, rel=0.2), (
                symbol, by_formula[symbol], measured)

    def test_expansion_is_not_taken_from_the_models_volume_minimum(self, qha):
        """Pinned as a finding. The model's own optimum volumes give copper
        4.3e-6 /K, an order of magnitude below the measured 5.0e-5; anything
        that size would mean the identity had been abandoned."""
        by_formula = dict(zip(qha.obs["formula"],
                              qha.obs["thermal_expansion_qha_emt"]))
        assert by_formula["Cu"] > 1e-5

    def test_the_heat_capacity_approaches_dulong_petit(self, qha):
        """Three k_B per atom well above the Debye temperature, and the Debye
        integral rather than that constant below it."""
        k_b = 1.380649e-23
        capacity = qha.obs["heat_capacity_300K_emt"].to_numpy(dtype=float)
        n_atoms = qha.obs["nsites"].to_numpy(dtype=float)
        assert (capacity < 3.0 * n_atoms * k_b).all()
        assert (capacity > 1.5 * n_atoms * k_b).all()

    def test_expansion_is_reported_as_a_curve_too(self, qha):
        """A single number at 300 K hides that the expansion falls away as the
        Debye temperature is approached from below."""
        grid = mv.grid_of(qha, "thermal_expansion")
        curve = np.asarray(qha.obsm["thermal_expansion_emt"], dtype=float)
        assert curve.shape == (qha.n_obs, grid.size)
        assert (np.diff(curve, axis=1) >= -1e-12).all()

    def test_too_few_volumes_is_refused(self):
        md = mv.datasets.metals(["Cu"])
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="at least five"):
            mv.prop.quasiharmonic(md, level="emt", scales=[0.98, 1.0, 1.02])


class TestCostAndSupplyRisk:
    @pytest.fixture(scope="module")
    def economics(self):
        from pymatgen.core import Lattice, Structure

        def cell(symbols):
            return Structure(Lattice.cubic(5.0), symbols,
                             [[0, 0, 0], [.5, .5, .5], [.25, .25, .25],
                              [.5, 0, 0], [0, .5, 0]][:len(symbols)])
        md = mv.data.from_structures([cell(["Fe", "Fe", "O", "O", "O"]),
                                      cell(["Pt", "Pt", "O", "O", "O"]),
                                      cell(["Cu", "Cu"])])
        mv.pp.describe(md)
        mv.prop.cost(md)
        mv.prop.supply_risk(md)
        return md

    def test_platinum_is_orders_of_magnitude_dearer_than_iron(self, economics):
        """The case this exists to catch: no process optimisation closes five
        orders of magnitude."""
        by_formula = dict(zip(economics.obs["formula"],
                              economics.obs["cost_per_kg"]))
        assert by_formula["Pt2O3"] > 1000 * by_formula["Fe2O3"]

    def test_the_units_are_recorded(self, economics):
        assert economics.uns["units"]["cost_per_kg"] == "USD/kg"
        assert economics.uns["units"]["cost_per_mol"] == "USD/mol"

    def test_supply_risk_is_a_separate_question_from_price(self, economics):
        """Concentration and cost are different axes; both are reported."""
        assert set(economics.obs["supply_risk"]) <= {"low", "medium", "high"}
        by_formula = dict(zip(economics.obs["formula"],
                              economics.obs["hhi_reserve"]))
        assert by_formula["Pt2O3"] > by_formula["Fe2O3"]

    def test_a_screen_can_use_both(self, economics):
        mv.screen.pareto(economics, {"cost_per_kg": "min",
                                     "hhi_reserve": "min"}, name="affordable")
        assert "affordable" in economics.obs


class TestNeutronDiffraction:
    @pytest.fixture(scope="module")
    def patterns(self):
        md = mv.datasets.metals(["Cu", "Al"])
        mv.pp.describe(md)
        mv.prop.xrd(md, two_theta=(20, 90), step=0.1)
        mv.prop.neutron(md, two_theta=(20, 90), step=0.1)
        return md

    def test_both_patterns_land_on_their_own_grids(self, patterns):
        assert patterns.obsm["xrd_calc"].shape == \
            patterns.obsm["neutron_calc"].shape
        assert mv.grid_of(patterns, "neutron") == pytest.approx(
            mv.grid_of(patterns, "xrd"))

    def test_neutrons_and_x_rays_do_not_agree(self, patterns):
        """Neutrons scatter off nuclei and X-rays off electrons, so the
        intensities differ even where the peak positions do not — which is why
        having both is not redundant."""
        xrd = np.asarray(patterns.obsm["xrd_calc"], dtype=float)
        nd = np.asarray(patterns.obsm["neutron_calc"], dtype=float)
        assert not np.allclose(xrd, nd, atol=1.0)

    def test_the_peaks_sit_at_the_same_angles(self, patterns):
        """Bragg's law does not care what is scattering, so the allowed
        reflections are at the same angles in both patterns.

        Which of them is *strongest* is a different question and the answer
        differs — copper's tallest X-ray line is at 43.4 degrees and its
        tallest neutron line at 38.5 — because the scattering factors reweight
        the same reflections. That reweighting is the whole reason to compute
        both."""
        grid = mv.grid_of(patterns, "xrd")

        def peak_angles(curve):
            interior = np.arange(1, curve.size - 1)
            local = interior[(curve[1:-1] > curve[:-2])
                             & (curve[1:-1] > curve[2:])
                             & (curve[1:-1] > 0.05 * curve.max())]
            return grid[local]

        for row in range(patterns.n_obs):
            xrd_peaks = peak_angles(
                np.asarray(patterns.obsm["xrd_calc"][row], dtype=float))
            nd_peaks = peak_angles(
                np.asarray(patterns.obsm["neutron_calc"][row], dtype=float))
            assert len(nd_peaks) and len(xrd_peaks)
            for angle in nd_peaks:
                assert np.min(np.abs(xrd_peaks - angle)) < 0.5, (
                    f"neutron peak at {angle} has no X-ray counterpart in "
                    f"{xrd_peaks}")


class TestElectronDiffraction:
    """The F-centring extinction rule, from the intensities."""

    @staticmethod
    def _crystals():
        from pymatgen.core import Lattice, Structure
        return [
            Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
            Structure(Lattice.cubic(3.0), ["Po"], [[0, 0, 0]]),
        ]

    @pytest.fixture(scope="class")
    def patterns(self):
        md = mv.data.from_structures(self._crystals())
        mv.pp.describe(md)
        mv.prop.tem(md, r_max=1.2, step=0.02)
        return md

    def test_face_centring_extinguishes_three_quarters_of_them(self, patterns):
        """A face-centred lattice allows only all-even or all-odd hkl, which is
        one reflection in four. Simple cubic extinguishes none."""
        counts = patterns.obs["tem_n_reflections_calc"].to_numpy(dtype=float)
        assert counts[0] / counts[1] == pytest.approx(0.25, abs=0.02)

    def test_the_strongest_reflection_obeys_the_rule(self, patterns):
        """(010) is extinct in fcc and allowed in simple cubic, so the two
        crystals disagree about their brightest spot."""
        strongest = list(patterns.obs["tem_strongest_calc"])
        assert strongest[0] == "(0, -2, 0)"
        assert strongest[1] == "(0, -1, 0)"

    def test_the_zone_axis_is_recorded(self, patterns):
        """The same crystal down [001] and [111] gives different patterns, so
        a pattern without its axis is not a result."""
        assert set(patterns.obs["tem_zone_axis"]) == {"0,0,1"}
        assert patterns.uns["grids"]["tem"]["beam_direction"] == "0,0,1"

    def test_a_different_zone_axis_gives_a_different_pattern(self):
        md = mv.data.from_structures(self._crystals()[:1])
        mv.pp.describe(md)
        mv.prop.tem(md, r_max=1.2, step=0.02, beam_direction=(1, 1, 1),
                    level="along111")
        assert md.obs["tem_zone_axis"].iloc[0] == "1,1,1"
        assert md.obs["tem_strongest_along111"].iloc[0] != "(0, -2, 0)"

    def test_nothing_failed_silently(self, patterns):
        """The first version of this function raised NameError on every
        structure because `warnings` was not imported, and a bare except
        turned that into NaN. The count is now reported."""
        assert mv.level_info(patterns, "calc")["n_failed"] == 0

"""Results computed elsewhere, reduced to what a screen can use.

Nothing in matverse computes a shielding tensor, a piezoelectric tensor or a
dielectric function — those come from a DFT code. What matverse does is the step
after: turn a tensor into the parameters a spectrum is described by, check it
against the crystal symmetry, and turn a dielectric function into a solar
efficiency.

That step is arithmetic with conventions in it, which is exactly the kind of
thing worth testing against textbook answers rather than against stored output.
Every number asserted here has a source outside this repository: the Haeberlen
and Herzfeld-Berger definitions, the measured piezoelectric constants of
alpha-quartz, and the Shockley-Queisser limit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")

#: Principal values chosen so every convention has a round hand answer:
#: sigma_iso 30, zeta 30, eta 1/3, span 50, skew -0.6.
PRINCIPAL = np.diag([10.0, 20.0, 60.0])

#: alpha-quartz, measured: d11 = 2.3 pC/N, d14 = -0.67 pC/N.
QUARTZ_D11, QUARTZ_D14 = 2.3, -0.67

#: The Shockley-Queisser maximum: about 33% at a gap near 1.34 eV.
SQ_PEAK_GAP = 1.34


@pytest.fixture(scope="module")
def sited():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    return md, mv.multi.sites(md)


@pytest.fixture(scope="module")
def quartz():
    from pymatgen.core import Lattice, Structure
    structure = Structure.from_spacegroup(
        "P3121", Lattice.hexagonal(4.913, 5.405), ["Si", "O"],
        [[0.4697, 0.0, 0.0], [0.4135, 0.2669, 0.1191]])
    md = mv.data.from_structures([structure])
    mv.pp.describe(md)
    return md


def _quartz_voigt(d11=QUARTZ_D11, d14=QUARTZ_D14):
    """The class-32 piezoelectric matrix, which is what quartz is."""
    return np.array([[[d11, -d11, 0.0, d14, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, -d14, -2.0 * d11],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])


class TestShielding:
    """Conventions, checked against their definitions."""

    @pytest.fixture(scope="class")
    def shielded(self, sited):
        md, sites = sited
        sites = sites.copy()
        mv.prop.nmr(md, sites, np.tile(PRINCIPAL, (sites.n_obs, 1, 1)),
                    level="pbe")
        return sites

    def test_the_isotropic_value_is_the_trace_over_three(self, shielded):
        assert shielded.obs["shielding_iso_pbe"].iloc[0] == pytest.approx(30.0)

    def test_haeberlen_zeta_is_sigma33_minus_isotropic(self, shielded):
        assert shielded.obs["shielding_anisotropy_pbe"].iloc[0] == \
            pytest.approx(60.0 - 30.0)

    def test_the_asymmetry_is_the_ratio_of_the_other_two(self, shielded):
        """eta = (sigma_22 - sigma_11) / zeta = (20 - 10) / 30."""
        assert shielded.obs["shielding_asymmetry_pbe"].iloc[0] == \
            pytest.approx(1.0 / 3.0)

    def test_span_and_skew_are_the_herzfeld_berger_pair(self, shielded):
        """Span is the full width; skew says which side sigma_22 falls on.
        Here it is below the isotropic value, so the skew is negative."""
        assert shielded.obs["shielding_span_pbe"].iloc[0] == \
            pytest.approx(60.0 - 10.0)
        assert shielded.obs["shielding_skew_pbe"].iloc[0] == \
            pytest.approx(3.0 * (20.0 - 30.0) / 50.0)

    def test_the_tensor_is_kept_not_only_its_reduction(self, shielded):
        """The five parameters are lossy; the tensor is what was given."""
        stored = shielded.obsm["shielding_tensor_pbe"][0].reshape(3, 3)
        assert stored == pytest.approx(PRINCIPAL)

    def test_it_lands_on_the_sites_object(self, sited):
        """Per-atom, so the material axis has no room for it."""
        md, _ = sited
        assert "shielding_iso_pbe" not in md.obs

    def test_a_wrong_shaped_input_is_refused(self, sited):
        md, sites = sited
        with pytest.raises(ValueError, match="one 3x3 tensor per row"):
            mv.prop.nmr(md, sites.copy(), np.zeros((2, 3, 3)), level="pbe")


class TestElectricFieldGradient:
    @pytest.fixture(scope="class")
    def gradients(self, sited):
        md, sites = sited
        sites = sites.copy()
        mv.prop.efg(md, sites, np.tile(np.diag([-1.0, -2.0, 3.0]),
                                       (sites.n_obs, 1, 1)), level="pbe")
        return sites

    def test_vzz_is_the_largest_by_magnitude(self, gradients):
        assert gradients.obs["efg_vzz_pbe"].iloc[0] == pytest.approx(3.0)

    def test_the_asymmetry_uses_the_ordering_convention(self, gradients):
        """|Vzz| >= |Vyy| >= |Vxx|, so eta = (Vxx - Vyy)/Vzz lands in [0, 1]."""
        eta = gradients.obs["efg_asymmetry_pbe"].to_numpy(dtype=float)
        assert eta[0] == pytest.approx(1.0 / 3.0)
        assert ((eta >= 0.0) & (eta <= 1.0)).all()

    def test_the_coupling_constant_needs_a_nucleus(self, gradients):
        """It is a property of the isotope, not of the calculation, so it is
        looked up from the element on the site.

        Its sign follows the sign of the nuclear quadrupole moment and is not
        an invariant — aluminium's comes out negative on the same gradient
        that gives copper a positive one — so only the magnitude is asserted.
        """
        coupling = gradients.obs["efg_coupling_pbe"].to_numpy(dtype=float)
        assert np.isfinite(coupling).all()
        assert (np.abs(coupling) > 0).all()


class TestPiezoelectric:
    @pytest.fixture(scope="class")
    def measured(self, quartz):
        md = quartz.copy()
        mv.prop.piezoelectric(md, _quartz_voigt(), level="exp")
        return md

    def test_the_longitudinal_maximum_recovers_d11(self, measured):
        """For class 32 the longitudinal response in the basal plane is
        d11 cos(3 theta), so its maximum is d11 itself."""
        assert measured.obs["piezo_max_longitudinal_exp"].iloc[0] == \
            pytest.approx(QUARTZ_D11, rel=0.01)

    def test_the_best_direction_is_in_the_basal_plane(self, measured):
        direction = np.array([float(x) for x in
                              measured.obs["piezo_max_direction_exp"].iloc[0]
                              .split(",")])
        assert abs(direction[2]) < 0.05, direction

    def test_quartz_accepts_its_own_tensor(self, measured):
        assert bool(measured.obs["piezo_symmetry_valid_exp"].iloc[0])

    def test_a_centrosymmetric_crystal_rejects_it(self):
        """Piezoelectricity is forbidden by inversion symmetry, so a non-zero
        tensor on fcc copper is an error rather than a discovery — and the
        check is what says so."""
        md = mv.datasets.metals(["Cu"])
        mv.pp.describe(md)
        mv.prop.piezoelectric(md, _quartz_voigt(), level="bogus")
        assert not bool(md.obs["piezo_symmetry_valid_bogus"].iloc[0])

    def test_voigt_and_full_notation_agree(self, quartz):
        """The shear columns carry a factor of two, and getting it wrong is
        the classic way to be off by two on d14."""
        from matverse.prop import _from_voigt

        voigt = _quartz_voigt()
        full = np.stack([_from_voigt(voigt[0])])
        by_voigt, by_full = quartz.copy(), quartz.copy()
        mv.prop.piezoelectric(by_voigt, voigt, level="a")
        mv.prop.piezoelectric(by_full, full, level="b")
        assert by_voigt.obs["piezo_max_longitudinal_a"].iloc[0] == \
            pytest.approx(by_full.obs["piezo_max_longitudinal_b"].iloc[0])

    def test_a_wrong_shaped_input_is_refused(self, quartz):
        with pytest.raises(ValueError, match="Voigt notation"):
            mv.prop.piezoelectric(quartz.copy(), np.zeros((1, 4, 4)))


class TestDielectric:
    @pytest.fixture(scope="class")
    def optical(self):
        md = mv.datasets.metals(["Cu", "Al", "Ni"])
        mv.pp.describe(md)
        grid = np.linspace(0.3, 4.0, 400)
        gaps = [1.34, 1.10, 2.00]
        eps1 = np.tile(4.0, (md.n_obs, grid.size))
        eps2 = np.stack([np.where(grid >= g, 6.0, 0.0) for g in gaps])
        mv.prop.dielectric(md, grid, eps1, eps2, level="pbe")
        md.obs["band_gap_pbe"] = gaps
        return md

    def test_the_refractive_index_is_the_root_of_epsilon(self, optical):
        """Below the gap the imaginary part is zero, so n = sqrt(eps_1)."""
        assert optical.obs["refractive_index_pbe"].to_numpy() == \
            pytest.approx(2.0)

    def test_absorption_is_zero_below_the_gap(self, optical):
        grid = mv.grid_of(optical, "absorption")
        alpha = np.asarray(optical.obsm["absorption_pbe"], dtype=float)
        below = grid < 1.10
        assert alpha[:, below] == pytest.approx(0.0)

    def test_the_dielectric_function_survives_the_derivation(self, optical):
        """alpha is derived, so eps stays recoverable — a spectrum that has
        been through the definition twice cannot be reconstructed."""
        assert "dielectric_real_pbe" in optical.obsm
        assert "dielectric_imag_pbe" in optical.obsm

    def test_the_curves_share_one_energy_grid(self, optical):
        for quantity in ("dielectric_real", "dielectric_imag", "absorption",
                         "extinction"):
            assert mv.grid_of(optical, quantity) == pytest.approx(
                mv.grid_of(optical, "absorption"))


class TestSolarEfficiency:
    @pytest.fixture(scope="class")
    def cells(self):
        md = mv.datasets.metals(["Cu", "Al", "Ni"])
        mv.pp.describe(md)
        grid = np.linspace(0.3, 4.0, 800)
        gaps = [SQ_PEAK_GAP, 0.90, 2.50]
        eps1 = np.tile(4.0, (md.n_obs, grid.size))
        eps2 = np.stack([np.where(grid >= g, 6.0, 0.0) for g in gaps])
        mv.prop.dielectric(md, grid, eps1, eps2, level="pbe")
        md.obs["band_gap_pbe"] = gaps
        mv.prop.slme(md, level="pbe", thickness=5e-6)
        return md

    def test_a_perfect_absorber_reproduces_shockley_queisser(self, cells):
        """The whole calibration of this function: a step absorption edge with
        no indirect gap is the Shockley-Queisser idealisation, and at 1.34 eV
        it gives the textbook 33%."""
        assert cells.obs["slme_pbe"].iloc[0] == pytest.approx(33.7, abs=1.0)

    def test_the_limit_peaks_at_the_right_gap(self, cells):
        """1.34 eV beats both 0.90 and 2.50 — the shape of the SQ curve."""
        limit = cells.obs["sq_limit_pbe"].to_numpy(dtype=float)
        assert limit[0] > limit[1] and limit[0] > limit[2]

    def test_it_is_reported_as_a_percentage(self, cells):
        """An efficiency is quoted in percent, and the unit is recorded rather
        than left to the reader."""
        assert cells.uns["units"]["slme_pbe"] == "percent"
        assert (cells.obs["slme_pbe"] > 1.0).all()

    def test_a_weak_absorber_falls_short_of_its_own_gap_limit(self):
        """This is why SLME exists rather than screening on the gap: two
        materials with the same gap and different absorption strength are not
        the same candidate."""
        md = mv.datasets.metals(["Cu", "Al", "Ni"])
        mv.pp.describe(md)
        grid = np.linspace(0.3, 4.0, 800)
        eps1 = np.tile(4.0, (md.n_obs, grid.size))
        eps2 = np.stack([np.where(grid >= SQ_PEAK_GAP, s, 0.0)
                         for s in (6.0, 0.05, 0.005)])
        mv.prop.dielectric(md, grid, eps1, eps2, level="pbe")
        md.obs["band_gap_pbe"] = [SQ_PEAK_GAP] * 3
        mv.prop.slme(md, level="pbe", thickness=5e-7)

        slme = md.obs["slme_pbe"].to_numpy(dtype=float)
        limit = md.obs["sq_limit_pbe"].to_numpy(dtype=float)
        assert limit == pytest.approx(limit[0])       # same gap, same ceiling
        assert slme[0] > slme[1] > slme[2]            # absorption decides
        assert slme[1] < 0.5 * limit[1]

    def test_an_indirect_gap_costs_efficiency(self):
        """The radiative fraction falls as exp(-(Eg_direct - Eg_indirect)/kT),
        which is the honest penalty silicon pays."""
        md = mv.datasets.metals(["Cu", "Al", "Ni"])
        mv.pp.describe(md)
        grid = np.linspace(0.3, 4.0, 800)
        eps1 = np.tile(4.0, (md.n_obs, grid.size))
        eps2 = np.stack([np.where(grid >= SQ_PEAK_GAP, 6.0, 0.0)] * 3)
        mv.prop.dielectric(md, grid, eps1, eps2, level="pbe")
        md.obs["band_gap_pbe"] = [SQ_PEAK_GAP] * 3
        md.obs["gap_indirect"] = [SQ_PEAK_GAP, SQ_PEAK_GAP - 0.2,
                                  SQ_PEAK_GAP - 0.4]
        mv.prop.slme(md, level="pbe", thickness=5e-7,
                     indirect_key="gap_indirect")

        slme = md.obs["slme_pbe"].to_numpy(dtype=float)
        assert slme[0] > slme[1] > slme[2]

    def test_it_refuses_without_an_absorption_spectrum(self):
        md = mv.datasets.metals(["Cu"])
        mv.pp.describe(md)
        md.obs["band_gap_pbe"] = [1.3]
        with pytest.raises(ValueError, match="mv.prop.dielectric"):
            mv.prop.slme(md, level="pbe")

    def test_it_refuses_without_a_gap(self):
        md = mv.datasets.metals(["Cu"])
        mv.pp.describe(md)
        grid = np.linspace(0.3, 4.0, 200)
        mv.prop.dielectric(md, grid, np.tile(4.0, (1, 200)),
                           np.tile(1.0, (1, 200)), level="pbe")
        with pytest.raises(ValueError, match="needs a gap"):
            mv.prop.slme(md, level="pbe")


class TestEnergyCorrections:
    """The corrections that decide whether a hull is right.

    The numbers are the published MP2020 values, so they can be checked without
    running anything: the oxide anion correction is -0.687 eV per oxygen, and
    the +U correction applies only where MP itself would have applied a U.
    """

    #: MP2020, eV per oxygen atom.
    OXIDE_ANION = -0.687

    @staticmethod
    def _cell(symbols):
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(5.0), symbols,
                         [[0, 0, 0], [.5, .5, .5], [.25, .25, .25],
                          [.5, 0, 0], [0, .5, 0]][:len(symbols)])

    @pytest.fixture(scope="class")
    def corrected(self):
        md = mv.data.from_structures([
            self._cell(["Fe", "Fe", "O", "O", "O"]),
            self._cell(["Al", "Al", "O", "O", "O"]),
            self._cell(["Cu", "Cu"])])
        mv.pp.describe(md)
        md.obs["energy_pbe"] = [-50.0, -60.0, -10.0]
        mv.thermo.corrections(md, level="pbe")
        return md

    def test_an_oxide_with_no_u_gets_only_the_anion_correction(self, corrected):
        """Al2O3: three oxygens, no transition metal MP applies a U to."""
        by_formula = dict(zip(corrected.obs["formula"],
                              corrected.obs["correction_pbe-mp2020"]))
        assert by_formula["Al2O3"] == pytest.approx(3 * self.OXIDE_ANION,
                                                    abs=0.01)

    def test_an_elemental_metal_gets_nothing(self, corrected):
        by_formula = dict(zip(corrected.obs["formula"],
                              corrected.obs["correction_pbe-mp2020"]))
        assert by_formula["Cu"] == pytest.approx(0.0)

    def test_iron_oxide_gets_the_anion_and_the_u_correction(self, corrected):
        """Fe2O3 moves by 6.6 eV, of which 2.1 is the anion term and the rest
        the +U term on two irons. Ignoring this is not a small error."""
        by_formula = dict(zip(corrected.obs["formula"],
                              corrected.obs["correction_pbe-mp2020"]))
        assert by_formula["Fe2O3"] < 3 * self.OXIDE_ANION - 3.0
        assert by_formula["Fe2O3"] == pytest.approx(-6.57, abs=0.1)

    def test_the_run_type_is_inferred_by_mps_own_rule(self, corrected):
        """+U where MP would use it — a transition metal in its table together
        with oxygen or fluorine — and plain GGA otherwise."""
        by_formula = dict(zip(corrected.obs["formula"],
                              corrected.obs["run_type_pbe-mp2020"]))
        assert by_formula["Fe2O3"] == "GGA+U"
        assert by_formula["Al2O3"] == "GGA"
        assert by_formula["Cu"] == "GGA"

    def test_the_corrected_energy_is_a_new_level(self, corrected):
        """A raw energy and a corrected one are different quantities, so the
        corrected one gets its own level rather than overwriting the column it
        came from — the same rule that keeps emt and pbe apart."""
        assert "energy_pbe" in corrected.obs
        assert "energy_pbe-mp2020" in corrected.obs
        info = mv.level_info(corrected, "pbe-mp2020")
        assert info["kind"] == "corrected"
        assert info["reference"] == "pbe"

    def test_the_correction_is_the_difference(self, corrected):
        raw = corrected.obs["energy_pbe"].to_numpy(dtype=float)
        new = corrected.obs["energy_pbe-mp2020"].to_numpy(dtype=float)
        delta = corrected.obs["correction_pbe-mp2020"].to_numpy(dtype=float)
        assert new - raw == pytest.approx(delta, abs=1e-6)

    def test_an_unknown_scheme_is_refused(self, corrected):
        with pytest.raises(ValueError, match="unknown scheme"):
            mv.thermo.corrections(corrected.copy(), level="pbe",
                                  scheme="handwaving")

    def test_it_refuses_without_an_energy(self):
        md = mv.datasets.metals(["Cu"])
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="nothing to correct"):
            mv.thermo.corrections(md, level="pbe")


class TestPolarization:
    """The test is branch reconstruction, because that is the whole job.

    A Berry-phase calculation returns polarization modulo a quantum, so the
    values along a switching path come back scattered. Feeding in a known
    smooth path with a random multiple of the quantum added to each point, and
    requiring the smooth path back, checks the one thing that is easy to get
    wrong and impossible to notice.
    """

    A = 4.0
    QUANTUM = 100.136          # uC/cm^2 for this cell, from the lattice

    @classmethod
    def _path(cls, n=7):
        from pymatgen.core import Lattice, Structure
        cells = [Structure(Lattice.cubic(cls.A),
                           ["Ba", "Ti", "O", "O", "O"],
                           [[0, 0, 0], [.5, .5, .5 + x], [.5, .5, 0],
                            [.5, 0, .5], [0, .5, .5]])
                 for x in np.linspace(0.0, 0.04, n)]
        return mv.data.from_structures(cells)

    @classmethod
    def _as_p_elec(cls, values):
        """uC/cm^2 along c into the e*angstrom vectors Polarization wants."""
        scale = cls.A ** 3 / 1602.1766208
        out = np.zeros((len(values), 3))
        out[:, 2] = np.asarray(values, dtype=float) * scale
        return out

    def test_the_branch_is_reconstructed_from_scattered_input(self):
        n = 7
        md = self._path(n)
        smooth = np.linspace(0.0, 30.0, n)
        rng = np.random.RandomState(0)
        folded = smooth + self.QUANTUM * rng.randint(-2, 3, n)
        assert np.abs(folded - smooth).max() > self.QUANTUM, \
            "the input must actually be scattered or this tests nothing"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.prop.polarization(md, self._as_p_elec(folded), np.zeros((n, 3)))
        recovered = md.obs["polarization_c"].to_numpy()
        # pymatgen carries the electronic term with the opposite sign.
        # atol=1e-3, because the conversion constant this test uses to turn
        # uC/cm^2 into e*angstrom is a rounded one — the reconstruction itself
        # is recovering values scattered over +/-200 to within a thousandth.
        assert np.allclose(np.abs(recovered), smooth, atol=1e-3)

    def test_the_spontaneous_polarization_is_the_end_to_end_change(self):
        n = 7
        md = self._path(n)
        smooth = np.linspace(0.0, 30.0, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.prop.polarization(md, self._as_p_elec(smooth), np.zeros((n, 3)))
        record = md.uns["polarization"]["polarization"]
        assert record["spontaneous_norm"] == pytest.approx(30.0, abs=1e-6)
        assert record["unit"] == "uC/cm^2"

    def test_a_path_that_does_not_move_is_not_polar(self):
        n = 5
        md = self._path(n)
        mv.prop.polarization(md, np.zeros((n, 3)), np.zeros((n, 3)))
        assert md.uns["polarization"]["polarization"]["spontaneous_norm"] == \
            pytest.approx(0.0, abs=1e-9)

    def test_a_coarse_path_is_warned_about(self):
        """When the change is a sizeable fraction of the quantum, 'nearest
        branch' is a guess. Saying so is the difference between a reading and
        a number."""
        n = 4
        md = self._path(n)
        big = np.linspace(0.0, 60.0, n)
        with pytest.warns(UserWarning, match="quantum"):
            mv.prop.polarization(md, self._as_p_elec(big), np.zeros((n, 3)))
        assert md.uns["polarization"]["polarization"]["fraction_of_quantum"] \
            > 0.25

    def test_a_fine_path_passes_quietly(self):
        n = 9
        md = self._path(n)
        small = np.linspace(0.0, 5.0, n)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mv.prop.polarization(md, self._as_p_elec(small),
                                 np.zeros((n, 3)))

    def test_one_structure_is_not_a_path(self):
        md = self._path(1)
        with pytest.raises(ValueError, match="at least two structures"):
            mv.prop.polarization(md, np.zeros((1, 3)), np.zeros((1, 3)))

    def test_the_shapes_must_match_the_rows(self):
        md = self._path(5)
        with pytest.raises(ValueError, match="one vector"):
            mv.prop.polarization(md, np.zeros((3, 3)), np.zeros((5, 3)))

    def test_the_quantum_is_recorded_beside_the_answer(self):
        n = 7
        md = self._path(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.prop.polarization(md, self._as_p_elec(np.linspace(0, 30, n)),
                                 np.zeros((n, 3)))
        record = md.uns["polarization"]["polarization"]
        assert record["quantum"][2] == pytest.approx(self.QUANTUM, rel=1e-3)
        assert record["n_images"] == n

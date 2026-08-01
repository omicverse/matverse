"""Electronic structure: the bands axis, and what is derived from it.

The band structures here come from a nearest-neighbour tight-binding model
evaluated on a real high-symmetry path. That is a genuine calculation rather
than a fixture — the dispersion is the textbook fcc one — which means the tests
can assert physics: a band crossing the Fermi level makes a metal, a rigid shift
opens a gap of exactly the size it was shifted by, and a gap whose extrema sit
at the same k-point is direct.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")


def _dispersion(structure, line_density=8, t=1.0):
    """Nearest-neighbour tight-binding dispersion on the structure's own path."""
    from pymatgen.symmetry.bandstructure import HighSymmKpath

    path = HighSymmKpath(structure)
    kpoints, _ = path.get_kpoints(line_density=line_density,
                                  coords_are_cartesian=False)
    k = np.asarray(kpoints, dtype=float)
    energies = -2.0 * t * (np.cos(2 * np.pi * k[:, 0])
                           + np.cos(2 * np.pi * k[:, 1])
                           + np.cos(2 * np.pi * k[:, 2]))
    return path, kpoints, energies


def _band_structure(structure, rows, kpoints, path):
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
    from pymatgen.electronic_structure.core import Spin

    return BandStructureSymmLine(
        kpoints, {Spin.up: np.vstack(rows)},
        structure.lattice.reciprocal_lattice, efermi=0.0,
        labels_dict=dict(path.kpath["kpoints"]), structure=structure)


def _metallic(structure, n_bands=4, spacing=3.0):
    """Bands that straddle the Fermi level, which is what a metal is."""
    path, kpoints, dispersion = _dispersion(structure)
    rows = [dispersion + spacing * b - spacing * (n_bands // 2)
            for b in range(n_bands)]
    return _band_structure(structure, rows, kpoints, path)


def _semiconductor(structure, gap=1.7, direct=True, n_each=2, spacing=2.0):
    """A gap of exactly ``gap`` eV, direct or indirect on request.

    Valence bands are pushed down so the highest touches zero; conduction bands
    are pushed up so the lowest touches ``gap``. For a direct gap the
    conduction dispersion is inverted, which puts its minimum at the same
    k-point as the valence maximum.
    """
    path, kpoints, dispersion = _dispersion(structure)

    rows = []
    for b in range(n_each):
        rows.append(dispersion - dispersion.max() - spacing * b)
    conduction = -dispersion if direct else dispersion
    for b in range(n_each):
        rows.append(conduction - conduction.min() + gap + spacing * b)
    return _band_structure(structure, rows, kpoints, path)


@pytest.fixture(scope="module")
def metals():
    md = mv.datasets.metals(["Cu", "Al"])
    mv.pp.describe(md)
    return md


class TestKpath:
    def test_every_material_gets_a_path(self, metals):
        mv.elec.kpath(metals, line_density=8)
        assert (metals.obs["n_kpoints"] > 0).all()
        assert metals.uns["kpath"]["n_failed"] == 0

    def test_the_fcc_path_is_the_conventional_one(self, metals):
        mv.elec.kpath(metals, line_density=8)
        labels = metals.obs["kpath_labels"].iloc[0]
        for point in ("X", "W", "K", "L"):
            assert point in labels

    def test_the_convention_is_recorded(self, metals):
        """Setyawan-Curtarolo and Hinuma disagree on several Bravais lattices,
        so a path whose convention is unstated is not reproducible."""
        mv.elec.kpath(metals, line_density=8)
        assert metals.uns["kpath"]["path_type"] == "setyawan_curtarolo"
        assert metals.uns["kpath"]["convention"]

    def test_an_unknown_convention_is_refused(self, metals):
        with pytest.raises(ValueError, match="unknown path_type"):
            mv.elec.kpath(metals, path_type="vibes")


class TestBandsAxis:
    @pytest.fixture(scope="class")
    def bands(self, metals):
        structures = mv.structures(metals)
        return mv.elec.bands(metals, [_metallic(s) for s in structures],
                             level="tb", n_points=120)

    def test_one_row_per_band_per_material(self, bands, metals):
        assert bands.n_obs == 4 * metals.n_obs
        assert bands.n_vars == 120
        assert set(bands.obs["material"]) == set(map(str, metals.obs_names))

    def test_it_is_its_own_axis(self, bands):
        assert bands.uns[mv.elec.AXIS_KEY] == "bands"

    def test_energies_are_relative_to_the_fermi_level(self, bands):
        """An absolute eigenvalue means nothing across codes, so zero is E_F
        everywhere on this axis."""
        assert bands.uns["bands"]["energy_reference"].startswith("Fermi")
        assert (bands.obs["band_minimum"] < 0).any()
        assert (bands.obs["band_maximum"] > 0).any()

    def test_the_abscissa_is_a_path_fraction(self, bands):
        fraction = bands.var["path_fraction"].to_numpy(dtype=float)
        assert fraction[0] == pytest.approx(0.0)
        assert fraction[-1] == pytest.approx(1.0)
        assert (np.diff(fraction) > 0).all()

    def test_materials_with_different_paths_still_share_a_matrix(self, bands,
                                                                 metals):
        """Cu and Al produce different numbers of k-points; resampling onto a
        path fraction is what makes them one block."""
        mv.elec.kpath(metals, line_density=8)
        assert metals.obs["n_kpoints"].nunique() > 1
        assert np.asarray(bands.X).shape == (bands.n_obs, 120)

    def test_a_missing_run_is_a_missing_row_not_a_dropped_material(self,
                                                                   metals):
        structures = mv.structures(metals)
        partial = [_metallic(structures[0]), None]
        bands = mv.elec.bands(metals, partial, level="tb", n_points=60)
        assert set(bands.obs["material"]) == {str(metals.obs_names[0])}
        assert metals.n_obs == 2

    def test_a_length_mismatch_is_refused(self, metals):
        with pytest.raises(ValueError, match="one per row"):
            mv.elec.bands(metals, [None], level="tb")


class TestBandFeatures:
    def test_a_band_crossing_the_fermi_level_is_a_metal(self, metals):
        structures = mv.structures(metals)
        bands = mv.elec.bands(metals, [_metallic(s) for s in structures],
                              level="tb")
        mv.elec.band_features(bands, metals, level="tb")

        assert metals.obs["is_metal_tb"].all()
        assert (metals.obs["band_gap_tb"].to_numpy(dtype=float) == 0.0).all()

    def test_a_rigid_shift_opens_exactly_the_gap_it_was_given(self, metals):
        """The model is constructed with a known gap, so this is a real check
        rather than a self-consistent one."""
        structures = mv.structures(metals)
        gapped = [_semiconductor(s, gap=1.7) for s in structures]
        bands = mv.elec.bands(metals, gapped, level="gapped")
        mv.elec.band_features(bands, metals, level="gapped")

        gaps = metals.obs["band_gap_gapped"].to_numpy(dtype=float)
        assert not metals.obs["is_metal_gapped"].any()
        assert gaps == pytest.approx(np.full_like(gaps, 1.7), abs=0.05)

    def test_the_edges_bracket_the_gap(self, metals):
        structures = mv.structures(metals)
        gapped = [_semiconductor(s, gap=1.7) for s in structures]
        bands = mv.elec.bands(metals, gapped, level="gapped")
        mv.elec.band_features(bands, metals, level="gapped")

        vbm = metals.obs["vbm_gapped"].to_numpy(dtype=float)
        cbm = metals.obs["cbm_gapped"].to_numpy(dtype=float)
        assert (vbm <= 0).all() and (cbm >= 0).all()
        assert (cbm - vbm == pytest.approx(
            metals.obs["band_gap_gapped"].to_numpy(dtype=float), abs=1e-6))

    def test_direct_and_indirect_are_told_apart(self, metals):
        """The model puts the conduction minimum at the valence maximum when
        asked for a direct gap and somewhere else when not, so this checks the
        k-point comparison rather than restating it."""
        structures = mv.structures(metals)

        for wanted in (True, False):
            level = "direct" if wanted else "indirect"
            built = [_semiconductor(s, gap=1.7, direct=wanted)
                     for s in structures]
            bands = mv.elec.bands(metals, built, level=level)
            mv.elec.band_features(bands, metals, level=level)
            assert metals.obs[f"is_direct_{level}"].all() == wanted

    def test_a_screen_can_reach_the_gap(self, metals):
        structures = mv.structures(metals)
        gapped = [_semiconductor(s, gap=1.7) for s in structures]
        bands = mv.elec.bands(metals, gapped, level="gapped")
        mv.elec.band_features(bands, metals, level="gapped")
        mv.screen.filter(metals, band_gap_gapped__gt=1.0, name="wide_gap")
        assert metals.obs["wide_gap"].all()

    def test_it_refuses_the_wrong_axis(self, metals):
        with pytest.raises(ValueError, match="not a bands object"):
            mv.elec.band_features(metals, metals, level="tb")


class TestTransport:
    def test_it_names_the_install_rather_than_returning_zeros(self, metals):
        """A thermoelectric screen that silently produces zeros is worse than
        one that refuses."""
        pytest.importorskip  # noqa: B018 - documented intent
        try:
            import BoltzTraP2                            # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="BoltzTraP2"):
                mv.elec.transport(metals, metals, level="tb")


class TestXPS:
    """The thing XPS adds over a density of states is cross-section weighting,
    so that is what these check: two identical DOS peaks on different orbitals
    must come out in the tabulated ratio, not equal."""

    #: Yeh and Lindau, as pymatgen ships them.
    CU_D = 0.0012
    O_P = 6.0e-05

    @staticmethod
    def _dos(cu_at=-3.0, o_at=-6.0, width=0.3, scale=1.0):
        import numpy as np
        from pymatgen.core import Lattice, Structure
        from pymatgen.electronic_structure.core import Orbital, Spin
        from pymatgen.electronic_structure.dos import CompleteDos, Dos
        energies = np.linspace(-10, 5, 401)
        cu = np.exp(-0.5 * ((energies - cu_at) / width) ** 2) * scale
        ox = np.exp(-0.5 * ((energies - o_at) / width) ** 2) * scale
        st = Structure(Lattice.cubic(4.2), ["Cu", "O"],
                       [[0, 0, 0], [.5, .5, .5]])
        return st, CompleteDos(
            st, Dos(0.0, energies, {Spin.up: cu + ox}),
            {st[0]: {Orbital.dxy: {Spin.up: cu}},
             st[1]: {Orbital.px: {Spin.up: ox}}})

    def test_identical_dos_peaks_come_out_in_the_cross_section_ratio(self):
        """The whole point. Equal contributions to the DOS, twenty to one in
        the photoemission, because copper's 3d is seen twenty times as
        strongly as oxygen's 2p."""
        st, dos = self._dos()
        md = mv.data.from_structures([st])
        mv.elec.xps(md, [dos], level="model")
        grid = mv.grid_of(md, "xps")
        y = md.obsm["xps_model"][0]
        at_cu = y[int(np.argmin(abs(grid - 3.0)))]
        at_o = y[int(np.argmin(abs(grid - 6.0)))]
        assert at_cu / at_o == pytest.approx(self.CU_D / self.O_P, rel=1e-3)

    def test_the_axis_is_binding_energy(self):
        """A state 3 eV below the Fermi level appears at +3 eV, not -3."""
        st, dos = self._dos()
        md = mv.data.from_structures([st])
        mv.elec.xps(md, [dos], level="model")
        assert float(md.obs["xps_peak_model"].iloc[0]) == pytest.approx(3.0,
                                                                       abs=0.1)
        assert (mv.grid_of(md, "xps") > 0).any()

    def test_the_stronger_emitter_wins_even_when_it_is_the_smaller_peak(self):
        """Make the oxygen peak ten times the copper one in the DOS. XPS must
        still report copper as the main line, because twenty beats ten."""
        import numpy as np
        from pymatgen.core import Lattice, Structure
        from pymatgen.electronic_structure.core import Orbital, Spin
        from pymatgen.electronic_structure.dos import CompleteDos, Dos
        energies = np.linspace(-10, 5, 401)
        cu = np.exp(-0.5 * ((energies + 3.0) / 0.3) ** 2)
        ox = 10.0 * np.exp(-0.5 * ((energies + 6.0) / 0.3) ** 2)
        st = Structure(Lattice.cubic(4.2), ["Cu", "O"],
                       [[0, 0, 0], [.5, .5, .5]])
        dos = CompleteDos(st, Dos(0.0, energies, {Spin.up: cu + ox}),
                          {st[0]: {Orbital.dxy: {Spin.up: cu}},
                           st[1]: {Orbital.px: {Spin.up: ox}}})
        md = mv.data.from_structures([st])
        mv.elec.xps(md, [dos], level="model")
        assert float(md.obs["xps_peak_model"].iloc[0]) == pytest.approx(3.0,
                                                                       abs=0.1)

    def test_rows_share_one_grid(self):
        st, first = self._dos(cu_at=-3.0)
        _, second = self._dos(cu_at=-4.0)
        md = mv.data.from_structures([st, st])
        mv.elec.xps(md, [first, second], level="model")
        assert md.obsm["xps_model"].shape == (2, 301)
        peaks = md.obs["xps_peak_model"].to_numpy()
        assert peaks[0] == pytest.approx(3.0, abs=0.15)
        assert peaks[1] == pytest.approx(4.0, abs=0.15)

    def test_one_dos_per_row_is_required(self):
        st, dos = self._dos()
        md = mv.data.from_structures([st, st])
        with pytest.raises(ValueError, match="one per row"):
            mv.elec.xps(md, [dos], level="model")

    def test_a_row_without_a_dos_is_recorded_not_guessed(self):
        st, dos = self._dos()
        md = mv.data.from_structures([st, st])
        mv.elec.xps(md, [dos, None], level="model")
        assert np.isnan(float(md.obs["xps_peak_model"].iloc[1]))
        assert any("no DOS" in e for e in md.uns["xps"]["model"]["errors"])

    def test_nothing_usable_says_so(self):
        st, _ = self._dos()
        md = mv.data.from_structures([st])
        with pytest.raises(ValueError, match="no XPS could be built"):
            mv.elec.xps(md, [None], level="model")


def _has_boltztrap2() -> bool:
    try:
        import BoltzTraP2  # noqa: F401
        from pymatgen.electronic_structure.boltztrap2 import (  # noqa: F401
            BztTransportProperties)
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _has_boltztrap2(),
                    reason="BoltzTraP2 absent (conda-forge carries it only "
                           "to a py310 build)")
class TestTransport:
    """A symmetric two-band model has a Seebeck coefficient that is odd in the
    chemical potential and a conductivity that is even in it. Neither depends
    on any convention, so they are the test."""

    @staticmethod
    def _model():
        from pymatgen.core import Lattice, Structure
        from pymatgen.electronic_structure.bandstructure import BandStructure
        from pymatgen.electronic_structure.core import Spin
        cell = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
        n = 8
        kpoints = np.array([[i / n, j / n, k / n]
                            for i in range(n) for j in range(n)
                            for k in range(n)])
        energy = 3.0 * np.sum((kpoints - 0.5) ** 2, axis=1)
        bands = {Spin.up: np.vstack([-energy - 0.5, energy + 0.5])}
        return cell, BandStructure(kpoints, bands,
                                   cell.lattice.reciprocal_lattice,
                                   efermi=0.0, structure=cell)

    @classmethod
    def _at(cls, mu):
        cell, band_structure = cls._model()
        md = mv.data.from_structures([cell])
        mv.elec.transport(md, [band_structure], level="m", mu=mu)
        return (float(md.obs["seebeck_m"].iloc[0]),
                float(md.obs["sigma_over_tau_m"].iloc[0]), md)

    def test_a_symmetric_band_has_no_seebeck_at_its_centre(self):
        seebeck, _sigma, _md = self._at(0.0)
        assert seebeck == pytest.approx(0.0, abs=1e-6)

    def test_electrons_and_holes_give_opposite_signs(self):
        n_type, sigma_n, _ = self._at(0.6)
        p_type, sigma_p, _ = self._at(-0.6)
        assert n_type == pytest.approx(-p_type, rel=1e-6)
        assert sigma_n == pytest.approx(sigma_p, rel=1e-6)
        assert n_type < 0 < p_type

    def test_heavier_doping_lowers_the_seebeck_coefficient(self):
        """More carriers, less entropy per carrier. The trade-off against
        conductivity is the whole difficulty of thermoelectrics."""
        light, sigma_light, _ = self._at(0.6)
        heavy, sigma_heavy, _ = self._at(0.9)
        assert abs(heavy) < abs(light)
        assert sigma_heavy > sigma_light

    def test_the_gap_conducts_nothing(self):
        """Exponentially small, and it underflows to zero. A true answer and
        a useless one, which is why mu is a parameter."""
        _seebeck, sigma, _md = self._at(0.0)
        assert sigma == pytest.approx(0.0, abs=1e-12)

    def test_the_power_factor_is_the_two_of_them(self):
        seebeck, sigma, md = self._at(0.6)
        assert float(md.obs["power_factor_m"].iloc[0]) == \
            pytest.approx(seebeck ** 2 * sigma * 1e-12, rel=1e-9)

    def test_the_doping_and_temperature_are_recorded(self):
        _s, _c, md = self._at(0.6)
        record = md.uns["transport"]["m"]
        assert record["mu"] == 0.6
        assert record["temperature"] == 300.0
        assert "independent of tau" in record["caveat"]

    def test_one_band_structure_per_row(self):
        cell, band_structure = self._model()
        md = mv.data.from_structures([cell, cell])
        with pytest.raises(ValueError, match="one per row"):
            mv.elec.transport(md, [band_structure], level="m")

    def test_a_missing_band_structure_is_recorded(self):
        cell, band_structure = self._model()
        md = mv.data.from_structures([cell, cell])
        mv.elec.transport(md, [band_structure, None], level="m", mu=0.6)
        assert np.isnan(float(md.obs["seebeck_m"].iloc[1]))
        assert any("no band structure" in e
                   for e in md.uns["transport"]["m"]["errors"])

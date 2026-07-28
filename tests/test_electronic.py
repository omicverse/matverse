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

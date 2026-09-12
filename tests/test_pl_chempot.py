"""mv.pl.chempot: the domains mv.thermo.chempot_diagram stored, and the
windows mv.thermo.chempot_limits stored."""

from __future__ import annotations

import pytest

import matverse as mv


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg")


def _fcc(symbol, a):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def _b2(a_sym, b_sym, a):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [a_sym, b_sym],
                     [[0, 0, 0], [.5, .5, .5]])


def _l12(host, guest, a):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


@pytest.fixture(scope="module")
def binary():
    """Al-Ni with formation energies supplied: Al3Ni at -1.8, AlNi at -1.4."""
    md = mv.data.from_structures([_l12("Al", "Ni", 3.78), _b2("Al", "Ni", 2.89),
                                  _fcc("Al", 4.05), _fcc("Ni", 3.52)])
    mv.pp.describe(md)
    md.obs["energy_demo"] = [-1.8, -1.4, 0.0, 0.0]
    mv.thermo.chempot_diagram(md, level="demo")
    mv.thermo.chempot_limits(md, level="demo")
    return md


@pytest.fixture(scope="module")
def ternary():
    md = mv.data.from_structures([_fcc("Al", 4.05), _fcc("Cu", 3.61),
                                  _fcc("Ni", 3.52), _b2("Al", "Ni", 2.89)])
    mv.calc.energy(md, level="emt")
    mv.thermo.chempot_diagram(md, level="emt")
    return md


class TestDiagram:
    def test_a_binary_is_segments_in_the_plane(self, binary):
        ax = mv.pl.chempot(binary)
        assert ax._matverse_n_domains == 4
        assert len(ax.lines) == 4
        assert ax.get_xlabel() == "Δμ(Al) (eV)"
        assert ax.get_ylabel() == "Δμ(Ni) (eV)"
        assert "demo" in ax.get_title() and "cut at" in ax.get_title()
        # The artificial floor at -50 eV is not drawn; the axis stops one eV
        # below the lowest physical vertex (-1.8).
        assert ax.get_xlim()[0] == pytest.approx(-2.8)
        for line in ax.lines:
            assert (line.get_xdata() >= -2.8 - 1e-9).all()

    def test_the_floor_is_an_argument(self, binary):
        ax = mv.pl.chempot(binary, limit=-4.0)
        assert ax.get_ylim()[0] == pytest.approx(-4.0)

    def test_a_ternary_is_polygons_in_a_volume(self, ternary):
        ax = mv.pl.chempot(ternary)
        assert ax.name == "3d"
        assert ax._matverse_n_domains >= 3
        assert len(ax.collections) == ax._matverse_n_domains
        assert ax.get_zlabel().startswith("Δμ(")

    def test_says_what_is_missing(self):
        md = mv.data.from_compositions(["AlNi"])
        with pytest.raises(ValueError, match="mv.thermo.chempot_diagram"):
            mv.pl.chempot(md)
        with pytest.raises(ValueError, match="kind must be"):
            mv.pl.chempot(md, kind="triangle")


class TestWindow:
    def test_one_bar_per_phase_with_a_window(self, binary):
        ax = mv.pl.chempot(binary, kind="window", element="Ni")
        # The elemental references have no window; the two compounds do.
        assert ax._matverse_n_windows == 2
        assert [t.get_text() for t in ax.get_yticklabels()] == ["Al3Ni", "AlNi"]
        assert ax.get_xlabel() == "μ(Ni) (eV)"
        assert "bounded by this dataset" in ax.get_title()

    def test_the_element_defaults_to_the_first_seen(self, binary):
        assert mv.pl.chempot(binary, kind="window").get_xlabel() == "μ(Al) (eV)"

    def test_says_what_is_missing(self, binary):
        with pytest.raises(ValueError, match="no window for element 'O'"):
            mv.pl.chempot(binary, kind="window", element="O")
        md = mv.data.from_compositions(["AlNi"])
        with pytest.raises(ValueError, match="mv.thermo.chempot_limits"):
            mv.pl.chempot(md, kind="window")

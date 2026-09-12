"""mv.pl.pourbaix draws the map mv.thermo.pourbaix stores.

The producer needs Materials Project's aqueous entries and a key, so the map
here is synthetic and shaped exactly as mv.thermo.pourbaix deposits it. The
plot is what is under test; the deposit's shape is pinned by the probe of the
producer's own claim.
"""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg")


@pytest.fixture
def aqueous():
    md = mv.data.from_compositions(["Fe2O3", "TiO2"])
    md.obs["name"] = ["Fe2O3", "TiO2"]
    ph = np.linspace(-2.0, 16.0, 37)
    potential = np.linspace(-2.0, 3.0, 26)
    P, E = np.meshgrid(ph, potential)
    md.obs["pourbaix_decomposition"] = [0.35, np.nan]
    md.uns["pourbaix"] = {
        "ph": 7.0, "potential": 0.0, "n_failed": 1,
        "errors": ["TiO2: KeyError: no matching Pourbaix entry"],
        "ph_grid": ph, "potential_grid": potential, "unit": "eV/atom",
        # A band of stability along a Nernst line, as an oxide would show.
        "maps": {"0": 0.6 * np.abs(E - (0.8 - 0.059 * P))},
    }
    return md


class TestPourbaix:
    def test_draws_the_surface_the_window_and_the_point(self, aqueous):
        ax = mv.pl.pourbaix(aqueous)
        assert ax.collections, "the filled contours"
        assert len(ax.lines) == 2, "the two water-window lines"
        assert ax.get_xlabel() == "pH"
        assert "V vs SHE" in ax.get_ylabel()
        labels = [t.get_text() for t in ax.texts]
        assert any("0.35 eV/atom" in t for t in labels), \
            "the value at the evaluated point is written next to it"
        assert "Fe2O3" in ax.get_title()
        colorbar = ax.figure.axes[-1]
        assert "eV/atom" in colorbar.get_ylabel()

    def test_the_threshold_is_an_argument(self, aqueous):
        ax = mv.pl.pourbaix(aqueous, threshold=0.2)
        assert ax._matverse_threshold == 0.2
        assert "0.2" in ax.get_title()

    def test_a_row_without_a_map_is_named(self, aqueous):
        with pytest.raises(ValueError, match="no Pourbaix map stored for 'TiO2'"):
            mv.pl.pourbaix(aqueous, row="TiO2")
        with pytest.raises(ValueError, match="no Pourbaix map stored for 'TiO2'"):
            mv.pl.pourbaix(aqueous, row=1)

    def test_says_what_is_missing(self):
        md = mv.data.from_compositions(["Fe2O3"])
        with pytest.raises(ValueError, match="mv.thermo.pourbaix"):
            mv.pl.pourbaix(md)


class TestProducerSignature:
    def test_the_producer_takes_the_map_grid(self):
        """The map is deposited by mv.thermo.pourbaix; without a key it must
        still refuse before touching the network, with the new arguments
        accepted."""
        import inspect
        params = inspect.signature(mv.thermo.pourbaix).parameters
        assert {"ph_range", "potential_range", "n_grid"} <= set(params)
        md = mv.data.from_compositions(["Fe2O3"])
        try:
            with pytest.raises(ValueError, match="MP_API_KEY"):
                mv.thermo.pourbaix(md, api_key=None, n_grid=5)
        except ImportError:
            pytest.skip("mp-api is not installed")

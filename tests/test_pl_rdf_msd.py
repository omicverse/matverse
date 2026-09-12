"""mv.pl.rdf_msd, and the MSD trace mv.md.run now keeps for it."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg")


def _frames(md, n=20):
    cell = mv.structures(md, "input")[0]
    frames = np.tile(np.array(cell.frac_coords), (n, 1, 1))
    return frames + np.random.default_rng(0).normal(0, 0.01, frames.shape)


@pytest.fixture(scope="module")
def dynamic():
    md = mv.datasets.metals(["Cu"], supercell=(2, 2, 2))
    mv.pp.describe(md)
    mv.md.run(md, level="emt", steps=40, equilibration=10, sample_every=5)
    return md


@pytest.fixture(scope="module")
def diffusing(dynamic):
    md = dynamic.copy()
    pytest.importorskip("pymatgen.analysis.diffusion.aimd.rdf")
    mv.md.rdf(md, _frames(md), species="Cu", reference="Cu", r_max=6.0)
    return md


class TestTheProducerKeepsTheTrace:
    def test_run_deposits_the_msd_on_the_sampling_grid(self, dynamic):
        assert "md_msd_trace_emt" in dynamic.obsm
        trace = dynamic.obsm["md_msd_trace_emt"]
        assert trace.shape == (1, 8)
        assert np.isfinite(trace).all() and (trace >= 0).all()
        np.testing.assert_array_equal(
            mv.grid_of(dynamic, "md_msd_trace"),
            mv.grid_of(dynamic, "md_temperature_trace"))
        assert dynamic.uns["grids"]["md_msd_trace"]["unit"] == "ps"
        assert trace[0, -1] == pytest.approx(
            float(dynamic.obs["msd_emt"].iloc[0])), \
            "the scalar MSD is the last point of the trace"

    def test_a_rerun_of_a_different_length_replaces_the_trace(self, dynamic):
        md = dynamic.copy()
        mv.md.run(md, level="emt", steps=20, equilibration=10, sample_every=5)
        assert md.obsm["md_msd_trace_emt"].shape == (1, 4)
        assert len(mv.grid_of(md, "md_msd_trace")) == 4

    def test_the_claim_is_in_the_registry(self):
        entry = mv.registry.get("mv.md.run")
        assert "md_msd_trace_{level}" in entry["produces"]["obsm"]


class TestRdfMsd:
    def test_both_panels_with_units(self, diffusing):
        ax = mv.pl.rdf_msd(diffusing, level="emt")
        msd = ax._matverse_msd_ax
        assert ax.figure is msd.figure and ax is not msd
        assert "Å" in ax.get_xlabel() and ax.get_ylabel() == "g(r)"
        assert "ps" in msd.get_xlabel() and "Å²" in msd.get_ylabel()
        assert len(ax.lines) >= 1
        assert len(msd.lines) == 2, "the MSD and the 6Dt line"
        assert "D = " in msd.get_legend().get_texts()[0].get_text()
        assert "Cu" in ax.get_title() and "emt" in msd.get_title()

    def test_the_slope_is_the_recorded_diffusivity(self, diffusing):
        from matverse.md import _A2_PER_PS_TO_CM2_PER_S
        msd = mv.pl.rdf_msd(diffusing, level="emt")._matverse_msd_ax
        dashed = [l for l in msd.lines if l.get_linestyle() == "--"][0]
        x, y = dashed.get_xdata(), dashed.get_ydata()
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        d = float(diffusing.obs["diffusivity_emt"].iloc[0])
        assert slope == pytest.approx(6.0 * d / _A2_PER_PS_TO_CM2_PER_S)

    def test_one_panel_alone(self, dynamic, diffusing):
        ax = mv.pl.rdf_msd(dynamic, level="emt", which="msd")
        assert "Å²" in ax.get_ylabel() and not hasattr(ax, "_matverse_msd_ax")
        ax = mv.pl.rdf_msd(diffusing, which="rdf")
        assert ax.get_ylabel() == "g(r)"

    def test_says_which_producer_is_missing(self, dynamic, diffusing):
        with pytest.raises(ValueError, match="mv.md.rdf"):
            mv.pl.rdf_msd(dynamic, level="emt")
        with pytest.raises(ValueError, match="mv.md.run"):
            mv.pl.rdf_msd(diffusing, level="not-run")
        with pytest.raises(ValueError, match="which must be"):
            mv.pl.rdf_msd(diffusing, which="all")

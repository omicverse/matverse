"""mv.pl.neb: the profile mv.neb.barrier recorded, with the barrier on it."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg")


@pytest.fixture(scope="module")
def barred():
    md = mv.datasets.metals(["Cu"])
    mv.pp.describe(md)
    mv.neb.hop_endpoints(md, "Cu", supercell=(1, 1, 1), key_added="hop")
    mv.neb.barrier(md, "hop_initial", "hop_final", level="emt", n_images=5,
                   steps=5)
    return md


class TestNeb:
    def test_draws_the_images_and_annotates_the_barrier(self, barred):
        ax = mv.pl.neb(barred, level="emt")
        (line,) = [l for l in ax.lines if len(l.get_xdata()) == 5]
        assert line.get_marker() == "o", "images are points, not a curve"
        np.testing.assert_allclose(line.get_xdata(), np.linspace(0, 1, 5))
        assert "eV" in ax.get_ylabel()
        assert "path coordinate" in ax.get_xlabel()
        assert "emt" in ax.get_title()
        barrier = float(barred.obs["barrier_emt"].iloc[0])
        assert ax._matverse_barriers == {"Cu": pytest.approx(barrier)}
        assert any(f"{barrier:.2f} eV" == t.get_text() for t in ax.texts)

    def test_an_unconverged_band_is_dashed_and_says_so(self, barred):
        md = barred.copy()
        md.obs["neb_converged_emt"] = [False]
        ax = mv.pl.neb(md, level="emt")
        (line,) = [l for l in ax.lines if len(l.get_xdata()) == 5]
        assert line.get_linestyle() == "--"
        assert "not converged" in line.get_label()

    def test_a_failed_band_is_skipped_rather_than_drawn(self, barred):
        md = barred.copy()
        md.obsm["neb_profile_emt"] = np.full_like(md.obsm["neb_profile_emt"],
                                                  np.nan)
        ax = mv.pl.neb(md, level="emt")
        assert ax._matverse_barriers == {}

    def test_says_what_is_missing(self):
        md = mv.datasets.metals(["Cu"])
        with pytest.raises(ValueError, match="mv.neb.barrier"):
            mv.pl.neb(md, level="emt")

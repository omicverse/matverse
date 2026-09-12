"""mv.pl.phonon: the DOS mv.prop.phonon stored, and the dispersion beside it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import matverse as mv


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg")


@pytest.fixture(scope="module")
def vibrating():
    md = mv.datasets.metals(["Cu", "Al"])
    mv.pp.describe(md)
    mv.prop.phonon(md, level="emt", supercell=(1, 1, 1))
    return md


def _dispersion_like(names, n_points=20, n_branches=3):
    """The layout mv.prop.dispersion returns, built without phonopy."""
    rows, material = [], []
    for name in names:
        for b in range(n_branches):
            rows.append(np.abs(np.sin(np.linspace(0, np.pi, n_points))) * (b + 1))
            material.append(name)
    ph = AnnData(
        X=np.vstack(rows),
        obs=pd.DataFrame({"material": pd.Categorical(material)},
                         index=[f"{m}:{i}" for i, m in enumerate(material)]),
        var=pd.DataFrame({"path_fraction": np.linspace(0, 1, n_points)},
                         index=[f"q{i}" for i in range(n_points)]))
    ph.uns["y_label"] = "frequency (THz)"
    ph.uns["path_labels"] = {name: {0.0: "Γ", 0.5: "X", 1.0: "Γ"} for name in names}
    return ph


class TestPhonon:
    def test_draws_one_curve_per_row_with_units(self, vibrating):
        ax = mv.pl.phonon(vibrating, level="emt")
        assert len(ax.lines) == 2
        assert "THz" in ax.get_xlabel()
        assert "1/THz" in ax.get_ylabel()
        assert "emt" in ax.get_title()
        assert "commensurate" in ax.get_title(), "the grid records the method"
        assert set(ax._matverse_n_imaginary) == {"Cu", "Al"}

    def test_the_dispersion_shares_the_frequency_axis(self, vibrating):
        ph = _dispersion_like([str(n) for n in vibrating.obs_names])
        ax = mv.pl.phonon(vibrating, level="emt", rows=[0], dispersion=ph)
        left = ax._matverse_dispersion_ax
        assert left.get_shared_y_axes().joined(left, ax)
        assert left._matverse_n_bands == 3, "only the requested row's branches"
        assert [t.get_text() for t in left.get_xticklabels()] == ["Γ", "X", "Γ"]
        assert "1/THz" in ax.get_xlabel()
        assert len(ax.lines) == 1

    def test_imaginary_modes_are_named_in_the_legend(self, vibrating):
        md = vibrating.copy()
        md.obs["n_imaginary_modes_emt"] = [2, 0]
        ax = mv.pl.phonon(md, level="emt")
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts[0].endswith("(2 imaginary)") and texts[1] == "Al"

    def test_says_what_is_missing(self, vibrating):
        bare = mv.datasets.metals(["Cu"])
        with pytest.raises(ValueError, match="mv.prop.phonon"):
            mv.pl.phonon(bare, level="emt")
        with pytest.raises(ValueError, match="mv.prop.phonon"):
            mv.pl.phonon(vibrating, level="not-run")

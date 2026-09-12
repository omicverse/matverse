"""mv.pl.wulff, and the polyhedron mv.surf.wulff now keeps for it."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg")


@pytest.fixture(scope="module")
def shaped():
    bulk = mv.datasets.metals(["Cu"])
    mv.pp.describe(bulk)
    mv.calc.energy(bulk, level="emt")
    facets = mv.surf.slabs(bulk, max_index=1)
    mv.calc.energy(facets, level="emt")
    mv.surf.surface_energy(facets, bulk, level="emt")
    mv.surf.wulff(facets, bulk, level="emt")
    return bulk, facets


class TestTheProducerKeepsTheShape:
    def test_every_face_on_the_particle_is_stored(self, shaped):
        bulk, facets = shaped
        shape = bulk.uns["wulff"]["emt"]["0"]
        families = list(shape["face_miller"])
        # fcc cut to max_index=1: 8 (111), 6 (100), 12 (110) faces.
        assert sorted(set(families)) == ["1_0_0", "1_1_0", "1_1_1"]
        assert len(families) == 26
        vertices = np.asarray(shape["vertices"])
        index = np.asarray(shape["face_index"])
        assert vertices.shape == (len(index), 3)
        assert index.max() == 25 and (np.bincount(index) >= 3).all()
        expressed = facets.obs.set_index("miller")["wulff_area_fraction_emt"]
        for miller, fraction in shape["area_fractions"].items():
            assert fraction == pytest.approx(float(expressed[miller]))

    def test_the_claim_is_back_in_the_registry(self):
        entry = mv.registry.get("mv.surf.wulff")
        assert entry["produces"]["bulk.uns"] == ["wulff"]

    def test_it_survives_h5ad(self, shaped, tmp_path):
        import anndata
        bulk, _ = shaped
        bulk.write_h5ad(tmp_path / "bulk.h5ad")
        back = anndata.read_h5ad(tmp_path / "bulk.h5ad")
        assert np.asarray(back.uns["wulff"]["emt"]["0"]["vertices"]).shape \
            == np.asarray(bulk.uns["wulff"]["emt"]["0"]["vertices"]).shape


class TestWulff:
    def test_draws_the_polyhedron_coloured_by_family(self, shaped):
        bulk, _ = shaped
        ax = mv.pl.wulff(bulk, level="emt")
        assert ax.name == "3d"
        assert ax._matverse_n_faces == 26
        (collection,) = ax.collections
        assert len(collection.get_facecolor()) == 26
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert len(labels) == 3
        assert any(l.startswith("(1 1 1)") and "% of the surface" in l
                   for l in labels)
        assert "Cu" in ax.get_title() and "emt" in ax.get_title()

    def test_a_row_can_be_named(self, shaped):
        bulk, _ = shaped
        assert mv.pl.wulff(bulk, level="emt", row="Cu")._matverse_n_faces == 26

    def test_says_what_is_missing(self, shaped):
        bulk, _ = shaped
        bare = mv.datasets.metals(["Cu"])
        with pytest.raises(ValueError, match="mv.surf.wulff"):
            mv.pl.wulff(bare, level="emt")
        stale = bulk.copy()
        stale.uns["wulff"]["emt"]["0"] = {"n_facets": 3, "anisotropy": 0.05}
        with pytest.raises(ValueError, match="older mv.surf.wulff"):
            mv.pl.wulff(stale, level="emt")

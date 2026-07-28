"""Working past memory: streaming construction and chunked compute.

Alexandria is 5.06M entries and OMat24 is roughly 110M calculations. None of
that is exercised here — these tests use a few hundred structures, because what
needs testing is the *shape* of the code path, not the size. A constructor that
takes a list, or an operation that decodes every structure before doing
anything, rules out those corpora regardless of how much memory the machine has.
"""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


def _library(n: int):
    """A stream of n structures spanning eleven elements."""
    from pymatgen.core import Lattice, Structure

    elements = ["Al", "Cu", "Ni", "Ag", "Au", "Pd", "Pt", "C", "N", "O", "H"]
    for i in range(n):
        yield Structure(Lattice.cubic(3.5 + 0.01 * (i % 20)),
                        [elements[i % len(elements)]] * 4,
                        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


class TestStreamingConstruction:
    def test_a_stream_becomes_a_dataset(self):
        md = mv.data.from_iterable(_library(250), chunk_size=40)
        assert md.n_obs == 250
        assert md.uns["X_is"] == "composition_atoms_reduced"

    def test_the_element_axis_is_unioned_across_blocks(self):
        """A block containing an element no earlier block had must widen the
        axis rather than fail."""
        md = mv.data.from_iterable(_library(250), chunk_size=3)
        assert md.n_vars == 11
        assert set(md.var_names) >= {"Al", "H", "O"}

    def test_chunking_does_not_change_the_result(self):
        one = mv.data.from_iterable(_library(60), chunk_size=1000)
        many = mv.data.from_iterable(_library(60), chunk_size=7)
        assert one.shape == many.shape
        assert sorted(one.var_names) == sorted(many.var_names)
        column = [list(one.var_names).index(e) for e in many.var_names]
        assert np.allclose(one.X.toarray()[:, column], many.X.toarray())

    def test_it_stops_at_max_n(self):
        md = mv.data.from_iterable(_library(10_000), chunk_size=50, max_n=120)
        assert md.n_obs == 120

    def test_obs_names_are_unique_across_blocks(self):
        md = mv.data.from_iterable(_library(90), chunk_size=10)
        assert len(set(md.obs_names)) == md.n_obs

    def test_the_conventions_survive_concatenation(self):
        md = mv.data.from_iterable(_library(90), chunk_size=10)
        for key in ("features", "levels", "provenance"):
            assert key in md.uns
        mv.pp.describe(md)                       # still an ordinary dataset
        assert "formula" in md.obs

    def test_an_empty_stream_says_so(self):
        with pytest.raises(ValueError, match="no structures"):
            mv.data.from_iterable(iter([]))


class TestWindowedDecode:
    def test_only_the_requested_rows_are_decoded(self):
        md = mv.data.from_iterable(_library(200), chunk_size=50)
        window = mv.structures(md, rows=[0, 5, 10])
        assert len(window) == 3
        assert window[0].composition.reduced_formula == "Al"

    def test_a_boolean_mask_works_too(self):
        md = mv.data.from_iterable(_library(20), chunk_size=5)
        mask = np.zeros(md.n_obs, dtype=bool)
        mask[[1, 3]] = True
        assert len(mv.structures(md, rows=mask)) == 2

    def test_a_window_matches_the_full_decode(self):
        md = mv.data.from_iterable(_library(30), chunk_size=10)
        everything = mv.structures(md)
        window = mv.structures(md, rows=[7, 12])
        assert window[0] == everything[7]
        assert window[1] == everything[12]


class TestChunkedCompute:
    @pytest.fixture
    def streamed(self):
        md = mv.data.from_iterable(_library(120), chunk_size=40)
        mv.pp.describe(md)
        return md

    def test_chunks_cover_every_row_exactly_once(self, streamed):
        seen = 0
        for start, block in mv.utils.chunks(streamed, 25):
            assert block.n_obs <= 25
            seen += block.n_obs
        assert seen == streamed.n_obs

    def test_results_merge_back_onto_the_parent(self, streamed):
        report = mv.utils.map_chunks(
            streamed, lambda block: mv.calc.energy(block, level="emt"), size=30)
        assert report["n_processed"] == streamed.n_obs
        assert "energy_emt" in streamed.obs
        assert np.isfinite(
            streamed.obs["energy_emt"].to_numpy(dtype=float)).all()

    def test_the_chunked_result_matches_the_whole(self, streamed):
        mv.calc.energy(streamed, level="emt")
        whole = streamed.obs["energy_emt"].to_numpy(dtype=float).copy()
        del streamed.obs["energy_emt"]

        mv.utils.map_chunks(
            streamed, lambda block: mv.calc.energy(block, level="emt"), size=17)
        assert np.allclose(streamed.obs["energy_emt"].to_numpy(dtype=float),
                           whole)

    def test_structure_variants_merge_back(self, streamed):
        mv.utils.map_chunks(
            streamed,
            lambda block: mv.calc.relax(block, level="emt", fmax=0.3, steps=5),
            size=40)
        assert "relaxed_emt" in mv.variants(streamed)
        assert len(mv.structures(streamed, "relaxed_emt")) == streamed.n_obs

    def test_feature_blocks_merge_back(self, streamed):
        mv.utils.map_chunks(
            streamed, lambda block: mv.feat.element_stats(block), size=40)
        assert streamed.obsm["X_element_stats"].shape[0] == streamed.n_obs
        assert np.isfinite(streamed.obsm["X_element_stats"]).any()

    def test_skip_if_resumes_rather_than_restarts(self, streamed):
        mv.utils.map_chunks(
            streamed, lambda block: mv.calc.energy(block, level="emt"), size=30)
        values = streamed.obs["energy_emt"].to_numpy(dtype=float).copy()
        values[:45] = np.nan                     # pretend a job died partway
        streamed.obs["energy_emt"] = values

        report = mv.utils.map_chunks(
            streamed, lambda block: mv.calc.energy(block, level="emt"),
            size=30, skip_if="energy_emt")
        assert report["n_skipped"] > 0
        assert report["n_processed"] < streamed.n_obs
        assert np.isfinite(
            streamed.obs["energy_emt"].to_numpy(dtype=float)).all()

    def test_a_failing_block_is_recorded_not_raised(self, streamed):
        def explode(block):
            if block.n_obs and str(block.obs_names[0]) == "60":
                raise RuntimeError("simulated calculator failure")
            mv.calc.energy(block, level="emt")

        report = mv.utils.map_chunks(streamed, explode, size=30)
        assert len(report["errors"]) == 1
        assert "simulated calculator failure" in report["errors"][0]
        assert report["n_processed"] == streamed.n_obs - 30

    def test_checkpointing_between_blocks(self, streamed, tmp_path):
        from matverse._core import records

        path = tmp_path / "run.h5ad"
        mv.utils.map_chunks(
            streamed, lambda block: mv.calc.energy(block, level="emt"),
            size=40, checkpoint_to=path)
        assert path.exists()
        assert len(records(streamed.uns, "checkpoints")) == 3

        import anndata
        back = anndata.read_h5ad(path)
        assert np.isfinite(
            back.obs["energy_emt"].to_numpy(dtype=float)).all()

    def test_a_bad_chunk_size_is_rejected(self, streamed):
        with pytest.raises(ValueError, match="at least 1"):
            list(mv.utils.chunks(streamed, 0))

"""Plotting, supervised models with honest splits, and design campaigns."""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


@pytest.fixture(scope="module")
def alloy_library():
    """Seven elementals plus every ordered binary, so splits have groups."""
    from pymatgen.core import Lattice, Structure

    def fcc(symbol, a):
        return Structure(Lattice.cubic(a), [symbol] * 4,
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    def l12(host, guest, a):
        return Structure(Lattice.cubic(a), [guest, host, host, host],
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    elements = ["Al", "Cu", "Ni", "Ag", "Au", "Pd", "Pt"]
    out = [fcc(e, 3.5 + 0.1 * i) for i, e in enumerate(elements)]
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            out.append(l12(a, b, 3.6 + 0.02 * i))
    return out


@pytest.fixture
def learned(alloy_library):
    """A library with descriptors and a learnable target."""
    md = mv.data.from_structures(list(alloy_library))
    mv.pp.describe(md)
    mv.feat.element_stats(md)
    rng = np.random.default_rng(0)
    md.obs["target"] = (md.obs["density"].to_numpy(dtype=float)
                        + rng.normal(0, 0.02, md.n_obs))
    return md


class TestSplits:
    def test_composition_split_keeps_relatives_together(self, learned):
        mv.model.split(learned, strategy="composition")
        from matverse._core import structures
        formulas = np.asarray([s.composition.reduced_formula
                               for s in structures(learned)])
        folds = learned.obs["split"].astype(str).to_numpy()
        for formula in set(formulas):
            sides = set(folds[formulas == formula])
            assert len(sides) == 1, f"{formula} landed on both sides"

    def test_random_split_is_recorded_as_leaky(self, learned):
        mv.model.split(learned, strategy="random")
        assert learned.uns["split"]["leaky"] is True
        mv.model.split(learned, strategy="composition")
        assert learned.uns["split"]["leaky"] is False

    def test_prototype_split_groups_structure_types(self, learned):
        mv.model.split(learned, strategy="prototype")
        assert learned.uns["split"]["n_groups"] < learned.n_obs

    def test_element_holdout_takes_everything_containing_it(self, learned):
        mv.model.split(learned, strategy="element", holdout="Ni")
        folds = learned.obs["split"].astype(str).to_numpy()
        has_ni = np.asarray(learned[:, "Ni"].X.todense()).ravel() > 0
        assert (folds[has_ni] == "test").all()
        assert (folds[~has_ni] == "train").all()

    def test_element_holdout_needs_an_element(self, learned):
        with pytest.raises(ValueError, match="needs holdout"):
            mv.model.split(learned, strategy="element")

    def test_unknown_element_is_named(self, learned):
        with pytest.raises(ValueError, match="not on the element axis"):
            mv.model.split(learned, strategy="element", holdout="Xe")

    def test_unknown_strategy_lists_the_options(self, learned):
        with pytest.raises(ValueError, match="unknown strategy"):
            mv.model.split(learned, strategy="whatever")


class TestFit:
    def test_prediction_is_a_level_of_theory(self, learned):
        mv.model.split(learned, strategy="composition")
        mv.model.fit(learned, target="target")
        assert "target_rf_pred" in learned.obs
        info = mv.level_info(learned, "rf_pred")
        assert info["kind"] == "model" and info["surrogate"] is True
        assert info["split_strategy"] == "composition"

    def test_ensemble_reports_an_uncertainty(self, learned):
        mv.model.split(learned, strategy="composition")
        mv.model.fit(learned, target="target", model="rf")
        assert "target_rf_pred_std" in learned.obs
        assert (learned.obs["target_rf_pred_std"] >= 0).all()

    def test_a_model_without_an_ensemble_reports_none(self, learned):
        mv.model.split(learned, strategy="composition")
        mv.model.fit(learned, target="target", model="ridge",
                     level="ridge_pred")
        assert "target_ridge_pred_std" not in learned.obs
        assert mv.level_info(learned, "ridge_pred")["uncertainty"] is None

    def test_scores_land_on_the_object(self, learned):
        mv.model.split(learned, strategy="composition")
        mv.model.fit(learned, target="target")
        scores = learned.uns["model"]["rf_pred"]["test_scores"]
        assert scores["n"] > 0
        assert set(scores) >= {"mae", "rmse", "r2"}

    def test_it_learns_something(self, learned):
        """Density from element statistics is genuinely predictable."""
        mv.model.split(learned, strategy="composition")
        mv.model.fit(learned, target="target")
        assert learned.uns["model"]["rf_pred"]["test_scores"]["r2"] > 0.3

    def test_fit_needs_a_split(self, learned):
        with pytest.raises(ValueError, match="mv.model.split"):
            mv.model.fit(learned, target="target")

    def test_fit_needs_descriptors(self, learned):
        mv.model.split(learned, strategy="composition")
        with pytest.raises(ValueError, match="element_stats"):
            mv.model.fit(learned, target="target", use_rep="X_nothing")

    def test_available_reports_the_estimators(self):
        table = mv.model.available()
        assert table["rf"]["method"] == "random forest"

    def test_a_registered_model_is_usable(self, learned):
        from sklearn.linear_model import LinearRegression
        mv.model.register_model("ols", LinearRegression, method="OLS")
        mv.model.split(learned, strategy="composition")
        mv.model.fit(learned, target="target", model="ols", level="ols_pred")
        assert mv.level_info(learned, "ols_pred")["method"] == "OLS"


class TestCrossValidate:
    def test_random_splits_flatter_the_model(self, learned):
        """The whole reason mv.model.split defaults to grouping."""
        mv.model.cross_validate(learned, target="target", seeds=(0, 1, 2))
        results = learned.uns["cross_validate"]["results"]
        assert "random" in results and "composition" in results
        assert results["random"]["mae"]["mean"] < \
            results["composition"]["mae"]["mean"]
        assert learned.uns["cross_validate"]["leakage_mae"] > 0

    def test_it_cleans_up_after_itself(self, learned):
        mv.model.split(learned, strategy="element", holdout="Ni")
        mv.model.cross_validate(learned, target="target", seeds=(0,))
        assert "_cv_split" not in learned.obs
        assert "target__cv" not in learned.obs
        assert "_cv" not in learned.uns.get("levels", {})
        # The caller's own split survives.
        assert learned.uns["split"]["strategy"] == "element"


class TestCampaign:
    @pytest.fixture
    def campaign(self, learned):
        truth = learned.obs["target"].to_numpy(dtype=float).copy()
        learned.obs["truth"] = truth
        seeded = np.full(learned.n_obs, np.nan)
        seeded[:5] = truth[:5]
        learned.obs["objective"] = seeded
        mv.opt.start(learned, objective="objective", goal="min")
        return learned

    def _round(self, md, method="ucb", n=4):
        md.obs["split"] = np.where(np.asarray(md.obs["observed"], dtype=bool),
                                   "train", "test")
        mv.model.fit(md, target="objective", level="pred")
        mv.opt.suggest(md, n=n, method=method, predicted="objective_pred",
                       uncertainty="objective_pred_std")
        picks = np.where(np.asarray(md.obs["selected"], dtype=bool))[0]
        truth = md.obs["truth"].to_numpy(dtype=float)
        mv.opt.observe(md, values={str(md.obs_names[i]): truth[i]
                                   for i in picks})
        return picks

    def test_start_marks_what_is_known(self, campaign):
        assert int(np.asarray(campaign.obs["observed"], dtype=bool).sum()) == 5
        assert campaign.uns["campaign"]["round"] == 0

    def test_a_round_selects_only_unobserved_candidates(self, campaign):
        before = np.asarray(campaign.obs["observed"], dtype=bool).copy()
        picks = self._round(campaign)
        assert not before[picks].any()
        assert len(picks) == 4

    def test_rounds_accumulate_observations(self, campaign):
        for _ in range(3):
            self._round(campaign)
        assert int(np.asarray(campaign.obs["observed"], dtype=bool).sum()) == 17
        assert campaign.uns["campaign"]["round"] == 3

    def test_the_loop_finds_the_optimum(self, campaign):
        truth = campaign.obs["truth"].to_numpy(dtype=float)
        for _ in range(4):
            self._round(campaign)
        history = mv.opt.history(campaign)
        assert history["best_so_far"].iloc[-1] == pytest.approx(
            float(truth.min()), abs=1e-6)

    def test_history_is_a_table_of_rounds(self, campaign):
        self._round(campaign)
        self._round(campaign)
        history = mv.opt.history(campaign)
        assert list(history["round"]) == [1, 2]
        assert history.attrs["objective"] == "objective"

    def test_a_campaign_survives_a_save(self, campaign, tmp_path):
        """Rounds are stored so h5ad can write them; a list of dicts cannot."""
        self._round(campaign)
        self._round(campaign)
        for column in ("split", "truth"):
            if column in campaign.obs:
                del campaign.obs[column]
        path = tmp_path / "campaign.h5ad"
        campaign.write_h5ad(path)

        import anndata
        back = anndata.read_h5ad(path)
        history = mv.opt.history(back)
        assert list(history["round"]) == [1, 2]
        assert back.uns["campaign"]["objective"] == "objective"

    def test_uncertainty_methods_refuse_without_a_sigma(self, campaign):
        campaign.obs["bare"] = np.zeros(campaign.n_obs)
        with pytest.raises(ValueError, match="needs an uncertainty"):
            mv.opt.suggest(campaign, n=3, method="ucb", predicted="bare")

    def test_greedy_works_without_a_sigma(self, campaign):
        campaign.obs["bare"] = np.arange(campaign.n_obs, dtype=float)
        mv.opt.suggest(campaign, n=3, method="greedy", predicted="bare")
        assert int(np.asarray(campaign.obs["selected"], dtype=bool).sum()) == 3

    def test_random_is_available_as_the_baseline(self, campaign):
        mv.opt.suggest(campaign, n=3, method="random", predicted="objective")
        assert int(np.asarray(campaign.obs["selected"], dtype=bool).sum()) == 3

    def test_diversified_batches_are_spread_out(self, campaign):
        mv.pp.normalize_composition(campaign)
        mv.tl.pca(campaign, n_comps=3)
        campaign.obs["bare"] = np.arange(campaign.n_obs, dtype=float)
        mv.opt.suggest(campaign, n=4, method="greedy", predicted="bare",
                       diversify=True)
        picked = np.where(np.asarray(campaign.obs["selected"], dtype=bool))[0]
        Z = campaign.obsm["X_pca"][picked]
        spread = np.linalg.norm(Z[:, None] - Z[None, :], axis=2).max()

        mv.opt.suggest(campaign, n=4, method="greedy", predicted="bare")
        greedy = np.where(np.asarray(campaign.obs["selected"], dtype=bool))[0]
        Zg = campaign.obsm["X_pca"][greedy]
        assert spread >= np.linalg.norm(Zg[:, None] - Zg[None, :],
                                        axis=2).max()

    def test_observe_needs_a_suggestion(self, campaign):
        with pytest.raises(ValueError, match="mv.opt.suggest"):
            mv.opt.observe(campaign)

    def test_suggest_needs_a_campaign(self, learned):
        with pytest.raises(ValueError, match="mv.opt.start"):
            mv.opt.suggest(learned, n=3)


class TestPlots:
    """Every plot draws without error and returns an axis.

    Deliberately not pixel comparisons: those break on a matplotlib upgrade and
    catch nothing that matters. What is worth pinning is that each function
    accepts the object a pipeline actually produces.
    """

    @pytest.fixture(autouse=True)
    def _headless(self):
        import matplotlib
        matplotlib.use("Agg")

    @pytest.fixture
    def plotted(self, md):
        mv.pp.describe(md)
        mv.calc.energy(md, level="emt")
        with pytest.warns(UserWarning):
            mv.thermo.hull(md, level="emt")
        mv.pp.normalize_composition(md)
        mv.tl.pca(md, n_comps=2)
        mv.screen.pareto(md, {"e_above_hull_emt": "min", "density": "min"})
        md.obs["group"] = ["a", "b", "a", "b", "a", "b"]
        mv.tl.rank_elements_groups(md, "group")
        mv.prop.xrd(md, two_theta=(10, 50), step=0.1)
        md.var["n_materials"] = np.asarray((md.X > 0).sum(axis=0)).ravel()
        md.obs["energy_per_atom_shifted"] = \
            md.obs["energy_per_atom_emt"].to_numpy(dtype=float) + 0.1
        return md

    def test_set_style_changes_the_defaults(self):
        import matplotlib.pyplot as plt

        before = dict(plt.rcParams)
        try:
            mv.pl.set_style(dpi=123, fontsize=9, quiet=True)
            assert plt.rcParams["figure.dpi"] == 123
            assert plt.rcParams["font.size"] == 9
        finally:
            plt.rcParams.update(before)

    def test_set_style_leaves_later_settings_alone(self):
        """It touches rcParams and nothing else, so a figure styled by hand
        afterwards still wins."""
        import matplotlib.pyplot as plt

        before = dict(plt.rcParams)
        try:
            mv.pl.set_style(quiet=True)
            plt.rcParams["font.size"] = 20
            assert plt.rcParams["font.size"] == 20
        finally:
            plt.rcParams.update(before)

    def test_periodic_table(self, plotted):
        ax = mv.pl.periodic_table(plotted, color="n_materials")
        assert ax.get_figure() is not None

    def test_periodic_table_accepts_bare_values(self, plotted):
        ax = mv.pl.periodic_table(plotted, values=np.arange(plotted.n_vars),
                                  label="rank")
        assert ax is not None

    def test_periodic_table_rejects_a_wrong_length(self, plotted):
        with pytest.raises(ValueError, match="values for"):
            mv.pl.periodic_table(plotted, values=[1.0, 2.0])

    def test_rank_elements_groups(self, plotted):
        ax = mv.pl.rank_elements_groups(plotted, group="a")
        assert len(ax.patches) > 0

    def test_hull_marks_a_closed_system(self, plotted):
        ax = mv.pl.hull(plotted, level="emt", x="Al")
        assert "relative" in ax.get_title()

    def test_parity_annotates_the_error(self, plotted):
        ax = mv.pl.parity(plotted, "energy_per_atom", "emt", "shifted")
        assert "MAE" in ax.get_title()

    def test_pareto(self, plotted):
        ax = mv.pl.pareto(plotted, "e_above_hull_emt", "density")
        assert ax.get_xlabel() == "e_above_hull_emt"

    def test_embedding_with_a_categorical(self, plotted):
        ax = mv.pl.embedding(plotted, color="group")
        assert ax.get_legend() is not None

    def test_embedding_with_a_continuous_colour(self, plotted):
        ax = mv.pl.embedding(plotted, color="density")
        assert ax is not None

    def test_spectra(self, plotted):
        ax = mv.pl.spectra(plotted, "xrd", rows=[0, 1])
        assert len(ax.lines) == 2

    def test_provenance(self, plotted):
        ax = mv.pl.provenance(plotted)
        assert len(ax.patches) == len(mv.provenance(plotted))

    def test_a_plot_says_what_is_missing(self, md):
        with pytest.raises(ValueError, match="mv.thermo.hull"):
            mv.pl.hull(md, level="emt")


class TestRegistryNaming:
    def test_a_bare_name_shared_by_two_functions_is_ambiguous(self):
        """mv.pl mirrors mv.tl, the way scanpy pairs them.

        Neither owns the bare word, so the registry withdraws it from exact
        lookup rather than awarding it to whichever module imported first.
        """
        assert mv.registry.is_ambiguous("rank_elements_groups")
        assert mv.registry.get("rank_elements_groups") is None
        assert "names more than one function" in \
            mv.describe("rank_elements_groups")

    def test_both_stay_reachable_by_their_full_name(self):
        assert mv.registry.get("mv.tl.rank_elements_groups") is not None
        assert mv.registry.get("mv.pl.rank_elements_groups") is not None

    def test_an_explicit_alias_outranks_a_derived_name(self):
        """mv.screen.pareto claims 'pareto'; mv.pl.pareto only derives it."""
        entry = mv.registry.get("pareto")
        assert entry is not None
        assert entry["public_name"] == "mv.screen.pareto"


class TestGeneralPlots:
    """The three plots the tutorials were hand-drawing.

    Every assertion is about what ended up on the axis — how many points, how
    many lines, where the Fermi level sits — rather than that the call
    returned. A plot function that silently drops half its data returns an
    axis just as happily as one that does not.
    """

    @pytest.fixture
    def md(self):
        from pymatgen.core import Lattice, Structure
        cells = [Structure(Lattice.cubic(a), [s] * 4,
                           [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
                 for s, a in (("Cu", 3.61), ("Al", 4.05), ("Ni", 3.52),
                              ("Au", 4.08))]
        out = mv.data.from_structures(cells)
        mv.pp.describe(out)
        mv.calc.energy(out, level="emt")
        return out

    @pytest.fixture
    def band_object(self):
        import pandas as pd
        from anndata import AnnData
        n_points, n_bands = 60, 5
        X = np.vstack([np.cos(np.linspace(0, 3, n_points)) * k - 1.0
                       for k in range(1, n_bands + 1)])
        obs = pd.DataFrame(
            {"material": pd.Categorical(["Cu"] * 3 + ["Al"] * 2),
             "band_index": [0, 1, 2, 0, 1]},
            index=[f"b{i}" for i in range(n_bands)])
        var = pd.DataFrame({"path_fraction": np.linspace(0, 1, n_points)},
                           index=[str(i) for i in range(n_points)])
        return AnnData(X=X, obs=obs, var=var)

    # --- scatter -------------------------------------------------------

    def test_scatter_draws_every_row(self, md):
        ax = mv.pl.scatter(md, "volume", "energy_emt")
        assert len(ax.collections[0].get_offsets()) == md.n_obs

    def test_scatter_puts_the_columns_on_the_axes(self, md):
        ax = mv.pl.scatter(md, "volume", "energy_emt")
        assert ax.get_xlabel() == "volume"
        assert ax.get_ylabel() == "energy_emt"

    def test_a_numeric_colour_gets_a_colourbar(self, md):
        ax = mv.pl.scatter(md, "volume", "energy_emt", color="energy_emt")
        assert len(ax.figure.axes) > 1, "no colourbar was added"

    def test_a_categorical_colour_gets_a_legend(self, md):
        ax = mv.pl.scatter(md, "volume", "energy_emt", color="formula")
        assert len(ax.get_legend().get_texts()) == md.n_obs

    def test_too_many_categories_is_refused(self, md):
        md.obs["many"] = [f"g{i}" for i in range(md.n_obs)]
        big = mv.data.from_structures(mv.structures(md) * 4)
        big.obs["x"] = np.arange(big.n_obs, dtype=float)
        big.obs["y"] = np.arange(big.n_obs, dtype=float)
        big.obs["many"] = [f"g{i}" for i in range(big.n_obs)]
        with pytest.raises(ValueError, match="legend nobody can read"):
            mv.pl.scatter(big, "x", "y", color="many")

    def test_annotating_a_crowd_is_refused(self, md):
        big = mv.data.from_structures(mv.structures(md) * 20)
        big.obs["x"] = np.arange(big.n_obs, dtype=float)
        big.obs["y"] = np.arange(big.n_obs, dtype=float)
        big.obs["name"] = [str(i) for i in range(big.n_obs)]
        with pytest.raises(ValueError, match="on top of one another"):
            mv.pl.scatter(big, "x", "y", annotate="name")

    def test_scatter_names_a_missing_column(self, md):
        with pytest.raises(ValueError, match="absent"):
            mv.pl.scatter(md, "volume", "not_a_column")

    def test_scatter_works_on_the_sites_axis(self, md):
        """It only asks for obs columns, which is why it works on any axis."""
        sites = mv.multi.sites(md)
        sites.obs["a"] = np.arange(sites.n_obs, dtype=float)
        sites.obs["b"] = np.arange(sites.n_obs, dtype=float) ** 2
        ax = mv.pl.scatter(sites, "a", "b")
        assert len(ax.collections[0].get_offsets()) == sites.n_obs

    # --- bands ---------------------------------------------------------

    def test_bands_draws_one_line_per_band(self, band_object):
        ax = mv.pl.bands(band_object)
        assert ax._matverse_n_bands == band_object.n_obs

    def test_the_fermi_level_is_drawn_at_zero(self, band_object):
        ax = mv.pl.bands(band_object)
        dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
        assert len(dashed) == 1
        assert dashed[0].get_ydata()[0] == pytest.approx(0.0)

    def test_the_fermi_line_can_be_turned_off(self, band_object):
        ax = mv.pl.bands(band_object, highlight_fermi=False)
        assert not [ln for ln in ax.lines if ln.get_linestyle() == "--"]

    def test_selecting_a_material_draws_only_its_bands(self, band_object):
        ax = mv.pl.bands(band_object, materials=["Cu"])
        assert ax._matverse_n_bands == 3

    def test_an_unknown_material_is_named(self, band_object):
        with pytest.raises(ValueError, match="no bands for"):
            mv.pl.bands(band_object, materials=["Fe"])

    def test_high_symmetry_labels_land_on_the_ticks(self, band_object):
        ax = mv.pl.bands(band_object,
                         labels={0.0: "G", 0.5: "X", 1.0: "L"})
        assert [t.get_text() for t in ax.get_xticklabels()] == ["G", "X", "L"]

    def test_a_non_bands_object_is_refused(self, md):
        with pytest.raises(ValueError, match="not a bands object"):
            mv.pl.bands(md)

    # --- distribution --------------------------------------------------

    def test_distribution_uses_the_bins_asked_for(self, md):
        ax = mv.pl.distribution(md, "volume", bins=12)
        assert len(ax.patches) == 12

    def test_non_finite_values_are_dropped_and_counted(self, md):
        """A column that is half NaN otherwise makes a histogram that looks
        like a narrow distribution rather than a missing one."""
        md.obs["patchy"] = [1.0, np.nan, 3.0, np.nan]
        ax = mv.pl.distribution(md, "patchy")
        assert ax._matverse_dropped == 2
        assert "non-finite" in ax.get_xlabel()

    def test_groups_share_their_bins(self, md):
        """Two histograms on different edges are two pictures, not a
        comparison."""
        ax = mv.pl.distribution(md, "volume", by="formula", bins=10)
        edges = {round(float(p.get_x()), 9) for p in ax.patches}
        assert len(edges) == 10, "the groups were binned separately"

    def test_an_all_nan_column_is_refused(self, md):
        md.obs["empty"] = [np.nan] * md.n_obs
        with pytest.raises(ValueError, match="no finite value"):
            mv.pl.distribution(md, "empty")

    def test_too_many_groups_is_refused(self, md):
        big = mv.data.from_structures(mv.structures(md) * 3)
        big.obs["v"] = np.arange(big.n_obs, dtype=float)
        big.obs["many"] = [f"g{i}" for i in range(big.n_obs)]
        with pytest.raises(ValueError, match="hides all of them"):
            mv.pl.distribution(big, "v", by="many")

    # --- bar -----------------------------------------------------------

    def test_bar_draws_one_per_category(self, md):
        ax = mv.pl.scatter(md, "formula", "energy_emt", kind="bar")
        assert len(ax.patches) == md.n_obs

    def test_bar_heights_are_the_data(self, md):
        ax = mv.pl.scatter(md, "formula", "energy_emt", kind="bar")
        heights = sorted(p.get_height() for p in ax.patches)
        assert np.allclose(heights, sorted(md.obs["energy_emt"]))

    def test_bar_colours_by_a_categorical_column(self, md):
        """Three tutorials wanted exactly this — blue for the ones that
        passed, grey for the rest — and hand-drew it because color= did
        nothing on the bar branch."""
        md.obs["passes"] = [True, False, True, False]
        ax = mv.pl.scatter(md, "formula", "energy_emt", kind="bar",
                           color="passes")
        shades = {tuple(np.round(p.get_facecolor(), 3)) for p in ax.patches}
        assert len(shades) == 2
        assert len(ax.get_legend().get_texts()) == 2

    def test_a_bar_over_a_continuous_axis_is_refused(self, md):
        """That plot is a histogram, and mv.pl.distribution draws it."""
        with pytest.raises(ValueError, match="is numeric"):
            mv.pl.scatter(md, "volume", "energy_emt", kind="bar")

    def test_an_unknown_kind_is_refused(self, md):
        with pytest.raises(ValueError, match="'scatter' or 'bar'"):
            mv.pl.scatter(md, "formula", "energy_emt", kind="violin")

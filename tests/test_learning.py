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

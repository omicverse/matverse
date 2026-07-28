"""``mv.model`` — supervised property prediction.

A property model is a level of theory like any other. It gets a record in
``uns['levels']`` saying what it was trained on and how far it was validated, and
its predictions land in ``obs['<property>_<level>']`` beside PBE and the
spectrometer. Nothing about the object treats "predicted" as special, which is
the point: a model's number and a DFT number should not be mixed by accident, and
they cannot be here without naming both.

The split is the part worth caring about
----------------------------------------
Random train/test splits are the field's most common silent methodological
failure. A materials dataset is full of near-duplicates — the same composition in
a different setting, the same prototype with one element swapped — so a random
split puts relatives on both sides and reports a number that will not survive
contact with a genuinely new material.

:func:`split` therefore defaults to grouping by **composition**, and offers
grouping by prototype or holding out an element entirely. The cost is a worse
number. The number is the one you would have got anyway, later, from someone
else's data.

Backends
--------
Only ``sklearn`` is wired, because a random forest on element statistics is the
baseline every serious model has to beat and it needs no GPU. Graph networks and
fine-tuned interatomic potentials are the real tools for large data, and belong
behind :func:`register_model` rather than vendored — matverse owns the object and
the protocol, not the architectures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import record, set_level
from ._registry import register_function

#: name -> (factory, metadata). Populated by :func:`register_model`.
_MODELS: dict[str, tuple] = {}

SPLIT_STRATEGIES = ("composition", "prototype", "element", "random")


@register_function(
    aliases=["register model", "add estimator", "custom model",
             "register predictor"],
    category="model",
    description="Make an estimator available to mv.model.fit under a name, "
                "with the metadata its predictions will carry.",
    examples=["mv.model.register_model('gbr', lambda: "
              "GradientBoostingRegressor(), method='gradient boosting')"],
    related=["mv.model.fit", "mv.model.available"],
)
def register_model(name: str, factory, *, method: str | None = None,
                   uncertainty: str | None = None, **extra) -> None:
    """Register an estimator. ``factory()`` returns something with fit/predict."""
    _MODELS[name] = (factory, {"method": method or name,
                               "uncertainty": uncertainty, **extra})


def _builtin(name: str):
    """Estimators matverse knows how to build, if sklearn is installed."""
    if name in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestRegressor
        return (lambda: RandomForestRegressor(n_estimators=200,
                                              random_state=0),
                {"method": "random forest",
                 "uncertainty": "spread across trees (uncalibrated)"})
    if name in ("ridge",):
        from sklearn.linear_model import RidgeCV
        return (lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)),
                {"method": "ridge regression", "uncertainty": None})
    if name in ("gbr", "gradient_boosting"):
        from sklearn.ensemble import GradientBoostingRegressor
        return (lambda: GradientBoostingRegressor(random_state=0),
                {"method": "gradient boosting", "uncertainty": None})
    raise KeyError(
        f"unknown model {name!r}. Available here: {sorted(available())}. "
        f"Register your own with mv.model.register_model({name!r}, ...).")


def _get(name: str):
    return _MODELS[name] if name in _MODELS else _builtin(name)


@register_function(
    aliases=["available models", "which estimators", "list models"],
    category="model",
    description="List the estimators this installation can actually fit.",
    examples=["mv.model.available()"],
    related=["mv.model.register_model", "mv.model.fit"],
)
def available(check_imports: bool = True) -> dict:
    """Estimators runnable here, mapped to their metadata."""
    out = {name: dict(meta) for name, (_, meta) in _MODELS.items()}
    for name in ("rf", "ridge", "gbr"):
        if name in out:
            continue
        try:
            _, meta = _builtin(name)
            out[name] = dict(meta)
        except Exception as exc:
            if check_imports:
                out[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return out


@register_function(
    aliases=["split", "train test split", "leakage aware split", "grouped split",
             "cross validation split", "holdout"],
    category="model",
    description="Assign each material to a train or test fold, grouping "
                "near-duplicates on the same side so the score is not inflated "
                "by relatives appearing in both.",
    requires={"structures": ["input"]},
    produces={"obs": ["{key_added}"], "uns": ["split"]},
    dispatch="strategy='composition' groups by reduced formula; 'prototype' by "
             "formula anonymised and space group; 'element' holds out every "
             "material containing a named element; 'random' groups nothing and "
             "is recorded as the leaky baseline it is",
    examples=["mv.model.split(md)",
              "mv.model.split(md, strategy='element', holdout='Ni')"],
    related=["mv.model.fit", "mv.model.cross_validate"],
    notes="Defaults to grouping by composition. A random split puts the same "
          "composition in a different setting on both sides of the line and "
          "reports a number that will not survive a genuinely new material; "
          "when you ask for one anyway it is recorded as leaky so a later "
          "reader can see which kind of number they are looking at.",
)
def split(md: AnnData, strategy: str = "composition", test_size: float = 0.2,
          holdout=None, seed: int = 0, key_added: str = "split") -> None:
    """Deposit a train/test assignment that respects material similarity."""
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; use "
                         f"{list(SPLIT_STRATEGIES)}")

    if strategy == "element":
        assignment, meta = _element_holdout(md, holdout)
    else:
        groups = _groups(md, strategy)
        assignment, meta = _grouped_split(groups, test_size, seed)

    md.obs[key_added] = pd.Categorical(assignment,
                                       categories=["train", "test"])
    md.uns["split"] = {
        "strategy": strategy,
        "leaky": strategy == "random",
        "n_train": int((assignment == "train").sum()),
        "n_test": int((assignment == "test").sum()),
        "seed": seed,
        **meta,
    }
    record(md, "model.split", strategy=strategy, test_size=test_size,
           holdout=holdout, seed=seed)


def _groups(md: AnnData, strategy: str) -> np.ndarray:
    """The unit that must not be split across the train/test line."""
    from ._core import structures

    if strategy == "random":
        return np.arange(md.n_obs).astype(str)
    S = structures(md, "input")
    if strategy == "composition":
        return np.asarray([s.composition.reduced_formula for s in S])
    if strategy == "prototype":
        return np.asarray([_prototype(s) for s in S])
    raise ValueError(strategy)                            # pragma: no cover


def _prototype(structure) -> str:
    """Anonymised formula plus space group — 'the same structure type'.

    ``AB3`` in space group 221 covers CuAl3 and AlCu3 alike, which is what makes
    holding one out a real test of whether a model learned chemistry or learned
    the prototype.
    """
    try:
        anonymised = structure.composition.anonymized_formula
    except Exception:
        anonymised = structure.composition.reduced_formula
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        number = int(SpacegroupAnalyzer(structure, symprec=0.1)
                     .get_space_group_number())
    except Exception:
        number = 0
    return f"{anonymised}-{number}"


def _grouped_split(groups: np.ndarray, test_size: float, seed: int):
    """Whole groups to one side, until the test share is reached."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups.tolist())))
    rng.shuffle(unique)

    counts = {g: int((groups == g).sum()) for g in unique}
    target = test_size * len(groups)
    test_groups, taken = set(), 0
    for g in unique:
        if taken >= target:
            break
        test_groups.add(g)
        taken += counts[g]

    assignment = np.where(np.isin(groups, list(test_groups)), "test", "train")
    return assignment, {"n_groups": int(len(unique)),
                        "n_test_groups": int(len(test_groups))}


def _element_holdout(md: AnnData, holdout):
    """Every material containing the named element goes to test.

    The hardest honest split: it asks whether a model can say anything about
    chemistry it has never seen, which is what a discovery campaign needs and
    what a random split never measures.
    """
    if holdout is None:
        raise ValueError("strategy='element' needs holdout='<element>'")
    if md.n_vars == 0:
        raise ValueError("this object has no element axis (build_X=False)")
    wanted = [holdout] if isinstance(holdout, str) else list(holdout)
    names = list(map(str, md.var_names))
    missing = [e for e in wanted if e not in names]
    if missing:
        raise ValueError(f"element(s) {missing} are not on the element axis "
                         f"({names})")

    raw = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    columns = [names.index(e) for e in wanted]
    contains = (np.asarray(raw, dtype=float)[:, columns] > 0).any(axis=1)
    return (np.where(contains, "test", "train"),
            {"holdout": wanted, "n_containing": int(contains.sum())})


@register_function(
    aliases=["fit", "train model", "fit predictor", "learn property",
             "property prediction"],
    category="model",
    description="Fit an estimator to predict one property from a descriptor "
                "block, evaluate it on the held-out fold, and deposit the "
                "predictions as their own level of theory.",
    requires={"obs": ["{target}", "{split_key}"], "obsm": ["{use_rep}"]},
    produces={"obs": ["{target}_{level}"], "levels": ["{level}"],
              "uns": ["model"]},
    prerequisites=["mv.model.split", "mv.feat.element_stats"],
    dispatch="model= selects the estimator; each registered one is a separate "
             "route with its own uncertainty behaviour",
    examples=["mv.model.fit(md, target='band_gap_pbe')",
              "mv.model.fit(md, target='band_gap_pbe', model='ridge', "
              "level='ridge_pred')"],
    related=["mv.model.split", "mv.model.cross_validate", "mv.pl.parity"],
    notes="A prediction is a level of theory. It gets a record saying what it "
          "was trained on and how it was split, so a predicted number and a DFT "
          "number cannot be averaged together by accident.",
)
def fit(md: AnnData, target: str, use_rep: str = "X_element_stats",
        model: str = "rf", split_key: str = "split",
        level: str | None = None) -> None:
    """Train on the ``train`` fold, predict everywhere, score on ``test``."""
    if target not in md.obs:
        raise ValueError(f"obs[{target!r}] absent; available: "
                         f"{list(md.obs.columns)}")
    if use_rep not in md.obsm:
        raise ValueError(f"obsm[{use_rep!r}] absent; run "
                         f"mv.feat.element_stats first")
    if split_key not in md.obs:
        raise ValueError(f"obs[{split_key!r}] absent; run mv.model.split first")

    factory, meta = _get(model)
    level = level or f"{model}_pred"

    X = np.nan_to_num(np.asarray(md.obsm[use_rep], dtype=float), nan=0.0,
                      posinf=0.0, neginf=0.0)
    y = md.obs[target].to_numpy(dtype=float)
    folds = md.obs[split_key].astype(str).to_numpy()
    train = (folds == "train") & np.isfinite(y)
    test = (folds == "test") & np.isfinite(y)
    if train.sum() < 2:
        raise ValueError(f"only {int(train.sum())} usable training rows; a "
                         f"model needs at least two")

    estimator = factory()
    estimator.fit(X[train], y[train])
    predicted = np.asarray(estimator.predict(X), dtype=float)

    md.obs[f"{target}_{level}"] = predicted
    spread = _tree_spread(estimator, X)
    if spread is not None:
        md.obs[f"{target}_{level}_std"] = spread

    scores = _score(y[test], predicted[test]) if test.any() else {}
    md.uns.setdefault("model", {})[level] = {
        "target": target, "use_rep": use_rep, "model": model,
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "split": dict(md.uns.get("split", {})),
        "test_scores": scores,
    }
    set_level(md, level, kind="model", method=meta["method"],
              reference=_reference_of(md, target), surrogate=True,
              license=None, uncertainty=meta.get("uncertainty"),
              trained_on=target, split_strategy=md.uns.get("split", {})
              .get("strategy"), test_scores=scores)
    record(md, "model.fit", target=target, model=model, level=level)


def _reference_of(md: AnnData, target: str) -> str | None:
    """What the model was trained to reproduce — the target's own level."""
    for name in sorted(md.uns.get("levels", {}), key=len, reverse=True):
        if target.endswith(f"_{name}"):
            return name
    return target


def _tree_spread(estimator, X: np.ndarray):
    """Spread across an ensemble's members, when there is one to take."""
    members = getattr(estimator, "estimators_", None)
    if not members:
        return None
    try:
        stacked = np.vstack([np.asarray(m.predict(X), dtype=float)
                             for m in members])
    except Exception:
        return None
    return stacked.std(axis=0)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return {}
    residual = y_true[ok] - y_pred[ok]
    variance = float(np.var(y_true[ok]))
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.mean(residual ** 2) / variance)
        if variance > 0 else float("nan"),
    }


@register_function(
    aliases=["cross validate", "cross validation", "compare splits",
             "how leaky", "validate model"],
    category="model",
    description="Score one estimator under several splitting strategies at "
                "once, showing how much of an apparent score comes from "
                "relatives on both sides of the train/test line.",
    requires={"obs": ["{target}"], "obsm": ["{use_rep}"]},
    produces={"uns": ["cross_validate"]},
    prerequisites=["mv.feat.element_stats"],
    examples=["mv.model.cross_validate(md, target='band_gap_pbe')"],
    related=["mv.model.split", "mv.model.fit"],
    notes="Report the grouped number. The gap between it and the random number "
          "is the size of the leak, and publishing only the random one is how "
          "a model that memorised prototypes gets reported as a model that "
          "learned chemistry.",
)
def cross_validate(md: AnnData, target: str, use_rep: str = "X_element_stats",
                   model: str = "rf",
                   strategies=("random", "composition", "prototype"),
                   test_size: float = 0.2, seeds=(0, 1, 2)) -> None:
    """Score under several splits and record the gap between them."""
    original_split = md.obs["split"].copy() if "split" in md.obs else None
    original_uns = dict(md.uns.get("split", {}))

    results: dict[str, dict] = {}
    for strategy in strategies:
        runs = []
        for seed in seeds:
            split(md, strategy=strategy, test_size=test_size, seed=seed,
                  key_added="_cv_split")
            try:
                fit(md, target=target, use_rep=use_rep, model=model,
                    split_key="_cv_split", level="_cv")
            except ValueError:
                continue
            scores = md.uns.get("model", {}).get("_cv", {}).get("test_scores", {})
            if scores:
                runs.append(scores)
        if runs:
            results[strategy] = {
                metric: {"mean": float(np.mean([r[metric] for r in runs])),
                         "std": float(np.std([r[metric] for r in runs]))}
                for metric in ("mae", "rmse", "r2")
            }
            results[strategy]["n_seeds"] = len(runs)

    _cleanup(md, "_cv_split", "_cv", target)
    if original_split is not None:
        md.obs["split"] = original_split
        md.uns["split"] = original_uns

    if not results:
        # Every fit failed. Returning an empty table would read as "the model
        # scored nothing" rather than "nothing was ever fitted", so re-raise the
        # reason by attempting one fit without the guard.
        split(md, strategy=strategies[0], test_size=test_size, seed=seeds[0],
              key_added="_cv_split")
        try:
            fit(md, target=target, use_rep=use_rep, model=model,
                split_key="_cv_split", level="_cv")
        finally:
            _cleanup(md, "_cv_split", "_cv", target)
            if original_split is not None:
                md.obs["split"] = original_split
                md.uns["split"] = original_uns
        raise ValueError(
            f"no split produced a scoreable fit for {target!r}; every test fold "
            f"was empty or unusable")

    leak = None
    if "random" in results and "composition" in results:
        leak = (results["composition"]["mae"]["mean"]
                - results["random"]["mae"]["mean"])

    md.uns["cross_validate"] = {
        "target": target, "model": model, "use_rep": use_rep,
        "seeds": list(seeds), "results": results,
        "leakage_mae": leak,
        "note": "leakage_mae is grouped MAE minus random MAE. A large positive "
                "value means the random split was flattering the model.",
    }
    record(md, "model.cross_validate", target=target, model=model,
           strategies=list(strategies))


def _cleanup(md: AnnData, split_key: str, level: str, target: str) -> None:
    """Remove the scratch columns cross_validate created."""
    for column in (split_key, f"{target}_{level}", f"{target}_{level}_std"):
        if column in md.obs:
            del md.obs[column]
    md.uns.get("model", {}).pop(level, None)
    md.uns.get("levels", {}).pop(level, None)


__all__ = ["split", "fit", "cross_validate", "register_model", "available",
           "SPLIT_STRATEGIES"]

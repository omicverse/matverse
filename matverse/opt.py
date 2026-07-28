"""``mv.opt`` — deciding what to compute next.

A screen ranks what you already have. A campaign chooses what to do next, runs
it, and folds the answer back in. The difference matters because compute is the
binding constraint in materials discovery: you cannot relax a million candidates
with DFT, so which few hundred you pick is the whole problem.

Everything here works on a **pool**: the candidates live in the object, some have
been measured and most have not, and each round selects a batch. That is
pool-based active learning rather than continuous optimisation, and it is the
right shape for materials — the search space is a list of structures, not a box
of real numbers.

``uns['campaign']`` records every round: what was selected, why, and what came
back. It is the provenance analogue for iterative work, where a linear list of
operations stops being enough.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import append_record, record, records
from ._registry import register_function

#: Acquisition functions, mapping (mean, sigma, best) -> score. Higher is better.
ACQUISITIONS = ("greedy", "uncertainty", "ucb", "ei", "random")


@register_function(
    aliases=["start campaign", "new campaign", "begin active learning",
             "initialise campaign", "campaign"],
    category="opt",
    description="Start a design campaign over the candidates in this object, "
                "declaring the objective and which candidates are already "
                "known.",
    produces={"obs": ["{observed_key}", "campaign_round"], "uns": ["campaign"]},
    examples=["mv.opt.start(md, objective='e_above_hull_emt', goal='min')"],
    related=["mv.opt.suggest", "mv.opt.observe"],
    notes="Separate from mv.screen because a screen is a decision about a fixed "
          "dataset and a campaign is a loop. The loop needs somewhere to record "
          "rounds, and a list of operations is not that shape.",
)
def start(md: AnnData, objective: str, goal: str = "min",
          observed=None, observed_key: str = "observed",
          name: str = "campaign") -> None:
    """Declare an objective and what is already known."""
    if goal not in ("min", "max"):
        raise ValueError(f"goal must be 'min' or 'max', got {goal!r}")

    if observed is None:
        known = np.isfinite(md.obs[objective].to_numpy(dtype=float)) \
            if objective in md.obs else np.zeros(md.n_obs, dtype=bool)
    else:
        known = np.asarray(observed, dtype=bool)

    md.obs[observed_key] = known
    md.obs["campaign_round"] = np.where(known, 0, -1).astype(float)
    md.uns[name] = {
        "objective": objective,
        "goal": goal,
        "observed_key": observed_key,
        "n_pool": int(md.n_obs),
        "round": 0,
        "rounds": {},
    }
    record(md, "opt.start", objective=objective, goal=goal,
           n_observed=int(known.sum()))


@register_function(
    aliases=["suggest", "acquisition", "what next", "select batch",
             "next experiments", "propose candidates", "active learning"],
    category="opt",
    description="Score every unobserved candidate with an acquisition function "
                "and select the next batch to compute or measure.",
    requires={"uns": ["{name}"]},
    produces={"obs": ["acquisition", "selected"], "uns": ["{name}"]},
    prerequisites=["mv.opt.start"],
    dispatch="method='greedy' takes the best predicted; 'uncertainty' takes the "
             "least known; 'ucb' and 'ei' balance the two; 'random' is the "
             "baseline a campaign has to beat",
    examples=["mv.opt.suggest(md, n=10)",
              "mv.opt.suggest(md, n=10, method='ei')"],
    related=["mv.opt.start", "mv.opt.observe", "mv.model.fit"],
    notes="Reads a prediction and its uncertainty from obs, so any level that "
          "produces both works — an ensemble of potentials via "
          "mv.calc.committee, or a fitted model via mv.model.fit. Without an "
          "uncertainty column only the greedy and random routes are honest, and "
          "the others refuse rather than pretending sigma is zero.",
)
def suggest(md: AnnData, n: int = 10, method: str = "ucb",
            predicted: str | None = None, uncertainty: str | None = None,
            beta: float = 1.0, seed: int = 0, name: str = "campaign",
            diversify: bool = False, use_rep: str = "X_pca") -> None:
    """Score the pool and flag the next batch in ``obs['selected']``."""
    campaign = _campaign(md, name)
    if method not in ACQUISITIONS:
        raise ValueError(f"unknown method {method!r}; use {list(ACQUISITIONS)}")

    observed = np.asarray(md.obs[campaign["observed_key"]], dtype=bool)
    pool = ~observed
    if not pool.any():
        raise ValueError("every candidate is already observed; nothing to suggest")

    mu, sigma = _predictions(md, campaign, predicted, uncertainty, method)
    sign = -1.0 if campaign["goal"] == "min" else 1.0
    best = _incumbent(md, campaign, observed, sign)

    score = _acquire(method, sign * mu, sigma, sign * best, beta, seed, md.n_obs)
    score = np.where(pool, score, -np.inf)

    chosen = _select(md, score, n, diversify, use_rep, pool)
    selected = np.zeros(md.n_obs, dtype=bool)
    selected[chosen] = True

    md.obs["acquisition"] = score
    md.obs["selected"] = selected
    campaign["pending"] = [str(md.obs_names[i]) for i in chosen]
    campaign["last_suggestion"] = {
        "method": method, "n_requested": int(n), "n_selected": int(len(chosen)),
        "beta": beta, "diversify": bool(diversify),
        "predicted": predicted, "uncertainty": uncertainty,
    }
    record(md, "opt.suggest", n=n, method=method, n_selected=len(chosen))


def _campaign(md: AnnData, name: str) -> dict:
    if name not in md.uns:
        raise ValueError(f"uns[{name!r}] absent; run mv.opt.start first")
    return md.uns[name]


def _predictions(md: AnnData, campaign: dict, predicted: str | None,
                 uncertainty: str | None, method: str):
    """The mean and sigma an acquisition function needs."""
    objective = campaign["objective"]
    mu_key = predicted or _first_present(
        md, [f"{objective}_pred", objective])
    if mu_key is None:
        raise ValueError(
            f"no prediction to act on. Pass predicted= naming an obs column, or "
            f"fit one with mv.model.fit(md, target={objective!r}).")
    # A column named explicitly and not present is a typo, and a bare KeyError
    # from the lookup below says only the name back. Say what it was for.
    if mu_key not in md.obs:
        raise ValueError(
            f"predicted={mu_key!r} is not an obs column. Available "
            f"predictions: {sorted(k for k in md.obs if k.endswith('_pred'))}")
    mu = md.obs[mu_key].to_numpy(dtype=float)

    sigma_key = uncertainty or _first_present(md, [f"{mu_key}_std"])
    if sigma_key is not None and sigma_key not in md.obs:
        raise ValueError(
            f"uncertainty={sigma_key!r} is not an obs column. Available: "
            f"{sorted(k for k in md.obs if k.endswith('_std'))}. An "
            f"acquisition that needs sigma refuses rather than treating it "
            f"as zero, because that would be a greedy run under another name.")
    if sigma_key is None:
        if method in ("uncertainty", "ucb", "ei"):
            raise ValueError(
                f"method={method!r} needs an uncertainty, and no "
                f"obs['{mu_key}_std'] exists. Produce one with "
                f"mv.model.fit using an ensemble estimator, or "
                f"mv.calc.committee, or use method='greedy'.")
        sigma = np.zeros_like(mu)
    else:
        sigma = md.obs[sigma_key].to_numpy(dtype=float)
    return np.nan_to_num(mu, nan=np.inf), np.nan_to_num(sigma, nan=0.0)


def _first_present(md: AnnData, candidates) -> str | None:
    for key in candidates:
        if key in md.obs:
            return key
    return None


def _incumbent(md: AnnData, campaign: dict, observed: np.ndarray,
               sign: float) -> float:
    """The best value seen so far, which expected improvement measures against."""
    objective = campaign["objective"]
    if objective not in md.obs or not observed.any():
        return -np.inf if sign > 0 else np.inf
    values = md.obs[objective].to_numpy(dtype=float)[observed]
    values = values[np.isfinite(values)]
    if not len(values):
        return -np.inf if sign > 0 else np.inf
    return float(values.max() if sign > 0 else values.min())


def _acquire(method: str, mu: np.ndarray, sigma: np.ndarray, best: float,
             beta: float, seed: int, n: int) -> np.ndarray:
    """Higher is better, always — the sign flip is applied by the caller."""
    if method == "random":
        return np.random.default_rng(seed).random(n)
    if method == "greedy":
        return mu
    if method == "uncertainty":
        return sigma
    if method == "ucb":
        return mu + beta * sigma
    if method == "ei":
        if not np.isfinite(best):
            return sigma
        from math import sqrt
        safe = np.maximum(sigma, 1e-12)
        z = (mu - best) / safe
        cdf = 0.5 * (1.0 + _erf(z / sqrt(2.0)))
        pdf = np.exp(-0.5 * z ** 2) / sqrt(2.0 * np.pi)
        return np.where(sigma > 0, (mu - best) * cdf + sigma * pdf, 0.0)
    raise ValueError(method)                              # pragma: no cover


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised error function, via scipy when present and numpy otherwise."""
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:                                   # pragma: no cover
        return np.vectorize(__import__("math").erf)(x)


def _select(md: AnnData, score: np.ndarray, n: int, diversify: bool,
            use_rep: str, pool: np.ndarray) -> np.ndarray:
    """Top ``n`` by score, optionally spread out in descriptor space.

    A batch of the ten highest-scoring candidates is often ten variations on one
    idea, and computing all ten answers one question rather than ten. Greedy
    farthest-point selection among a shortlist keeps the batch informative.
    """
    order = np.argsort(-score, kind="stable")
    order = order[np.isfinite(score[order])]
    n = int(min(n, len(order)))
    if n <= 0:
        return np.asarray([], dtype=int)
    if not diversify or use_rep not in md.obsm:
        return order[:n]

    shortlist = order[:min(len(order), max(n * 5, n))]
    Z = np.asarray(md.obsm[use_rep], dtype=float)[shortlist]
    chosen = [0]
    distances = np.linalg.norm(Z - Z[0], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(distances))
        if nxt in chosen:
            break
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(Z - Z[nxt], axis=1))
    return shortlist[np.asarray(chosen, dtype=int)]


@register_function(
    aliases=["observe", "record results", "fold in results", "close the loop",
             "report measurements"],
    category="opt",
    description="Fold the results of the last suggested batch back into the "
                "campaign, closing one round and recording what was learned.",
    requires={"obs": ["selected"], "uns": ["{name}"]},
    produces={"obs": ["campaign_round", "selected"], "uns": ["{name}"]},
    prerequisites=["mv.opt.suggest"],
    examples=["mv.opt.observe(md)",
              "mv.opt.observe(md, values={'12': 0.03, '47': 0.11})"],
    related=["mv.opt.suggest", "mv.opt.history"],
    notes="Values may already be in the objective column — the usual case when "
          "the round was a calculation rather than an experiment — in which "
          "case this only marks them observed and closes the round. It also "
          "writes the campaign's observed column, whose name is chosen at "
          "mv.opt.start and so cannot be named in a contract slot: the "
          "template would have nothing in this call's arguments to resolve "
          "against.",
)
def observe(md: AnnData, values=None, name: str = "campaign") -> None:
    """Mark the selected batch observed and close the round."""
    campaign = _campaign(md, name)
    if "selected" not in md.obs:
        raise ValueError("obs['selected'] absent; run mv.opt.suggest first")

    observed_key = campaign["observed_key"]
    objective = campaign["objective"]
    selected = np.asarray(md.obs["selected"], dtype=bool)
    if not selected.any():
        raise ValueError("no candidate is selected; run mv.opt.suggest first")

    if values is not None:
        # copy=True: from pandas 3.0 to_numpy may return a read-only view, and
        # this writes the observed values into it.
        column = (md.obs[objective].to_numpy(dtype=float, copy=True)
                  if objective in md.obs else np.full(md.n_obs, np.nan))
        names = list(map(str, md.obs_names))
        for key, value in dict(values).items():
            if str(key) not in names:
                raise KeyError(f"{key!r} is not a material in this object")
            column[names.index(str(key))] = float(value)
        md.obs[objective] = column

    known = np.asarray(md.obs[observed_key], dtype=bool) | selected
    md.obs[observed_key] = known
    campaign["round"] = int(campaign["round"]) + 1
    md.obs["campaign_round"] = np.where(
        selected & (md.obs["campaign_round"].to_numpy(dtype=float) < 0),
        float(campaign["round"]),
        md.obs["campaign_round"].to_numpy(dtype=float))

    sign = -1.0 if campaign["goal"] == "min" else 1.0
    best = _incumbent(md, campaign, known, sign)
    append_record(campaign, "rounds", {
        "round": int(campaign["round"]),
        "n_selected": int(selected.sum()),
        "n_observed": int(known.sum()),
        "best_so_far": float("nan") if not np.isfinite(best) else float(best),
        **dict(campaign.get("last_suggestion", {})),
    })
    campaign["pending"] = []
    md.obs["selected"] = np.zeros(md.n_obs, dtype=bool)
    record(md, "opt.observe", round=int(campaign["round"]),
           n_observed=int(known.sum()))


@register_function(
    aliases=["campaign history", "rounds", "optimisation history",
             "learning curve", "progress"],
    category="opt",
    description="Return the campaign's rounds as a table — how many candidates "
                "were observed each round and how the best value moved.",
    requires={"uns": ["campaign"]},
    prerequisites=["mv.opt.observe"],
    examples=["history = mv.opt.history(md)"],
    related=["mv.opt.observe", "mv.pl.parity"],
)
def history(md: AnnData, name: str = "campaign"):
    """The campaign as a DataFrame, one row per round."""
    import pandas as pd

    campaign = _campaign(md, name)
    rounds = records(campaign, "rounds")
    if not rounds:
        raise ValueError("no round has closed; run mv.opt.observe first")
    frame = pd.DataFrame(rounds)
    frame.attrs["objective"] = campaign["objective"]
    frame.attrs["goal"] = campaign["goal"]
    return frame


__all__ = ["start", "suggest", "observe", "history", "ACQUISITIONS"]

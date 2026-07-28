"""``mv.tl`` — analysis on the composition matrix.

``X`` is materials x elements, which is structurally a cells x genes matrix:
sparse, non-negative, and mostly zero because almost every material draws on
five or fewer of 118 columns. That correspondence is what this module spends.

The operation worth the whole bet is :func:`rank_elements_groups`. Screening
produces a partition — stable versus not, passed versus failed, cluster 3 versus
the rest — and the question that follows is always "what chemistry distinguishes
them". That is ``rank_genes_groups`` with the nouns changed, and answering it
from a DataFrame of Magpie averages requires writing the test by hand every
time.

Only ``pca`` and ``rank_elements_groups`` run on the core dependencies. Anything
needing a neighbour graph will use scikit-learn when it is installed and fall
back to an exact brute-force computation when it is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import record
from ._registry import register_function


def _matrix(md: AnnData, layer: str | None, dense: bool = True) -> np.ndarray:
    """The composition matrix to analyse, preferring atomic fractions.

    Raw counts scale with the size of the formula unit, so Fe2O3 and Fe4O6 would
    sit at different points in an unnormalised space despite being the same
    chemistry. Fractions are the right default for anything geometric.
    """
    if md.n_vars == 0:
        raise ValueError(
            "this object has no element axis; it was built with build_X=False. "
            "mv.tl operates on the composition matrix.")
    if layer is None:
        layer = "fraction" if "fraction" in md.layers else None
    M = md.layers[layer] if layer is not None else md.X
    if dense and hasattr(M, "toarray"):
        M = M.toarray()
    return np.asarray(M, dtype=float)


@register_function(
    aliases=["pca", "principal components", "chemical space", "ordination",
             "dimensionality reduction"],
    category="tl",
    description="Principal components of the composition matrix, giving a "
                "low-dimensional map of the chemical space a candidate library "
                "spans.",
    requires={"X": ["composition"]},
    produces={"obsm": ["X_pca"], "varm": ["PCs"], "uns": ["pca"]},
    examples=["mv.tl.pca(md)", "mv.tl.pca(md, n_comps=10)"],
    related=["mv.tl.neighbors", "mv.tl.cluster"],
    notes="Computed by exact SVD rather than a randomised solver: the element "
          "axis is at most 118 columns wide, so there is nothing to approximate.",
)
def pca(md: AnnData, n_comps: int = 10, layer: str | None = None,
        zero_center: bool = True) -> None:
    """Principal components of composition space."""
    M = _matrix(md, layer)
    n_comps = int(min(n_comps, min(M.shape) - 1)) if min(M.shape) > 1 else 1
    mean = M.mean(axis=0) if zero_center else np.zeros(M.shape[1])
    C = M - mean
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    md.obsm["X_pca"] = (U[:, :n_comps] * S[:n_comps]).astype(float)
    md.varm["PCs"] = Vt[:n_comps].T.astype(float)
    total = float((S ** 2).sum())
    md.uns["pca"] = {
        "variance": (S[:n_comps] ** 2 / max(len(M) - 1, 1)).astype(float),
        "variance_ratio": (S[:n_comps] ** 2 / total).astype(float)
        if total else np.zeros(n_comps),
        "layer": layer or ("fraction" if "fraction" in md.layers else "X"),
        "zero_center": bool(zero_center),
    }
    record(md, "tl.pca", n_comps=n_comps, layer=layer)


@register_function(
    aliases=["neighbors", "neighbours", "knn graph", "nearest neighbours",
             "similarity graph"],
    category="tl",
    description="Build a k-nearest-neighbour graph over materials in "
                "composition or descriptor space and store its connectivities "
                "and distances.",
    requires={"obsm": ["{use_rep}"]},
    produces={"obsp": ["connectivities", "distances"], "uns": ["neighbors"]},
    prerequisites=["mv.tl.pca"],
    examples=["mv.tl.neighbors(md)",
              "mv.tl.neighbors(md, n_neighbors=10, use_rep='X_pca')"],
    related=["mv.tl.cluster", "mv.tl.umap"],
)
def neighbors(md: AnnData, n_neighbors: int = 15, use_rep: str = "X_pca",
              metric: str = "euclidean") -> None:
    """A kNN graph over materials."""
    if use_rep not in md.obsm:
        raise ValueError(
            f"obsm[{use_rep!r}] absent; run mv.tl.pca first, or pass "
            f"use_rep= naming a block in obsm ({sorted(md.obsm)})")
    Z = np.asarray(md.obsm[use_rep], dtype=float)
    k = int(min(n_neighbors, len(Z) - 1)) if len(Z) > 1 else 1

    idx, dist = _knn(Z, k, metric)
    from scipy.sparse import csr_matrix

    n = len(Z)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    d = dist.ravel()
    distances = csr_matrix((d, (rows, cols)), shape=(n, n))

    # Gaussian kernel on each row's own scale, symmetrised — the same shape of
    # construction scanpy uses, kept simple because the graph here is small.
    sigma = np.maximum(dist[:, -1:], 1e-12)
    w = np.exp(-(dist / sigma) ** 2).ravel()
    conn = csr_matrix((w, (rows, cols)), shape=(n, n))
    conn = conn.maximum(conn.T)

    md.obsp["distances"] = distances
    md.obsp["connectivities"] = conn
    md.uns["neighbors"] = {"n_neighbors": k, "use_rep": use_rep,
                           "metric": metric}
    record(md, "tl.neighbors", n_neighbors=k, use_rep=use_rep, metric=metric)


def _knn(Z: np.ndarray, k: int, metric: str):
    """Exact k nearest neighbours, excluding self."""
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Z)
        dist, idx = nn.kneighbors(Z)
        return idx[:, 1:], dist[:, 1:]
    except ImportError:
        if metric != "euclidean":
            raise ImportError(
                f"metric={metric!r} needs scikit-learn; `pip install "
                f"matverse[analysis]`, or use metric='euclidean'")
        D = _pairwise_euclidean(Z)
        np.fill_diagonal(D, np.inf)
        idx = np.argsort(D, axis=1)[:, :k]
        return idx, np.take_along_axis(D, idx, axis=1)


def _pairwise_euclidean(Z: np.ndarray) -> np.ndarray:
    sq = (Z ** 2).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    return np.sqrt(np.maximum(D, 0.0))


@register_function(
    aliases=["cluster", "clustering", "leiden", "kmeans", "group materials",
             "chemical families"],
    category="tl",
    description="Partition materials into groups in composition or descriptor "
                "space, by Leiden community detection on the neighbour graph or "
                "by k-means.",
    produces={"obs": ["{key_added}"], "uns": ["cluster"]},
    dispatch="method='leiden' requires obsp['connectivities'] from "
             "mv.tl.neighbors, and igraph/leidenalg; method='kmeans' requires "
             "obsm[use_rep] from mv.tl.pca, and scikit-learn",
    examples=["mv.tl.cluster(md, method='kmeans', n_clusters=4)",
              "mv.tl.cluster(md, method='leiden', resolution=1.0)"],
    related=["mv.tl.rank_elements_groups", "mv.tl.neighbors", "mv.tl.pca"],
    notes="This function carries no requires claim, and the omission is a "
          "finding rather than an oversight. Its two routes consume different "
          "state — leiden reads the neighbour graph, kmeans reads the embedding "
          "directly — and the contract vocabulary has one requires field per "
          "function, not one per route. An unconditional claim on "
          "obsp['connectivities'] was probed and deleted because the kmeans "
          "route succeeds without it. Route-conditional dependencies are "
          "expressible only in the dispatch text, which is prose a caller reads "
          "rather than structure a tool can check.",
)
def cluster(md: AnnData, method: str = "leiden", key_added: str = "cluster",
            resolution: float = 1.0, n_clusters: int = 8,
            use_rep: str = "X_pca", seed: int = 0) -> None:
    """Group materials. ``method`` dispatches; each route has its own inputs."""
    if method == "leiden":
        labels = _leiden(md, resolution, seed)
    elif method == "kmeans":
        labels = _kmeans(md, n_clusters, use_rep, seed)
    else:
        raise ValueError(f"unknown method {method!r}; use 'leiden' or 'kmeans'")
    md.obs[key_added] = pd.Categorical([str(v) for v in labels])
    md.uns["cluster"] = {"method": method, "key": key_added, "seed": seed,
                         **({"resolution": resolution} if method == "leiden"
                            else {"n_clusters": n_clusters,
                                  "use_rep": use_rep})}
    record(md, "tl.cluster", method=method, key_added=key_added)


def _leiden(md: AnnData, resolution: float, seed: int) -> np.ndarray:
    if "connectivities" not in md.obsp:
        raise ValueError("obsp['connectivities'] absent; run mv.tl.neighbors first")
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError(
            "method='leiden' needs igraph and leidenalg (`pip install "
            "matverse[analysis]`); method='kmeans' needs only scikit-learn"
        ) from exc
    A = md.obsp["connectivities"].tocoo()
    keep = A.row < A.col
    graph = ig.Graph(n=md.n_obs,
                     edges=list(zip(A.row[keep].tolist(), A.col[keep].tolist())),
                     edge_attrs={"weight": A.data[keep].tolist()})
    part = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed)
    return np.asarray(part.membership)


def _kmeans(md: AnnData, n_clusters: int, use_rep: str, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError("method='kmeans' needs scikit-learn") from exc
    if use_rep not in md.obsm:
        raise ValueError(f"obsm[{use_rep!r}] absent; run mv.tl.pca first")
    Z = np.asarray(md.obsm[use_rep], dtype=float)
    k = int(min(n_clusters, len(Z)))
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)


@register_function(
    aliases=["rank elements groups", "differential elements",
             "element enrichment", "which elements", "marker elements",
             "rank genes groups"],
    category="tl",
    description="Find which elements are over-represented in each group of "
                "materials relative to the rest — the chemistry that "
                "distinguishes stable candidates from unstable ones, or one "
                "cluster from another.",
    requires={"obs": ["{groupby}"], "X": ["composition"]},
    produces={"uns": ["rank_elements_groups"]},
    examples=["mv.tl.rank_elements_groups(md, 'is_stable_emt')",
              "mv.tl.rank_elements_groups(md, 'cluster', method='fraction')"],
    related=["mv.tl.cluster", "mv.pl.periodic_table"],
    dispatch="method='presence' tests how often an element appears "
             "(Fisher exact); method='fraction' tests how much of it there is "
             "(Wilcoxon rank-sum on atomic fraction)",
    notes="The materials analogue of rank_genes_groups, and the operation that "
          "justifies making X the composition matrix. Reported for each group: "
          "the odds ratio or effect size, an uncorrected p-value, and a "
          "Benjamini-Hochberg q-value.",
)
def rank_elements_groups(md: AnnData, groupby: str, method: str = "presence",
                         groups=None, reference: str = "rest",
                         key_added: str = "rank_elements_groups") -> None:
    """Which elements characterise each group of materials."""
    if groupby not in md.obs:
        raise ValueError(f"obs[{groupby!r}] absent; available: "
                         f"{list(md.obs.columns)}")
    if md.n_vars == 0:
        raise ValueError("this object has no element axis (build_X=False)")
    if method not in ("presence", "fraction"):
        raise ValueError(f"unknown method {method!r}; use 'presence' or 'fraction'")

    labels = pd.Series(md.obs[groupby].to_numpy()).astype(str).to_numpy()
    all_groups = list(pd.unique(labels))
    wanted = [str(g) for g in (groups if groups is not None else all_groups)]
    missing = [g for g in wanted if g not in all_groups]
    if missing:
        raise ValueError(f"group(s) {missing} not in obs[{groupby!r}]; "
                         f"have {all_groups}")

    raw = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    raw = np.asarray(raw, dtype=float)
    present = raw > 0
    totals = raw.sum(axis=1, keepdims=True)
    values = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)

    elements = list(md.var_names)
    out: dict = {"params": {"groupby": groupby, "method": method,
                            "reference": reference},
                 "groups": wanted}
    for g in wanted:
        in_group = labels == g
        rest = ~in_group if reference == "rest" else labels == str(reference)
        if not in_group.any() or not rest.any():
            continue
        if method == "presence":
            frame = _presence_test(present, in_group, rest, elements)
        else:
            frame = _fraction_test(values, in_group, rest, elements)
        frame["group"] = g
        out[g] = frame.reset_index(drop=True)

    md.uns[key_added] = out
    record(md, "tl.rank_elements_groups", groupby=groupby, method=method)


def _presence_test(present: np.ndarray, in_group, rest,
                   elements) -> pd.DataFrame:
    """Fisher exact test on how often each element appears."""
    from scipy.stats import fisher_exact

    a = present[in_group].sum(axis=0).astype(float)      # in group, has element
    b = int(in_group.sum()) - a                          # in group, lacks it
    c = present[rest].sum(axis=0).astype(float)          # outside, has it
    d = int(rest.sum()) - c

    odds, pvals = np.ones(len(elements)), np.ones(len(elements))
    for j in range(len(elements)):
        table = [[a[j], b[j]], [c[j], d[j]]]
        if a[j] == 0 and c[j] == 0:
            continue
        try:
            odds[j], pvals[j] = fisher_exact(table, alternative="two-sided")
        except Exception:
            odds[j], pvals[j] = np.nan, 1.0

    frame = pd.DataFrame({
        "element": elements,
        "n_in_group": a.astype(int),
        "n_rest": c.astype(int),
        "frac_in_group": a / max(int(in_group.sum()), 1),
        "frac_rest": c / max(int(rest.sum()), 1),
        "odds_ratio": odds,
        "pval": pvals,
        "qval": _bh(pvals),
    })
    frame["log2_odds"] = np.log2(np.where(frame["odds_ratio"] > 0,
                                          frame["odds_ratio"], np.nan))
    return frame.sort_values(["pval", "odds_ratio"],
                             ascending=[True, False])


def _fraction_test(values: np.ndarray, in_group, rest,
                   elements) -> pd.DataFrame:
    """Wilcoxon rank-sum on how much of each element is present."""
    from scipy.stats import ranksums

    stats, pvals = np.zeros(len(elements)), np.ones(len(elements))
    for j in range(len(elements)):
        x, y = values[in_group, j], values[rest, j]
        if not x.any() and not y.any():
            continue
        try:
            stats[j], pvals[j] = ranksums(x, y)
        except Exception:
            stats[j], pvals[j] = np.nan, 1.0

    mean_in = values[in_group].mean(axis=0)
    mean_rest = values[rest].mean(axis=0)
    frame = pd.DataFrame({
        "element": elements,
        "mean_frac_in_group": mean_in,
        "mean_frac_rest": mean_rest,
        "diff": mean_in - mean_rest,
        "zscore": stats,
        "pval": pvals,
        "qval": _bh(pvals),
    })
    return frame.sort_values(["pval", "diff"], ascending=[True, False])


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


@register_function(
    aliases=["novelty", "distance to known", "is this new", "out of "
             "distribution", "nearest known material"],
    category="tl",
    description="Score how far each material sits from its nearest neighbour in "
                "a reference dataset, in composition space — the honest version "
                "of asking whether a generated candidate is new.",
    requires={"X": ["composition"]},
    produces={"obs": ["novelty_distance", "nearest_reference"]},
    examples=["mv.tl.novelty(md, reference=known)"],
    related=["mv.tl.pca", "mv.pp.dedup"],
    notes="Composition-space distance answers 'is this chemistry new', not 'is "
          "this structure new'. For the latter, dedup against the reference "
          "with mv.pp.dedup on a concatenated object.",
)
def novelty(md: AnnData, reference: AnnData,
            key_added: str = "novelty_distance") -> None:
    """Distance from each material to the closest reference composition."""
    if md.n_vars == 0 or reference.n_vars == 0:
        raise ValueError("both objects need an element axis (build_X=True)")

    elements = sorted(set(md.var_names) | set(reference.var_names))
    A = _aligned_fractions(md, elements)
    B = _aligned_fractions(reference, elements)
    if len(B) == 0:
        raise ValueError("reference is empty")

    D = _pairwise_euclidean_between(A, B)
    nearest = np.argmin(D, axis=1)
    md.obs[key_added] = D[np.arange(len(A)), nearest]
    md.obs["nearest_reference"] = [str(reference.obs_names[j]) for j in nearest]
    record(md, "tl.novelty", n_reference=int(reference.n_obs))


def _aligned_fractions(md: AnnData, elements: list[str]) -> np.ndarray:
    """Atomic fractions on a shared element vocabulary."""
    raw = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    out = np.zeros((md.n_obs, len(elements)), dtype=float)
    index = {el: j for j, el in enumerate(elements)}
    for j, el in enumerate(md.var_names):
        out[:, index[str(el)]] = raw[:, j]
    totals = out.sum(axis=1, keepdims=True)
    return np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)


def _pairwise_euclidean_between(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    sa = (A ** 2).sum(axis=1)[:, None]
    sb = (B ** 2).sum(axis=1)[None, :]
    return np.sqrt(np.maximum(sa + sb - 2.0 * (A @ B.T), 0.0))


__all__ = ["pca", "neighbors", "cluster", "rank_elements_groups", "novelty"]

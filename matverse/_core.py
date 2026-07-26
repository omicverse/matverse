"""Slot convention and provenance — the rules every namespace obeys.

matverse stores a materials dataset in an ``AnnData``. Not a subclass and not
a wrapper: the object you get back is an ``AnnData``, writable to ``h5ad`` and
readable by anything that speaks it.

    obs                     one row per material
    obsm                    descriptors, embeddings, per-level vector results
    obsp                    pairwise: similarity, hull adjacency
    uns['structures']       raw structures, keyed by variant
    uns['features']         which featuriser produced which block
    uns['calc']             per-level calculator parameters
    uns['provenance']       operations applied, in order

``X`` is left empty on purpose. AnnData ties ``X``'s width to ``var``, so it
cannot be widened in place, and every operation here writes in place. Features
therefore go to ``obsm``, whose width is free.

Two rules
---------
**Operations deposit; they do not return.** ``mv.struct.standardize(md)`` writes
``uns['structures']['primitive']`` rather than handing back an object the
caller must find a home for. After any step you can ask the object what is in
it, which is also what makes a run reproducible from the object alone.

**A result carries its level of theory in the slot name.** ``obs['energy_mace']``
and ``obs['energy_pbe']`` are different quantities, and ``uns['calc'][level]``
holds the parameters that produced each. Comparing a surrogate potential
against DFT then requires naming both, instead of silently averaging them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

CONTAINERS = ("obs", "var", "obsm", "obsp", "levels", "uns")


def new(structures: list, obs: pd.DataFrame | None = None,
        source: str = "structures") -> AnnData:
    """An empty dataset holding ``structures`` as the ``input`` variant."""
    n = len(structures)
    obs = pd.DataFrame(index=range(n)) if obs is None else obs.reset_index(drop=True)
    md = AnnData(X=np.zeros((n, 0)), obs=obs)
    md.obs_names = [str(i) for i in range(n)]
    md.uns["structures"] = {"input": list(structures)}
    md.uns["features"] = {}
    md.uns["calc"] = {}
    md.uns["provenance"] = [source]
    return md


def structures(md: AnnData, variant: str = "input") -> list:
    """Fetch a structure variant, with a useful error when it is absent."""
    have = md.uns.get("structures", {})
    if variant not in have:
        raise KeyError(
            f"no structure variant {variant!r}; have {sorted(have)}. "
            f"Operations deposit variants under a name — e.g. mv.struct.standardize "
            f"writes 'primitive'."
        )
    return have[variant]


def deposit_structures(md: AnnData, variant: str, value: list) -> None:
    md.uns.setdefault("structures", {})[variant] = list(value)


def require(md: AnnData, container: str, key: str, hint: str = "") -> Any:
    """Read a slot, or fail with what would have produced it.

    The error text names the operation to run, because the common failure in a
    deposit-style API is calling a step before the one that fills its input.
    """
    holder = getattr(md, container, None) if container != "uns" else md.uns
    if holder is None or key not in holder:
        msg = f"{container}[{key!r}] is not present"
        raise ValueError(f"{msg}; {hint}" if hint else msg)
    return holder[key]


def record(md: AnnData, op: str) -> None:
    md.uns.setdefault("provenance", []).append(op)


__all__ = ["new", "structures", "deposit_structures", "require", "record", "CONTAINERS"]

"""The periodic table as a ``var`` frame.

``X`` is materials x elements, so ``var`` is one row per element — the analogue
of gene metadata, and the thing that makes element-level results interpretable
without a lookup elsewhere.

Properties come from ``pymatgen.core.Element`` and are read defensively: several
are ``None`` for some elements, several carry units, and a few raise. A missing
value becomes ``NaN`` rather than an exception, because a dataset containing one
awkward element should not fail to build.

Economic and supply-risk properties (price, criticality, import reliance) are
deliberately **not** shipped. They are jurisdiction- and date-dependent, no
single authoritative open table exists, and inventing numbers to fill a column
would make ``mv.screen.filter(md, supply_risk__lt=0.3)`` return confident
nonsense. Attach your own with :func:`annotate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Scalar properties read from ``pymatgen.core.Element``.
ELEMENT_PROPERTIES = (
    "Z", "atomic_mass", "atomic_radius", "X", "group", "row",
    "melting_point", "boiling_point", "molar_volume",
    "thermal_conductivity", "electrical_resistivity",
    "average_ionic_radius", "max_oxidation_state", "min_oxidation_state",
)

#: Boolean classifications, useful as grouping keys.
ELEMENT_FLAGS = (
    "is_metal", "is_transition_metal", "is_alkali", "is_alkaline",
    "is_metalloid", "is_halogen", "is_noble_gas", "is_chalcogen",
    "is_lanthanoid", "is_actinoid", "is_rare_earth_metal",
)

_RENAME = {"X": "electronegativity", "row": "period"}


def _scalar(value) -> float:
    """Coerce a pymatgen property to a float, or NaN."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def element_frame(symbols) -> pd.DataFrame:
    """One row per element symbol, indexed by symbol.

    >>> element_frame(["Fe", "O"]).loc["Fe", "Z"]
    26.0
    """
    import warnings

    from pymatgen.core.periodic_table import Element

    symbols = list(symbols)
    rows, flags = [], []
    # pymatgen warns per missing property per element; for a whole table that is
    # hundreds of lines saying only "this element has no published value".
    warnings.filterwarnings("ignore", message="No data available for",
                            category=UserWarning)
    for sym in symbols:
        try:
            el = Element(sym)
        except Exception:
            rows.append({p: float("nan") for p in ELEMENT_PROPERTIES})
            flags.append({f: False for f in ELEMENT_FLAGS})
            continue
        row = {}
        for prop in ELEMENT_PROPERTIES:
            try:
                row[prop] = _scalar(getattr(el, prop, None))
            except Exception:
                row[prop] = float("nan")
        rows.append(row)
        flag = {}
        for name in ELEMENT_FLAGS:
            try:
                flag[name] = bool(getattr(el, name, False))
            except Exception:
                flag[name] = False
        flags.append(flag)

    df = pd.DataFrame(rows, index=pd.Index(symbols, name="element"))
    df = df.rename(columns=_RENAME)
    for name in ELEMENT_FLAGS:
        df[name] = [f[name] for f in flags]
    df["block"] = [_block(s) for s in symbols]
    return df


def _block(symbol: str) -> str:
    try:
        from pymatgen.core.periodic_table import Element
        return str(Element(symbol).block)
    except Exception:
        return ""


def annotate(md, table: pd.DataFrame, columns=None) -> None:
    """Join user-supplied element properties onto ``var``.

    The hook for everything this module refuses to ship — price, supply risk,
    criticality, toxicity, a lab's own availability flags. ``table`` is indexed
    by element symbol; elements absent from it get ``NaN``.

    >>> import pandas as pd
    >>> price = pd.DataFrame({"price_usd_kg": [21.0, 0.15]}, index=["Cu", "Fe"])
    >>> mv.elements.annotate(md, price)
    >>> mv.screen.filter(md, ...)   # now able to reference price_usd_kg
    """
    if md.n_vars == 0:
        raise ValueError(
            "this object has no element axis; it was built with build_X=False. "
            "Rebuild with mv.data.from_structures(..., build_X=True).")
    cols = list(table.columns if columns is None else columns)
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise KeyError(f"table has no column(s) {missing}")
    aligned = table.reindex(list(md.var_names))
    for col in cols:
        md.var[col] = aligned[col].to_numpy()


def periodic_table_layout() -> pd.DataFrame:
    """(group, period) coordinates for plotting, lanthanoids/actinoids placed
    in the conventional two extra rows below the main block."""
    from pymatgen.core.periodic_table import Element

    rows = []
    for z in range(1, 119):
        try:
            el = Element.from_Z(z)
        except Exception:
            continue
        group, period = el.group, el.row
        if el.is_lanthanoid:
            group, period = z - 54, 9          # La(57) -> group 3 of row 9
        elif el.is_actinoid:
            group, period = z - 86, 10
        rows.append({"element": el.symbol, "Z": z,
                     "group": int(group), "period": int(period)})
    return pd.DataFrame(rows).set_index("element")


__all__ = ["element_frame", "annotate", "periodic_table_layout",
           "ELEMENT_PROPERTIES", "ELEMENT_FLAGS"]

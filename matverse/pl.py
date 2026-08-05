"""``mv.pl`` — plots with defaults worth publishing.

Every function takes the object, draws onto an axis, and returns that axis, so a
plot composes into a figure someone else laid out. Nothing here calls
``plt.show``; a library that shows figures cannot be used to build one.

The plot that matters most is :func:`periodic_table`. It is the natural display
for :func:`~matverse.tl.rank_elements_groups`, in the way a dot plot is the
natural display for differential expression — the axis it draws on is the one
every materials scientist already reads.

matplotlib is an optional dependency, imported inside each function. Producing a
number and drawing it are different jobs, and a screening pipeline on a cluster
should not need a plotting stack to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import grid_of, structures
from ._registry import register_function


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:                            # pragma: no cover
        raise ImportError(
            "mv.pl needs matplotlib (`pip install matverse[plot]`). Everything "
            "that computes works without it.") from exc


def _axis(ax, figsize=(6.0, 4.0)):
    if ax is not None:
        return ax
    return _plt().subplots(figsize=figsize)[1]


#: Emoji map for status reporting, mirroring omicverse's plot_set.
EMOJI = {
    "start": "🔬", "settings": "⚙️", "warnings": "🚫", "gpu": "🖥️",
    "calc": "🧪", "logo": "🌟", "done": "✅",
}

#: The logo prints once per session, not once per call.
_has_printed_logo = False

LOGO = r"""
                   __
   ____ ___  ____ _/ /__   _____  _____________
  / __ `__ \/ __ `/ __/ | / / _ \/ ___/ ___/ _ \
 / / / / / / /_/ / /_ | |/ /  __/ /  (__  )  __/
/_/ /_/ /_/\__,_/\__/ |___/\___/_/  /____/\___/
"""


@register_function(
    aliases=["set style", "plot set", "mv_plot_set", "plotset", "绘图设置",
             "plotting defaults", "figure style"],
    category="pl",
    description="Configure plotting for matverse — matplotlib rcParams, "
                "inline figure format, warning suppression — and report which "
                "calculators and GPUs this installation can actually use.",
    examples=["mv.pl.set_style()",
              "mv.pl.set_style(dpi=150, fontsize=10)",
              "mv.pl.set_style(font_path='arial', figsize=8)"],
    related=["mv.pl.spectra", "mv.pl.periodic_table", "mv.calc.available"],
    notes="Call once at the top of a notebook. It touches rcParams and "
          "warning filters and nothing else, so anything set afterwards still "
          "wins and a figure built by hand is unaffected.\n\n"
          "The calculator and GPU report is the part worth reading. matverse "
          "ships one working calculator and dispatches the rest to whatever "
          "you installed, so 'which levels of theory can I actually run here' "
          "is a question with a different answer on every machine — and the "
          "answer decides what the session can do.",
)
def set_style(dpi: int = 100, dpi_save: int = 300, fontsize: int = 11,
              figsize=(6.0, 4.0), facecolor: str = "white",
              transparent: bool | None = None, grid: bool = False,
              font_path: str | None = None,
              ipython_format: str = "retina",
              suppress_warnings: bool = True,
              show_calculators: bool = True, show_gpu: bool = True,
              verbose: bool = True, quiet: bool | None = None) -> None:
    """Apply matverse's plotting defaults and report the environment.

    Returns ``None``; everything it does is to global state.
    """
    global _has_printed_logo

    # quiet= was the whole reporting switch in v0.1.14, which is on PyPI.
    # Removing it would break working notebooks for a rename, so it stays and
    # means "say nothing at all".
    if quiet is not None:
        verbose = show_calculators = show_gpu = not quiet

    plt = _plt()
    if verbose:
        print(f"{EMOJI['start']} Starting plot initialization...")

    # Inline figures at screen resolution, when there is a notebook to do it in.
    import builtins
    if getattr(builtins, "__IPYTHON__", False):
        try:
            from matplotlib_inline.backend_inline import set_matplotlib_formats
            set_matplotlib_formats(ipython_format)
        except (ImportError, AttributeError):
            # matplotlib_inline 0.2.x reaches for rcParams._get(), which newer
            # matplotlib removed. Fall back to configuring IPython directly.
            try:
                from IPython import get_ipython
                shell = get_ipython()
                if shell is not None:
                    shell.config.InlineBackend.figure_formats = {ipython_format}
            except Exception:
                pass

    if font_path is not None:
        _use_font(font_path, verbose=verbose)

    if isinstance(figsize, (int, float)):
        figsize = (figsize, figsize)

    settings = {
        "figure.dpi": dpi,
        "savefig.dpi": dpi_save,
        "figure.figsize": figsize,
        "figure.facecolor": facecolor,
        "axes.facecolor": facecolor,
        "savefig.bbox": "tight",
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "grid.alpha": 0.3,
    }
    if transparent is not None:
        settings["savefig.transparent"] = transparent
    plt.rcParams.update(settings)

    if suppress_warnings:
        import warnings
        for category in (UserWarning, FutureWarning, DeprecationWarning):
            warnings.simplefilter("ignore", category=category)

    if show_calculators:
        _report_calculators()
    if show_gpu:
        _report_gpu()

    if verbose and not _has_printed_logo:
        from . import __version__
        from ._registry import get_registry

        print(LOGO)
        print(f"🔖 Version: {__version__}   "
              f"🧮 Functions: {len(get_registry())}   "
              f"📚 Tutorials: https://matverse.readthedocs.io/")
        _has_printed_logo = True

    if verbose:
        print(f"{EMOJI['done']} set_style complete.\n")


def _use_font(font_path: str, verbose: bool = True) -> None:
    """Register a font file with matplotlib and make it the default."""
    import os

    import matplotlib as mpl
    from matplotlib import font_manager as fm

    if font_path.lower() in ("arial", "arial.ttf") and \
            not font_path.endswith(".ttf"):
        # matplotlib ships no Arial, and a paper that asks for it is common
        # enough to be worth fetching once and caching.
        import tempfile

        cached = os.path.join(tempfile.gettempdir(), "matverse_arial.ttf")
        if os.path.exists(cached):
            font_path = cached
        else:
            try:
                import requests

                url = ("https://github.com/kavin808/arial.ttf/raw/refs/"
                       "heads/master/arial.ttf")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(cached, "wb") as handle:
                    handle.write(response.content)
                font_path = cached
                if verbose:
                    print(f"{EMOJI['settings']} Arial cached at {cached}")
            except Exception as exc:
                print(f"{EMOJI['warnings']} could not fetch Arial ({exc}); "
                      f"keeping the default font")
                return

    try:
        fm.fontManager.addfont(font_path)
        name = fm.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        if verbose:
            print(f"{EMOJI['settings']} Font registered as {name!r}")
    except Exception as exc:
        print(f"{EMOJI['warnings']} could not set the font ({exc}); "
              f"keeping the default")


def _report_calculators() -> None:
    """Which levels of theory this installation can actually run."""
    from .calc import available

    try:
        entries = available()
    except Exception as exc:                              # pragma: no cover
        print(f"{EMOJI['warnings']} calculator detection failed: {exc}")
        return

    runnable = [name for name, meta in entries.items()
                if meta.get("importable", True)]
    print(f"{EMOJI['calc']} Calculators available: {len(runnable)}")
    for name in runnable:
        meta = entries[name]
        method = meta.get("method", name)
        licence = meta.get("license") or "unstated"
        print(f"    • {name} — {method} ({licence})")
    if not runnable:
        print(f"{EMOJI['warnings']} none — install ASE, or register one with "
              f"mv.calc.register_calculator")


def _report_gpu() -> None:
    """Report accelerators, because a machine-learned potential needs one."""
    try:
        import torch
    except ImportError:
        print(f"{EMOJI['gpu']} PyTorch absent — GPU detection skipped. "
              f"Machine-learned potentials need it.")
        return

    found = False
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"{EMOJI['gpu']} NVIDIA CUDA GPUs: {count}")
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            print(f"    • [CUDA {index}] {props.name} — "
                  f"{props.total_memory / 1024 ** 3:.1f} GB, "
                  f"compute {props.major}.{props.minor}")
        found = True
    if getattr(torch.backends, "mps", None) is not None and \
            torch.backends.mps.is_available():
        print(f"{EMOJI['gpu']} Apple Silicon MPS available")
        found = True
    if getattr(torch.version, "hip", None):
        print(f"{EMOJI['gpu']} AMD ROCm — HIP {torch.version.hip}")
        found = True
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        print(f"{EMOJI['gpu']} Intel XPU: {torch.xpu.device_count()}")
        found = True
    if not found:
        print(f"{EMOJI['warnings']} No GPU found — CPU only. Fine for EMT, "
              f"slow for a machine-learned potential.")


@register_function(
    aliases=["view structure", "show structure", "3d view", "visualise",
             "look at it", "render structure", "draw the cell"],
    category="pl",
    description="Draw a structure in three dimensions — interactive in a "
                "notebook, a static image otherwise.",
    requires={"structures": ["{source}"]},
    examples=["mv.pl.structure(md, 0)",
              "mv.pl.structure(md, 'LiFePO4', supercell=(2, 2, 2))",
              "mv.pl.structure(md, 0, backend='matplotlib')"],
    related=["mv.pl.periodic_table", "mv.env.coordination"],
    notes="Looking at a structure is how you catch the mistakes a number will "
          "not show you: a slab built upside down, a molecule that came out of "
          "a parser inside out, an interface with the film on the wrong side.\n\n"
          "py3Dmol renders in the browser and is interactive; the matplotlib "
          "fallback needs nothing extra and produces an axis like every other "
          "plot here, so it composes into a figure. Neither replaces VESTA or "
          "Crystal Toolkit for real inspection — this is for the quick look "
          "you take twenty times a day.",
)
def structure(md: AnnData, which=0, source: str = "input",
              backend: str = "auto", supercell=None, style: str = "stick",
              width: int = 480, height: int = 360, ax=None):
    """Draw one structure. Returns the viewer or the axis."""
    from ._core import structures as _structures

    names = [str(x) for x in md.obs.get("name", md.obs_names)]
    if isinstance(which, (int, np.integer)):
        index = int(which)
    else:
        text = str(which)
        pool = names if text in names else [str(x) for x in md.obs_names]
        if text not in pool:
            raise KeyError(f"{which!r} is not a row; rows are {pool[:8]}")
        index = pool.index(text)

    obj = _structures(md, source)[index]
    if supercell is not None and hasattr(obj, "lattice"):
        obj = obj.copy()
        obj.make_supercell(list(supercell))

    if backend == "auto":
        try:
            import py3Dmol                                # noqa: F401
            backend = "py3dmol"
        except ImportError:
            backend = "matplotlib"

    if backend == "py3dmol":
        return _view_py3dmol(obj, style, width, height)
    if backend == "matplotlib":
        return _view_matplotlib(obj, ax)
    raise ValueError(f"backend must be 'auto', 'py3dmol' or 'matplotlib', "
                     f"got {backend!r}")


def _view_py3dmol(obj, style: str, width: int, height: int):
    import py3Dmol

    from pymatgen.io.xyz import XYZ

    viewer = py3Dmol.view(width=width, height=height)
    if hasattr(obj, "lattice"):
        # CIF carries the cell, which is the point of drawing a crystal.
        from pymatgen.io.cif import CifWriter

        viewer.addModel(str(CifWriter(obj)), "cif")
        viewer.addUnitCell()
    else:
        viewer.addModel(str(XYZ(obj)), "xyz")
    viewer.setStyle({style: {}, "sphere": {"scale": 0.3}})
    viewer.zoomTo()
    return viewer


def _view_matplotlib(obj, ax=None):
    """A projection along the shortest axis. Crude, and needs nothing."""
    plt = _plt()
    from .elements import element_frame

    coords = np.asarray(obj.cart_coords, dtype=float)
    symbols = [str(s.specie.symbol) for s in obj]
    spread = coords.max(axis=0) - coords.min(axis=0)
    depth = int(np.argmin(spread))
    plane = [k for k in range(3) if k != depth]

    ax = _axis(ax, figsize=(4.8, 4.4))
    frame = element_frame(sorted(set(symbols)))
    radii = frame["atomic_radius"].to_dict() if "atomic_radius" in frame \
        else {}
    order = np.argsort(coords[:, depth])
    palette = plt.get_cmap("tab20")
    colours = {el: palette(i % 20) for i, el in enumerate(sorted(set(symbols)))}

    for i in order:
        radius = radii.get(symbols[i]) or 1.0
        ax.scatter(coords[i, plane[0]], coords[i, plane[1]],
                   s=260 * float(radius) ** 2, color=colours[symbols[i]],
                   edgecolors="#333", linewidths=0.6, zorder=2)
    for element, colour in colours.items():
        ax.scatter([], [], color=colour, edgecolors="#333", label=element)

    ax.set_aspect("equal")
    ax.set_xlabel(f"{'xyz'[plane[0]]} (Å)")
    ax.set_ylabel(f"{'xyz'[plane[1]]} (Å)")
    ax.legend(fontsize=8, loc="best")
    return ax


@register_function(
    aliases=["periodic table", "periodic table heatmap", "element map",
             "plot elements", "element heatmap"],
    category="pl",
    description="Colour the periodic table by a per-element value — an "
                "enrichment score, a count, a fitted correction — laying "
                "elements out in their conventional positions.",
    requires={"var": ["{color}"]},
    examples=["mv.pl.periodic_table(md, color='n_materials')",
              "mv.pl.periodic_table(md, values=scores, label='log2 odds')"],
    related=["mv.tl.rank_elements_groups", "mv.pp.harmonize"],
    notes="The display for rank_elements_groups. A bar chart of 118 categories "
          "is unreadable and throws away the structure a chemist reads a "
          "periodic table for — groups above one another, periods across.",
)
def periodic_table(md: AnnData, color: str | None = None, values=None,
                   label: str | None = None, cmap: str = "RdBu_r",
                   center: float | None = None, ax=None, show_symbols=True):
    """Heatmap over the periodic table. Returns the axis."""
    from .elements import periodic_table_layout

    plt = _plt()
    if values is None:
        if color is None:
            raise ValueError("pass color= naming a var column, or values=")
        if color not in md.var.columns:
            raise ValueError(f"var[{color!r}] absent; available: "
                             f"{list(md.var.columns)}")
        series = md.var[color]
        label = label or color
    else:
        values = np.asarray(values, dtype=float)
        if len(values) != md.n_vars:
            raise ValueError(f"got {len(values)} values for {md.n_vars} elements")
        import pandas as pd
        series = pd.Series(values, index=md.var_names)

    layout = periodic_table_layout()
    ax = _axis(ax, figsize=(11.0, 5.5))

    finite = series.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("nothing finite to plot")
    if center is not None:
        span = float(np.nanmax(np.abs(finite - center))) or 1.0
        vmin, vmax = center - span, center + span
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colours = plt.get_cmap(cmap)

    for symbol, row in layout.iterrows():
        x, y = int(row["group"]), int(row["period"])
        present = symbol in series.index
        value = float(series.get(symbol, np.nan)) if present else np.nan
        face = colours(norm(value)) if np.isfinite(value) else "#f2f2f2"
        ax.add_patch(plt.Rectangle((x - 0.46, -y - 0.46), 0.92, 0.92,
                                   facecolor=face,
                                   edgecolor="#999999" if present else "#e0e0e0",
                                   linewidth=0.6))
        if show_symbols:
            ax.text(x, -y, symbol, ha="center", va="center", fontsize=7,
                    color=_readable(face) if np.isfinite(value) else "#b0b0b0")

    ax.set_xlim(0.2, 18.8)
    ax.set_ylim(-10.8, -0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colours)
    mappable.set_array([])
    bar = ax.figure.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
    if label:
        bar.set_label(label)
    return ax


def _readable(face) -> str:
    """Black or white text, whichever the background can carry."""
    try:
        r, g, b = face[:3]
    except Exception:
        return "black"
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


@register_function(
    aliases=["plot rank elements", "plot enrichment", "enrichment plot",
             "marker elements plot"],
    category="pl",
    description="Draw the elements that characterise one group, ordered by "
                "significance, with the direction of enrichment shown.",
    requires={"uns": ["rank_elements_groups"]},
    prerequisites=["mv.tl.rank_elements_groups"],
    examples=["mv.pl.rank_elements_groups(md, group='True')"],
    related=["mv.tl.rank_elements_groups", "mv.pl.periodic_table"],
)
def rank_elements_groups(md: AnnData, group: str | None = None, n: int = 10,
                         key: str = "rank_elements_groups", ax=None):
    """Top elements for one group. Returns the axis."""
    if key not in md.uns:
        raise ValueError(f"uns[{key!r}] absent; run mv.tl.rank_elements_groups "
                         f"first")
    result = md.uns[key]
    groups = list(result.get("groups", []))
    chosen = str(group) if group is not None else (groups[0] if groups else None)
    if chosen not in result:
        raise ValueError(f"group {chosen!r} not in {groups}")

    frame = result[chosen].head(n).iloc[::-1]
    effect = ("log2_odds" if "log2_odds" in frame.columns else "diff")
    heights = frame[effect].to_numpy(dtype=float)
    heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

    ax = _axis(ax, figsize=(5.0, 0.35 * len(frame) + 1.2))
    ax.barh(range(len(frame)), heights,
            color=["#c0392b" if h > 0 else "#2c6e9b" for h in heights])
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["element"])
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_xlabel("log2 odds ratio" if effect == "log2_odds"
                  else "difference in mean atomic fraction")
    ax.set_title(f"elements characterising {chosen!r}")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


@register_function(
    aliases=["plot hull", "convex hull plot", "hull diagram",
             "stability plot", "phase diagram plot"],
    category="pl",
    description="Plot distance above the convex hull against composition for a "
                "binary system, showing which candidates lie on the hull.",
    requires={"obs": ["e_above_hull_{level}"]},
    prerequisites=["mv.thermo.hull"],
    examples=["mv.pl.hull(md, level='emt', x='Al')"],
    related=["mv.thermo.hull", "mv.screen.filter"],
    notes="A binary section. Ternary and higher need a simplex plot, which "
          "pymatgen's PDPlotter already draws well — this is for the case that "
          "comes up in a screen, where one axis is the composition variable.",
)
def hull(md: AnnData, level: str = "emt", x: str | None = None, ax=None,
         annotate: bool = True):
    """Energy above hull against atomic fraction of one element."""
    column = f"e_above_hull_{level}"
    if column not in md.obs:
        raise ValueError(f"obs[{column!r}] absent; run mv.thermo.hull("
                         f"md, level={level!r}) first")
    if md.n_vars == 0:
        raise ValueError("this object has no element axis (build_X=False)")

    element = x or str(md.var_names[0])
    if element not in list(md.var_names):
        raise ValueError(f"{element!r} is not on the element axis "
                         f"({list(md.var_names)})")

    raw = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    raw = np.asarray(raw, dtype=float)
    totals = raw.sum(axis=1, keepdims=True)
    fractions = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)
    xs = fractions[:, list(md.var_names).index(element)]
    ys = md.obs[column].to_numpy(dtype=float)
    stable = md.obs.get(f"is_stable_{level}", np.zeros(md.n_obs, bool))
    stable = np.asarray(stable, dtype=bool)

    ax = _axis(ax)
    ax.scatter(xs[~stable], ys[~stable], s=28, c="#8fa8bb",
               edgecolors="none", label="above hull")
    ax.scatter(xs[stable], ys[stable], s=46, c="#c0392b",
               edgecolors="white", linewidth=0.6, label="on hull")
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xlabel(f"atomic fraction {element}")
    ax.set_ylabel(f"E above hull ({level}, eV/atom)")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if annotate and "formula" in md.obs:
        for i in np.where(stable)[0]:
            ax.annotate(str(md.obs["formula"].iloc[i]), (xs[i], ys[i]),
                        fontsize=7, xytext=(0, 6),
                        textcoords="offset points", ha="center")
    if md.uns.get("phase_diagram", {}).get("closed_system", False):
        ax.set_title("hull closed over this dataset — relative, not absolute",
                     fontsize=8, color="#c0392b")
    return ax


@register_function(
    aliases=["parity plot", "predicted versus actual", "scatter predictions",
             "compare levels plot", "validation plot"],
    category="pl",
    description="Plot one quantity computed at two levels of theory against "
                "each other, with the error bars when an uncertainty is "
                "recorded and the parity line for reference.",
    requires={"obs": ["{quantity}_{a}", "{quantity}_{b}"]},
    examples=["mv.pl.parity(md, 'energy_per_atom', 'emt', 'mace-mpa')"],
    related=["mv.compare_levels", "mv.calc.committee"],
    notes="Draws the uncertainty when obs['<quantity>_<level>_std'] exists. A "
          "parity plot without error bars invites a reader to over-trust the "
          "scatter.",
)
def parity(md: AnnData, quantity: str, a: str, b: str, ax=None):
    """Scatter of one quantity at two levels, with MAE and RMSE annotated."""
    xa, xb = f"{quantity}_{a}", f"{quantity}_{b}"
    for column in (xa, xb):
        if column not in md.obs:
            raise ValueError(f"obs[{column!r}] absent; available: "
                             f"{[c for c in md.obs.columns if quantity in c]}")
    X = md.obs[xa].to_numpy(dtype=float)
    Y = md.obs[xb].to_numpy(dtype=float)
    err = md.obs[f"{xb}_std"].to_numpy(dtype=float) \
        if f"{xb}_std" in md.obs else None

    ax = _axis(ax, figsize=(4.6, 4.6))
    if err is not None:
        ax.errorbar(X, Y, yerr=err, fmt="none", ecolor="#b0c4d4",
                    elinewidth=0.9, capsize=0)
    ax.scatter(X, Y, s=34, c="#2c6e9b", edgecolors="white", linewidth=0.5)

    both = np.isfinite(X) & np.isfinite(Y)
    if both.any():
        lo = float(min(X[both].min(), Y[both].min()))
        hi = float(max(X[both].max(), Y[both].max()))
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="#666666", linewidth=0.9, linestyle="--")
        mae = float(np.mean(np.abs(X[both] - Y[both])))
        rmse = float(np.sqrt(np.mean((X[both] - Y[both]) ** 2)))
        ax.set_title(f"MAE {mae:.3g}   RMSE {rmse:.3g}", fontsize=9)

    ax.set_xlabel(f"{quantity} ({a})")
    ax.set_ylabel(f"{quantity} ({b})")
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _note_levels(ax, md, a, b)
    return ax


def _note_levels(ax, md: AnnData, a: str, b: str) -> None:
    """Say when two levels reproduce different methods, on the plot itself."""
    levels = md.uns.get("levels", {})
    ra = (levels.get(a) or {}).get("reference")
    rb = (levels.get(b) or {}).get("reference")
    if ra != rb and (ra or rb):
        ax.text(0.03, 0.95, f"{a}→{ra}   {b}→{rb}", transform=ax.transAxes,
                fontsize=7, color="#c0392b", va="top")


@register_function(
    aliases=["pareto plot", "trade off plot", "plot pareto front",
             "multi objective plot"],
    category="pl",
    description="Plot two objectives against each other with the non-dominated "
                "front highlighted and connected.",
    requires={"obs": ["{x}", "{y}", "{key}"]},
    prerequisites=["mv.screen.pareto"],
    examples=["mv.pl.pareto(md, 'e_above_hull_emt', 'density')"],
    related=["mv.screen.pareto"],
)
def pareto(md: AnnData, x: str, y: str, key: str = "pareto", ax=None):
    """Two objectives, with the first front drawn as a line."""
    for column in (x, y, key):
        if column not in md.obs:
            raise ValueError(f"obs[{column!r}] absent; run mv.screen.pareto "
                             f"first, or name existing columns")
    X = md.obs[x].to_numpy(dtype=float)
    Y = md.obs[y].to_numpy(dtype=float)
    front = np.asarray(md.obs[key], dtype=bool)

    ax = _axis(ax)
    ax.scatter(X[~front], Y[~front], s=26, c="#b8c6d1", edgecolors="none",
               label="dominated")
    order = np.argsort(X[front])
    ax.plot(X[front][order], Y[front][order], color="#c0392b", linewidth=1.0,
            marker="o", markersize=6, markeredgecolor="white", label="front")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


@register_function(
    aliases=["chemical space plot", "plot embedding", "umap plot", "pca plot",
             "scatter embedding"],
    category="pl",
    description="Scatter the materials in an embedding, coloured by any obs "
                "column, giving a map of the chemical space a library spans.",
    requires={"obsm": ["{use_rep}"]},
    prerequisites=["mv.tl.pca"],
    examples=["mv.pl.embedding(md, color='is_stable_emt')"],
    related=["mv.tl.pca", "mv.tl.cluster"],
)
def embedding(md: AnnData, color: str | None = None, use_rep: str = "X_pca",
              components=(0, 1), ax=None, cmap: str = "viridis",
              legend: bool = True):
    """Scatter an embedding. Returns the axis."""
    import pandas as pd

    if use_rep not in md.obsm:
        raise ValueError(f"obsm[{use_rep!r}] absent; run mv.tl.pca first")
    Z = np.asarray(md.obsm[use_rep], dtype=float)
    i, j = components
    ax = _axis(ax, figsize=(5.0, 4.4))

    if color is None:
        ax.scatter(Z[:, i], Z[:, j], s=34, c="#2c6e9b", edgecolors="white",
                   linewidth=0.5)
    else:
        if color not in md.obs:
            raise ValueError(f"obs[{color!r}] absent")
        values = md.obs[color]
        if not pd.api.types.is_numeric_dtype(values) or \
                pd.api.types.is_bool_dtype(values):
            categories = list(pd.unique(values.astype(str)))
            palette = _plt().get_cmap("tab10")
            for k, category in enumerate(categories):
                mask = values.astype(str).to_numpy() == category
                ax.scatter(Z[mask, i], Z[mask, j], s=34,
                           color=palette(k % 10), edgecolors="white",
                           linewidth=0.5, label=str(category))
            if legend:
                ax.legend(frameon=False, fontsize=8, title=color)
        else:
            points = ax.scatter(Z[:, i], Z[:, j], s=34,
                                c=values.to_numpy(dtype=float), cmap=cmap,
                                edgecolors="white", linewidth=0.5)
            ax.figure.colorbar(points, ax=ax, fraction=0.04, pad=0.02,
                               label=color)

    ax.set_xlabel(f"{use_rep}[{i}]")
    ax.set_ylabel(f"{use_rep}[{j}]")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


@register_function(
    aliases=["plot spectra", "plot xrd", "spectrum plot", "pattern plot",
             "plot grid quantity"],
    category="pl",
    description="Overlay grid-shaped curves — diffraction patterns, densities "
                "of states — for selected materials, optionally comparing a "
                "computed level against a measured one.",
    requires={"obsm": ["{quantity}_{levels}"], "uns": ["grids"]},
    prerequisites=["mv.prop.xrd"],
    examples=["mv.pl.spectra(md, 'xrd', rows=[0, 1, 2])",
              "mv.pl.spectra(md, 'xrd', rows=[0], levels=['calc', "
              "'experiment'])"],
    related=["mv.prop.xrd", "mv.exp.attach", "mv.prop.compare_grids"],
)
def spectra(md: AnnData, quantity: str = "xrd", levels=("calc",), rows=None,
            offset: float = 0.0, ax=None):
    """Overlay curves. ``offset`` stacks them for legibility."""
    grid = grid_of(md, quantity)
    indices = list(range(min(md.n_obs, 5))) if rows is None else list(rows)
    ax = _axis(ax, figsize=(7.0, 4.0))

    styles = ["-", "--", ":", "-."]
    for li, level in enumerate(levels):
        key = f"{quantity}_{level}"
        if key not in md.obsm:
            raise ValueError(f"obsm[{key!r}] absent")
        block = np.asarray(md.obsm[key], dtype=float)
        for k, i in enumerate(indices):
            label = f"{md.obs_names[i]}"
            if len(levels) > 1:
                label += f" ({level})"
            ax.plot(grid, block[i] + k * offset, linewidth=0.9,
                    linestyle=styles[li % len(styles)], label=label)

    unit = md.uns.get("grids", {}).get(quantity, {}).get("unit", "")
    ax.set_xlabel(f"{quantity} axis" + (f" ({unit})" if unit else ""))
    ax.set_ylabel("intensity" + (" (offset)" if offset else ""))
    ax.legend(frameon=False, fontsize=7, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


@register_function(
    aliases=["plot provenance", "pipeline diagram", "what ran", "show history"],
    category="pl",
    description="Draw the operations applied to this object in order, as a "
                "readable record of how the result was produced.",
    requires={"uns": ["provenance"]},
    examples=["mv.pl.provenance(md)"],
    related=["mv.provenance"],
    notes="A figure of the pipeline, taken from the object rather than from a "
          "lab notebook, which is the point of recording provenance in the "
          "object at all.",
)
def provenance(md: AnnData, ax=None):
    """The operation history as a labelled sequence. Returns the axis."""
    from ._core import provenance as _provenance

    steps = _provenance(md)
    if not steps:
        raise ValueError("uns['provenance'] is empty")

    ax = _axis(ax, figsize=(7.5, 0.42 * len(steps) + 0.8))
    for i, step in enumerate(steps):
        y = -i
        ax.add_patch(_plt().Rectangle((0.02, y - 0.32), 0.96, 0.64,
                                      facecolor="#eef3f7",
                                      edgecolor="#9db4c6", linewidth=0.7))
        ax.text(0.05, y, step, fontsize=8, va="center", family="monospace")
        if i:
            ax.annotate("", xy=(0.5, y + 0.34), xytext=(0.5, y + 0.68),
                        arrowprops={"arrowstyle": "-|>", "color": "#9db4c6",
                                    "linewidth": 0.8})
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(steps) + 0.4, 0.6)
    ax.axis("off")
    return ax


__all__ = ["set_style", "structure", "periodic_table", "rank_elements_groups", "hull",
           "parity", "pareto", "embedding", "spectra", "provenance"]


@register_function(
    aliases=["scatter", "plot", "xy plot", "two columns", "scatter plot",
             "plot one against another", "bar chart"],
    category="pl",
    description="Scatter any two obs columns, optionally coloured by a third "
                "and grouped by a categorical one.",
    requires={"obs": ["{x}", "{y}"]},
    examples=["mv.pl.scatter(md, 'volume', 'bulk_modulus_emt')",
              "mv.pl.scatter(md, 'nsites', 'energy_emt', color='formula')",
              "mv.pl.scatter(sites, 'coordination_number', 'force_emt')"],
    related=["mv.pl.pareto", "mv.pl.embedding", "mv.pl.hull",
             "mv.pl.distribution"],
    notes="The plot there was no function for. mv.pl.pareto draws two "
          "objectives with a front, mv.pl.embedding draws a projection, "
          "mv.pl.hull draws one particular pair — and the ordinary case of "
          "one column against another had to be hand-drawn, which is why "
          "matverse's own tutorials reached for matplotlib more often than "
          "for mv.pl.\\n\\n"
          "Works on any axis, because it only asks for obs columns: the "
          "material axis, the sites axis from mv.multi.sites, a facets object "
          "from mv.surf.slabs. That is the point of the columns being on obs "
          "in the first place.\\n\\n"
          "color= takes a numeric column and draws a colourbar, or a "
          "categorical one and draws a legend, deciding by dtype rather than "
          "by an argument. Passing a categorical column with more than a "
          "dozen levels makes an unreadable legend and says so.\\n\\n"
          "kind='bar' draws the same pair as bars, for the case where each "
          "category has one value — a bulk modulus per material, an area "
          "fraction per facet. It refuses a numeric x, because a bar chart "
          "over a continuous axis is a histogram and mv.pl.distribution "
          "draws those.\\n\\n"
          "A categorical x is laid out at integers and jittered, so one "
          "value per element or per facet is the same call rather than a "
          "different function — and coincident points stay visible instead "
          "of stacking into one.\n\n"
          "annotate= labels each point with a column, which is what makes a "
          "twenty-material screen readable and a two-thousand-material one "
          "unusable; it refuses above fifty points rather than drawing them "
          "on top of each other.",
)
def scatter(md: AnnData, x: str, y: str, color: str | None = None,
            size: str | None = None, annotate: str | None = None,
            kind: str = "scatter", log_x: bool = False, log_y: bool = False,
            cmap: str = "viridis", ax=None):
    """Scatter two obs columns. Returns the axis."""
    if kind not in ("scatter", "bar"):
        raise ValueError(f"kind must be 'scatter' or 'bar', got {kind!r}")
    for column in (x, y):
        if column not in md.obs:
            raise ValueError(f"obs[{column!r}] absent; the columns on this "
                             f"object are {sorted(md.obs.columns)[:8]}...")
    for column in (color, size, annotate):
        if column is not None and column not in md.obs:
            raise ValueError(f"obs[{column!r}] absent")

    # A categorical abscissa is the commonest case after a numeric one - one
    # value per element, per facet, per space group - and it is the same plot
    # with the categories laid out at integers and jittered so coincident
    # points are visible rather than one point.
    categories = None
    raw_x = md.obs[x]
    if raw_x.dtype.kind in "ifu":
        X = raw_x.to_numpy(dtype=float)
    else:
        categories = list(dict.fromkeys(map(str, raw_x)))
        position = {name: index for index, name in enumerate(categories)}
        X = np.array([position[str(v)] for v in raw_x], dtype=float)
        counts: dict[float, int] = {}
        jitter = np.zeros(len(X))
        for index, value in enumerate(X):
            seen = counts.get(value, 0)
            counts[value] = seen + 1
            jitter[index] = seen
        for value, total in counts.items():
            mask = X == value
            if total > 1:
                jitter[mask] = np.linspace(-0.18, 0.18, total)
            else:
                jitter[mask] = 0.0
        X = X + jitter

    Y = md.obs[y].to_numpy(dtype=float)
    ax = _axis(ax)

    marker_size = 36.0
    if size is not None:
        raw = md.obs[size].to_numpy(dtype=float)
        finite = np.isfinite(raw)
        if finite.any() and np.nanmax(raw) > np.nanmin(raw):
            spread = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw))
            marker_size = 18.0 + 120.0 * np.nan_to_num(spread)

    if kind == "bar":
        if categories is None:
            raise ValueError(
                f"kind='bar' needs a categorical x; obs[{x!r}] is numeric. "
                f"A bar chart over a continuous axis is a histogram — "
                f"mv.pl.distribution draws that.")
        heights = np.zeros(len(categories))
        slot = np.zeros(len(categories), dtype=int)
        for index, value in enumerate(md.obs[x].astype(str)):
            heights[categories.index(value)] = Y[index]
            slot[categories.index(value)] = index

        colours = "#4c72b0"
        if color is not None:
            values = md.obs[color]
            if values.dtype.kind in "ifu" and values.nunique() > 2:
                raise ValueError(
                    f"kind='bar' colours by a categorical column; "
                    f"obs[{color!r}] is numeric. Use kind='scatter' for a "
                    f"colourbar.")
            labels = np.asarray(values.astype(str))[slot]
            groups = list(dict.fromkeys(labels))
            palette = _plt().get_cmap("tab10")
            lookup = {g: palette(i % 10) for i, g in enumerate(groups)}
            colours = [lookup[g] for g in labels]

        ax.bar(range(len(categories)), heights, color=colours,
               edgecolor="white", linewidth=0.6)
        if color is not None:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor=lookup[g], label=g)
                               for g in groups], frameon=False, fontsize=8)
        ax.set_xticks(range(len(categories)))
        rotate = max(len(c) for c in categories) > 4
        ax.set_xticklabels(categories, rotation=45 if rotate else 0,
                           ha="right" if rotate else "center")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if log_y:
            ax.set_yscale("log")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        return ax

    if color is None:
        ax.scatter(X, Y, s=marker_size, c="#4c72b0", edgecolors="white",
                   linewidths=0.4)
    else:
        values = md.obs[color]
        numeric = values.dtype.kind in "ifu"
        if numeric:
            drawn = ax.scatter(X, Y, s=marker_size,
                               c=values.to_numpy(dtype=float), cmap=cmap,
                               edgecolors="white", linewidths=0.4)
            bar = ax.figure.colorbar(drawn, ax=ax, pad=0.02)
            bar.set_label(color, fontsize=9)
            bar.outline.set_visible(False)
        else:
            groups = list(dict.fromkeys(map(str, values)))
            if len(groups) > 12:
                raise ValueError(
                    f"obs[{color!r}] has {len(groups)} distinct values, which "
                    f"makes a legend nobody can read. Colour by a numeric "
                    f"column, or group the categories first.")
            palette = _plt().get_cmap("tab10")
            labels = np.asarray(list(map(str, values)))
            for index, group in enumerate(groups):
                mask = labels == group
                sizes = (marker_size if np.isscalar(marker_size)
                         else marker_size[mask])
                ax.scatter(X[mask], Y[mask], s=sizes, label=group,
                           color=palette(index % 10), edgecolors="white",
                           linewidths=0.4)
            ax.legend(frameon=False, fontsize=8)

    if annotate is not None:
        if md.n_obs > 50:
            raise ValueError(
                f"annotate= on {md.n_obs} points draws labels on top of one "
                f"another. Subset the object first, or leave it off.")
        for xi, yi, text in zip(X, Y, md.obs[annotate].astype(str)):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(text, (xi, yi), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")

    if categories is not None:
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45 if
                           max(len(c) for c in categories) > 4 else 0,
                           ha="right" if max(len(c) for c in categories) > 4
                           else "center")
        if log_x:
            raise ValueError(f"obs[{x!r}] is categorical; a log scale on it "
                             f"would be meaningless")
    elif log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


@register_function(
    aliases=["band structure plot", "plot bands", "band diagram",
             "electronic structure plot"],
    category="pl",
    description="Draw a band structure along its path, with the Fermi level "
                "marked and one colour per material.",
    requires={"var": ["path_fraction"], "obs": ["material"]},
    prerequisites=["mv.elec.bands"],
    examples=["mv.pl.bands(bands)",
              "mv.pl.bands(bands, materials=['Cu'], labels={0: 'Γ', 1: 'X'})"],
    related=["mv.elec.bands", "mv.elec.band_features", "mv.pl.spectra"],
    notes="A band structure is not a spectrum and mv.pl.spectra will not do "
          "it: the rows are bands rather than materials, they all share one "
          "abscissa, and the Fermi level is a line the reader needs.\\n\\n"
          "**The abscissa is the fraction along each material's own path**, "
          "not a wavevector. Two materials with different k-point counts are "
          "resampled onto one axis so the object can be a matrix at all, "
          "which means the horizontal position is comparable between "
          "materials only in the sense that both ran the same sequence of "
          "high-symmetry points. Read it as a path fraction and never as k.\\n\\n"
          "Energies are relative to the Fermi level, which is where "
          "mv.elec.bands puts them, so the line at zero is the Fermi level "
          "and needs no argument.\\n\\n"
          "labels= takes {path_fraction: name} and draws the high-symmetry "
          "ticks. It is an argument rather than something read off the "
          "object because mv.elec.bands does not keep the labels — the path "
          "came from whatever produced the band structure.",
)
def bands(bands_obj: AnnData, materials=None, labels: dict | None = None,
          highlight_fermi: bool = True, energy_range=None, ax=None):
    """Band structure along the path. Returns the axis."""
    if "path_fraction" not in bands_obj.var:
        raise ValueError("var['path_fraction'] absent; this is not a bands "
                         "object — build one with mv.elec.bands")
    if "material" not in bands_obj.obs:
        raise ValueError("obs['material'] absent; this is not a bands object")

    fraction = bands_obj.var["path_fraction"].to_numpy(dtype=float)
    names = list(dict.fromkeys(map(str, bands_obj.obs["material"])))
    if materials is not None:
        wanted = [str(m) for m in materials]
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise ValueError(f"no bands for {missing}; this object has "
                             f"{names}")
        names = wanted

    ax = _axis(ax)
    palette = ("#4c72b0", "#c1121f", "#2a9d8f", "#e9c46a", "#8e44ad",
               "#d35400")
    drawn = 0
    for index, name in enumerate(names):
        block = bands_obj[
            np.asarray(bands_obj.obs["material"]).astype(str) == name]
        values = np.asarray(block.X, dtype=float)
        colour = palette[index % len(palette)]
        for row_index, row in enumerate(values):
            ax.plot(fraction, row, color=colour, linewidth=0.9,
                    label=name if row_index == 0 else None)
            drawn += 1

    if highlight_fermi:
        ax.axhline(0.0, linestyle="--", color="#333333", linewidth=0.9)
    if labels:
        positions = sorted(labels)
        ax.set_xticks(positions)
        ax.set_xticklabels([labels[p] for p in positions])
        for position in positions:
            ax.axvline(position, color="#cccccc", linewidth=0.6, zorder=0)
    if energy_range is not None:
        ax.set_ylim(*energy_range)

    ax.set_xlim(float(fraction.min()), float(fraction.max()))
    ax.set_xlabel("fraction along the high-symmetry path"
                  if not labels else "")
    # A phonon dispersion is the same object with a different ordinate, and
    # mv.prop.dispersion says so in uns rather than leaving the axis labelled
    # as electron energies.
    ax.set_ylabel(str(bands_obj.uns.get("y_label", "E − E$_F$ (eV)")))
    if len(names) > 1:
        ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax._matverse_n_bands = drawn
    return ax


@register_function(
    aliases=["distribution", "histogram", "hist", "spread of a column",
             "how are they distributed"],
    category="pl",
    description="Histogram of one obs column, optionally split by a "
                "categorical one.",
    requires={"obs": ["{column}"]},
    examples=["mv.pl.distribution(sites, 'force_emt')",
              "mv.pl.distribution(md, 'e_above_hull_emt', by='crystal_system')"],
    related=["mv.pl.scatter", "mv.pl.periodic_table", "mv.pl.spectra"],
    notes="A mean is not a distribution, and the difference is usually the "
          "result. Twenty-six atoms with a mean force of 0.05 eV/Å can be "
          "twenty-six relaxed atoms or twenty-four relaxed atoms and two that "
          "are nowhere near, and a screen that ranks on the mean cannot tell "
          "you which.\\n\\n"
          "Works on any axis: the material axis, the sites axis, a facets or "
          "fragments object. Non-finite values are dropped and counted, "
          "because a column that is half NaN makes a histogram that looks "
          "like a narrow distribution rather than a missing one.\\n\\n"
          "by= overlays one histogram per category on shared bins, which is "
          "the comparison worth making — the same bins, or the two "
          "distributions are not being compared.",
)
def distribution(md: AnnData, column: str, by: str | None = None,
                 bins: int = 30, log_x: bool = False, ax=None):
    """Histogram of one obs column. Returns the axis."""
    if column not in md.obs:
        raise ValueError(f"obs[{column!r}] absent; this object has "
                         f"{sorted(md.obs.columns)[:8]}...")
    if by is not None and by not in md.obs:
        raise ValueError(f"obs[{by!r}] absent")

    values = md.obs[column].to_numpy(dtype=float)
    finite = np.isfinite(values)
    dropped = int((~finite).sum())
    if not finite.any():
        raise ValueError(f"obs[{column!r}] has no finite value to bin")

    ax = _axis(ax)
    # Shared edges, always: two histograms on different bins are two pictures,
    # not a comparison.
    edges = np.histogram_bin_edges(values[finite], bins=bins)

    if by is None:
        ax.hist(values[finite], bins=edges, color="#4c72b0",
                edgecolor="white", linewidth=0.5)
    else:
        labels = np.asarray(md.obs[by].astype(str))
        groups = list(dict.fromkeys(labels[finite]))
        if len(groups) > 8:
            raise ValueError(
                f"obs[{by!r}] has {len(groups)} categories; overlaying that "
                f"many histograms hides all of them. Group them first.")
        palette = _plt().get_cmap("tab10")
        for index, group in enumerate(groups):
            mask = finite & (labels == group)
            ax.hist(values[mask], bins=edges, alpha=0.55, label=group,
                    color=palette(index % 10), edgecolor="white",
                    linewidth=0.4)
        ax.legend(frameon=False, fontsize=8)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(column
                  + (f"  ({dropped} non-finite dropped)" if dropped else ""))
    ax.set_ylabel("count")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax._matverse_dropped = dropped
    return ax


@register_function(
    aliases=["space group distribution", "symmetry distribution",
             "spacegroup bar", "which space groups", "crystal system "
             "distribution", "how symmetric is this set"],
    category="pl",
    description="Distribution of space groups across a dataset, grouped by "
                "crystal system — what symmetry a generated or screened set "
                "actually has.",
    requires={"obs": ["{column}"]},
    examples=["mv.pl.spacegroups(built)",
              "mv.pl.spacegroups(md, column='spacegroup_number')",
              "mv.pl.spacegroups(built, column='requested_space_group')"],
    related=["mv.gen.from_symmetry", "mv.pp.describe", "mv.pl.distribution"],
    notes="A generated set has a symmetry distribution, and it is rarely the "
          "one that was asked for. mv.gen.from_symmetry records both the "
          "requested group and the one the structure actually has, and "
          "plotting the two side by side is how the difference becomes "
          "visible rather than a column nobody reads.\\n\\n"
          "Bars are grouped and coloured by crystal system rather than "
          "plotted as 230 flat categories, because the number itself carries "
          "no order a reader can use — 62 is not 'between' 61 and 63 in any "
          "sense that matters, but Pnma being orthorhombic does.",
)
def spacegroups(md: AnnData, column: str = "spacegroup_number",
                compare: str | None = None, top: int = 20, ax=None):
    """Space-group distribution grouped by crystal system. Returns the axis."""
    if column not in md.obs:
        raise ValueError(
            f"obs[{column!r}] absent; run mv.pp.symmetry(md) for "
            f"'spacegroup_number', or point column= at "
            f"mv.gen.from_symmetry's 'space_group' — this object has "
            f"{sorted(md.obs.columns)[:8]}...")
    if compare is not None and compare not in md.obs:
        raise ValueError(f"obs[{compare!r}] absent")

    numbers = pd.to_numeric(md.obs[column], errors="coerce").dropna()
    if numbers.empty:
        raise ValueError(f"obs[{column!r}] holds no usable space-group number")

    counts = numbers.astype(int).value_counts().sort_values(ascending=False)
    keep = counts.head(int(top))
    order = sorted(keep.index)

    ax = _axis(ax)
    systems = [_crystal_system(n) for n in order]
    palette = {"triclinic": "#8e44ad", "monoclinic": "#4c72b0",
               "orthorhombic": "#2a9d8f", "tetragonal": "#e9c46a",
               "trigonal": "#e76f51", "hexagonal": "#d35400",
               "cubic": "#c1121f"}
    positions = np.arange(len(order))
    width = 0.4 if compare is not None else 0.7
    ax.bar(positions - (width / 2 if compare is not None else 0),
           [keep[n] for n in order], width=width,
           color=[palette[s] for s in systems],
           label=column if compare is not None else None)

    if compare is not None:
        other = pd.to_numeric(md.obs[compare], errors="coerce").dropna()
        other = other.astype(int).value_counts()
        ax.bar(positions + width / 2, [int(other.get(n, 0)) for n in order],
               width=width, facecolor="none", edgecolor="#333333",
               linewidth=0.9, label=compare)
        ax.legend(frameon=False, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in order], rotation=90, fontsize=7)
    ax.set_xlabel("space group number")
    ax.set_ylabel("materials")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    seen = list(dict.fromkeys(systems))
    handles = [_plt().Line2D([], [], color=palette[s], linewidth=6, label=s)
               for s in seen]
    ax.add_artist(ax.legend(handles=handles, frameon=False, fontsize=7,
                            loc="upper right", title="crystal system",
                            title_fontsize=7))
    ax._matverse_n_groups = len(order)
    ax._matverse_dropped = int(len(counts) - len(keep))
    return ax


#: Space-group number ranges, in the international convention.
_CRYSTAL_SYSTEMS = ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                    (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
                    (230, "cubic"))


def _crystal_system(number: int) -> str:
    """The crystal system a space-group number belongs to."""
    for limit, name in _CRYSTAL_SYSTEMS:
        if number <= limit:
            return name
    return "cubic"


#: Voigt index pairs, in the order the 6x6 matrix uses.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _compliance_tensor(stiffness: np.ndarray) -> np.ndarray:
    """The rank-four compliance S_ijkl from a 6x6 stiffness matrix.

    The Voigt factors are the part worth stating: a compliance matrix carries
    a half on every shear index and a quarter on a shear-shear pair, where a
    stiffness matrix carries none. Inverting C and expanding without them
    gives a tensor that looks right and is wrong by a factor of four on the
    off-diagonal blocks — which shows up as an anisotropy that is not there.
    """
    compact = np.linalg.inv(np.asarray(stiffness, dtype=float).reshape(6, 6))
    full = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_VOIGT_PAIRS):
        for b, (k, l) in enumerate(_VOIGT_PAIRS):
            value = compact[a, b]
            if a >= 3:
                value *= 0.5
            if b >= 3:
                value *= 0.5
            for p, q in {(i, j), (j, i)}:
                for r, s in {(k, l), (l, k)}:
                    full[p, q, r, s] = value
    return full


def _youngs_along(compliance: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Young's modulus along each unit direction: 1 / (n n S n n)."""
    unit = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    denominator = np.einsum("ijkl,...i,...j,...k,...l->...", compliance,
                            unit, unit, unit, unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denominator) > 1e-12, 1.0 / denominator, np.nan)


@register_function(
    aliases=["elastic anisotropy", "directional youngs modulus", "young's "
             "modulus surface", "how anisotropic is it", "plot elastic tensor",
             "stiffness by direction"],
    category="pl",
    description="Young's modulus as a function of direction, in the three "
                "principal planes — what the full elastic tensor says that "
                "its isotropic averages do not.",
    requires={"obsm": ["elastic_tensor_{level}"]},
    prerequisites=["mv.prop.elastic"],
    examples=["mv.pl.elastic(md, level='emt')",
              "mv.pl.elastic(md, level='emt', row='Cu')"],
    related=["mv.prop.elastic", "mv.pl.spectra", "mv.screen.rank"],
    notes="mv.prop.elastic computes the whole 6x6 tensor and then reports "
          "four Voigt-Reuss-Hill averages — bulk, shear and Young's moduli "
          "and a Poisson ratio. Those are what a screen ranks on, and they "
          "are all isotropic: they are exactly the part of the tensor that "
          "survives averaging the anisotropy away. Computing the tensor and "
          "reading only the averages throws away the reason for computing "
          "it.\\n\\n"
          "Copper is the standard illustration and the suite checks against "
          "it: measured constants give a Young's modulus of 67 GPa along "
          "[100] and 191 along [111], a factor of 2.9, where the isotropic "
          "average is a single number near 120. A screen that ranks on the "
          "average is ranking materials whose stiffness varies threefold "
          "with direction as though it did not.\\n\\n"
          "The three curves are sections through the xy, xz and yz planes of "
          "the **crystal axes**, not of any conventional setting, so compare "
          "them between materials only when the cells were oriented the same "
          "way. ax._matverse_anisotropy carries max/min over the sampled "
          "sphere, which is one for an isotropic solid by construction.",
)
def elastic(md: AnnData, level: str = "emt", row=0, n_points: int = 361,
            ax=None):
    """Directional Young's modulus in three planes. Returns the axis."""
    key = f"elastic_tensor_{level}"
    if key not in md.obsm:
        raise ValueError(
            f"obsm[{key!r}] absent; run mv.prop.elastic(md, level={level!r}) "
            f"first")

    # obs['name'] first, the way mv.pl.structure resolves a row: a matverse
    # dataset carries the formula there and leaves obs_names as integers, so
    # looking only at obs_names sends "Cu" to int() and fails with a message
    # about base 10. The tutorial found this; the unit tests had named their
    # rows directly and never saw it.
    labels = [str(x) for x in md.obs.get("name", md.obs_names)]
    index_names = [str(x) for x in md.obs_names]
    key_row = str(row)
    if key_row in labels:
        index = labels.index(key_row)
    elif key_row in index_names:
        index = index_names.index(key_row)
    else:
        try:
            index = int(row)
        except (TypeError, ValueError):
            raise ValueError(
                f"no row {row!r}; this object has names {labels[:8]} and an "
                f"index of {index_names[:8]}") from None
    stiffness = np.asarray(md.obsm[key], dtype=float)[index]
    if not np.isfinite(stiffness).all():
        raise ValueError(
            f"the elastic tensor for row {labels[index]!r} is not finite; "
            f"mv.prop.elastic records a failed row that way")

    compliance = _compliance_tensor(stiffness)
    angle = np.linspace(0.0, 2.0 * np.pi, int(n_points))
    planes = {
        "xy": np.stack([np.cos(angle), np.sin(angle),
                        np.zeros_like(angle)], axis=-1),
        "xz": np.stack([np.cos(angle), np.zeros_like(angle),
                        np.sin(angle)], axis=-1),
        "yz": np.stack([np.zeros_like(angle), np.cos(angle),
                        np.sin(angle)], axis=-1),
    }

    ax = ax if ax is not None else _plt().subplots(
        figsize=(5.2, 5.2), subplot_kw={"projection": "polar"})[1]
    colours = ("#4c72b0", "#c1121f", "#2a9d8f")
    for (label, directions), colour in zip(planes.items(), colours):
        ax.plot(angle, _youngs_along(compliance, directions), color=colour,
                linewidth=1.4, label=label)

    # Anisotropy over the sphere rather than over the three sections, so the
    # number does not depend on which planes happen to be drawn.
    sphere = _fibonacci_sphere(512)
    over_sphere = _youngs_along(compliance, sphere)
    finite = over_sphere[np.isfinite(over_sphere) & (over_sphere > 0)]
    ratio = float(finite.max() / finite.min()) if finite.size else np.nan

    ax.set_title(f"{labels[index]} — Young's modulus by direction\n"
                 f"anisotropy {ratio:.2f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.15, 1.10))
    ax.grid(True, alpha=0.3)
    ax._matverse_anisotropy = ratio
    ax._matverse_extremes = (float(finite.min()), float(finite.max())) \
        if finite.size else (np.nan, np.nan)
    return ax


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Roughly equal-area directions on the unit sphere."""
    index = np.arange(int(count)) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / count)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1)


@register_function(
    aliases=["plot fermi surface", "draw fermi surface", "fermi surface 3d",
             "show the fermi surface"],
    category="pl",
    description="Draw the Fermi surface sheets stored by "
                "mv.elec.fermi_surface, in three dimensions.",
    requires={"uns": ["fermi_surface"]},
    prerequisites=["mv.elec.fermi_surface"],
    examples=["mv.pl.fermi_surface(md, level='pbe')",
              "mv.pl.fermi_surface(md, level='pbe', row='Cu', elevation=20)"],
    related=["mv.elec.fermi_surface", "mv.pl.bands", "mv.elec.band_features"],
    notes="Draws the mesh mv.elec.fermi_surface kept, rather than "
          "recomputing it. The interpolation behind a Fermi surface takes "
          "minutes, and a plotting function that repeats it on every call is "
          "not an interface worth having — so the vertices and faces are "
          "stored once and read here.\\n\\n"
          "The surface is clipped to the first Brillouin zone when "
          "mv.elec.fermi_surface was run with wigner_seitz, which is the "
          "default. A sphere wider than the zone therefore appears with flat "
          "faces where it crosses the boundary; those faces are physics, not "
          "a rendering artefact.\\n\\n"
          "Sheets are coloured separately because a count of them is the "
          "thing to read off: one closed sheet is a simple metal, several are "
          "pockets, and none is an insulator, which draws an empty axes "
          "rather than raising.",
)
def fermi_surface(md: AnnData, level: str = "dft", row=0,
                  azimuth: float = 45.0, elevation: float = 25.0, ax=None):
    """Fermi surface sheets in three dimensions. Returns the axis."""
    stored = (md.uns.get("fermi_surface") or {}).get(level)
    if stored is None:
        raise ValueError(
            f"uns['fermi_surface'][{level!r}] absent; run "
            f"mv.elec.fermi_surface(md, bandstructures, level={level!r}) "
            f"first")
    meshes = stored.get("meshes") or {}
    if not meshes:
        raise ValueError(
            "no mesh was kept; mv.elec.fermi_surface was run with "
            "keep_mesh=False, and the sheets cannot be redrawn without "
            "repeating the interpolation")

    names = [str(x) for x in md.obs_names]
    name = str(row) if str(row) in names else names[int(row)]
    sheets = meshes.get(name)
    if sheets is None:
        raise ValueError(
            f"no Fermi surface was stored for {name!r}; it has "
            f"{sorted(meshes)}")

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        ax = _plt().figure(figsize=(5.4, 5.0)).add_subplot(projection="3d")
    palette = ("#4c72b0", "#c1121f", "#2a9d8f", "#e9c46a", "#8e44ad")
    extent = 0.0
    for index, sheet in enumerate(sheets):
        vertices = np.asarray(sheet["vertices"], dtype=float)
        faces = np.asarray(sheet["faces"], dtype=int)
        if not len(faces):
            continue
        collection = Poly3DCollection(
            vertices[faces], alpha=0.75, linewidths=0.0,
            facecolor=palette[index % len(palette)])
        ax.add_collection3d(collection)
        extent = max(extent, float(np.abs(vertices).max()))

    if extent > 0:
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-extent, extent)
    ax.set_xlabel("$k_x$"); ax.set_ylabel("$k_y$"); ax.set_zlabel("$k_z$")
    ax.set_title(f"{name} — {len(sheets)} sheet"
                 f"{'s' if len(sheets) != 1 else ''}", fontsize=10)
    ax.view_init(elev=float(elevation), azim=float(azimuth))
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:                                      # pragma: no cover
        pass
    ax._matverse_n_sheets = len(sheets)
    return ax

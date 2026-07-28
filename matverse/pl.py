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
    requires={"obsm": ["{quantity}_{level}"], "uns": ["grids"]},
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

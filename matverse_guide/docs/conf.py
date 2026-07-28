"""Sphinx configuration for the matverse guide.

Mirrors the omicverse guide: Sphinx with ``sphinx_book_theme``, MyST notebooks
through ``myst-nb``, and ``sphinx-design`` cards on the landing page. Markdown is
the source format throughout — ``.md`` is routed to myst-nb so a tutorial and a
prose page are the same kind of file.
"""

import importlib.util
import inspect
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent

# Locate the matverse package. Layout 1 (local dev): the guide lives inside the
# repo, so HERE.parent.parent is the repo root holding matverse/. Layout 2: a
# sibling checkout named matverse-core, matching the omicverse arrangement.
_repo_root = HERE.parent.parent
if (_repo_root / "matverse").is_dir():
    sys.path.insert(0, str(_repo_root))
else:
    _core = _repo_root / "matverse-core"
    if _core.exists():
        sys.path.insert(0, str(_core))

# -- Project information -------------------------------------------------------
project = "matverse"
author = "matverse contributors"
copyright = f"{datetime.now():%Y}, matverse contributors"
release = "0.1.5"
version = release
repository_url = "https://github.com/matverse/matverse"
default_github_ref = "main"


def _fallback_github_ref() -> str:
    for key in ("READTHEDOCS_GIT_IDENTIFIER", "GITHUB_REF_NAME",
                "READTHEDOCS_VERSION_NAME"):
        value = os.environ.get(key)
        if value and value not in {"latest", "stable"}:
            return value
    return default_github_ref


try:
    from importlib.metadata import metadata as _pkg_meta
    _info = _pkg_meta("matverse")
    release = version = _info["Version"]
except Exception:
    pass

html_context = {
    "display_github": True,
    "github_user": "matverse",
    "github_repo": project,
    "github_version": _fallback_github_ref(),
    "conf_py_path": "/matverse_guide/docs/",
}

# -- Extensions ---------------------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "hoverxref.extension",
]

_ext_dir = HERE / "extensions"
if _ext_dir.exists():
    sys.path.insert(0, str(_ext_dir))
    for _p in sorted(_ext_dir.glob("*.py")):
        extensions.append(_p.stem)

# -- Autodoc / Napoleon -------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"

# Optional backends. matverse imports each of these inside the function that
# needs it, so autodoc only meets them when it reads a signature — but mocking
# them keeps a docs build from depending on the heavier half of the ecosystem.
autodoc_mock_imports = [
    "torch",
    "mace",
    "chgnet",
    "sevenn",
    "dscribe",
    "matminer",
    "mp_api",
    "igraph",
    "leidenalg",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
todo_include_todos = False

# sphinx_autodoc_typehints — skip rtype injection. Its docstring parser raises a
# SEVERE system message on any RST glitch, which aborts the whole build over one
# bad docstring; param-type injection is the part we actually rely on.
typehints_document_rtype = False
always_document_param_types = True

# -- MyST / myst-nb -----------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "substitution",
    "linkify",
]
myst_url_schemes = ("http", "https", "mailto")
myst_heading_anchors = 3
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

# -- Source files -------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    ".md": "myst-nb",
}
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]
needs_sphinx = "4.0"
nitpicky = False

# -- extlinks / intersphinx ---------------------------------------------------
extlinks = {
    "issue": (f"{repository_url}/issues/%s", "#%s"),
    "pr": (f"{repository_url}/pull/%s", "#%s"),
    "ghuser": ("https://github.com/%s", "@%s"),
}

# ASE and mudata are absent on purpose: neither publishes a reachable
# objects.inv, so mapping them means two 404s on every build and no cross
# references gained.
intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pymatgen": ("https://pymatgen.org/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

# -- HTML / sphinx_book_theme -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": default_github_ref,
    "path_to_docs": "matverse_guide/docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "navigation_with_keys": True,
}

pygments_style = "tango"
pygments_dark_style = "monokai"

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sphinx = False


# -- Linkcode -----------------------------------------------------------------
def _git(*args):
    return subprocess.check_output(["git", *args]).strip().decode()


_git_ref = None
try:
    _git_ref = _git("name-rev", "--name-only", "--no-undefined", "HEAD")
    _git_ref = re.sub(r"^(remotes/[^/]+|tags)/", "", _git_ref)
except Exception:
    pass
if not _git_ref or re.search(r"[\^~]", _git_ref):
    try:
        _git_ref = _git("rev-parse", "HEAD")
    except Exception:
        _git_ref = _fallback_github_ref()

_module_path = None
try:
    _spec = importlib.util.find_spec("matverse")
    if _spec and _spec.origin:
        _module_path = os.path.dirname(_spec.origin)
except Exception:
    pass


def linkcode_resolve(domain, info):
    """Resolve a Python object to its source URL on GitHub."""
    if domain != "py" or not _module_path:
        return None
    try:
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        if isinstance(obj, property):
            obj = inspect.unwrap(obj.fget)
        path = os.path.relpath(inspect.getsourcefile(obj), start=_module_path)
        src, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None
    return (f"{repository_url}/blob/{_git_ref}/matverse/{path}"
            f"#L{lineno}-L{lineno + len(src) - 1}")


def setup(app):
    """Regenerate the registry-driven API page before every build."""
    script = HERE / "_scripts" / "generate_api.py"
    if not script.exists():
        return
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(HERE))
    except Exception as exc:                              # pragma: no cover
        print(f"[matverse docs] could not regenerate api/user.md: {exc}")

# matverse guide

Documentation source for [matverse](../README.md).

Sphinx with `sphinx-book-theme`, MyST markdown through `myst-nb`, and
`sphinx-design` for the landing page cards — the same stack the omicverse guide
uses. Markdown is the source format throughout: `.md` is routed to myst-nb, so a
tutorial and a prose page are the same kind of file and a notebook can be dropped
in beside them.

## Build

```bash
pip install -r requirements.txt
pip install -e ..            # so autodoc can import matverse
cd docs
make html
```

Output lands in `docs/_build/html`.

## Layout

```
docs/
├── conf.py                  Sphinx configuration
├── index.md                 landing page
├── Installation_guide.md
├── Design.md                includes the repo-root DESIGN.md verbatim
├── Developer_guide.md       conventions for adding a function
├── Release_notes.md
├── api/
│   ├── index.md
│   └── user.md              GENERATED — do not edit
├── tutorials/
│   ├── index.md
│   ├── screening.ipynb
│   └── chemical_space.ipynb
├── _scripts/generate_api.py
├── _templates/autosummary/
└── _static/css/custom.css
```

## The API page is generated

`api/user.md` is written by `_scripts/generate_api.py` from matverse's
`@register_function` registry, and `conf.py` reruns it before every build. The
registry already holds the description, the aliases someone would search for, and
the slots each call reads and writes, so maintaining a second copy by hand would
only let the two drift.

A function added with its decorator appears in the docs on the next build. One
added without a decorator does not appear at all, which is the intended pressure.

To regenerate without a full build:

```bash
python docs/_scripts/generate_api.py
```

## Two things to know before editing

**Version pins are load-bearing.** `requirements.txt` holds sphinx below 8.0 and
sphinx-design below 0.7. Those pins came from the omicverse guide, which hit real
build failures on the newer lines; loosen them only with a build to show for it.

**`Design.md` is an include, not a copy.** It pulls `DESIGN.md` from the
repository root with a heading offset. Edit the root document.

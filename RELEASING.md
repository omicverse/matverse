# Releasing

Publishing is irreversible. PyPI will not let a version number be reused, even
after the file is deleted, so a mistake costs a version number permanently.
Nothing in `.github/workflows/publish.yml` runs on a push for that reason — a
release has to be created deliberately.

## One-time setup

The workflow uses **trusted publishing**, so no API token is stored anywhere.
GitHub proves the workflow's identity to PyPI over OIDC, which means there is no
secret to leak and none to rotate.

On [pypi.org](https://pypi.org/manage/account/publishing/), add a pending
publisher:

| Field | Value |
|---|---|
| PyPI project name | `matverse` |
| Owner | `omicverse` |
| Repository name | `matverse` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Repeat on [test.pypi.org](https://test.pypi.org/manage/account/publishing/) with
environment `testpypi`.

Then in the repository, under **Settings → Environments**, create `pypi` and
`testpypi`. Adding a required reviewer to `pypi` puts a human between a tag and
an upload, which is worth the ten seconds.

## Rehearse on TestPyPI first

**Actions → publish → Run workflow → target: testpypi.**

It builds, checks the metadata, installs the wheel into a clean virtualenv
outside the source tree, and uploads to TestPyPI. That last check matters more
than it looks: importing from the repository root passes even when the wheel is
missing half the package, because the source directory is already on the path.

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple matverse
```

## Release

```bash
# 1. bump the version in the three places that carry it
#      pyproject.toml            version
#      matverse/__init__.py      __version__
#      matverse_guide/docs/conf.py   release
# 2. write the entry in matverse_guide/docs/Release_notes.md
# 3. commit, then tag with a leading v
git tag v0.1.10
git push origin main --tags
```

Then create a GitHub release pointing at the tag. Publishing the release runs
the workflow.

The build job **fails if the tag and `matverse.__version__` disagree**. A tag
saying `v0.2.0` on a tree whose package says `0.1.9` produces a release nobody
can install by the name they were given, and that cannot be fixed after upload.

## Versioning

Patch-level increments: `0.1.9` → `0.1.10`. The library is pre-1.0 and the
substrate is still moving — `uns['calc']` became `uns['levels']`, structures
moved from `uns` to `obsm` — so the minor number is reserved for a change that
breaks a stored `h5ad`, and those are called out in the release notes.

## Before tagging

```bash
pytest -q                                              # the suite
pytest tests/test_contracts.py -q -s -k rate           # contract-verified rate
cd matverse_guide/docs && python -m sphinx -b html -E . _build/html
python -m build && python -m twine check dist/*
```

CI runs all of these on every pull request, so a green PR has already done it.
The contract rate is printed rather than only asserted, because a drop from
141/141 is a claim that stopped being true and is worth seeing in the log.

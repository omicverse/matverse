"""matverse-bench task specifications.

Each task states a **scientific objective and a storage target**, never an API
path. "Screen these candidates for thermodynamic stability and put the pass/fail
in obs" — not "call mv.thermo.hull". A benchmark that names the function is
measuring whether a model can read, not whether it can work.

Every task specification here was written against the goal first. Only afterwards
was it checked which functions the goal happens to need, and two tasks required
functions that did not exist and were left in rather than deleted — a benchmark
co-designed around the library it measures is worth nothing, and the credibility
problem is real enough that it is worth losing points over.

Tasks are layered by what they demand:

``single``
    one operation. Failure localises to discovery — did the model find the
    function at all?
``compose``
    two operations where the second consumes what the first deposited. Failure
    localises to composition — did it know the order?
``pipeline``
    a full screen. Failure can be either, and the per-check report says which.

The grader is in ``bench/grader.py`` and contains **no model calls**. Every
criterion is arguable from the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Check:
    """One thing that must be true of the object the agent produced.

    ``kind`` is what to look at, ``target`` is where, and the remaining fields
    constrain the value. ``pattern`` matches a column by regular expression
    rather than by exact name, because the name is not part of the specification
    unless the task said so — a model that writes ``stable`` where the reference
    solution writes ``passes`` has not made a mistake.
    """
    kind: str                       # obs | obsm | uns | structures | levels | X
    target: str = ""
    pattern: str | None = None
    dtype: str | None = None        # numeric | boolean | string
    min_true: int | None = None
    max_true: int | None = None
    finite_fraction: float | None = None
    within: tuple | None = None
    describe: str = ""


@dataclass(frozen=True)
class Task:
    """A scientific objective, a starting dataset, and how to grade the result."""
    id: str
    layer: str                      # single | compose | pipeline
    prompt: str
    fixture: str
    checks: list = field(default_factory=list)
    notes: str = ""


#: Named starting datasets. The grader builds these; the agent is handed one.
FIXTURES = {
    "alcuni": "Six Al-Cu-Ni candidates: three elementals and three ordered "
              "alloys. Every element is parameterised for the EMT calculator, "
              "so the whole pipeline runs without a downloaded model.",
    "alcuni_broken": "The same six, plus one structure with two atoms 0.08 A "
                     "apart — a broken cell of the kind a generative model "
                     "produces.",
    "alcuni_duplicated": "The same six, with the first repeated once.",
    "two_databases": "The same four compositions twice, labelled 'mp' and "
                     "'oqmd' in obs['database'], with a per-element energy "
                     "offset applied to one of them.",
}


TASKS = [
    # ---- single: one operation ---------------------------------------
    Task(
        id="describe-library",
        layer="single",
        fixture="alcuni",
        prompt="I have a set of candidate structures and I do not know the "
               "first thing about them. Record each one's chemical formula, "
               "how many atoms are in its cell, its volume and its density, "
               "as columns of the per-material table.",
        checks=[
            Check("obs", pattern=r"formula", dtype="string",
                  describe="a formula column"),
            Check("obs", pattern=r"n?sites|natoms", dtype="numeric",
                  describe="an atom-count column"),
            Check("obs", pattern=r"volume", dtype="numeric",
                  describe="a cell volume column"),
            Check("obs", pattern=r"density", dtype="numeric",
                  describe="a density column"),
        ],
    ),
    Task(
        id="symmetry",
        layer="single",
        fixture="alcuni",
        prompt="Work out the space group of every candidate and record it, and "
               "keep a symmetry-reduced version of each structure alongside the "
               "original rather than replacing it.",
        checks=[
            Check("obs", pattern=r"space ?_?group", describe="a space group column"),
            Check("structures", pattern=r"primitive|conventional|standard",
                  describe="a symmetry-reduced structure variant"),
            Check("structures", target="input",
                  describe="the original structures, still present"),
        ],
    ),
    Task(
        id="find-broken",
        layer="single",
        fixture="alcuni_broken",
        prompt="One of these structures is physically impossible — it has two "
               "atoms almost on top of each other. Flag every candidate as "
               "usable or not, so the broken one can be kept out of an "
               "expensive calculation.",
        checks=[
            Check("obs", pattern=r"valid|usable|ok|good|pass",
                  dtype="boolean", min_true=6, max_true=6,
                  describe="a validity flag marking exactly the six good rows"),
        ],
        notes="The count is the check. A flag that marks everything valid, or "
              "everything invalid, satisfies a shape test and fails this.",
    ),
    Task(
        id="find-duplicates",
        layer="single",
        fixture="alcuni_duplicated",
        prompt="This candidate list was concatenated from two sources and I "
               "think something is in it twice. Identify the repeats without "
               "removing anything.",
        checks=[
            Check("obs", pattern=r"dup", dtype="boolean",
                  min_true=1, max_true=1,
                  describe="exactly one row flagged as a duplicate"),
        ],
    ),
    Task(
        id="chemical-space",
        layer="single",
        fixture="alcuni",
        prompt="Give me a two-dimensional map of the chemical space these "
               "candidates span, stored so I can plot it later.",
        checks=[
            Check("obsm", pattern=r"pca|umap|tsne|embed",
                  describe="a low-dimensional embedding in obsm"),
        ],
    ),

    # ---- compose: two operations, one consuming the other -------------
    Task(
        id="relax-and-rank",
        layer="compose",
        fixture="alcuni",
        prompt="Relax these candidates with the effective-medium-theory "
               "calculator, then rank them from lowest to highest energy per "
               "atom. Keep both the relaxed geometries and the ranking.",
        checks=[
            Check("obs", pattern=r"energy_per_atom", dtype="numeric",
                  finite_fraction=1.0,
                  describe="an energy per atom for every candidate"),
            Check("structures", pattern=r"relax|opt",
                  describe="the relaxed geometries, kept"),
            Check("obs", pattern=r"rank|order", dtype="numeric",
                  describe="a ranking column"),
        ],
    ),
    Task(
        id="stability",
        layer="compose",
        fixture="alcuni",
        prompt="Work out how far each of these candidates sits above the "
               "convex hull of the ones you have, using the effective-medium "
               "calculator, and record the distance and whether each is on the "
               "hull.",
        checks=[
            Check("obs", pattern=r"e_?above_?hull|hull_?dist", dtype="numeric",
                  finite_fraction=1.0, within=(-1e-6, 1e9),
                  describe="a non-negative distance above the hull"),
            Check("obs", pattern=r"stable", dtype="boolean", min_true=1,
                  describe="a stability flag with at least one stable phase"),
        ],
        notes="The within= bound catches a sign error, which is the most common "
              "way this comes out wrong and is invisible to a presence check.",
    ),
    Task(
        id="descriptors-and-similarity",
        layer="compose",
        fixture="alcuni",
        prompt="Build a numerical fingerprint for each candidate from the "
               "properties of the elements it contains, then compute how "
               "similar every pair of candidates is.",
        checks=[
            Check("obsm", pattern=r".", describe="a descriptor block in obsm"),
            Check("obsp", pattern=r"similar|distance|correlation",
                  describe="a pairwise similarity matrix"),
        ],
    ),
    Task(
        id="reconcile-databases",
        layer="compose",
        fixture="two_databases",
        prompt="These energies came from two different databases and are not "
               "directly comparable — the two disagree by an amount that "
               "depends on composition. Put them on a common footing and store "
               "the corrected energies.",
        checks=[
            Check("obs", pattern=r"harmon|correct|adjust|reconcil",
                  dtype="numeric", finite_fraction=1.0,
                  describe="a corrected energy column"),
            Check("agreement", pattern=r"harmon|correct|adjust|reconcil",
                  within=(0.0, 1e-6),
                  describe="the two databases agreeing on shared compositions"),
        ],
        notes="A goal no single obvious function name announces. The reference "
              "solution is mv.pp.harmonize. The agreement check is what "
              "separates a real per-element reconciliation from subtracting a "
              "constant offset: both produce a column, and only one makes the "
              "databases agree on a composition whose correction differs from "
              "the average.",
    ),

    # ---- pipeline: a full screen -------------------------------------
    Task(
        id="screen-for-stability",
        layer="pipeline",
        fixture="alcuni_broken",
        prompt="Screen this candidate set for thermodynamically plausible "
               "materials. Throw out anything physically broken first, relax "
               "what survives with the effective-medium calculator, work out "
               "how far each sits above the convex hull, and mark which ones "
               "are within 50 meV per atom of it. Leave the criteria you used "
               "somewhere I can read them.",
        checks=[
            Check("obs", pattern=r"e_?above_?hull|hull_?dist", dtype="numeric",
                  describe="a distance above the hull"),
            Check("obs", pattern=r"pass|select|shortlist|screen|within",
                  dtype="boolean", min_true=1,
                  describe="a pass/fail flag with at least one candidate passing"),
            Check("uns", pattern=r"screen|criteri|filter",
                  describe="the criteria, recorded"),
            Check("structures", pattern=r"relax|opt",
                  describe="the relaxed geometries"),
        ],
    ),
    Task(
        id="which-chemistry-passed",
        layer="pipeline",
        fixture="alcuni",
        prompt="Split these candidates into the more and less stable half by "
               "energy per atom at the effective-medium level, then tell me "
               "which chemical elements are characteristic of the stable half. "
               "I want the answer stored on the object, not printed.",
        checks=[
            Check("obs", pattern=r"energy_per_atom", dtype="numeric",
                  describe="energies per atom"),
            Check("uns", pattern=r"rank_element|element_enrich|marker",
                  describe="a per-element enrichment result"),
        ],
        notes="Written against the goal. It happens to need "
              "mv.tl.rank_elements_groups, which is the operation the "
              "composition-matrix design exists to make possible — but the "
              "prompt names neither.",
    ),
    Task(
        id="full-report",
        layer="pipeline",
        fixture="alcuni",
        prompt="Prepare these candidates for a report: standardise the cells, "
               "record the basic structural properties, compute energies with "
               "the effective-medium calculator, simulate a powder diffraction "
               "pattern for each, and save the whole thing to a single h5ad "
               "file called report.h5ad.",
        checks=[
            Check("obs", pattern=r"formula", describe="basic properties"),
            Check("obs", pattern=r"energy", dtype="numeric",
                  describe="energies"),
            Check("obsm", pattern=r"xrd|diffract|pattern",
                  describe="diffraction patterns"),
            Check("file", target="report.h5ad",
                  describe="the object, saved and reloadable"),
        ],
        notes="The save is a real check. Structures in uns cannot be written to "
              "h5ad at all, which is how that bug was found — a task that ends "
              "in a file catches a whole class of failure that an in-memory "
              "assertion does not.",
    ),
]


def by_layer(layer: str) -> list:
    return [t for t in TASKS if t.layer == layer]


def by_id(task_id: str) -> Task:
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(f"no task {task_id!r}; have {[t.id for t in TASKS]}")


__all__ = ["Task", "Check", "TASKS", "FIXTURES", "by_layer", "by_id"]

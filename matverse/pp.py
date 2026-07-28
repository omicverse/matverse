"""``mv.pp`` — everything that happens before analysis.

Structure standardisation, quality control, filtering and deduplication. The
shape is borrowed from ``scanpy.pp`` because the operations genuinely
correspond: ``qc`` is ``calculate_qc_metrics``, ``filter_materials`` is
``filter_cells``, ``filter_elements`` is ``filter_genes``. A screening library
throws away bad rows and uninformative columns for the same reasons a
single-cell one does.

Every operation deposits and returns ``None``, except the two filters, which
return a subset because dropping rows cannot be done in place on an AnnData.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import deposit_structures, record, structures
from ._registry import register_function


@register_function(
    aliases=["standardize structure", "primitive cell", "conventional cell",
             "spacegroup", "symmetry analysis", "spglib"],
    category="pp",
    description="Reduce every structure to its primitive and conventional "
                "standard settings and record the space group, so that "
                "downstream comparisons are between equivalent cells.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["primitive", "conventional"],
              "obs": ["spacegroup", "spacegroup_number", "crystal_system",
                      "nsites_primitive"]},
    examples=["mv.pp.standardize(md)",
              "mv.pp.standardize(md, symprec=0.1)"],
    related=["mv.pp.describe", "mv.pp.dedup"],
)
def standardize(md: AnnData, source: str = "input", symprec: float = 0.01) -> None:
    """Primitive and conventional standard cells, plus symmetry.

    Named for what pymatgen calls it — ``get_primitive_standard_structure`` /
    ``get_conventional_standard_structure`` — rather than 'normalize', which in
    a dataset context reads as rescaling numbers.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    prim, conv, sg, num, sys_ = [], [], [], [], []
    for s in structures(md, source):
        try:
            sga = SpacegroupAnalyzer(s, symprec=symprec)
            prim.append(sga.get_primitive_standard_structure())
            conv.append(sga.get_conventional_standard_structure())
            sg.append(sga.get_space_group_symbol())
            num.append(int(sga.get_space_group_number()))
            sys_.append(sga.get_crystal_system())
        except Exception:
            # spglib fails on genuinely disordered or degenerate cells. Keeping
            # the input under the variant name is better than a ragged dataset.
            prim.append(s)
            conv.append(s)
            sg.append("")
            num.append(0)
            sys_.append("")
    deposit_structures(md, "primitive", prim)
    deposit_structures(md, "conventional", conv)
    md.obs["spacegroup"] = sg
    md.obs["spacegroup_number"] = num
    md.obs["crystal_system"] = sys_
    md.obs["nsites_primitive"] = [len(s) for s in prim]
    record(md, "pp.standardize", source=source, symprec=symprec)


@register_function(
    aliases=["describe structures", "formula", "density", "cell volume",
             "basic structure properties"],
    category="pp",
    description="Record per-material formula, site count, cell volume, density "
                "and element count into obs.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["formula", "nsites", "volume", "density", "n_elements",
                      "volume_per_atom"]},
    examples=["mv.pp.describe(md)"],
    related=["mv.pp.qc"],
)
def describe(md: AnnData, source: str = "input") -> None:
    """The columns every materials table has, computed once and named once."""
    S = structures(md, source)
    md.obs["formula"] = [s.composition.reduced_formula for s in S]
    md.obs["nsites"] = [len(s) for s in S]
    md.obs["volume"] = [float(s.volume) for s in S]
    md.obs["density"] = [float(s.density) for s in S]
    md.obs["n_elements"] = [len(s.composition.elements) for s in S]
    md.obs["volume_per_atom"] = [float(s.volume) / len(s) for s in S]
    record(md, "pp.describe", source=source)


@register_function(
    aliases=["qc", "quality control", "sanity check", "structure validity",
             "calculate qc metrics", "charge balance"],
    category="pp",
    description="Compute per-material quality-control metrics — minimum "
                "interatomic distance, ordered/disordered, charge neutrality, "
                "and an overall validity flag — so that unphysical structures "
                "are dropped before compute is spent on them.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["min_distance", "is_ordered", "is_charge_balanced",
                      "is_valid", "qc_reason"]},
    examples=["mv.pp.qc(md)",
              "mv.pp.qc(md, min_distance=0.7)"],
    related=["mv.pp.filter_materials", "mv.pp.describe"],
    notes="A generated or database structure that fails min_distance is almost "
          "always a broken cell rather than an exotic one; relaxing it wastes "
          "the calculator's time and pollutes the hull.",
)
def qc(md: AnnData, source: str = "input", min_distance: float = 0.5,
       require_charge_balance: bool = False) -> None:
    """Per-material validity metrics.

    ``min_distance`` is in angstrom and defaults deliberately low: 0.5 A catches
    structures that are broken, not structures that are merely unusual.
    """
    S = structures(md, source)
    dmin, ordered, balanced, valid, reason = [], [], [], [], []
    for s in S:
        try:
            d = float(np.min(s.distance_matrix[np.triu_indices(len(s), k=1)])) \
                if len(s) > 1 else float("inf")
        except Exception:
            d = float("nan")
        is_ord = bool(getattr(s, "is_ordered", True))
        bal = _charge_balanced(s)

        why = []
        if not (d != d) and d < min_distance:
            why.append(f"min_distance {d:.2f} < {min_distance}")
        if not is_ord:
            why.append("disordered (partial occupancy)")
        if require_charge_balance and bal is False:
            why.append("not charge balanced")

        dmin.append(d)
        ordered.append(is_ord)
        balanced.append(bal if bal is not None else True)
        valid.append(not why)
        reason.append("; ".join(why))

    md.obs["min_distance"] = dmin
    md.obs["is_ordered"] = ordered
    md.obs["is_charge_balanced"] = balanced
    md.obs["is_valid"] = valid
    md.obs["qc_reason"] = reason
    record(md, "pp.qc", source=source, min_distance=min_distance,
           require_charge_balance=require_charge_balance)


def _charge_balanced(structure) -> bool | None:
    """Whether an oxidation-state assignment sums to zero. ``None`` if pymatgen
    cannot assign states, which is common and not itself a defect."""
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer
        oxi = BVAnalyzer().get_oxi_state_decorated_structure(structure)
        total = sum(getattr(site.specie, "oxi_state", 0) or 0 for site in oxi)
        return bool(abs(total) < 1e-6)
    except Exception:
        return None


@register_function(
    aliases=["filter materials", "filter cells", "drop invalid structures",
             "subset materials"],
    category="pp",
    description="Return the subset of materials passing a validity flag or an "
                "explicit boolean mask.",
    requires={"obs": ["{flag}"]},
    examples=["md = mv.pp.filter_materials(md)",
              "md = mv.pp.filter_materials(md, flag='passes')"],
    related=["mv.pp.qc", "mv.pp.filter_elements", "mv.screen.filter"],
    notes="Returns rather than deposits, because AnnData cannot drop rows in "
          "place. Use mv.screen.filter when you want the record kept and the "
          "rows retained.",
)
def filter_materials(md: AnnData, flag: str = "is_valid",
                     mask=None) -> AnnData:
    """Drop rows. The one place matverse returns instead of depositing."""
    if mask is None:
        if flag not in md.obs:
            raise ValueError(
                f"obs[{flag!r}] absent; run mv.pp.qc first, or pass mask=")
        mask = md.obs[flag].to_numpy(dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    out = md[mask].copy()
    n_dropped = int((~mask).sum())
    record(out, "pp.filter_materials", flag=flag, n_dropped=n_dropped)
    return out


@register_function(
    aliases=["filter elements", "filter genes", "drop absent elements",
             "prune element axis"],
    category="pp",
    description="Drop element columns that appear in fewer than a minimum "
                "number of materials, shrinking the composition axis to the "
                "chemistry this library actually spans.",
    requires={"X": ["composition"]},
    examples=["md = mv.pp.filter_elements(md, min_materials=2)"],
    related=["mv.pp.filter_materials", "mv.tl.rank_elements_groups"],
)
def filter_elements(md: AnnData, min_materials: int = 1) -> AnnData:
    """Drop elements present in fewer than ``min_materials`` materials.

    After subsetting a library, most of the periodic table is a column of zeros.
    Dropping those is exactly ``filter_genes`` and for the same reason: they
    contribute nothing and they dilute every per-element statistic.
    """
    if md.n_vars == 0:
        raise ValueError("this object has no element axis (built with "
                         "build_X=False); nothing to filter")
    counts = _nonzero_per_column(md.X)
    keep = counts >= int(min_materials)
    out = md[:, keep].copy()
    out.var["n_materials"] = counts[keep]
    record(out, "pp.filter_elements", min_materials=min_materials,
           n_dropped=int((~keep).sum()))
    return out


def _nonzero_per_column(X) -> np.ndarray:
    if hasattr(X, "getnnz"):
        return np.asarray((X > 0).sum(axis=0)).ravel()
    return np.asarray((np.asarray(X) > 0).sum(axis=0)).ravel()


@register_function(
    aliases=["atomic fraction", "normalize composition", "normalise composition",
             "fractional composition"],
    category="pp",
    description="Store the composition as atomic fractions in a layer, leaving "
                "the raw atom counts in X.",
    requires={"X": ["composition"]},
    produces={"layers": ["fraction"]},
    examples=["mv.pp.normalize_composition(md)"],
    related=["mv.tl.pca"],
    notes="A layer rather than a replacement, because a hull needs the counts "
          "and a chemical-space map needs the fractions.",
)
def normalize_composition(md: AnnData) -> None:
    """Row-normalise the composition matrix into ``layers['fraction']``."""
    if md.n_vars == 0:
        raise ValueError("this object has no element axis (built with "
                         "build_X=False)")
    X = md.X
    if hasattr(X, "toarray"):
        from scipy.sparse import csr_matrix, diags
        totals = np.asarray(X.sum(axis=1)).ravel()
        inv = np.divide(1.0, totals, out=np.zeros_like(totals),
                        where=totals > 0)
        md.layers["fraction"] = csr_matrix(diags(inv) @ X)
    else:
        A = np.asarray(X, dtype=float)
        totals = A.sum(axis=1, keepdims=True)
        md.layers["fraction"] = np.divide(A, totals, out=np.zeros_like(A),
                                          where=totals > 0)
    record(md, "pp.normalize_composition")


@register_function(
    aliases=["deduplicate", "dedup", "remove duplicate structures",
             "structure matching", "unique structures"],
    category="pp",
    description="Flag duplicate structures by grouping on a composition and "
                "symmetry fingerprint before running pymatgen's StructureMatcher "
                "within each group, so the comparison stays linear in the "
                "number of candidates.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["duplicate_of", "is_duplicate"], "uns": ["dedup"]},
    examples=["mv.pp.dedup(md)",
              "mv.pp.dedup(md, source='primitive')"],
    related=["mv.pp.standardize"],
    notes="A naive all-pairs StructureMatcher is quadratic and unusable past a "
          "few thousand structures. Blocking on (reduced formula, space group) "
          "first makes the expensive comparison local.",
)
def dedup(md: AnnData, source: str = "input", symprec: float = 0.1,
          ltol: float = 0.2, stol: float = 0.3, angle_tol: float = 5.0) -> None:
    """Mark near-duplicate structures, keeping the first of each group."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    S = structures(md, source)
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True, scale=True)

    blocks: dict[tuple, list[int]] = {}
    for i, s in enumerate(S):
        blocks.setdefault(_fingerprint(s, symprec), []).append(i)

    representative = list(range(len(S)))
    for members in blocks.values():
        reps: list[int] = []
        for i in members:
            for r in reps:
                try:
                    same = matcher.fit(S[i], S[r])
                except Exception:
                    same = False
                if same:
                    representative[i] = r
                    break
            else:
                reps.append(i)

    names = list(md.obs_names)
    md.obs["duplicate_of"] = [names[r] for r in representative]
    md.obs["is_duplicate"] = [r != i for i, r in enumerate(representative)]
    md.uns["dedup"] = {
        "source": source, "n_blocks": len(blocks),
        "n_duplicates": int(sum(r != i for i, r in enumerate(representative))),
        "n_unique": len(set(representative)),
        "matcher": {"ltol": ltol, "stol": stol, "angle_tol": angle_tol},
    }
    record(md, "pp.dedup", source=source, symprec=symprec)


def _fingerprint(structure, symprec: float) -> tuple:
    """A cheap blocking key: same formula and space group, or no comparison."""
    formula = structure.composition.reduced_formula
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sg = int(SpacegroupAnalyzer(structure, symprec=symprec)
                 .get_space_group_number())
    except Exception:
        sg = 0
    return (formula, sg)


@register_function(
    aliases=["supercell", "make supercell", "expand cell"],
    category="pp",
    description="Build a supercell of every structure and deposit it under its "
                "own variant name.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{name}"]},
    examples=["mv.pp.supercell(md, [2, 2, 2])",
              "mv.pp.supercell(md, 2, name='2x')"],
    related=["mv.pp.standardize", "mv.pp.rattle"],
)
def supercell(md: AnnData, scaling, source: str = "input",
              name: str | None = None) -> None:
    """Deposit a supercell variant."""
    out = []
    for s in structures(md, source):
        c = s.copy()
        c.make_supercell(scaling)
        out.append(c)
    key = name or _supercell_name(scaling)
    deposit_structures(md, key, out)
    record(md, "pp.supercell", scaling=scaling, source=source, name=key)


def _supercell_name(scaling) -> str:
    if hasattr(scaling, "__iter__"):
        return "supercell_" + "x".join(str(int(v)) for v in scaling)
    return f"supercell_{int(scaling)}"


@register_function(
    aliases=["rattle", "perturb structures", "random displacement",
             "training set generation"],
    category="pp",
    description="Randomly displace every atom by a fixed standard deviation "
                "and deposit the perturbed structures, for building an "
                "interatomic-potential training or validation set.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{name}"]},
    examples=["mv.pp.rattle(md, stdev=0.05, seed=0)"],
    related=["mv.pp.supercell", "mv.calc.energy"],
)
def rattle(md: AnnData, stdev: float = 0.03, source: str = "input",
           name: str = "rattled", seed: int | None = None) -> None:
    """Deposit a randomly perturbed variant. ``stdev`` is in angstrom."""
    rng = np.random.default_rng(seed)
    out = []
    for s in structures(md, source):
        c = s.copy()
        for i in range(len(c)):
            c.translate_sites([i], rng.normal(0.0, stdev, 3),
                              frac_coords=False, to_unit_cell=False)
        out.append(c)
    deposit_structures(md, name, out)
    record(md, "pp.rattle", stdev=stdev, source=source, name=name, seed=seed)


@register_function(
    aliases=["strain", "apply strain", "deform cell", "elastic deformation"],
    category="pp",
    description="Apply an isotropic or tensor strain to every cell and deposit "
                "the deformed structures.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{name}"]},
    examples=["mv.pp.strain(md, 0.01)",
              "mv.pp.strain(md, [[0.01, 0, 0], [0, 0, 0], [0, 0, 0]])"],
    related=["mv.pp.supercell"],
)
def strain(md: AnnData, amount, source: str = "input",
           name: str = "strained") -> None:
    """Deposit a strained variant. Scalar ``amount`` means isotropic strain."""
    eps = np.asarray(amount, dtype=float)
    if eps.ndim == 0:
        eps = float(eps) * np.eye(3)
    if eps.shape != (3, 3):
        raise ValueError("strain must be a scalar or a 3x3 tensor")
    F = np.eye(3) + eps
    out = []
    for s in structures(md, source):
        c = s.copy()
        c.lattice = c.lattice.__class__(np.asarray(c.lattice.matrix) @ F.T)
        out.append(c)
    deposit_structures(md, name, out)
    record(md, "pp.strain", source=source, name=name)


__all__ = ["standardize", "describe", "qc", "filter_materials",
           "filter_elements", "normalize_composition", "dedup", "supercell",
           "rattle", "strain"]

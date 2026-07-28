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

import warnings

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
    aliases=["harmonize", "harmonise", "batch correction", "cross database",
             "reconcile energies", "database offset", "elemental corrections"],
    category="pp",
    description="Reconcile energies computed by different databases by fitting "
                "a per-element offset for each one against a reference, using "
                "the compositions they have in common.",
    requires={"obs": ["{batch_key}", "{energy_key}"], "X": ["composition"]},
    produces={"obs": ["{energy_key}_harmonized"], "uns": ["harmonize"]},
    examples=["mv.pp.harmonize(md, batch_key='database', "
              "energy_key='energy_per_atom_dft')",
              "mv.pp.harmonize(md, batch_key='database', reference='mp')"],
    related=["mv.thermo.hull", "mv.tl.rank_elements_groups"],
    notes="Formation energies from Materials Project, OQMD and Alexandria carry "
          "systematic offsets from differing pseudopotentials, cutoffs and "
          "correction schemes. That is a batch effect with a compositional "
          "structure, and this fits it the way the field already does by hand — "
          "as per-element reference corrections — rather than by aligning "
          "distributions. It cannot repair a difference that is not linear in "
          "composition, and it says so in uns['harmonize'].",
)
def harmonize(md: AnnData, batch_key: str, energy_key: str,
              reference: str | None = None, match_on: str = "formula",
              key_added: str | None = None,
              min_anchors_per_element: int = 1) -> None:
    """Fit and apply per-element energy offsets between databases.

    The model is the one the field already uses::

        E_db(x) = E_ref(x) + sum_element  fraction_element * delta[db, element]

    fitted by least squares on **anchors** — compositions present in both the
    reference database and another one. Everything from that database is then
    shifted by its fitted offset.

    What this does not do: repair a difference that is not linear in
    composition. Two databases that disagree because one relaxed with a
    different functional will disagree structure by structure, and a
    compositional offset absorbs only the average of that. The fit residual is
    reported so the size of what is left is visible.
    """
    for column in (batch_key, energy_key):
        if column not in md.obs:
            raise ValueError(f"obs[{column!r}] absent; available: "
                             f"{list(md.obs.columns)}")
    if md.n_vars == 0:
        raise ValueError("this object has no element axis (build_X=False); "
                         "harmonize fits offsets per element")

    batches = pd.Series(md.obs[batch_key].to_numpy()).astype(str).to_numpy()
    energies = md.obs[energy_key].to_numpy(dtype=float)
    present = list(pd.unique(batches))
    if len(present) < 2:
        raise ValueError(f"obs[{batch_key!r}] has only {present}; there is "
                         f"nothing to harmonise against")
    ref = str(reference) if reference is not None else _largest(batches)
    if ref not in present:
        raise ValueError(f"reference {ref!r} not in obs[{batch_key!r}] "
                         f"({present})")

    keys = _match_keys(md, match_on)
    fractions = _fractions(md)

    rows: dict[str, list] = {b: [] for b in present if b != ref}
    targets: dict[str, list] = {b: [] for b in present if b != ref}
    n_anchor_groups = 0

    for key in pd.unique(keys):
        group = keys == key
        in_ref = group & (batches == ref) & ~np.isnan(energies)
        if not in_ref.any():
            continue
        ref_energy = float(np.mean(energies[in_ref]))
        anchored = False
        for batch in rows:
            member = group & (batches == batch) & ~np.isnan(energies)
            if not member.any():
                continue
            rows[batch].append(fractions[member].mean(axis=0))
            targets[batch].append(float(np.mean(energies[member])) - ref_energy)
            anchored = True
        n_anchor_groups += int(anchored)

    offsets, diagnostics = {}, {}
    for batch in rows:
        A = np.asarray(rows[batch], dtype=float)
        y = np.asarray(targets[batch], dtype=float)
        if not len(y):
            offsets[batch] = np.zeros(md.n_vars)
            diagnostics[batch] = {"n_anchors": 0, "rmse_before": None,
                                  "rmse_after": None,
                                  "note": "no shared composition with the "
                                          "reference; left uncorrected"}
            continue
        covered = (A > 0).sum(axis=0)
        delta, *_ = np.linalg.lstsq(A, y, rcond=None)
        # An element seen in too few anchors gets whatever least squares felt
        # like; zero is the honest value for "not determined by the data".
        delta = np.where(covered >= min_anchors_per_element, delta, 0.0)
        offsets[batch] = delta
        residual = y - A @ delta
        diagnostics[batch] = {
            "n_anchors": int(len(y)),
            "n_elements_fitted": int((covered >= min_anchors_per_element).sum()),
            "rmse_before": float(np.sqrt(np.mean(y ** 2))),
            "rmse_after": float(np.sqrt(np.mean(residual ** 2))),
            "underdetermined": bool(len(y) < (covered > 0).sum()),
        }

    corrected = energies.copy()
    for batch, delta in offsets.items():
        member = batches == batch
        corrected[member] = energies[member] - fractions[member] @ delta

    name = key_added or f"{energy_key}_harmonized"
    md.obs[name] = corrected
    md.uns["harmonize"] = {
        "batch_key": batch_key,
        "energy_key": energy_key,
        "reference": ref,
        "match_on": match_on,
        "n_anchor_groups": int(n_anchor_groups),
        "elements": list(map(str, md.var_names)),
        "offsets": {b: np.asarray(d, dtype=float) for b, d in offsets.items()},
        "diagnostics": diagnostics,
    }
    if not n_anchor_groups:
        warnings.warn(
            f"no composition is shared between {ref!r} and the other databases, "
            f"so no offset could be fitted and obs[{name!r}] is a copy of "
            f"obs[{energy_key!r}]. Harmonisation needs overlap.", stacklevel=2)
    record(md, "pp.harmonize", batch_key=batch_key, energy_key=energy_key,
           reference=ref, n_anchor_groups=n_anchor_groups)


def _largest(batches: np.ndarray) -> str:
    """Default reference: the database contributing the most rows."""
    values, counts = np.unique(batches, return_counts=True)
    return str(values[int(np.argmax(counts))])


def _match_keys(md: AnnData, match_on: str) -> np.ndarray:
    """What counts as 'the same material' across databases."""
    if match_on == "formula":
        S = structures(md, "input")
        return np.asarray([s.composition.reduced_formula for s in S])
    if match_on in md.obs:
        return pd.Series(md.obs[match_on].to_numpy()).astype(str).to_numpy()
    raise ValueError(f"match_on={match_on!r} is neither 'formula' nor a column "
                     f"of obs ({list(md.obs.columns)})")


def _fractions(md: AnnData) -> np.ndarray:
    raw = md.X.toarray() if hasattr(md.X, "toarray") else np.asarray(md.X)
    raw = np.asarray(raw, dtype=float)
    totals = raw.sum(axis=1, keepdims=True)
    return np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)


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
    aliases=["defects", "vacancies", "substitutions", "point defects",
             "enumerate defects", "doping", "make defects"],
    category="pp",
    description="Enumerate point defects — vacancies and substitutions — in a "
                "supercell of every structure, returning them as a new dataset "
                "with the parent and defect type recorded.",
    requires={"structures": ["{source}"]},
    examples=["defective = mv.pp.defects(md, kinds=('vacancy',))",
              "defective = mv.pp.defects(md, substitutions={'Al': ['Mg']})"],
    related=["mv.calc.relax", "mv.pp.supercell"],
    notes="Returns a new dataset rather than depositing, because there are more "
          "defects than parents. Symmetry-inequivalent sites are enumerated "
          "once each; without that a 32-atom supercell yields 32 identical "
          "vacancies and wastes a calculator on 31 of them.\n\n"
          "Defect *formation energies* need charge states, a chemical "
          "potential and a finite-size correction, which is what doped and "
          "pydefect exist for. This builds the structures.",
)
def defects(md: AnnData, source: str = "input", supercell=(2, 2, 2),
            kinds=("vacancy",), substitutions: dict | None = None,
            symprec: float = 0.1) -> AnnData:
    """Enumerate point defects. Returns a new dataset of defective cells."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from .data import from_structures  # noqa: F401  (imported for clarity)

    unknown = set(kinds) - {"vacancy", "substitution"}
    if unknown:
        raise ValueError(f"unknown defect kind(s) {sorted(unknown)}; use "
                         f"'vacancy' and 'substitution'")
    if "substitution" in kinds and not substitutions:
        raise ValueError("kinds includes 'substitution' but no substitutions "
                         "were given, e.g. substitutions={'Al': ['Mg']}")

    out, rows = [], []
    for name, structure in zip(md.obs_names, structures(md, source)):
        cell = structure.copy()
        cell.make_supercell(list(supercell))
        for site_index in _inequivalent_sites(cell, symprec):
            symbol = str(cell[site_index].specie.symbol)
            if "vacancy" in kinds:
                defective = cell.copy()
                defective.remove_sites([site_index])
                out.append(defective)
                rows.append({"parent": str(name), "defect": "vacancy",
                             "site": int(site_index), "removed": symbol,
                             "added": ""})
            for replacement in (substitutions or {}).get(symbol, []):
                defective = cell.copy()
                defective.replace(site_index, replacement)
                out.append(defective)
                rows.append({"parent": str(name), "defect": "substitution",
                             "site": int(site_index), "removed": symbol,
                             "added": str(replacement)})

    if not out:
        raise ValueError("no defect was generated; check kinds= and "
                         "substitutions=")

    import pandas as _pd
    defective = from_structures(out, _pd.DataFrame(rows))
    record(defective, "pp.defects", source=source, supercell=list(supercell),
           kinds=list(kinds), n_parents=int(md.n_obs))
    return defective


def _inequivalent_sites(structure, symprec: float) -> list:
    """One representative per symmetry-equivalent set of sites.

    Falls back to every site when symmetry cannot be determined, which is safe
    but expensive — a 32-atom elemental supercell then yields 32 identical
    vacancies instead of one.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        dataset = SpacegroupAnalyzer(structure,
                                     symprec=symprec).get_symmetry_dataset()
        equivalent = getattr(dataset, "equivalent_atoms", None)
        if equivalent is None:
            equivalent = dataset["equivalent_atoms"]
        seen, out = set(), []
        for i, group in enumerate(equivalent):
            if group not in seen:
                seen.add(group)
                out.append(i)
        return out
    except Exception:
        return list(range(len(structure)))


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


__all__ = ["standardize", "describe", "qc", "filter_materials", "harmonize",
           "defects",
           "filter_elements", "normalize_composition", "dedup", "supercell",
           "rattle", "strain"]

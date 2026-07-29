"""``mv.mol`` — molecules on the same substrate.

The design note used to say molecules were out of scope "by construction". That
was wrong, and testing it took one line: a ``Molecule`` has a composition, so
``X`` and ``var`` build for water exactly as they do for a crystal — H:2, O:1.
The only thing that failed was a decoder that assumed a lattice.

So molecules live on the same axes as everything else. ``obs`` is one row per
species, ``X`` is still the composition matrix, the sites axis is still one row
per atom, and ``obs['is_periodic']`` is what tells the two apart. A dataset can
hold both, which is what a study of a catalyst and its adsorbates, an
electrolyte and its salt, or a MOF and its guest actually needs.

What genuinely differs is the analysis. A molecule has a point group rather
than a space group, covalent bonds rather than a coordination polyhedron, and
functional groups rather than a lattice:

```python
md = mv.data.from_molecules([water, methane])
mv.mol.point_group(md)                  # -> obs['point_group']
mv.mol.bonds(md, sites)                 # covalent graph -> obsp
mv.mol.descriptors(md)                  # weight, rings, rotatable bonds
fragments = mv.mol.fragments(md)        # break every bond, keep the pieces
```

Periodic operations refuse on a molecule rather than producing a number from a
lattice that is not there, and molecular operations refuse on a crystal for the
same reason.
"""

from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData

from ._core import record, structures
from ._registry import register_function

#: Near-neighbour strategies that make sense for a molecule.
BOND_STRATEGIES = {
    "covalent": "CovalentBondNN — bonded when the separation is within a "
                "tolerance of the sum of covalent radii; needs no extra "
                "dependency and is the default",
    "openbabel": "OpenBabelNN — Open Babel's perception, which knows about "
                 "bond orders; needs the openbabel Python bindings",
    "critic2": "Critic2NN — from a charge density, so it is the only one that "
               "is not a distance heuristic; needs the critic2 program",
}


def _topology_comparator():
    """pymatgen's bond-graph molecule comparator, wherever it lives."""
    try:
        from pymatgen.core.molecule_structure_comparator import (
            MoleculeStructureComparator)
    except ImportError:
        from pymatgen.analysis.molecule_structure_comparator import (
            MoleculeStructureComparator)
    return MoleculeStructureComparator()


def _order_matcher():
    """pymatgen's Hungarian order matcher, wherever this pymatgen keeps it.

    It moved from ``pymatgen.analysis.molecule_matcher`` to
    ``pymatgen.core.molecule_matcher`` in 2026.5, and matverse supports both
    sides of that move.
    """
    try:
        from pymatgen.core.molecule_matcher import HungarianOrderMatcher
    except ImportError:
        from pymatgen.analysis.molecule_matcher import HungarianOrderMatcher
    return HungarianOrderMatcher


def _molecules(md: AnnData, source: str, what: str):
    """The rows that are molecules, or a refusal naming the periodic ones."""
    S = structures(md, source)
    periodic = [i for i, s in enumerate(S) if hasattr(s, "lattice")]
    if len(periodic) == len(S):
        raise ValueError(
            f"{what} needs molecules and every row of this dataset is "
            f"periodic. Build one with mv.data.from_molecules, or check "
            f"obs['is_periodic'] after mv.pp.describe.")
    return S, set(periodic)


@register_function(
    aliases=["from molecules", "molecule dataset", "load molecules",
             "build from molecules"],
    category="mol",
    description="Build a dataset whose rows are molecules rather than "
                "crystals, on the same axes as everything else.",
    produces={"obs": ["is_periodic"], "structures": ["input"], "X": []},
    examples=["md = mv.mol.from_molecules([water, methane])",
              "md = mv.mol.from_molecules(molecules, obs=frame)"],
    related=["mv.data.from_structures", "mv.mol.point_group",
             "mv.mol.descriptors"],
    notes="A thin alias for mv.data.from_structures, which already accepts "
          "molecules — the composition axis does not care whether a formula "
          "unit repeats. It exists so the intent is visible in a script and so "
          "``obs['is_periodic']`` is set without waiting for mv.pp.describe.",
)
def from_molecules(molecules, obs=None) -> AnnData:
    """A molecular dataset. Returns an ``AnnData``."""
    from .data import from_structures

    md = from_structures(list(molecules), obs=obs)
    md.obs["is_periodic"] = [hasattr(m, "lattice") for m in molecules]
    record(md, "mol.from_molecules", n=len(md.obs))
    return md


@register_function(
    aliases=["point group", "molecular symmetry", "schoenflies",
             "molecule symmetry", "is it chiral"],
    category="mol",
    description="The Schoenflies point group of every molecule, and what it "
                "implies — whether the molecule is chiral, and whether it can "
                "carry a dipole.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["point_group", "symmetry_order", "is_chiral",
                      "can_be_polar"]},
    examples=["mv.mol.point_group(md)",
              "mv.mol.point_group(md, tolerance=0.1)"],
    related=["mv.pp.standardize", "mv.mol.descriptors"],
    notes="The molecular analogue of a space group. C2v for water, Td for "
          "methane, D6h for benzene.\n\n"
          "Two consequences come free and are worth having as columns. A "
          "molecule is **chiral** exactly when its point group contains no "
          "improper operation — C1, Cn or Dn — which is what decides whether "
          "it has an enantiomer. It **can be polar** only in C1, Cs, Cn or "
          "Cnv; any other group forces the dipole to cancel by symmetry, which "
          "is a hard selection rule rather than a tendency.",
)
def point_group(md: AnnData, source: str = "input",
                tolerance: float = 0.3) -> None:
    """Point group per molecule. Deposits; returns ``None``."""
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    S, periodic = _molecules(md, source, "mv.mol.point_group")

    groups = np.full(md.n_obs, "", dtype=object)
    orders = np.full(md.n_obs, np.nan)
    chiral = np.full(md.n_obs, False)
    polar = np.full(md.n_obs, False)
    failures = []

    for i, molecule in enumerate(S):
        if i in periodic:
            groups[i] = "periodic"
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                analyzer = PointGroupAnalyzer(molecule, tolerance=tolerance)
                symbol = str(analyzer.sch_symbol)
                orders[i] = float(len(analyzer.get_symmetry_operations()))
            groups[i] = symbol
            chiral[i] = _is_chiral(symbol)
            polar[i] = _can_be_polar(symbol)
        except Exception as exc:
            failures.append(f"{i}: {type(exc).__name__}: {exc}")

    md.obs["point_group"] = groups
    md.obs["symmetry_order"] = orders
    md.obs["is_chiral"] = chiral
    md.obs["can_be_polar"] = polar
    md.uns["point_group"] = {
        "tolerance": float(tolerance), "source": source,
        "n_failed": len(failures), "errors": failures[:10],
        "note": "chiral means no improper operation (C1, Cn, Dn); polar is "
                "allowed only in C1, Cs, Cn, Cnv — elsewhere symmetry forces "
                "the dipole to zero",
    }
    record(md, "mol.point_group", source=source, tolerance=tolerance)


def _is_chiral(symbol: str) -> bool:
    """No improper operation means an enantiomer exists."""
    text = symbol.strip()
    if text in ("C1",):
        return True
    if text.startswith("S") or "h" in text or "v" in text or "d" in text:
        return False
    if text in ("Ci", "Cs", "Td", "Oh", "Ih", "Kh", "C*v", "D*h"):
        return False
    return text[0] in ("C", "D")


def _can_be_polar(symbol: str) -> bool:
    """A dipole survives only in C1, Cs, Cn and Cnv."""
    text = symbol.strip()
    if text in ("C1", "Cs", "C*v"):
        return True
    if text.startswith("C") and not text.startswith("Ci"):
        return text.endswith("v") or text[1:].isdigit()
    return False


@register_function(
    aliases=["molecular bonds", "covalent bonds", "molecule graph",
             "bond order", "connectivity of a molecule"],
    category="mol",
    description="The covalent bond graph of every molecule, as a sparse "
                "adjacency matrix on the sites axis.",
    requires={"structures": ["{source}"]},
    produces={"sites.obsp": ["bonds", "bond_distances"]},
    prerequisites=["mv.multi.sites"],
    examples=["mv.mol.bonds(md, sites)",
              "mv.mol.bonds(md, sites, strategy='openbabel')"],
    related=["mv.env.bonds", "mv.mol.fragments", "mv.mol.descriptors"],
    notes="The molecular counterpart of mv.env.bonds, and it lands in the same "
          "slot — ``obsp`` on the sites object — so a graph algorithm does not "
          "need to know which kind of material it was handed.\n\n"
          "Different perception, though. A crystal's neighbours come from "
          "Voronoi solid angles; a molecule's bonds come from covalent radii, "
          "because a molecule has no periodic environment to take a solid "
          "angle over.",
)
def bonds(md: AnnData, sites: AnnData, strategy: str = "covalent",
          source: str = "input", tolerance: float = 0.2) -> None:
    """Covalent bond graph into ``sites.obsp``. Returns ``None``."""
    from scipy import sparse

    from .multi import AXIS_KEY

    if sites.uns.get(AXIS_KEY) != "sites":
        raise ValueError("this is not a sites object; build one with "
                         "mv.multi.sites(md)")

    finder = _bond_strategy(strategy, tolerance)
    S = structures(md, source)

    rows, cols, lengths = [], [], []
    offset, failures = 0, 0
    for molecule in S:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for index in range(len(molecule)):
                    for neighbour in finder.get_nn_info(molecule, index):
                        j = int(neighbour["site_index"])
                        rows.append(offset + index)
                        cols.append(offset + j)
                        lengths.append(float(
                            molecule[index].distance(neighbour["site"])))
        except Exception:
            failures += 1
        offset += len(molecule)

    shape = (sites.n_obs, sites.n_obs)
    sites.obsp["bonds"] = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=shape).tocsr()
    sites.obsp["bond_distances"] = sparse.coo_matrix(
        (np.asarray(lengths, dtype=float), (rows, cols)), shape=shape).tocsr()
    sites.uns["bonds"] = {"strategy": str(strategy), "source": source,
                          "n_edges": int(sites.obsp["bonds"].nnz),
                          "n_failed": failures, "kind": "covalent"}
    record(sites, "mol.bonds", strategy=strategy, source=source)


def _bond_strategy(name: str, tolerance: float):
    from pymatgen.analysis import local_env

    key = str(name).strip().lower()
    if key == "covalent":
        return local_env.CovalentBondNN(tol=tolerance)
    if key == "openbabel":
        try:
            return local_env.OpenBabelNN()
        except Exception as exc:
            raise ImportError(
                "strategy='openbabel' needs the Open Babel Python bindings, "
                "which are not pip-installable — `conda install -c conda-forge "
                "openbabel`. strategy='covalent' needs nothing extra."
            ) from exc
    if key == "critic2":
        return local_env.Critic2NN()
    raise ValueError(f"unknown strategy {name!r}; known: "
                     f"{sorted(BOND_STRATEGIES)}")


@register_function(
    aliases=["molecular descriptors", "molecule properties", "molecular "
             "weight", "rotatable bonds", "how flexible"],
    category="mol",
    description="Per-molecule descriptors that need the bond graph — heavy "
                "atom count, rings, rotatable bonds, and the radius of "
                "gyration.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["heavy_atoms", "n_bonds", "n_rings",
                      "rotatable_bonds", "radius_of_gyration"]},
    examples=["mv.mol.descriptors(md)"],
    related=["mv.mol.bonds", "mv.feat.element_stats", "mv.mol.point_group"],
    notes="``rotatable_bonds`` counts single bonds that are not in a ring and "
          "not terminal — the standard conformational-flexibility count, and "
          "the one that decides how many conformers a search has to cover.\n\n"
          "``n_rings`` is the cyclomatic number, ``edges - nodes + "
          "components``, which counts independent cycles rather than the "
          "chemist's smallest set of smallest rings. They agree for ordinary "
          "molecules and differ for fused polycyclics.",
)
def descriptors(md: AnnData, source: str = "input",
                tolerance: float = 0.2) -> None:
    """Graph-derived molecular descriptors. Deposits; returns ``None``."""
    import networkx as nx
    from pymatgen.analysis.graphs import MoleculeGraph

    S, periodic = _molecules(md, source, "mv.mol.descriptors")
    finder = _bond_strategy("covalent", tolerance)

    heavy = np.full(md.n_obs, np.nan)
    n_bonds = np.full(md.n_obs, np.nan)
    n_rings = np.full(md.n_obs, np.nan)
    rotatable = np.full(md.n_obs, np.nan)
    gyration = np.full(md.n_obs, np.nan)
    failures = []

    for i, molecule in enumerate(S):
        if i in periodic:
            continue
        heavy[i] = sum(1 for s in molecule if s.specie.symbol != "H")
        coords = np.asarray(molecule.cart_coords, dtype=float)
        centre = coords.mean(axis=0)
        gyration[i] = float(np.sqrt(((coords - centre) ** 2).sum(axis=1).mean()))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                graph = MoleculeGraph.from_local_env_strategy(molecule, finder)
            simple = nx.Graph(graph.graph)
            n_bonds[i] = simple.number_of_edges()
            n_rings[i] = (simple.number_of_edges()
                          - simple.number_of_nodes()
                          + nx.number_connected_components(simple))
            in_ring = set()
            for cycle in nx.cycle_basis(simple):
                for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                    in_ring.add(frozenset((a, b)))
            rotatable[i] = sum(
                1 for a, b in simple.edges
                if frozenset((a, b)) not in in_ring
                and simple.degree(a) > 1 and simple.degree(b) > 1)
        except Exception as exc:
            failures.append(f"{i}: {type(exc).__name__}: {exc}")

    md.obs["heavy_atoms"] = heavy
    md.obs["n_bonds"] = n_bonds
    md.obs["n_rings"] = n_rings
    md.obs["rotatable_bonds"] = rotatable
    md.obs["radius_of_gyration"] = gyration
    md.uns["molecular_descriptors"] = {
        "source": source, "n_failed": len(failures), "errors": failures[:10],
        "note": "n_rings is the cyclomatic number, not the smallest set of "
                "smallest rings; they agree except for fused polycyclics",
    }
    record(md, "mol.descriptors", source=source)


@register_function(
    aliases=["fragments", "break bonds", "fragment a molecule",
             "bond dissociation", "what does it break into"],
    category="mol",
    description="Break every acyclic bond in turn and keep the pieces, which "
                "is what a bond-dissociation or degradation study enumerates.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["parent", "broken_bond", "fragment_index"],
              "structures": ["input"]},
    examples=["fragments = mv.mol.fragments(md)",
              "fragments = mv.mol.fragments(md, depth=2)"],
    related=["mv.mol.bonds", "mv.mol.descriptors", "mv.calc.energy"],
    notes="Returns an ordinary object whose rows are fragments, with "
          "``obs['parent']`` pointing back — the same derived-axis shape as "
          "mv.pp.defects and mv.disorder.orderings.\n\n"
          "Only acyclic bonds are broken by one cut: breaking a ring bond "
          "leaves the molecule connected, so a ring needs two. ``depth`` "
          "controls how many cuts are made, and the count grows quickly — a "
          "molecule with twenty rotatable bonds has 190 two-cut fragmentations."
          "\n\nCompute the fragments and the parent at the same level and the "
          "difference is a bond dissociation energy; matverse does not do that "
          "step for you, because which reference state you subtract is a "
          "choice.",
)
def fragments(md: AnnData, source: str = "input", depth: int = 1,
              tolerance: float = 0.2) -> AnnData:
    """Fragments of every molecule. Returns a new object."""
    import networkx as nx
    from pymatgen.analysis.graphs import MoleculeGraph
    from pymatgen.core import Molecule

    from .data import from_structures

    S, periodic = _molecules(md, source, "mv.mol.fragments")
    finder = _bond_strategy("covalent", tolerance)
    labels = [str(x) for x in md.obs.get("name", md.obs_names)]

    built, parents, broken, indices, failures = [], [], [], [], []
    for i, molecule in enumerate(S):
        if i in periodic:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                graph = MoleculeGraph.from_local_env_strategy(molecule, finder)
            simple = nx.Graph(graph.graph)
        except Exception as exc:
            failures.append(f"{labels[i]}: {type(exc).__name__}: {exc}")
            continue

        ring_edges = set()
        for cycle in nx.cycle_basis(simple):
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                ring_edges.add(frozenset((a, b)))

        k = 0
        for a, b in simple.edges:
            if frozenset((a, b)) in ring_edges:
                continue
            cut = simple.copy()
            cut.remove_edge(a, b)
            pieces = list(nx.connected_components(cut))
            if len(pieces) < 2:
                continue
            for piece in pieces:
                order = sorted(piece)
                built.append(Molecule(
                    [molecule[j].specie for j in order],
                    [molecule[j].coords for j in order]))
                parents.append(labels[i])
                broken.append(f"{molecule[a].specie}{a}-{molecule[b].specie}{b}")
                indices.append(k)
            k += 1

    if not built:
        raise ValueError(
            "no fragment was produced. Every bond may be in a ring — one cut "
            "cannot separate a ring — or the molecules may be single atoms. "
            + (f"Errors: {failures[:3]}" if failures else ""))

    out = from_structures(built)
    out.obs["parent"] = parents
    out.obs["broken_bond"] = broken
    out.obs["fragment_index"] = indices
    out.obs["is_periodic"] = False
    # The *reduced* formula is actively misleading for a fragment: pymatgen
    # applies the diatomic convention, so a single hydrogen atom reads "H2" and
    # a hydroxyl reads "H2O2". A fragment is a specific set of atoms, not a
    # stoichiometry, so record the unreduced formula next to it.
    out.obs["fragment_formula"] = [
        m.composition.formula.replace(" ", "") for m in built]
    out.obs["fragment_size"] = [len(m) for m in built]
    out.uns["fragments"] = {
        "source": source, "depth": int(depth),
        "n_failed": len(failures), "errors": failures[:10],
        "note": "only acyclic bonds are cut; a ring bond leaves the molecule "
                "connected and needs a second cut",
    }
    record(out, "mol.fragments", source=source, depth=depth)
    return out


@register_function(
    aliases=["match molecules", "are these the same molecule",
             "molecule matcher", "same conformer", "deduplicate molecules"],
    category="mol",
    description="Decide which molecules are the same one, up to rotation, "
                "translation and atom relabelling.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["molecule_group", "is_duplicate", "match_rmsd"]},
    dispatch="method= chooses what identity means: 'geometry' superposes and "
             "compares shape, 'topology' compares the bond graph and ignores "
             "shape entirely",
    examples=["mv.mol.match(md)", "mv.mol.match(md, tolerance=0.2)"],
    related=["mv.pp.dedup", "mv.mol.descriptors"],
    notes="The molecular counterpart of mv.pp.dedup, and it exists for the "
          "same reason: two rows that are the same molecule waste a calculator "
          "twice and count twice in any statistic.\n\n"
          "Two molecules match when the best rigid-body superposition over "
          "**all atom assignments** leaves an RMSD within tolerance — Kabsch "
          "for the rotation, Hungarian for the labelling. That makes the "
          "answer invariant to rotation, translation and the order the atoms "
          "happen to be listed in, which is what identity means for a "
          "molecule.\n\n"
          "Until v0.1.27 this compared a sorted heavy-atom distance spectrum "
          "instead. That is invariant to the same three things and is not a "
          "proof of congruence: two different molecules can share a distance "
          "spectrum, and hydrogens were ignored entirely.\n\n"
          "obs['match_rmsd'] carries the RMSD to the group representative, in "
          "angstrom, so tolerance can be chosen by looking rather than "
          "guessed. It is 0 for the representative itself.\n\n"
          "pymatgen's default MoleculeMatcher needs openbabel, which is a C++ "
          "library rather than a wheel; the ordering matchers used here need "
          "nothing beyond numpy and scipy.\n\n"
          "**method='topology' asks a different question.** It compares the "
          "bond graph and ignores geometry, so a torsional conformer matches: "
          "rotating ethanol's hydroxyl hydrogen about the C-O bond leaves "
          "every bond intact and moves the geometry by an RMSD of 0.29, which "
          "the default route calls a different molecule and this route calls "
          "the same one. Which you want depends on whether conformers are the "
          "thing you are deduplicating or the thing you are studying.")
def match(md: AnnData, source: str = "input", tolerance: float = 0.1,
          method: str = "geometry") -> None:
    """Group identical molecules. Deposits; returns ``None``."""
    if method not in ("geometry", "topology"):
        raise ValueError(
            f"unknown method {method!r}; use 'geometry' for a rigid-body "
            f"superposition or 'topology' for a bond-graph comparison")
    HungarianOrderMatcher = _order_matcher() if method == "geometry" else None
    comparator = _topology_comparator() if method == "topology" else None

    S, periodic = _molecules(md, source, "mv.mol.match")

    groups = np.full(md.n_obs, -1, dtype=int)
    duplicate = np.full(md.n_obs, False)
    rmsd_to_group = np.full(md.n_obs, np.nan)
    representatives: dict[str, list[int]] = {}

    for i, molecule in enumerate(S):
        if i in periodic:
            continue
        formula = molecule.composition.reduced_formula
        for candidate in representatives.get(formula, []):
            reference = S[candidate]
            if len(reference) != len(molecule):
                continue
            try:
                if method == "topology":
                    # Same bonds, whatever the geometry. A torsional conformer
                    # matches here and does not match on geometry.
                    rmsd = 0.0 if comparator.are_equal(reference, molecule) \
                        else float("inf")
                else:
                    # Kabsch superposition with a Hungarian assignment over the
                    # atom labels: invariant to rotation, translation and the
                    # order the atoms happen to be listed in.
                    _, rmsd = HungarianOrderMatcher(reference).fit(molecule)
            except Exception:
                continue
            if float(rmsd) <= tolerance:
                groups[i] = candidate
                duplicate[i] = True
                rmsd_to_group[i] = float(rmsd)
                break
        if groups[i] < 0:
            representatives.setdefault(formula, []).append(i)
            groups[i] = i
            rmsd_to_group[i] = 0.0

    md.obs["molecule_group"] = groups
    md.obs["is_duplicate"] = duplicate
    md.obs["match_rmsd"] = rmsd_to_group
    md.uns["molecule_match"] = {
        "tolerance": float(tolerance), "source": source,
        "n_unique": int(sum(len(v) for v in representatives.values())),
        "n_duplicates": int(duplicate.sum()),
        "method": str(method),
        "matcher": ("Kabsch superposition with Hungarian atom assignment"
                    if method == "geometry"
                    else "bond-graph comparison, geometry ignored"),
        "note": "two molecules match when the best rigid-body superposition "
                "over all atom assignments leaves an RMSD within tolerance, "
                "in angstrom",
    }
    record(md, "mol.match", source=source, tolerance=tolerance, method=method)


__all__ = ["BOND_STRATEGIES", "from_molecules", "point_group", "bonds",
           "bond_lengths",
           "descriptors", "fragments", "match"]


@register_function(
    aliases=["bond lengths", "bond length check", "are the bonds sensible",
             "unusual bonds", "geometry check", "bond deviation"],
    category="mol",
    description="Compare every covalent bond against its tabulated length and "
                "report how far the geometry departs from it, which is the "
                "cheapest check that a generated molecule is a molecule.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["mean_bond_deviation", "max_bond_deviation",
                      "n_unusual_bonds", "n_bonds_measured",
                      "bond_lengths_ok"]},
    examples=["mv.mol.bond_lengths(md)",
              "mv.mol.bond_lengths(md, tolerance=0.1)"],
    related=["mv.mol.bonds", "mv.gen.validate", "mv.pp.qc"],
    notes="mv.pp.qc catches atoms on top of one another. This catches the "
          "subtler failure: bonds that exist but are the wrong length. A "
          "generated molecule with a 1.9 angstrom C-C bond is not a strained "
          "conformer, it is not a molecule — and no minimum-distance check "
          "will say so, because 1.9 angstrom is a perfectly ordinary distance "
          "between two atoms that are not bonded.\\n\\n"
          "Compared against the **single-bond** length throughout, so a double "
          "or triple bond reads as short by design: C=C at 1.34 against a "
          "tabulated 1.54 is a deviation of 0.2, and that is information "
          "rather than an error. Read max_bond_deviation with that in mind, "
          "and n_unusual_bonds as 'worth looking at' rather than 'wrong'.\\n\\n"
          "A pair with no tabulated length is skipped rather than guessed. The "
          "table covers the common organic pairs and little else, so this is a "
          "check for molecules rather than for coordination compounds.\n\n"
          "**Read n_bonds_measured first.** The bonds are found by covalent "
          "radius, so a badly stretched geometry does not report long bonds — "
          "it stops having bonds at all. Ethanol scaled by 1.25 leaves one "
          "measurable bond out of eight, and that count is a louder signal "
          "than any deviation computed from the one that survived.",
)
def bond_lengths(md: AnnData, source: str = "input",
                 tolerance: float = 0.15) -> None:
    """Bond lengths against their tabulated values. Deposits; returns ``None``."""
    from pymatgen.analysis.local_env import CovalentBondNN
    from pymatgen.core.bonds import get_bond_length

    S, periodic = _molecules(md, source, "mv.mol.bond_lengths")
    finder = CovalentBondNN()

    mean_dev = np.full(md.n_obs, np.nan)
    max_dev = np.full(md.n_obs, np.nan)
    unusual = np.full(md.n_obs, np.nan)
    measured_count = np.full(md.n_obs, np.nan)
    ok = np.full(md.n_obs, False)
    skipped = 0

    for i, molecule in enumerate(S):
        if i in periodic:
            continue
        deviations = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for index in range(len(molecule)):
                    for neighbour in finder.get_nn_info(molecule, index):
                        other = neighbour["site_index"]
                        if other <= index:
                            continue        # each bond once
                        a = molecule[index].specie.symbol
                        b = molecule[other].specie.symbol
                        try:
                            expected = float(get_bond_length(a, b, 1))
                        except Exception:
                            skipped += 1
                            continue
                        measured = float(molecule[index].distance(
                            molecule[other]))
                        deviations.append(abs(measured - expected))
        except Exception:
            continue
        measured_count[i] = len(deviations)
        if not deviations:
            continue
        values = np.asarray(deviations, dtype=float)
        mean_dev[i] = float(values.mean())
        max_dev[i] = float(values.max())
        unusual[i] = int((values > tolerance).sum())
        ok[i] = bool(unusual[i] == 0)

    md.obs["mean_bond_deviation"] = mean_dev
    md.obs["max_bond_deviation"] = max_dev
    md.obs["n_unusual_bonds"] = unusual
    md.obs["n_bonds_measured"] = measured_count
    md.obs["bond_lengths_ok"] = ok
    md.uns["bond_lengths"] = {
        "source": source, "tolerance": float(tolerance),
        "n_pairs_without_a_table_entry": int(skipped),
        "note": "measured against the tabulated single-bond length, so a "
                "double or triple bond reads as short by design",
    }
    record(md, "mol.bond_lengths", source=source, tolerance=tolerance)

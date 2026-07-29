"""``mv.iface`` — interfaces between two materials.

A battery, a transistor and a coated turbine blade are all made of interfaces.
Two materials that are each perfectly stable can react the moment they touch,
and two that do not react may still be unable to grow on one another because
their lattices do not match. Both failures kill devices, and neither is visible
from either material alone.

The namespace answers the two questions separately, because they are separate:

===============================  ==========================================
will the lattices match?          :func:`match` — Zur and McGill's algorithm
what happens at the contact?      :func:`reactivity` — the interfacial hull
can I build the cell?             :func:`build` — the coherent interface
===============================  ==========================================

A pairing is not a material, so it gets its own object: rows are *pairs*, with
``obs['film']`` and ``obs['substrate']`` pointing back at the parent. That is
the same shape as ``mv.multi.sites`` and ``mv.surf.slabs`` — a derived axis
with foreign keys — and everything written for the material axis works on it.

```python
pairs = mv.iface.match(md)                      # every film/substrate pairing
mv.iface.reactivity(pairs, md, level='emt')     # do they react on contact?
interfaces = mv.iface.build(md, film=1, substrate=0)
```
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from ._core import deposit_structures, record, structures
from ._registry import register_function

#: Set on a pairs object so functions can tell the axes apart.
AXIS_KEY = "matverse_axis"


def _require_pairs(pairs: AnnData) -> None:
    if pairs.uns.get(AXIS_KEY) != "interface_pairs":
        raise ValueError(
            "this is not a pairs object; build one with mv.iface.match(md). "
            "An interface is a property of two materials, so it needs the axis "
            "whose rows are pairs.")


@register_function(
    aliases=["lattice match", "epitaxy", "zsl", "substrate matching",
             "can these grow together", "interface match", "film substrate"],
    category="iface",
    description="Find the epitaxial lattice matches between every pair of "
                "materials, by the Zur and McGill algorithm, and report the "
                "strain each match would need.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["film", "substrate", "film_miller", "substrate_miller",
                      "match_area", "von_mises_strain", "n_matches"]},
    examples=["pairs = mv.iface.match(md)",
              "pairs = mv.iface.match(md, max_area=200.0, max_strain=0.05)"],
    related=["mv.iface.build", "mv.iface.reactivity", "mv.surf.slabs"],
    notes="Zur and McGill (1984) search over supercells of both surfaces for "
          "one that fits within a tolerance. The answer is a **list** of "
          "matches per orientation pair, not a yes or no — which is why the "
          "result is its own axis rather than a column.\n\n"
          "``von_mises_strain`` is the number to screen on. Epitaxy needs it "
          "below roughly 5%; above that the film relaxes by making "
          "dislocations instead, and the coherent interface you were "
          "designing does not exist.\n\n"
          "The search is over pairs, so cost is quadratic in the number of "
          "materials and grows fast with max_area. Start small.",
)
def match(md: AnnData, source: str = "input", max_area: float = 400.0,
          max_strain: float = 0.05, film_max_miller: int = 1,
          substrate_max_miller: int = 1, lowest: bool = True) -> AnnData:
    """Epitaxial matches between every ordered pair. Returns a pairs object."""
    from pymatgen.analysis.interfaces.substrate_analyzer import (
        SubstrateAnalyzer)

    analyzer = SubstrateAnalyzer(film_max_miller=film_max_miller,
                                 substrate_max_miller=substrate_max_miller,
                                 max_area=max_area)
    S = structures(md, source)
    # Label by obs['name'] when the dataset carries one — "Cu on Al" is
    # readable and "0 on 1" is not — but keep the row index alongside so the
    # pairing can always be resolved back to a row without a lookup.
    names = [str(x) for x in md.obs.get("name", md.obs_names)]

    rows, failures = [], []
    for i, film in enumerate(S):
        for j, substrate in enumerate(S):
            if i == j:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    found = list(analyzer.calculate(
                        film=film, substrate=substrate, lowest=lowest))
            except Exception as exc:
                failures.append(f"{names[i]}/{names[j]}: "
                                f"{type(exc).__name__}: {exc}")
                continue

            if not found:
                rows.append({"film": names[i], "substrate": names[j],
                             "film_index": i, "substrate_index": j,
                             "film_miller": "", "substrate_miller": "",
                             "match_area": np.nan, "von_mises_strain": np.nan,
                             "n_matches": 0})
                continue

            best = min(found, key=lambda m: getattr(m, "von_mises_strain",
                                                    np.inf))
            rows.append({
                "film": names[i], "substrate": names[j],
                "film_index": i, "substrate_index": j,
                "film_miller": "_".join(map(str, best.film_miller)),
                "substrate_miller": "_".join(map(str, best.substrate_miller)),
                "match_area": float(getattr(best, "match_area", np.nan)),
                "von_mises_strain": float(getattr(best, "von_mises_strain",
                                                  np.nan)),
                "n_matches": len(found),
            })

    if not rows:
        raise ValueError(
            "no pairings could be evaluated; the dataset needs at least two "
            "materials, and the search may need a larger max_area")

    frame = pd.DataFrame(rows)
    frame.index = pd.Index([f"{r.film}|{r.substrate}"
                            for r in frame.itertuples()], dtype=object)
    out = AnnData(X=np.zeros((len(frame), 0), dtype=np.float32), obs=frame)
    out.uns[AXIS_KEY] = "interface_pairs"
    out.uns["provenance"] = []
    out.uns["match"] = {
        "max_area": float(max_area), "max_strain": float(max_strain),
        "source": source, "algorithm": "Zur and McGill (1984), via pymatgen "
                                       "SubstrateAnalyzer",
        "n_failed": len(failures), "errors": failures[:10],
        "note": "von_mises_strain below ~0.05 is where coherent epitaxy is "
                "plausible; above it the film relaxes by making dislocations",
    }
    out.obs["epitaxial"] = (
        out.obs["von_mises_strain"].to_numpy(dtype=float) <= max_strain)
    record(out, "iface.match", source=source, max_area=max_area,
           max_strain=max_strain)
    return out


@register_function(
    aliases=["interface reaction", "do they react", "interfacial reactivity",
             "contact stability", "electrolyte compatibility"],
    category="iface",
    description="Whether two materials react on contact, and the most "
                "exothermic reaction they can undergo, from the phase diagram "
                "of their combined chemistry.",
    requires={"pairs.obs": ["film_index", "substrate_index"],
              "md.obs": ["energy_per_atom_{level}"]},
    produces={"pairs.obs": ["reaction_energy_{level}", "reaction_{level}",
                            "reacts_{level}"]},
    prerequisites=["mv.iface.match", "mv.thermo.hull"],
    examples=["mv.iface.reactivity(pairs, md, level='emt')"],
    related=["mv.iface.match", "mv.thermo.hull", "mv.thermo.reaction"],
    notes="This is the question that decides whether a solid electrolyte works. "
          "Li-metal anodes and most sulfide electrolytes are each stable, and "
          "they destroy each other on contact — a fact invisible to a hull "
          "computed on either one alone.\n\n"
          "The mixing energy is minimised over all ratios of the two "
          "compositions, so the reported number is the worst case rather than "
          "the 1:1 case. A value of zero means the pair is inert with respect "
          "to the phases **in this dataset**, which is a weaker claim than "
          "inert: pass a complete phase diagram to make it a real one.",
)
def reactivity(pairs: AnnData, md: AnnData, level: str = "emt",
               source: str = "input") -> None:
    """Interfacial reaction energies. Deposits on ``pairs``; returns ``None``."""
    from pymatgen.analysis.interface_reactions import InterfacialReactivity
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    _require_pairs(pairs)

    key = f"energy_per_atom_{level}"
    if key not in md.obs:
        raise ValueError(
            f"obs[{key!r}] absent; an interfacial reaction is read off a phase "
            f"diagram, so run mv.calc.relax(md, level={level!r}) first")

    S = structures(md, source)
    energies = md.obs[key].to_numpy(dtype=float)
    entries = [ComputedEntry(s.composition, e * len(s))
               for s, e in zip(S, energies) if np.isfinite(e)]
    if len(entries) < 2:
        raise ValueError("need at least two materials with finite energies to "
                         "build a phase diagram")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagram = PhaseDiagram(entries)

    film_index = pairs.obs["film_index"].to_numpy(dtype=int)
    substrate_index = pairs.obs["substrate_index"].to_numpy(dtype=int)

    worst = np.full(pairs.n_obs, np.nan)
    equations = np.full(pairs.n_obs, "", dtype=object)
    errors = []

    for row in range(pairs.n_obs):
        c1 = S[film_index[row]].composition
        c2 = S[substrate_index[row]].composition
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                analysis = InterfacialReactivity(c1, c2, diagram, norm=True)
                found = list(analysis.get_kinks())
            if found:
                # get_kinks yields (index, x, energy, reaction, e_above_hull)
                best = min(found, key=lambda k: float(k[2]))
                worst[row] = float(best[2])
                equations[row] = str(best[3])
            errors.append("")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    pairs.obs[f"reaction_energy_{level}"] = worst
    pairs.obs[f"reaction_{level}"] = equations
    pairs.obs[f"reacts_{level}"] = worst < -1e-6
    pairs.obs[f"reactivity_error_{level}"] = errors
    pairs.uns["reactivity"] = {
        "level": str(level), "n_entries": len(entries),
        "closed_system": True,
        "note": "energies are minimised over all mixing ratios, so this is the "
                "worst case rather than the 1:1 case; a hull built only from "
                "this dataset makes 'inert' a claim about these phases only",
    }
    record(pairs, "iface.reactivity", level=level, source=source)


@register_function(
    aliases=["build interface", "coherent interface", "make an interface",
             "heterostructure", "epitaxial cell"],
    category="iface",
    description="Build the actual interface cells for one film/substrate "
                "orientation, one per distinct termination.",
    requires={"structures": ["{source}"]},
    produces={"obs": ["film", "substrate", "film_miller", "substrate_miller",
                      "termination"],
              "structures": ["input"]},
    prerequisites=["mv.iface.match"],
    examples=["interfaces = mv.iface.build(md, film=1, substrate=0)",
              "interfaces = mv.iface.build(md, film='Al', substrate='Ni', "
              "film_miller=(1, 1, 1))"],
    related=["mv.iface.match", "mv.surf.slabs", "mv.calc.relax"],
    notes="Returns an ordinary materials object, so the interfaces relax, "
          "screen and plot like anything else — an interface is a structure, "
          "and matverse has no reason to treat it as a special kind of thing.\n\n"
          "One cell per **termination**: cutting the same orientation at a "
          "different plane gives a different interface with a different energy, "
          "and which one forms depends on growth conditions. Enumerating them "
          "is the honest thing to do, in the same way mv.surf.slabs enumerates "
          "surface terminations.",
)
def build(md: AnnData, film, substrate, film_miller=(1, 1, 1),
          substrate_miller=(1, 1, 1), source: str = "input",
          gap: float = 2.0, vacuum: float = 20.0,
          max_terminations: int = 4) -> AnnData:
    """Interface cells for one orientation pair. Returns a materials object."""
    from pymatgen.analysis.interfaces.coherent_interfaces import (
        CoherentInterfaceBuilder)

    from .data import from_structures

    names = list(map(str, md.obs_names))
    labels = [str(x) for x in md.obs.get("name", md.obs_names)]

    def _resolve(which):
        if isinstance(which, (int, np.integer)):
            return int(which)
        text = str(which)
        for pool in (names, labels):
            if text in pool:
                return pool.index(text)
        raise KeyError(f"{which!r} is not a row of this dataset; rows are "
                       f"{names} with names {labels}")

    i, j = _resolve(film), _resolve(substrate)
    S = structures(md, source)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder = CoherentInterfaceBuilder(
            substrate_structure=S[j], film_structure=S[i],
            film_miller=tuple(film_miller),
            substrate_miller=tuple(substrate_miller))
        terminations = list(builder.terminations)[:max_terminations]
        built, used = [], []
        for termination in terminations:
            try:
                for cell in builder.get_interfaces(
                        termination=termination, gap=gap, vacuum_over_film=vacuum):
                    built.append(cell)
                    used.append(str(termination))
                    break
            except Exception:
                continue

    if not built:
        raise ValueError(
            f"no coherent interface could be built for {labels[i]}"
            f"{tuple(film_miller)} on {labels[j]}{tuple(substrate_miller)}. "
            f"Check mv.iface.match first — a pairing with a large "
            f"von_mises_strain has no coherent cell to build.")

    out = from_structures([c for c in built])
    out.obs["film"] = labels[i]
    out.obs["substrate"] = labels[j]
    out.obs["film_miller"] = "_".join(map(str, film_miller))
    out.obs["substrate_miller"] = "_".join(map(str, substrate_miller))
    out.obs["termination"] = used
    out.uns["interface"] = {
        "film": labels[i], "substrate": labels[j],
        "film_miller": list(film_miller),
        "substrate_miller": list(substrate_miller),
        "gap": float(gap), "vacuum": float(vacuum),
        "n_terminations": len(terminations),
    }
    record(out, "iface.build", film=labels[i], substrate=labels[j],
           film_miller=tuple(film_miller))
    return out


__all__ = ["AXIS_KEY", "match", "reactivity", "build"]

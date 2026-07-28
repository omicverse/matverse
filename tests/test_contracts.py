"""Every contract claim, verified by execution.

The audit metric in common use credits a registry field for being *present*, not
for being *true*. These tests refuse that: each ``produces`` claim is checked by
running the call and looking, each ``requires`` claim by deleting the slot and
confirming the call fails, and each ``prerequisites`` claim by omitting the
upstream call.

A claim that fails here is meant to be deleted from the decorator. The point of
the exercise is a registry whose claims are all earned.
"""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv
from matverse._probe import ProbeReport, audit, probe_call, probe_prerequisite


class TestRegistryIntegrity:
    def test_every_public_function_is_registered(self):
        missing = []
        for name in ("data", "pp", "feat", "tl", "calc", "thermo", "screen"):
            module = getattr(mv, name)
            for attr in module.__all__:
                func = getattr(module, attr)
                if not callable(func) or isinstance(func, type):
                    continue          # exception classes are not operations
                if mv.registry.get(f"matverse.{name}.{attr}") is None and \
                        mv.registry.get(attr) is None:
                    missing.append(f"mv.{name}.{attr}")
        assert not missing, f"undecorated public functions: {missing}"

    def test_aliases_do_not_collide(self):
        """Registered at import; a collision would have raised already."""
        seen: dict[str, str] = {}
        for entry in mv.registry.entries():
            for alias in entry["aliases"]:
                key = alias.lower()
                assert key not in seen or seen[key] == entry["full_name"], (
                    f"alias {alias!r} claimed by both {seen.get(key)} and "
                    f"{entry['full_name']}")
                seen[key] = entry["full_name"]

    def test_every_entry_has_a_description_and_an_example(self):
        """The two fields the omicverse ablation ranked highest."""
        thin = [e["public_name"] for e in mv.registry.entries()
                if not e["description"] or not e["examples"]]
        assert not thin, f"entries missing description or examples: {thin}"

    def test_lookup_by_intent_finds_the_right_function(self):
        assert mv.find("thermodynamic stability")[0] == "mv.thermo.hull"
        assert mv.find("which elements distinguish")[0] == \
            "mv.tl.rank_elements_groups"
        assert mv.find("relax structures")[0] == "mv.calc.relax"

    def test_producers_of_resolves_a_templated_slot(self):
        names = [e["public_name"]
                 for e in mv.registry.producers_of("e_above_hull_emt")]
        assert "mv.thermo.hull" in names

    def test_describe_renders_the_slot_convention(self):
        text = mv.describe("mv.calc.relax")
        assert "obsm['structures']['relaxed_{level}']" in text
        assert "obs['energy_{level}']" in text


def _probe_everything(make_md) -> ProbeReport:
    """Run every probeable call once and collect the claims."""
    report = ProbeReport()

    def energised():
        md = make_md()
        mv.calc.energy(md, level="emt")
        return md

    def patterned():
        md = make_md()
        mv.prop.xrd(md, two_theta=(10, 40), step=0.1)
        return md

    def two_levels():
        md = patterned()
        mv.exp.attach(md, "xrd", md.obsm["xrd_calc"], mv.grid_of(md, "xrd"))
        return md

    def batched():
        """Two databases sharing every composition, so harmonize has anchors."""
        import pandas as pd
        md = make_md()
        doubled = mv.data.from_structures(
            mv.structures(md) + mv.structures(md),
            obs=pd.DataFrame({
                "database": ["mp"] * md.n_obs + ["oqmd"] * md.n_obs,
                "e": np.concatenate([np.zeros(md.n_obs),
                                     np.full(md.n_obs, 0.1)]),
            }))
        return doubled

    def relaxed():
        md = make_md()
        mv.calc.relax(md, level="emt", fmax=0.2, steps=20)
        return md

    def targeted():
        md = make_md()
        mv.pp.describe(md)
        mv.feat.element_stats(md)
        return md

    def splittable():
        md = targeted()
        mv.model.split(md, strategy="composition")
        return md

    def campaigning():
        md = targeted()
        known = np.zeros(md.n_obs, dtype=bool)
        known[:2] = True                       # leave a pool to suggest from
        mv.opt.start(md, objective="volume", observed=known)
        return md

    def vibrating():
        md = make_md()
        mv.calc.relax(md, level="emt", fmax=0.05, steps=30)
        return md

    def phononed():
        md = vibrating()
        mv.prop.phonon(md, level="emt", source="relaxed_emt",
                       supercell=(1, 1, 1))
        return md

    def suggested():
        md = campaigning()
        mv.opt.suggest(md, n=2, method="greedy", predicted="volume")
        return md

    def featured():
        md = make_md()
        mv.feat.element_stats(md)
        return md

    def grouped():
        md = make_md()
        mv.pp.describe(md)
        md.obs["group"] = ["a", "b", "a", "b", "a", "b"]
        return md

    def embedded():
        md = make_md()
        mv.pp.normalize_composition(md)
        mv.tl.pca(md, n_comps=2)
        return md

    cases = [
        (mv.pp.standardize, make_md, (), {}),
        (mv.pp.describe, make_md, (), {}),
        (mv.pp.qc, make_md, (), {}),
        (mv.pp.normalize_composition, make_md, (), {}),
        (mv.pp.dedup, make_md, (), {}),
        (mv.pp.supercell, make_md, ([2, 1, 1],), {"name": "big"}),
        (mv.pp.rattle, make_md, (), {"seed": 0}),
        (mv.pp.strain, make_md, (0.01,), {}),
        (mv.feat.element_stats, make_md, (), {}),
        (mv.feat.similarity, featured, (), {}),
        (mv.calc.energy, make_md, (), {"level": "emt"}),
        (mv.calc.relax, make_md, (), {"level": "emt", "fmax": 0.2,
                                      "steps": 20}),
        (mv.thermo.hull, energised, (), {"level": "emt"}),
        (mv.screen.rank, grouped, (), {"by": "volume"}),
        (mv.tl.pca, make_md, (), {"n_comps": 2}),
        (mv.tl.neighbors, embedded, (), {"n_neighbors": 3}),
        (mv.tl.cluster, embedded, (), {"method": "kmeans", "n_clusters": 2}),
        (mv.tl.rank_elements_groups, grouped, ("group",), {}),
        (mv.pp.harmonize, batched, (), {"batch_key": "database",
                                        "energy_key": "e",
                                        "reference": "mp"}),
        (mv.prop.xrd, make_md, (), {"two_theta": (10, 40), "step": 0.1}),
        (mv.prop.rdf, make_md, (), {"r_max": 6.0}),
        (mv.prop.compare_grids, two_levels, ("xrd", "calc", "experiment"), {}),
        (mv.exp.measure, make_md, ("band_gap", [0.0] * 6), {}),
        (mv.exp.match_xrd, patterned, ([1.0] * 10, list(range(10, 20))), {}),
        (mv.gen.validate, make_md, (), {}),
        (mv.prop.elastic, relaxed, (), {"level": "emt",
                                        "source": "relaxed_emt"}),
        (mv.model.split, featured, (), {"strategy": "composition"}),
        (mv.model.fit, splittable, (), {"target": "volume",
                                       "level": "rf_pred"}),
        (mv.model.cross_validate, targeted, (), {"target": "volume",
                                                 "seeds": (0,),
                                                 "strategies": ("random",)}),
        (mv.opt.start, targeted, (), {"objective": "volume"}),
        (mv.opt.suggest, campaigning, (), {"n": 2, "method": "greedy",
                                           "predicted": "volume"}),
        (mv.opt.observe, suggested, (), {}),
        (mv.utils.set_units, targeted, (), {"column": "volume",
                                            "unit": "angstrom^3"}),
        (mv.utils.convert, targeted, (), {"column": "volume",
                                          "unit": "eV",
                                          "key_added": "volume_ev"}),
        (mv.prop.phonon, vibrating, (), {"level": "emt",
                                         "source": "relaxed_emt",
                                         "supercell": (1, 1, 1)}),
        (mv.prop.free_energy, phononed, (), {"level": "emt"}),
        (mv.thermo.chempot_limits, energised, (), {"level": "emt"}),
        (mv.multi.aggregate, None, (), {}),          # placeholder, see below
    ]
    cases = [case for case in cases if case[1] is not None]
    for func, factory, args, kwargs in cases:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report.results += probe_call(func, factory, *args, **kwargs)
    return report


@pytest.fixture(scope="module")
def probe_report(request):
    """Probe everything once; the individual tests read the same report."""
    structures = request.getfixturevalue("_probe_structures")

    def make_md():
        return mv.data.from_structures(list(structures))
    return _probe_everything(make_md)


@pytest.fixture(scope="module")
def _probe_structures():
    from pymatgen.core import Lattice, Structure

    def fcc(sym, a):
        return Structure(Lattice.cubic(a), [sym] * 4,
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    def l12(host, guest, a):
        return Structure(Lattice.cubic(a), [guest, host, host, host],
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    return [fcc("Al", 4.05), fcc("Cu", 3.61), fcc("Ni", 3.52),
            l12("Al", "Cu", 3.90), l12("Cu", "Al", 3.70),
            Structure(Lattice.cubic(2.89), ["Al", "Ni"],
                      [[0, 0, 0], [.5, .5, .5]])]


class TestContracts:
    def test_every_claim_probed_is_verified(self, probe_report):
        assert probe_report.results, "no claims were probed"
        assert not probe_report.failed, "\n" + probe_report.summary()

    def test_contract_verified_rate_is_reported(self, probe_report):
        print("\n" + probe_report.summary())
        assert probe_report.rate == 1.0

    def test_both_contract_kinds_are_exercised(self, probe_report):
        kinds = probe_report.by_kind()
        assert kinds["produces"][1] >= 30
        assert kinds["requires"][1] >= 10


class TestPrerequisites:
    def test_hull_really_needs_energies(self, make_md):
        result = probe_prerequisite(
            mv.thermo.hull, make_md, mv.calc.energy,
            name="mv.thermo.hull", level="emt",
            upstream_kwargs={"level": "emt"})
        assert result.verified, result.detail

    def test_neighbors_really_needs_pca(self, make_md):
        result = probe_prerequisite(
            mv.tl.neighbors, make_md, mv.tl.pca,
            name="mv.tl.neighbors", n_neighbors=3,
            upstream_kwargs={"n_comps": 2})
        assert result.verified, result.detail

    def test_similarity_really_needs_a_feature_block(self, make_md):
        result = probe_prerequisite(
            mv.feat.similarity, make_md, mv.feat.element_stats,
            name="mv.feat.similarity")
        assert result.verified, result.detail

    def test_cluster_has_no_unconditional_prerequisite(self, make_md):
        """The route-conditional case, pinned as a finding.

        mv.tl.cluster consumes different state on each route: leiden reads the
        neighbour graph, kmeans reads the embedding. So no unconditional
        prerequisite on mv.tl.neighbors is true, and the registry carries none.
        This test exists to fail loudly if someone adds one back.
        """
        def with_pca():
            md = make_md()
            mv.tl.pca(md, n_comps=2)
            return md

        entry = mv.registry.get("mv.tl.cluster")
        assert not entry["requires"], (
            "mv.tl.cluster must carry no requires claim; its two routes need "
            "different state and the contract has one field per function")
        assert not entry["prerequisites"]
        assert "leiden" in entry["dispatch"] and "kmeans" in entry["dispatch"]

        result = probe_prerequisite(
            mv.tl.cluster, with_pca, mv.tl.neighbors,
            name="mv.tl.cluster", method="kmeans", n_clusters=2,
            upstream_kwargs={"n_neighbors": 3})
        assert not result.verified
        assert "succeeded without it" in result.detail


class TestAudit:
    def test_coverage_is_reported_beside_the_probe_rate(self):
        report = audit()
        print("\nregistry audit:", report["n_entries"], "entries,",
              report["n_claims"], "contract claims")
        for field, value in sorted(report["coverage"].items()):
            print(f"  {field:15s} {value:.0%}")
        assert report["coverage"]["description"] == 1.0
        assert report["coverage"]["examples"] == 1.0
        assert report["n_claims"] > 50

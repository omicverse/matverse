"""Amorphous content from neat / spiked specimen pairs (mv.exp.qpa_amorphous).

Synthetic powders: corundum (the only Al carrier) + stishovite-type SiO2 +
silica glass (a broad hump, Si again). Patterns come from the engine's forward
model with a different Debye-Waller B per phase, zero shift, displacement and
Poisson counts. Standards: corundum (already in the powder: standard addition)
and zincite (foreign, strongly absorbing).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytest.importorskip("xraydb")

import matverse as mv
from matverse import _qpa as Q
from matverse import _qpa_amorphous as A

RAD = Q.parse_radiation([(1.540598, 1.0), (1.544426, 0.5)])
GRID = np.arange(15.0, 90.0 + 1e-9, 0.02)
B_TRUE = {"corundum": 0.3, "stishovite": 0.9, "zincite": 0.6}


def _structures():
    from pymatgen.core import Lattice, Structure
    corundum = Structure.from_spacegroup(
        "R-3c", Lattice.hexagonal(4.759, 12.99), ["Al", "O"],
        [[0, 0, 0.352], [0.306, 0, 0.25]])
    stishovite = Structure.from_spacegroup(
        "P4_2/mnm", Lattice.tetragonal(4.18, 2.666), ["Si", "O"],
        [[0, 0, 0], [0.306, 0.306, 0]])
    zincite = Structure.from_spacegroup(
        "P6_3mc", Lattice.hexagonal(3.25, 5.207), ["Zn", "O"],
        [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.382]])
    rutile = Structure.from_spacegroup(
        "P4_2/mnm", Lattice.tetragonal(4.594, 2.959), ["Ti", "O"],
        [[0, 0, 0], [0.305, 0.305, 0]])
    return {"corundum": corundum, "stishovite": stishovite, "zincite": zincite,
            "rutile": rutile}


def _pattern(structs, crystalline, glass, rng, zero=0.0, disp=0.0):
    """Counts for crystalline masses (any unit) + glass mass (hump)."""
    mix = np.zeros_like(GRID)
    for name, w in crystalline.items():
        s = structs[name]
        st = A.debye_waller(Q.compute_sticks(s, RAD.primary, (GRID[0], GRID[-1]),
                                             cutoff=0.0), B_TRUE[name], RAD.primary)
        zmv = float(s.composition.weight) * float(s.volume)
        mix += (w / zmv) * Q.render(st, GRID, RAD, zero, disp, 0.0,
                                    (0.004, -0.001, 0.005), 0.45, 1.0)
    top = mix.max()
    hump = glass * 0.35 * top * np.exp(-0.5 * ((GRID - 22.0) / 4.0) ** 2)
    bg = 0.04 * top * np.exp(-(GRID - GRID[0]) / 25.0) + 0.02 * top
    return rng.poisson((mix + hump + bg) / top * 30000.0).astype(float)


def _oxides(c, q, g, spike=None, w=0.0):
    """Oxide wt% of a specimen (original powder c/q/g, plus spike w)."""
    al, si = c * (1 - w), (q + g) * (1 - w)
    ox = {"Al2O3": al, "SiO2": si}
    if spike == "corundum":
        ox["Al2O3"] += w
    elif spike == "zincite":
        ox["ZnO"] = w
    return {k: 100 * v for k, v in ox.items()}


TRUTH = {"c": 0.30, "q": 0.45, "g": 0.25}


def _campaign(rng, xrf_bias=0.0):
    s = _structures()
    c, q, g = TRUTH["c"], TRUTH["q"], TRUTH["g"]
    spec, std, xrf = {}, {}, {}
    # r1: corundum spike (standard addition), r2: zincite spike, r3: neat only
    for name, sp, w in (("r1", "corundum", 0.2), ("r2", "zincite", 0.2),
                        ("r3", None, 0.0)):
        neat = _pattern(s, {"corundum": c, "stishovite": q}, g, rng,
                        zero=float(rng.uniform(-0.02, 0.02)))
        spec[name] = {"batch": "B1", "specimens": {"neat": (GRID, neat)}}
        if sp:
            cr = {"corundum": c * (1 - w), "stishovite": q * (1 - w)}
            cr[sp] = cr.get(sp, 0.0) + w
            spiked = _pattern(s, cr, g * (1 - w), rng, disp=0.02)
            spec[name]["specimens"]["spiked"] = (GRID, spiked)
            std[name] = {"phase": sp, "weight_fraction": w, "specimen": "spiked"}
            ox = _oxides(c, q, g, sp, w)
            xrf[name] = {"specimen": "spiked", "oxide_wt_percent": ox}
        else:
            std[name] = None
            xrf[name] = {"specimen": "neat", "oxide_wt_percent": _oxides(c, q, g)}
        xrf[name]["oxide_wt_percent"]["Al2O3"] += xrf_bias
    return s, spec, std, xrf


def _rmse(got, truth):
    keys = set(got) | set(truth)
    return math.sqrt(sum((got.get(k, 0) - truth.get(k, 0)) ** 2 for k in keys)
                     / len(keys))


class TestEstimators:
    def test_standard_addition_is_exact_on_exact_ratios(self):
        # powder: k 0.2, other 0.5, glass 0.3; spike w = 0.25 of the spiked mass
        w = 0.25
        s = w / (1 - w)
        w0 = 0.2 / 0.7
        w1 = (0.2 + s) / (0.7 + s)
        g, sig = A.g_iso(w0, w1, w, 0.015)
        assert g == pytest.approx(0.3, abs=1e-12) and sig > 0

    def test_internal_standard_is_exact_on_exact_ratio(self):
        w = 0.1
        s = w / (1 - w)
        z = s / (s + 0.6)                     # 0.6 crystalline per unit powder
        g, _ = A.g_xeno(z, w, 0.015)
        assert g == pytest.approx(0.4, abs=1e-12)

    def test_closure_drops_a_foreign_spike_and_subtracts_an_iso_spike(self):
        from pymatgen.core import Composition
        el, f, _ = A.original_chemistry({"Al2O3": 24.0, "SiO2": 56.0, "ZnO": 20.0},
                                        "zincite", 0.2, Composition("ZnO"),
                                        {"Al", "Si"})
        assert el["Al"] == pytest.approx(0.30 * Composition("Al2O3").get_wt_fraction("Al"))
        assert f == pytest.approx(100 / 80)
        el, f, _ = A.original_chemistry({"Al2O3": 44.0, "SiO2": 56.0}, "corundum",
                                        0.2, Composition("Al2O3"), {"Al", "Si"})
        assert el["Al"] == pytest.approx(0.30 * Composition("Al2O3").get_wt_fraction("Al"))

    def test_chemistry_scale_only_grows_on_disagreement(self):
        gkey = {"a": "A", "b": "B"}
        agree = [dict(sample=x, estimator=e, G=0.3, sigma=0.02, used=True)
                 for x in "ab" for e in ("iso_spike", "element_balance")]
        assert A.chemistry_scale(agree, gkey) == 1.0
        off = [dict(sample="a", estimator="iso_spike", G=0.30, sigma=0.01, used=True),
               dict(sample="a", estimator="element_balance", G=0.20, sigma=0.01, used=True),
               dict(sample="b", estimator="iso_spike", G=0.30, sigma=0.01, used=True),
               dict(sample="b", estimator="element_balance", G=0.40, sigma=0.01, used=True)]
        assert A.chemistry_scale(off, gkey) > 5.0

    def test_spiked_role_from_entry_names_or_single_specimen(self):
        assert A.spiked_role({"neat": 0, "spiked": 0}, {}) == "spiked"
        assert A.spiked_role({"a": 0, "b": 0}, {"specimen": "b"}) == "b"
        assert A.spiked_role({"only": 0}, {}) == "only"
        with pytest.raises(ValueError):
            A.spiked_role({"a": 0, "b": 0}, {})


class TestQpaAmorphous:
    def test_recovers_glass_with_both_spike_kinds_and_biased_xrf(self):
        rng = np.random.default_rng(3)
        s, spec, std, xrf = _campaign(rng, xrf_bias=2.0)
        names = ["corundum", "stishovite", "zincite", "rutile"]
        md = mv.data.from_structures([s[n] for n in names])
        table = mv.exp.qpa_amorphous(
            md, spec, RAD, names=names, internal_standard=std,
            bulk_chemistry=xrf, chemistry_sigma=1.5, verbose=False, n_jobs=1)
        truth = {"corundum": TRUTH["c"], "stishovite": TRUTH["q"],
                 "amorphous": TRUTH["g"]}
        for sample in ("r1", "r2", "r3"):
            sub = table[table["sample"] == sample]
            got = dict(zip(sub.phase, sub.weight_fraction))
            assert set(got) == set(truth), (sample, got)
            assert _rmse(got, truth) < 0.02, (sample, got)
            assert sum(got.values()) == pytest.approx(1.0)
        est = md.uns["qpa_amorphous"]["estimators"]
        kinds = set(est[est.used].estimator)
        assert {"iso_spike", "element_balance"} <= kinds
        # the Si balance is a lower bound (glass is silica) and is not used
        si = est[est.detail.str.startswith("Si")]
        assert len(si) and not si.used.any()
        # zincite absorbs far more than the matrix: reported, not used
        xs = est[est.estimator == "internal_standard"]
        assert len(xs) == 1 and not bool(xs.used.iloc[0])
        b = json.loads(md.uns["qpa_amorphous"]["specimens"].b_iso.iloc[0])
        assert b["stishovite"] > b["corundum"]

    def test_json_files_and_relative_paths(self, tmp_path):
        rng = np.random.default_rng(4)
        s, spec, std, xrf = _campaign(rng)
        files = {}
        for name, v in spec.items():
            files[name] = {"batch": v["batch"], "specimens": {}}
            for role, (tt, y) in v["specimens"].items():
                p = tmp_path / "patterns" / f"{name}_{role}.xy"
                p.parent.mkdir(exist_ok=True)
                np.savetxt(p, np.c_[tt, y], header="two_theta intensity")
                files[name]["specimens"][role] = f"patterns/{p.name}"
        (tmp_path / "samples.json").write_text(json.dumps({"samples": files}))
        (tmp_path / "std.json").write_text(json.dumps({"samples": std}))
        names = ["corundum", "stishovite", "zincite"]
        md = mv.data.from_structures([s[n] for n in names])
        table = mv.exp.qpa_amorphous(
            md, str(tmp_path / "samples.json"), RAD, names=names,
            internal_standard=str(tmp_path / "std.json"),
            phases=["corundum", "stishovite"], verbose=False, n_jobs=1)
        assert set(table["sample"]) == {"r1", "r2", "r3"}
        g = md.uns["qpa_amorphous"]["groups"]
        assert len(g) == 1 and g.G.iloc[0] == pytest.approx(TRUTH["g"], abs=0.03)

    def test_chemistry_needs_its_accuracy(self):
        rng = np.random.default_rng(5)
        s, spec, std, xrf = _campaign(rng)
        md = mv.data.from_structures([s["corundum"], s["stishovite"], s["zincite"]])
        with pytest.raises(ValueError, match="chemistry_sigma"):
            mv.exp.qpa_amorphous(md, spec, RAD,
                                 names=["corundum", "stishovite", "zincite"],
                                 internal_standard=std, bulk_chemistry=xrf,
                                 verbose=False)

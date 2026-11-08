"""Amorphous content from specimen pairs, element balance and replicate groups.

Numerics behind :func:`matverse.exp.qpa_amorphous`. A sample is described by
one or two measured specimens of the same powder: the powder as prepared
("neat") and the same powder after a known mass of a crystalline standard was
mixed in ("spiked"). Whole-pattern fitting gives only *relative* crystalline
weights per specimen; the non-crystalline fraction ``G`` of the original
powder comes from estimators that tie those ratios to a known mass:

iso-phase spike (standard addition)
    The standard ``k`` is a phase the powder already contains. With
    ``s = w / (1 - w)`` spike mass per unit original mass and ``R = W_k /
    (1 - W_k)`` the ratio of ``k`` to all other crystalline mass in a pattern,
    neat gives ``R0 = x_k / X`` and spiked ``R1 = (x_k + s) / X`` where ``X`` is
    the other crystalline mass per unit original mass. So ``X = s / (R1 - R0)``,
    the total crystalline mass is ``X / (1 - W0_k)`` and ``G = 1 - X / (1 -
    W0_k)``. The neat pattern is the external reference; one spiked pattern
    alone only gives the total ``k``.

xeno-phase spike (classical internal standard)
    The standard is foreign to the powder, with relative weight ``z`` in the
    spiked pattern: crystalline mass per unit original mass is ``s (1 - z) /
    z``. Accurate only when the standard absorbs like the matrix or particle
    sizes allow a Brindley correction: a strongly absorbing standard is
    under-represented, ``z`` reads low, and ``G`` is biased LOW (often
    negative). Without particle sizes it is reported but not used once the
    linear-absorption contrast exceeds ``max_absorption_contrast``.

element balance (bulk chemistry, e.g. XRF)
    An element carried by exactly one identified crystalline phase ``p`` (and
    not by the amorphous material) fixes ``x_p`` absolutely; ``G = 1 - x_p /
    W0_p``. When the amorphous material also carries the element, ``x_p`` is
    over-estimated and ``G`` is too LOW, so each element estimate is a lower
    bound: per specimen the largest one is kept unless the amorphous
    composition is given. The analysed specimen's spike is removed first:
    for a foreign standard its components are dropped and the closed (sum
    100) analysis renormalised, since the standard's weighed mass is known
    exactly and its reading only distorts the closure; for an iso-phase
    standard its element mass is subtracted.

Replicate groups (same recipe, same raw-material lots) share ``G``. Estimates
are combined by inverse variance; the element balance of standardless
chemistry is a systematic error within a group, so the group's element
estimates are averaged and enter once with their mean sigma, not as
independent measurements.

References: Hill & Howard, J. Appl. Cryst. 20 (1987) 467 (ZMV scale-to-mass);
Bish & Howard, J. Appl. Cryst. 21 (1988) 86 (internal standards); Brindley,
Philos. Mag. 36 (1945) 347 (microabsorption); Madsen, Scarlett & Kern, Z.
Kristallogr. 226 (2011) 944 (amorphous QPA); Copeland & Bragg, Anal. Chem. 30
(1958) 196 (the addition method); Alexander & Klug, Anal. Chem. 20 (1948) 886.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _qpa as Q


# ----------------------------------------------------------------------------
# Joint whole-pattern refinement of specimens of the same powder
# ----------------------------------------------------------------------------

@dataclass
class JointOptions:
    n_cheb: int = 12                  # Chebyshev background order (per pattern)
    width_range: tuple = (0.5, 2.0)   # per-phase FWHM multiplier
    b_range: tuple = (0.0, 3.0)       # per-phase isotropic Debye-Waller B, Å^2
    b0: float = 0.5
    max_nfev: int = 200


@dataclass
class JointFit:
    names: list                       # phases of this pattern
    weights_rel: dict                 # Hill-Howard (+Brindley) relative weights
    scales: dict
    rwp: float
    zero: float
    displacement: float
    tau: dict = field(default_factory=dict)


def _background_basis(grid, n_cheb, diffuse_fwhm):
    u = 2.0 * (grid - grid.min()) / (grid.max() - grid.min()) - 1.0
    cheb = np.polynomial.chebyshev.chebvander(u, n_cheb)
    return np.column_stack([cheb, 1.0 / grid, Q.diffuse_basis(grid, diffuse_fwhm)])


def _bounded_linear(L, y, n_pos):
    """Counting-weighted linear solve: the first n_pos columns (phase scales)
    non-negative, the background columns (Chebyshev, 1/2θ, humps) free."""
    from scipy.optimize import lsq_linear
    w = 1.0 / np.sqrt(np.maximum(y, 1.0))
    A = L * w[:, None]
    sc = np.linalg.norm(A, axis=0) + 1e-30
    lo = np.r_[np.zeros(n_pos), np.full(L.shape[1] - n_pos, -np.inf)]
    r = lsq_linear(A / sc, y * w, bounds=(lo, np.full(L.shape[1], np.inf)),
                   method="bvls")
    return r.x / sc


def debye_waller(st: Q.Sticks, b_iso: float, wavelength: float) -> Q.Sticks:
    """Sticks attenuated by exp(-2 B (sin θ / λ)^2)."""
    s = np.sin(np.radians(st.pos / 2.0)) / wavelength
    return Q.Sticks(st.pos, st.inten * np.exp(-2.0 * b_iso * s * s), st.cos_po)


def fit_joint(patterns, phase_sets, sticks: dict, props: dict, radiation,
              opts: Q.FitOptions, jopts: JointOptions | None = None,
              particle_um=None):
    """Refine several patterns of specimens made from the same powders.

    Shared: Caglioti U, V, W and η (one instrument), and per phase the lattice
    strain, March-Dollase ratio, an FWHM multiplier (phase-specific
    broadening) and an isotropic Debye-Waller B (high-angle intensity fall-off,
    which otherwise biases Hill-Howard scales by several percent between
    phases of different stiffness). Per pattern: zero shift, displacement,
    Chebyshev + 1/2θ + diffuse-hump background and the phase scales (linear,
    solved at every step, counting-statistics weights).

    ``props[name] = (zmv, mu_um, rho)``. Returns (list of JointFit, info).
    """
    from scipy.optimize import least_squares
    jopts = jopts or JointOptions()
    names = []
    for ps in phase_sets:
        for n in ps:
            if n not in names:
                names.append(n)
    k, npat = len(names), len(patterns)
    bases = [_background_basis(np.asarray(tt, float), jopts.n_cheb, opts.diffuse_fwhm)
             for tt, _ in patterns]
    lw = np.log(jopts.width_range)
    p0 = np.r_[opts.uvw0, 0.5, np.zeros(2 * npat), np.zeros(k), np.zeros(k),
               np.zeros(k), np.full(k, jopts.b0)]
    lo = np.r_[opts.uvw_lo, 0.0, np.tile([-opts.zero_bound, -opts.disp_bound], npat),
               np.full(k, -opts.strain_bound), np.full(k, np.log(opts.po_range[0])),
               np.full(k, lw[0]), np.full(k, jopts.b_range[0])]
    hi = np.r_[opts.uvw_hi, 1.0, np.tile([opts.zero_bound, opts.disp_bound], npat),
               np.full(k, opts.strain_bound), np.full(k, np.log(opts.po_range[1])),
               np.full(k, lw[1]), np.full(k, jopts.b_range[1])]
    o = 4 + 2 * npat

    def design(p, i):
        uvw = np.asarray(p[0:3])
        tt = np.asarray(patterns[i][0], float)
        cols = []
        for n in phase_sets[i]:
            j = names.index(n)
            st = debye_waller(sticks[n], p[o + 3 * k + j], radiation.primary)
            f2 = np.exp(2.0 * p[o + 2 * k + j])
            cols.append(Q.render(st, tt, radiation, p[4 + 2 * i], p[5 + 2 * i],
                                 p[o + j], tuple(uvw * f2), p[3],
                                 float(np.exp(p[o + k + j]))))
        return np.column_stack(cols + [bases[i]])

    def residual(p):
        out = []
        for i, (_, y) in enumerate(patterns):
            L = design(p, i)
            c = _bounded_linear(L, y, len(phase_sets[i]))
            out.append((L @ c - y) / np.sqrt(np.maximum(y, 1.0)))
        return np.concatenate(out)

    sol = least_squares(residual, np.clip(p0, lo, hi), bounds=(lo, hi),
                        method="trf", max_nfev=jopts.max_nfev)
    p = sol.x
    fits = []
    for i, (tt, y) in enumerate(patterns):
        y = np.asarray(y, float)
        L = design(p, i)
        c = _bounded_linear(L, y, len(phase_sets[i]))
        r = (L @ c - y) / np.sqrt(np.maximum(y, 1.0))
        rwp = float(np.sqrt(np.sum(r * r) / np.sum(y * y / np.maximum(y, 1.0))))
        ps = list(phase_sets[i])
        sc = c[:len(ps)]
        w, _, tau = Q.hill_howard_masses(
            sc, [props[n][0] for n in ps], [props[n][1] for n in ps],
            [props[n][2] for n in ps], particle_um)
        fits.append(JointFit(names=ps, weights_rel=dict(zip(ps, map(float, w))),
                             scales=dict(zip(ps, map(float, sc))), rwp=rwp,
                             zero=float(p[4 + 2 * i]), displacement=float(p[5 + 2 * i]),
                             tau=dict(zip(ps, map(float, tau)))))
    info = {"b_iso": dict(zip(names, map(float, p[o + 3 * k:o + 4 * k]))),
            "width_factor": dict(zip(names, map(float, np.exp(p[o + 2 * k:o + 3 * k])))),
            "po_ratio": dict(zip(names, map(float, np.exp(p[o + k:o + 2 * k])))),
            "strain": dict(zip(names, map(float, p[o:o + k]))),
            "nfev": int(sol.nfev)}
    return fits, info


# ----------------------------------------------------------------------------
# Estimators of the amorphous fraction G of the original powder
# ----------------------------------------------------------------------------

def _prop(fun, args, rel):
    """fun(*args) and its 1-sigma for independent relative errors on args."""
    base = fun(*args)
    var = 0.0
    for i, e in enumerate(rel):
        if not e:
            continue
        a = list(args)
        a[i] = a[i] * (1.0 + e)
        var += (fun(*a) - base) ** 2
    return float(base), float(np.sqrt(var))


def g_iso(w0_k, w1_k, spike_wt, ratio_sigma):
    """Standard addition of a phase already present; neat = external reference."""
    s = spike_wt / (1.0 - spike_wt)

    def f(r0, r1):
        return 1.0 - s / (r1 - r0) * (1.0 + r0)    # (1 + R0) = 1 / (1 - W0_k)

    r0 = w0_k / (1.0 - w0_k)
    r1 = w1_k / (1.0 - w1_k)
    if not (np.isfinite(r0) and np.isfinite(r1)) or r1 <= r0:
        return float("nan"), float("nan")
    return _prop(f, (r0, r1), (ratio_sigma, ratio_sigma))


def g_xeno(z, spike_wt, ratio_sigma):
    """Classical internal standard foreign to the powder."""
    s = spike_wt / (1.0 - spike_wt)
    if not 0.0 < z < 1.0:
        return float("nan"), float("nan")

    def f(r):                                      # r = z / (1 - z)
        return 1.0 - s / r

    return _prop(f, (z / (1.0 - z),), (ratio_sigma,))


def absorption_contrast(mu_spike, mu_phases: dict, weights: dict, rho: dict):
    """|μ_spike − μ̄| / μ̄ with μ̄ the volume-weighted μ of the powder's phases."""
    vols = {n: weights[n] / rho[n] for n in weights if weights[n] > 0}
    tot = sum(vols.values())
    if tot <= 0:
        return float("nan")
    mu_bar = sum(vols[n] * mu_phases[n] for n in vols) / tot
    return float(abs(mu_spike - mu_bar) / mu_bar)


def _element_fractions(component_wt: dict):
    """{component formula: wt (any unit summing ~100)} -> {element: mass fraction}
    (oxygen dropped from oxide components, an element given alone kept)."""
    from pymatgen.core import Composition
    out = {}
    for comp, v in component_wt.items():
        c = Composition(comp)
        els = [e for e in c.elements]
        for e in els:
            if e.symbol == "O" and len(els) > 1:
                continue
            out[e.symbol] = out.get(e.symbol, 0.0) + float(v) / 100.0 * c.get_wt_fraction(e)
    return out


def _cations(formula):
    from pymatgen.core import Composition
    els = {e.symbol for e in Composition(formula).elements}
    return els - {"O"} if len(els) > 1 else els


def original_chemistry(component_wt: dict, spike=None, spike_wt=0.0,
                       spike_comp=None, native_elements=()):
    """Element mass fractions of the ORIGINAL powder from an analysis of the
    neat or spiked specimen. Returns (fractions, factor, note) where ``factor``
    converts an absolute component error of the analysis into the original
    basis (per unit mass)."""
    if not spike or spike_wt <= 0:
        return _element_fractions(component_wt), 1.0, ""
    sp_el = {e.symbol for e in spike_comp.elements} - {"O"}
    native = set(native_elements)
    if sp_el and sp_el <= native:                  # iso-phase: subtract
        el = _element_fractions(component_wt)
        for e in sp_el:
            el[e] = el.get(e, 0.0) - spike_wt * spike_comp.get_wt_fraction(e)
        return ({e: v / (1.0 - spike_wt) for e, v in el.items()},
                1.0 / (1.0 - spike_wt), "spike element mass subtracted")
    if sp_el and not (sp_el & native):             # foreign: drop + reclose
        keep = {c: v for c, v in component_wt.items()
                if not (_cations(c) <= sp_el)}
        tot = sum(keep.values())
        if tot <= 0:
            return {}, float("nan"), "nothing left after removing the spike"
        f = 100.0 / tot
        return (_element_fractions({c: v * f for c, v in keep.items()}), f,
                "spike components dropped, analysis re-closed")
    return {}, float("nan"), "spike shares only some elements with the powder"


def g_elements(el_orig: dict, factor, sigma_abs: dict, weights: dict, comps: dict,
               exclude=()):
    """Per-element estimates of G from single-carrier elements.

    ``sigma_abs[element]`` is the absolute 1-sigma of the element mass fraction
    in the analysed specimen. Returns a list of (element, phase, G, sigma)."""
    out = []
    present = [n for n, v in weights.items() if v > 0]
    for e in sorted(el_orig):
        if e in exclude:
            continue
        carriers = [n for n in present
                    if e in {x.symbol for x in comps[n].elements}]
        if len(carriers) != 1:
            continue
        p = carriers[0]
        f = comps[p].get_wt_fraction(e)
        if f <= 0 or weights[p] <= 0:
            continue
        x_p = el_orig[e] / f
        g = 1.0 - x_p / weights[p]
        sig = sigma_abs.get(e, float("nan")) * factor / f / weights[p]
        out.append((e, p, float(g), float(sig)))
    return out


def chemistry_scale(est, gkey, min_groups=2):
    """Birge-type factor (>= 1) on the element-balance sigmas.

    Standardless chemistry is often biased beyond its stated accuracy, and the
    bias is shared by a group. Where a group has both a diffraction estimate
    (standard addition / internal standard) and an element balance, their
    difference measures that bias; λ is chosen so the differences are
    consistent (chi2 per group = 1) given both sigmas. Needs ``min_groups``
    such groups, else 1."""
    diffs = []
    for gk in set(gkey.values()):
        rows = [r for r in est if gkey[r["sample"]] == gk and r["used"]]
        xr = [(r["G"], r["sigma"]) for r in rows if r["estimator"] != "element_balance"]
        ch = [(r["G"], r["sigma"]) for r in rows if r["estimator"] == "element_balance"]
        if not xr or not ch:
            continue
        gx, sx, _ = pool(xr)
        gc = float(np.mean([g for g, _ in ch]))
        sc = float(np.mean([e for _, e in ch]))
        if np.isfinite(gx) and np.isfinite(gc) and sc > 0:
            diffs.append((gc - gx, sc, sx))
    if len(diffs) < min_groups:
        return 1.0

    def chi2(lam):
        return float(np.mean([d * d / (lam * lam * sc * sc + sx * sx)
                              for d, sc, sx in diffs]))

    if chi2(1.0) <= 1.0:
        return 1.0
    lo, hi = 1.0, 2.0
    while chi2(hi) > 1.0 and hi < 1e3:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if chi2(mid) > 1.0 else (lo, mid)
    return float(hi)


def pool(estimates):
    """Inverse-variance mean of [(G, sigma)], with chi2 per degree of freedom."""
    est = [(g, s) for g, s in estimates if np.isfinite(g) and np.isfinite(s) and s > 0]
    if not est:
        return float("nan"), float("nan"), float("nan")
    g = np.array([e[0] for e in est])
    w = 1.0 / np.array([e[1] for e in est]) ** 2
    m = float(np.sum(w * g) / w.sum())
    sig = float(1.0 / np.sqrt(w.sum()))
    chi2 = float(np.sum(w * (g - m) ** 2) / (len(g) - 1)) if len(g) > 1 else float("nan")
    return m, sig, chi2


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

_SPIKED_HINTS = ("spik", "std", "standard", "added", "addition", "doped")
_SPREAD_HINTS = ("dev", "sigma", "err", "uncert", "spread", "stdev", "_sd")


def _load_json(obj):
    import json
    from pathlib import Path
    if isinstance(obj, (str, Path)) and Path(obj).is_file():
        return json.loads(Path(obj).read_text()), Path(obj).parent
    return obj, None


def _unwrap(obj):
    if isinstance(obj, dict) and isinstance(obj.get("samples"), dict):
        return obj["samples"]
    return obj


def parse_specimens(specimens, base_dir=None):
    """{sample: {role: (two_theta, intensity)}} and {sample: group or None}."""
    from pathlib import Path
    obj, here = _load_json(specimens)
    base = Path(base_dir) if base_dir is not None else here
    obj = _unwrap(obj)
    if not isinstance(obj, dict):
        raise ValueError("specimens must be {sample: {role: file or (2θ, I)}}, "
                         "or a JSON file of that shape")
    pats, groups = {}, {}
    for s, v in obj.items():
        if not isinstance(v, dict):
            v = {"neat": v}
        grp = v.get("group", v.get("batch"))
        spec = v.get("specimens", v.get("patterns", v))
        roles = {}
        for role, src in spec.items():
            if role in ("group", "batch") or src is None:
                continue
            if isinstance(src, (str, Path)):
                path = Path(src)
                if not path.is_absolute() and base is not None:
                    path = base / path
                roles[str(role)] = Q.read_pattern(path)
            elif isinstance(src, dict):
                continue
            else:
                tt, y = src
                roles[str(role)] = (np.asarray(tt, float), np.asarray(y, float))
        if roles:
            pats[str(s)] = roles
            groups[str(s)] = None if grp is None else str(grp)
    return pats, groups


def spiked_role(roles, std_entry):
    """Which role of a sample carries the standard (None if no standard)."""
    if std_entry is None:
        return None
    named = std_entry.get("specimen") if isinstance(std_entry, dict) else None
    if named and named in roles:
        return named
    hits = [r for r in roles if any(h in r.lower() for h in _SPIKED_HINTS)]
    if len(hits) == 1:
        return hits[0]
    if len(roles) == 1:
        return next(iter(roles))
    raise ValueError(f"cannot tell which of {sorted(roles)} is the spiked "
                     f"specimen; name it in the standard entry ('specimen')")


def _is_composition(d):
    from pymatgen.core import Composition
    if not isinstance(d, dict) or not d:
        return False
    try:
        for k, v in d.items():
            Composition(str(k))
            float(v)
    except Exception:
        return False
    return True


def parse_chemistry(entry):
    """(values, spread, specimen) from one sample's chemistry entry."""
    if entry is None:
        return None, {}, None
    if _is_composition(entry):
        return {str(k): float(v) for k, v in entry.items()}, {}, None
    values, spread = None, {}
    for k, v in entry.items():
        if not _is_composition(v):
            continue
        if any(h in k.lower() for h in _SPREAD_HINTS):
            spread = {str(a): float(b) for a, b in v.items()}
        elif values is None:
            values = {str(a): float(b) for a, b in v.items()}
    spec = entry.get("specimen") if isinstance(entry.get("specimen"), str) else None
    return values, spread, spec


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def _joint_job(patterns, phase_sets, sticks, props, radiation, opts, jopts,
               particle_um):
    return fit_joint(patterns, phase_sets, sticks, props, radiation, opts, jopts,
                     particle_um)


def run(structs: dict, pats: dict, groups: dict, standards: dict, chemistry: dict,
        chem_sigma, radiation, fo: Q.FitOptions, *, particle_um=None,
        amorphous_exclude=(), phases=None, ratio_sigma=0.015,
        max_absorption_contrast=0.5, detection_limit=0.03, n_jobs=1,
        blas_threads=1, jopts: JointOptions | None = None, log=print,
        std_entries=None):
    """Quantify every sample. Returns (rows, estimator rows, group rows,
    specimen rows)."""
    samples = sorted(pats)
    std_entries = std_entries or {}
    std_role = {s: (spiked_role(pats[s], std_entries.get(s) or {})
                    if s in standards else None) for s in samples}
    neat_role = {}
    for s in samples:
        others = [r for r in pats[s] if r != std_role[s]]
        if len(others) > 1:
            raise ValueError(f"{s}: more than one unspiked specimen {others}")
        neat_role[s] = others[0] if others else None

    libs = {}

    def lib_for(tt):
        key = (len(tt), round(float(tt[0]), 6), round(float(tt[-1]), 6))
        if key not in libs:
            libs[key] = Q.Library(structs, radiation, (float(tt[0]), float(tt[-1])),
                                  fo, solid_solutions=False)
        return libs[key]

    # 1. identification on the neat specimen (spiked one, with the standard
    #    forced, if the sample has no neat pattern)
    items, who = [], []
    for s in samples:
        if phases is not None:
            continue
        role = neat_role[s] or std_role[s]
        tt, y = pats[s][role]
        sp = standards[s][0] if role == std_role[s] and standards.get(s) else None
        items.append(((f"{s}:{role}", tt, y, lib_for(tt)),
                      dict(std_phase=sp, std_wt=standards[s][1] if sp else 0.0,
                           opts=fo)))
        who.append(s)
    ident = {}
    if items:
        log(f"identifying phases in {len(items)} pattern(s) "
            f"({n_jobs} worker(s))")
        for s, f in zip(who, Q.run_fits(items, n_jobs=n_jobs,
                                        blas_threads=blas_threads)):
            ident[s] = f
    found = {}
    for s in samples:
        if phases is not None:
            found[s] = [p for p in phases]
            continue
        f = ident[s]
        sp = standards[s][0] if standards.get(s) else None
        keep = [n for n, w in f.weights_rel.items() if w > 0]
        if neat_role[s] is None and sp in keep:
            keep.remove(sp)                    # spiked-only: the standard is added
        found[s] = sorted(keep)
        log(f"{s}: identified {', '.join(found[s])}")

    # 2. joint refinement of each sample's specimens
    jobs, roles_of = [], {}
    for s in samples:
        roles, sets = [], []
        sp = standards[s][0] if standards.get(s) else None
        if neat_role[s]:
            roles.append(neat_role[s])
            sets.append(list(found[s]))
        if std_role[s]:
            roles.append(std_role[s])
            sets.append(sorted(set(found[s]) | {sp}))
        tts = [pats[s][r] for r in roles]
        lib = lib_for(tts[0][0])
        names = sorted(set().union(*sets))
        sticks = {n: Q.compute_sticks(structs[n], radiation.primary,
                                      lib.tt_range, fo.po_axis, fo.peak_cutoff)
                  for n in names}
        props = {n: (lib.zmv[n], lib.mu[n], lib.rho[n]) for n in names}
        d = (particle_um or {}).get(s)
        jobs.append(((tts, sets, sticks, props, radiation, fo, jopts,
                      float(d) if d else None), {}))
        roles_of[s] = roles
    log(f"joint refinement of {len(jobs)} sample(s)")
    joint = dict(zip(samples, Q.run_fits(jobs, n_jobs=n_jobs,
                                         blas_threads=blas_threads,
                                         fn=_joint_job)))
    spec_rows, rel0, rel1 = [], {}, {}
    for s in samples:
        fits, info = joint[s]
        roles = roles_of[s]
        for f, r in zip(fits, roles):
            spec_rows.append({"sample": s, "specimen": r, "phases": list(f.names),
                              "weights_rel": f.weights_rel, "rwp": f.rwp,
                              "zero_shift": f.zero, "displacement_coef": f.displacement,
                              "brindley_tau": f.tau, "b_iso": info["b_iso"],
                              "width_factor": info["width_factor"],
                              "po_ratio": info["po_ratio"]})
        byrole = dict(zip(roles, fits))
        if neat_role[s]:
            rel0[s] = dict(byrole[neat_role[s]].weights_rel)
        if std_role[s]:
            rel1[s] = dict(byrole[std_role[s]].weights_rel)
        if s not in rel0:                      # spiked only: drop the standard
            sp = standards[s][0]
            rest = {n: v for n, v in rel1[s].items() if n != sp}
            t = sum(rest.values()) or 1.0
            rel0[s] = {n: v / t for n, v in rest.items()}
        if any(info["b_iso"][n] >= (jopts or JointOptions()).b_range[1] - 1e-6
               for n in info["b_iso"]):
            log(f"{s}: a Debye-Waller B hit its upper bound — check the phase set")

    # 3. estimators
    lib0 = next(iter(libs.values())) if libs else None
    est = []
    for s in samples:
        g_ = groups.get(s)
        if std_role[s]:
            sp, w = standards[s]
            if sp in rel0[s] and rel0[s][sp] > 0 and neat_role[s]:
                g, e = g_iso(rel0[s][sp], rel1[s][sp], w, ratio_sigma)
                est.append(dict(sample=s, group=g_, estimator="iso_spike",
                                detail=sp, G=g, sigma=e, used=bool(np.isfinite(g)),
                                note="" if np.isfinite(g) else
                                "spiked ratio not above neat ratio"))
            elif sp not in rel0[s] or neat_role[s] is None:
                z = rel1[s].get(sp, 0.0)
                g, e = g_xeno(z, w, ratio_sigma)
                mu = {n: lib0.mu[n] for n in rel0[s]}
                rho = {n: lib0.rho[n] for n in rel0[s]}
                c = absorption_contrast(lib0.mu[sp], mu, rel0[s], rho)
                sized = bool((particle_um or {}).get(s))
                ok = bool(np.isfinite(g)) and (sized or not np.isfinite(c)
                                               or c <= max_absorption_contrast)
                note = (f"absorption contrast {c:.2f}"
                        + ("" if ok or not np.isfinite(g) else
                           f" > {max_absorption_contrast}: microabsorption-biased "
                           f"without particle sizes, not used")
                        + ("; Brindley-corrected" if sized else ""))
                est.append(dict(sample=s, group=g_, estimator="internal_standard",
                                detail=sp, G=g, sigma=e, used=ok, note=note))
        chem = chemistry.get(s)
        if chem and chem[0]:
            values, spread, spec = chem
            spec = spec if spec in pats[s] else (neat_role[s] or std_role[s])
            sp, w, comp_k = None, 0.0, None
            if spec == std_role[s] and standards.get(s):
                sp, w = standards[s]
                comp_k = lib0.comps[sp]
            tot = sum(values.values())
            f = 100.0 / tot if 0 < tot <= 1.5 else 1.0     # fractions -> percent
            values = {c: v * f for c, v in values.items()}
            from pymatgen.core import Composition
            sig_el = {}
            for comp, v in values.items():
                sc = float(np.hypot(float(chem_sigma) * f, spread.get(comp, 0.0) * f))
                cc = Composition(comp)
                for e in cc.elements:
                    if e.symbol == "O" and len(cc.elements) > 1:
                        continue
                    sig_el[e.symbol] = float(np.hypot(
                        sig_el.get(e.symbol, 0.0),
                        sc / 100.0 * cc.get_wt_fraction(e)))
            native = {e.symbol for n in rel0[s] for e in lib0.comps[n].elements} - {"O"}
            el, fac, note = original_chemistry(values, sp, w, comp_k, native)
            ge = g_elements(el, fac, sig_el, rel0[s], lib0.comps,
                            exclude=amorphous_exclude)
            best = max(ge, key=lambda t: t[2]) if (ge and not amorphous_exclude) else None
            for e, p, g, sg in ge:
                used = (best is None) or (e == best[0])
                est.append(dict(sample=s, group=g_, estimator="element_balance",
                                detail=f"{e} in {p}", G=g, sigma=sg, used=used,
                                note=note + ("" if used else
                                             "; lower than another element's "
                                             "estimate (element likely also in "
                                             "the amorphous fraction)")))

    # 4. pooling within groups (each sample its own group if none given)
    gkey = {s: (groups.get(s) if groups.get(s) is not None else f"__{s}")
            for s in samples}
    lam = chemistry_scale(est, gkey)
    if lam > 1.0:
        log(f"element-balance sigmas scaled x{lam:.2f}: chemistry disagrees with "
            f"the diffraction estimators by more than its stated accuracy")
        for r in est:
            if r["estimator"] == "element_balance":
                r["sigma"] = r["sigma"] * lam
                r["note"] = (r["note"] + f"; sigma x{lam:.2f}").lstrip("; ")
    group_rows, G = [], {}
    for gk in sorted(set(gkey.values())):
        mem = [s for s in samples if gkey[s] == gk]
        rows = [r for r in est if r["sample"] in mem and r["used"]]
        terms = [(r["G"], r["sigma"]) for r in rows if r["estimator"] != "element_balance"]
        ch = [(r["G"], r["sigma"]) for r in rows if r["estimator"] == "element_balance"]
        if ch:
            # systematic within a group: averaged, entered once
            terms.append((float(np.mean([g for g, _ in ch])),
                          float(np.mean([e for _, e in ch]))))
        g, sg, chi2 = pool(terms)
        G[gk] = g
        group_rows.append(dict(group=None if gk.startswith("__") else gk,
                               samples=mem, G=g, sigma=sg, chi2=chi2,
                               n_estimates=len(terms)))
        log(f"{'group ' + gk if not gk.startswith('__') else gk[2:]}: "
            f"amorphous {g:.4f} ± {sg:.4f} from {len(terms)} estimate(s)")

    # 5. report
    out = []
    for s in samples:
        g = G[gkey[s]]
        comp = {}
        if np.isfinite(g):
            g = float(np.clip(g, 0.0, 1.0))
            comp = {n: (1.0 - g) * v for n, v in rel0[s].items()}
            comp["amorphous"] = g
        else:
            log(f"{s}: no absolute estimator — relative crystalline weights only")
            comp = dict(rel0[s])
        kept = {n: v for n, v in comp.items() if v >= detection_limit}
        t = sum(kept.values()) or 1.0
        for n, v in sorted(kept.items(), key=lambda kv: -kv[1]):
            out.append((s, n, float(v / t)))
    return out, est, group_rows, spec_rows

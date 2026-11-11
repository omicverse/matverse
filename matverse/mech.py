"""``mv.mech`` — atomistic solid mechanics: stiffness, cleavage, stability.

The quantities a fracture, dislocation or indentation calculation is built on,
computed with any ASE calculator on a single ``Atoms`` object:

* :func:`elastic_constants` — the stiffness tensor by finite strain, with the
  **internal (inner) relaxation** of the basis switched on or off. For a lattice
  with more than one atom per primitive cell (diamond, wurtzite, hcp, most
  compounds) the clamped-ion and relaxed-ion constants differ — in diamond-
  structure crystals C44 by a factor of up to two — and only the relaxed ones
  describe a continuum that is in equilibrium with its own atoms.
* :func:`cleavage_energies` — every inequivalent way of cutting a crystal along
  one plane, with its unrelaxed and relaxed surface energy, so the termination
  a crack actually selects (the lowest) is found rather than assumed.
* :func:`check_gradient` — central finite differences of an energy against an
  analytic derivative (forces, configurational forces, stresses). Every hand-
  derived derivative in a coupled atomistic model should pass this before it
  drives a solver.
* :func:`bracket_fold` — quasi-static continuation of an equilibrium branch in
  a load parameter until it ceases to exist (snap-through, crack advance or
  healing, buckling), bisected to a requested load resolution, returning the
  last on-branch and first off-branch states.

These functions take and return plain objects (``Atoms``, arrays, dicts) rather
than depositing into an AnnData: they act on one structure inside a larger
calculation, not on a dataset of materials. ``mv.prop.elastic`` is the
dataset-level counterpart.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from ._registry import register_function

#: eV/angstrom^3 -> GPa.
_EV_PER_A3_TO_GPA = 160.21766208
#: eV/angstrom^2 -> J/m^2.
_EV_PER_A2_TO_J_PER_M2 = 16.021766208
#: Voigt order used by ASE: xx, yy, zz, yz, xz, xy.
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _optimizer(name: str):
    from ase import optimize
    try:
        return getattr(optimize, name)
    except AttributeError as exc:  # pragma: no cover - user error path
        raise ValueError(f"unknown ASE optimizer {name!r}") from exc


def _relax_positions(atoms, fmax: float, steps: int, optimizer: str) -> tuple:
    """Relax positions at fixed cell; return (converged, max force)."""
    opt = _optimizer(optimizer)(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    fmax_now = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    return converged or fmax_now <= fmax, fmax_now


def _strain_matrix(j: int, amount: float) -> np.ndarray:
    """Symmetric small strain for Voigt component j (engineering shear)."""
    a, b = _VOIGT[j]
    eps = np.zeros((3, 3))
    if a == b:
        eps[a, a] = amount
    else:
        eps[a, b] = eps[b, a] = amount / 2.0
    return eps


def _stress_at(atoms, calculator, F: np.ndarray, relax: bool, fmax: float,
               steps: int, optimizer: str) -> tuple:
    deformed = atoms.copy()
    deformed.calc = calculator
    deformed.set_cell(np.asarray(atoms.get_cell()) @ F.T, scale_atoms=True)
    ok, fnow = True, 0.0
    if relax:
        ok, fnow = _relax_positions(deformed, fmax, steps, optimizer)
    return np.asarray(deformed.get_stress(voigt=True), dtype=float), ok, fnow


def _cubic_summary(C: np.ndarray) -> dict:
    c11 = float(np.mean([C[0, 0], C[1, 1], C[2, 2]]))
    c12 = float(np.mean([C[0, 1], C[0, 2], C[1, 2]]))
    c44 = float(np.mean([C[3, 3], C[4, 4], C[5, 5]]))
    off = np.abs(C - _cubic_tensor(c11, c12, c44)).max()
    return {"C11": c11, "C12": c12, "C44": c44,
            "bulk_modulus": (c11 + 2.0 * c12) / 3.0,
            "zener_anisotropy": 2.0 * c44 / (c11 - c12) if c11 != c12 else np.inf,
            "max_deviation_from_cubic": float(off)}


def _cubic_tensor(c11: float, c12: float, c44: float) -> np.ndarray:
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    np.fill_diagonal(C[:3, :3], c11)
    C[3, 3] = C[4, 4] = C[5, 5] = c44
    return C


@register_function(
    aliases=["relaxed elastic constants",
             "relaxed-ion elastic constants", "clamped-ion elastic constants",
             "internal relaxation stiffness", "inner displacement elastic",
             "C11 C12 C44", "stiffness of one structure"],
    category="mech",
    description="Elastic stiffness tensor (GPa) of one ASE structure by central "
                "finite strains, with the internal relaxation of the atomic "
                "basis at each strain switched on (relaxed-ion) or off "
                "(clamped-ion), plus cubic C11/C12/C44 when the cell axes are "
                "the cubic axes.",
    examples=["res = mv.mech.elastic_constants(atoms, calc)",
              "res = mv.mech.elastic_constants(atoms, calc, relax_internal=True,"
              " also_clamped=True); res['C11'], res['C44'], "
              "res['clamped']['C44']"],
    related=["mv.prop.elastic", "mv.mech.cleavage_energies"],
    notes="Relax the reference cell first (lattice constant at the potential's "
          "own minimum): the residual stress is reported, and a stiffness "
          "taken about a stressed reference carries that stress in it. For a "
          "multilattice the relaxed-ion constants are the ones a continuum "
          "(LEFM, Cauchy-Born far field) must use; the clamped-ion ones are "
          "stiffer, mostly in shear. The cubic summary averages the three "
          "equivalent entries and reports how far the tensor is from cubic "
          "form, so a cell that is not aligned with the cubic axes is visible.",
)
def elastic_constants(atoms, calculator=None, strain: float = 1e-3,
                      relax_internal: bool = True, also_clamped: bool = False,
                      fmax: float = 1e-5, steps: int = 1000,
                      optimizer: str = "BFGS") -> dict:
    """Stiffness tensor of one structure, relaxed-ion by default.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic reference cell, ideally stress-free.
    calculator : ase calculator, optional
        Defaults to ``atoms.calc``.
    strain : float
        Amplitude of each central-difference strain (engineering shear for
        the Voigt shear components).
    relax_internal : bool
        Relax atomic positions at fixed strained cell before reading stress.
    also_clamped : bool
        Additionally return the clamped-ion tensor under ``'clamped'``.
    fmax, steps, optimizer
        Convergence of the inner relaxation (ASE optimizer name).

    Returns
    -------
    dict with ``C`` (6x6, GPa, ASE Voigt order xx,yy,zz,yz,xz,xy), cubic
    ``C11``/``C12``/``C44``/``bulk_modulus``/``zener_anisotropy``,
    ``residual_stress`` (GPa, reference cell), ``relax_internal``,
    ``max_inner_force`` and ``all_inner_relaxations_converged``.
    """
    calc = calculator if calculator is not None else atoms.calc
    if calc is None:
        raise ValueError("pass a calculator or attach one to atoms")
    ref = atoms.copy()
    ref.calc = calc
    residual = np.asarray(ref.get_stress(voigt=True)) * _EV_PER_A3_TO_GPA

    def tensor(relax: bool):
        C = np.zeros((6, 6))
        ok_all, worst = True, 0.0
        for j in range(6):
            eps = _strain_matrix(j, strain)
            sp, ok1, f1 = _stress_at(ref, calc, np.eye(3) + eps, relax, fmax,
                                     steps, optimizer)
            sm, ok2, f2 = _stress_at(ref, calc, np.eye(3) - eps, relax, fmax,
                                     steps, optimizer)
            C[:, j] = (sp - sm) / (2.0 * strain)
            ok_all &= ok1 and ok2
            worst = max(worst, f1, f2)
        C = 0.5 * (C + C.T) * _EV_PER_A3_TO_GPA
        return C, ok_all, worst

    C, ok, worst = tensor(relax_internal)
    out = {"C": C, **_cubic_summary(C), "residual_stress": residual,
           "relax_internal": bool(relax_internal), "strain": strain,
           "max_inner_force": worst, "all_inner_relaxations_converged": ok}
    if also_clamped:
        Cc, _, _ = tensor(False)
        out["clamped"] = {"C": Cc, **_cubic_summary(Cc)}
    return out


def _plane_heights(h: np.ndarray, period: float, tol: float) -> np.ndarray:
    """Distinct plane heights in [0, period), sorted."""
    h = np.sort(np.mod(h, period))
    planes = [h[0]]
    for value in h[1:]:
        if value - planes[-1] > tol:
            planes.append(value)
    if len(planes) > 1 and (planes[0] + period - planes[-1]) <= tol:
        planes.pop()
    return np.asarray(planes)


@register_function(
    aliases=["cleavage energies", "cleavage termination energy", "surface termination",
             "lowest energy cleavage plane", "fracture surface energy",
             "which cut of a plane", "shuffle glide termination",
             "surface energy of a cut"],
    category="mech",
    description="Enumerate every inequivalent cut of a crystal along one "
                "(hkl) plane — one per distinct inter-plane gap — and return "
                "the unrelaxed and position-relaxed surface energy (J/m^2) of "
                "each, lowest relaxed first.",
    examples=["cuts = mv.mech.cleavage_energies(bulk, (1, 1, 1), calc)",
              "gamma = cuts[0]['gamma_relaxed']   # the termination a crack "
              "selects"],
    related=["mv.surf.surface_energy", "mv.mech.elastic_constants"],
    notes="A plane family can usually be cut in more than one place: between "
          "widely spaced planes (few bonds cut) or narrowly spaced ones (many). "
          "Fracture follows the lowest-energy cut, so the Griffith load uses "
          "that gamma. Each slab has two identical surfaces made by the same "
          "cut; gamma = (E_slab - N e_bulk) / (2 A) with e_bulk from the same "
          "oriented periodic cell, so the reference is consistent. Converge "
          "min_thickness and vacuum; relaxation is of positions at fixed cell "
          "and is local (no reconstruction search).",
)
def cleavage_energies(bulk, miller: Sequence[int], calculator=None,
                      min_thickness: float = 20.0, vacuum: float = 12.0,
                      relax: bool = True, fmax: float = 1e-3, steps: int = 2000,
                      optimizer: str = "BFGS", tol: float = 0.05) -> list:
    """Surface energy of every inequivalent cut along ``miller``.

    Parameters
    ----------
    bulk : ase.Atoms
        Bulk crystal (conventional or primitive) at its equilibrium lattice
        constant for this calculator.
    miller : (h, k, l)
        Plane indices with respect to ``bulk``'s cell.
    calculator : ase calculator, optional
        Defaults to ``bulk.calc``.
    min_thickness : float
        Minimum slab thickness along the normal, angstrom.
    vacuum : float
        Vacuum added along the normal, angstrom.
    relax : bool
        Also relax positions of each slab and report ``gamma_relaxed``.
    tol : float
        Height tolerance (angstrom) for grouping atoms into planes.

    Returns
    -------
    list of dicts, sorted by relaxed (else unrelaxed) energy, each with
    ``gap`` (width of the gap cut, angstrom), ``cut_height``,
    ``gamma_unrelaxed``, ``gamma_relaxed`` (J/m^2), ``n_atoms``, ``area``,
    ``slab`` (the relaxed slab, or unrelaxed when ``relax=False``),
    ``converged``.
    """
    from ase.build import surface

    calc = calculator if calculator is not None else bulk.calc
    if calc is None:
        raise ValueError("pass a calculator or attach one to bulk")
    # The smallest number of layers that makes a *periodic* oriented cell
    # depends on the stacking (ABC along fcc/diamond <111> needs three):
    # ase.build.surface(periodic=True) returns a mis-stacked cell otherwise.
    # Accept the first one whose energy per atom equals the bulk's.
    probe = bulk.copy()
    probe.calc = calc
    e_ref = probe.get_potential_energy() / len(probe)
    unit = None
    for layers in range(1, 25):
        trial = surface(bulk, tuple(int(i) for i in miller), layers,
                        periodic=True)
        trial.calc = calc
        if abs(trial.get_potential_energy() / len(trial) - e_ref) < 1e-6:
            unit = trial
            break
    if unit is None:
        raise ValueError(f"no periodic oriented cell found for {miller}; "
                         "pass a conventional bulk cell")
    a1, a2, u3 = np.asarray(unit.get_cell())
    normal = np.cross(a1, a2)
    area = float(np.linalg.norm(normal))
    normal /= area
    repeats = max(1, int(np.ceil(min_thickness / abs(u3 @ normal))))
    cell_atoms = unit.repeat((1, 1, repeats))
    a3 = np.asarray(cell_atoms.get_cell())[2]
    period = float(abs(a3 @ normal))

    ref = cell_atoms.copy()
    ref.calc = calc
    e_bulk = ref.get_potential_energy() / len(ref)

    heights = ref.get_positions() @ normal
    planes = _plane_heights(heights, period, tol)
    gaps = np.diff(np.append(planes, planes[0] + period))
    symbols = np.asarray(ref.get_chemical_symbols())

    def plane_key(level: float) -> tuple:
        on = np.abs(((heights - level + period / 2) % period) - period / 2) <= tol
        return tuple(sorted(symbols[on]))

    seen, candidates = set(), []
    for i, gap in enumerate(gaps):
        below, above = planes[i], planes[(i + 1) % len(planes)]
        key = (round(float(gap) / max(tol, 1e-6)), plane_key(below),
               plane_key(above))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((float(gap), float(below + gap / 2.0)))

    out = []
    for gap, cut in candidates:
        slab = ref.copy()
        pos = slab.get_positions()
        h = pos @ normal
        shift = np.floor((h - cut) / period)
        sign = 1.0 if a3 @ normal > 0 else -1.0
        pos -= sign * shift[:, None] * a3[None, :]
        pos -= (cut * normal)[None, :]
        slab.set_positions(pos)
        slab.set_cell([a1, a2, normal * (period + vacuum)], scale_atoms=False)
        slab.pbc = (True, True, True)
        slab.center(axis=2)
        slab.calc = calc
        e0 = slab.get_potential_energy()
        g0 = (e0 - len(slab) * e_bulk) / (2.0 * area) * _EV_PER_A2_TO_J_PER_M2
        entry = {"gap": gap, "cut_height": cut, "gamma_unrelaxed": float(g0),
                 "n_atoms": len(slab), "area": area, "converged": True}
        if relax:
            ok, _ = _relax_positions(slab, fmax, steps, optimizer)
            e1 = slab.get_potential_energy()
            entry["gamma_relaxed"] = float(
                (e1 - len(slab) * e_bulk) / (2.0 * area) * _EV_PER_A2_TO_J_PER_M2)
            entry["converged"] = ok
        entry["slab"] = slab
        out.append(entry)
    key = "gamma_relaxed" if relax else "gamma_unrelaxed"
    return sorted(out, key=lambda e: e[key])


@register_function(
    aliases=["check gradient", "finite difference check", "gradient check",
             "verify analytic forces", "verify derivative",
             "configurational force check", "fd test"],
    category="mech",
    description="Compare an analytic derivative of a scalar energy against a "
                "central finite difference on sampled components, and report "
                "the worst relative and absolute error.",
    examples=["rep = mv.mech.check_gradient(E, x, analytic=-forces.ravel(), "
              "n_samples=30)",
              "rep = mv.mech.check_gradient(E, x, gradient=dE, h=1e-4); "
              "rep['max_rel_err']"],
    related=["mv.mech.bracket_fold"],
    notes="Pass forces as analytic=-forces (forces are minus the gradient), or "
          "set sign=-1. The relative error is |analytic - fd| / max(|fd|, "
          "floor): components whose derivative is near zero are judged on the "
          "absolute scale floor instead. Pick h so that the energy change is "
          "well above round-off (1e-4 to 1e-3 angstrom for interatomic "
          "energies of a few thousand atoms in float64); check two values of h "
          "— an error that does not shrink with h is a real bug, one that does "
          "is truncation.",
)
def check_gradient(energy: Callable[[np.ndarray], float], x,
                   gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                   analytic=None, h: float = 1e-4,
                   indices: Optional[Sequence[int]] = None, n_samples: int = 20,
                   sign: float = 1.0, floor: float = 1e-3, seed: int = 0) -> dict:
    """Finite-difference audit of an analytic derivative.

    ``energy(x)`` takes an array of the shape of ``x``; the derivative is with
    respect to its flattened components. Supply the analytic derivative as an
    array (``analytic``) or a callable (``gradient``); ``sign=-1`` treats it as
    a force. Returns ``max_rel_err``, ``max_abs_err``, ``indices``,
    ``analytic``, ``fd`` and ``h``.
    """
    x = np.array(x, dtype=float)
    if analytic is None:
        if gradient is None:
            raise ValueError("give analytic= or gradient=")
        analytic = gradient(x)
    a = sign * np.asarray(analytic, dtype=float).ravel()
    if a.size != x.size:
        raise ValueError("analytic derivative must have the size of x")
    if indices is None:
        rng = np.random.default_rng(seed)
        k = min(int(n_samples), x.size)
        idx = rng.choice(x.size, size=k, replace=False)
    else:
        idx = np.asarray(indices, dtype=int).ravel()
    flat = x.ravel()
    fd = np.empty(len(idx))
    for n, i in enumerate(idx):
        xp = flat.copy(); xp[i] += h
        xm = flat.copy(); xm[i] -= h
        fd[n] = (energy(xp.reshape(x.shape)) - energy(xm.reshape(x.shape))) / (2 * h)
    an = a[idx]
    abs_err = np.abs(an - fd)
    rel = abs_err / np.maximum(np.abs(fd), floor)
    return {"max_rel_err": float(rel.max()), "max_abs_err": float(abs_err.max()),
            "rel_err": rel, "indices": idx, "analytic": an, "fd": fd, "h": h}


def _call_solve(solve, load, guess):
    result = solve(load, guess)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(
            result[1], (bool, np.bool_)):
        return result[0], bool(result[1])
    return result, True


def _distance(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim >= 2 and a.shape[-1] in (2, 3):
        return float(np.linalg.norm(a - b, axis=-1).max())
    return float(np.abs(a - b).max())


@register_function(
    aliases=["bracket fold", "fold bracketing", "turning point",
             "loss of equilibrium", "quasi-static ramp", "load continuation",
             "snap-through load", "lattice trapping limit",
             "instability load", "critical load bisection"],
    category="mech",
    description="Follow a quasi-static equilibrium branch in a load parameter "
                "from a converged seed until the branch is lost (solver fails "
                "or the state jumps), then bisect between the last on-branch "
                "and first off-branch load to a requested resolution.",
    examples=["res = mv.mech.bracket_fold(solve, load0=1.0, state0=x0, "
              "step=0.02, jump=0.5, resolution=0.01, direction=+1)",
              "res['fold'], res['trapped_state'], res['detrapped_state']"],
    related=["mv.mech.check_gradient"],
    notes="solve(load, guess) must return the equilibrium state reached from "
          "guess (or (state, converged)). The branch is 'kept' at a new load "
          "when the solver converged and observable(state) moved by at most "
          "`jump` from the previous on-branch state (max per-row norm for "
          "(N, 2|3) arrays, max abs otherwise) — choose `jump` well above the "
          "smooth drift per step and below the size of the event (for a crack, "
          "a fraction of a lattice period). Every probe is warm-started from "
          "the last on-branch state, never from an off-branch one, so the "
          "bracket is of the branch, not of the solver. Run it twice — "
          "direction=+1 and -1 — for the upper and lower fold of the same "
          "branch. 'fold' is the bracket midpoint.",
)
def bracket_fold(solve: Callable[[float, Any], Any], load0: float, state0,
                 step: float, jump: float, resolution: float = 0.01,
                 direction: int = 1,
                 observable: Optional[Callable[[Any], np.ndarray]] = None,
                 max_load: Optional[float] = None, max_steps: int = 500,
                 callback: Optional[Callable[[dict], None]] = None) -> dict:
    """Bracket the load at which an equilibrium branch ceases to exist.

    Returns a dict with ``trapped_load``, ``trapped_state``,
    ``detrapped_load``, ``detrapped_state``, ``fold`` (midpoint), ``width``,
    ``found`` (False if ``max_load`` / ``max_steps`` stopped the ramp first)
    and ``history`` (one record per solve: load, converged, distance,
    on_branch, phase).
    """
    obs = observable or (lambda s: s)
    d = 1.0 if direction >= 0 else -1.0
    history = []

    def probe(load, guess_state, phase):
        state, ok = _call_solve(solve, load, guess_state)
        dist = _distance(obs(state), obs(guess_state)) if ok else np.inf
        on = bool(ok and dist <= jump)
        rec = {"load": float(load), "converged": ok, "distance": dist,
               "on_branch": on, "phase": phase}
        history.append(rec)
        if callback is not None:
            callback(rec)
        return state, on

    lo, s_lo = float(load0), state0
    hi, s_hi = None, None
    for _ in range(int(max_steps)):
        nxt = lo + d * step
        if max_load is not None and (nxt - max_load) * d > 0:
            break
        s_new, on = probe(nxt, s_lo, "ramp")
        if on:
            lo, s_lo = nxt, s_new
        else:
            hi, s_hi = nxt, s_new
            break
    if hi is None:
        return {"found": False, "trapped_load": lo, "trapped_state": s_lo,
                "detrapped_load": None, "detrapped_state": None, "fold": None,
                "width": None, "history": history}

    while abs(hi - lo) > resolution:
        mid = 0.5 * (lo + hi)
        s_mid, on = probe(mid, s_lo, "bisect")
        if on:
            lo, s_lo = mid, s_mid
        else:
            hi, s_hi = mid, s_mid
    return {"found": True, "trapped_load": lo, "trapped_state": s_lo,
            "detrapped_load": hi, "detrapped_state": s_hi,
            "fold": 0.5 * (lo + hi), "width": abs(hi - lo), "history": history}


__all__ = ["elastic_constants", "cleavage_energies", "check_gradient",
           "bracket_fold"]

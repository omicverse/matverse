"""``mv.neb`` — migration barriers.

A diffusivity from molecular dynamics tells you how fast an ion moves at the
temperature you simulated. A migration barrier tells you *why*, and extrapolates
to temperatures no simulation reached — which is why barriers, not diffusivities,
are what a solid-electrolyte screen ranks on.

The method is nudged elastic band: interpolate a chain of images between two
endpoints, relax them under a spring constraint that keeps them spread along the
path, and read the highest energy. With climbing-image enabled the top image
converges onto the saddle rather than near it.

Where the barrier lives
-----------------------
A barrier belongs to a **path**, not to a material, and a material has several.
That is a third axis, like sites, and matverse handles it the same way: the
scalar summaries — lowest barrier, mean barrier, number of paths — go onto
``obs`` where a screen can filter on them, and the full energy profile goes onto
the grid axis as ``obsm['neb_profile_{level}']``. The per-path detail is
returned rather than deposited, because it does not fit the material axis.

Read this before trusting a barrier
-----------------------------------
Universal machine-learned potentials **systematically soften the potential
energy surface and under-predict migration barriers**. Published benchmarks put
MACE and SevenNet at 0.07–0.08 eV mean absolute error against DFT for
transition-metal-free systems, rising to about 0.20 eV once transition metals
are involved — and the error has a sign, so a screen ranked on MLIP barriers
promotes candidates that DFT would reject. Every level records this in
``uns['levels'][level]['note']`` when the level is a surrogate.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from ._core import deposit_grid, record, set_level, structures
from ._registry import register_function


@register_function(
    aliases=["neb", "migration barrier", "nudged elastic band", "activation "
             "barrier", "diffusion barrier", "transition state", "saddle point"],
    category="neb",
    description="Compute the migration barrier between two endpoint structures "
                "for every material by climbing-image nudged elastic band, and "
                "record the barrier and the energy profile along the path.",
    requires={"structures": ["{initial}", "{final}"]},
    produces={"obs": ["barrier_{level}", "barrier_reverse_{level}",
                      "reaction_energy_{level}", "neb_converged_{level}"],
              "obsm": ["neb_profile_{level}"],
              "uns": ["grids"], "levels": ["{level}"]},
    prerequisites=["mv.calc.relax"],
    dispatch="level= selects the calculator, as for mv.calc.energy",
    examples=["mv.neb.barrier(md, initial='relaxed_emt', final='hopped_emt')",
              "mv.neb.barrier(md, initial='start', final='end', n_images=7)"],
    related=["mv.neb.hop_endpoints", "mv.md.conductivity", "mv.md.run"],
    notes="Endpoints must be relaxed and must correspond atom for atom. An NEB "
          "between two geometries whose atoms are ordered differently "
          "interpolates every atom into the wrong place and returns a large "
          "meaningless barrier.\n\n"
          "The reverse barrier is reported alongside the forward one because "
          "they differ whenever the endpoints are inequivalent, and quoting "
          "only one of them hides which direction the ion actually struggles "
          "to go.",
)
def barrier(md: AnnData, initial: str, final: str, level: str = "emt",
            n_images: int = 5, fmax: float = 0.1, steps: int = 100,
            climb: bool = True, spring: float = 0.1,
            interpolation: str = "idpp") -> None:
    """Climbing-image NEB between two structure variants.

    ``interpolation`` defaults to IDPP rather than linear, and the difference is
    not cosmetic. Straight-line interpolation moves every atom along a chord, so
    a hopping atom is driven through its neighbours rather than around them; the
    band starts with overlapping atoms, and the barrier it converges to — if it
    converges — is several eV of interatomic repulsion rather than a migration
    barrier. IDPP first relaxes the images under a pairwise-distance objective,
    which gives a path that goes around.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    from .calc import _get

    if n_images < 3:
        raise ValueError(f"n_images must be at least 3 (two endpoints and a "
                         f"midpoint), got {n_images}")

    factory, meta = _get(level)
    adaptor = AseAtomsAdaptor()
    coordinate = np.linspace(0.0, 1.0, n_images)

    forward, reverse, reaction, converged, profiles, failed = [], [], [], [], [], 0

    for start, end in zip(structures(md, initial), structures(md, final)):
        try:
            energies, ok = _run_neb(start, end, adaptor, factory, n_images,
                                    fmax, steps, climb, spring, interpolation)
        except Exception:
            failed += 1
            forward.append(np.nan); reverse.append(np.nan)
            reaction.append(np.nan); converged.append(False)
            profiles.append(np.full(n_images, np.nan))
            continue

        relative = energies - energies[0]
        forward.append(float(relative.max()))
        reverse.append(float((energies - energies[-1]).max()))
        reaction.append(float(relative[-1]))
        converged.append(bool(ok))
        profiles.append(relative)

    md.obs[f"barrier_{level}"] = forward
    md.obs[f"barrier_reverse_{level}"] = reverse
    md.obs[f"reaction_energy_{level}"] = reaction
    md.obs[f"neb_converged_{level}"] = converged
    deposit_grid(md, "neb_profile", level, np.vstack(profiles), coordinate,
                 unit="fractional path coordinate")

    set_level(md, level, **{**meta, **_softening_note(meta)},
              initial=initial, final=final, n_images=n_images,
              climb=bool(climb), interpolation=interpolation, n_failed=failed)
    record(md, "neb.barrier", level=level, initial=initial, final=final,
           n_images=n_images)


def _softening_note(meta: dict) -> dict:
    """Attach the surrogate-barrier caveat to the level that earned it.

    Appended to whatever note the calculator already carries rather than
    replacing it, so a level does not lose its own description by being used
    for a barrier.
    """
    if not meta.get("surrogate"):
        return {}
    caveat = (
        "Barriers from a surrogate potential are systematically low: "
        "machine-learned potentials soften the potential energy surface, with "
        "roughly 0.07-0.08 eV MAE against DFT for transition-metal-free "
        "systems and about 0.20 eV with transition metals. The error has a "
        "sign, so a screen ranked on these promotes candidates DFT would "
        "reject.")
    existing = str(meta.get("note") or "").strip()
    return {"note": f"{existing} {caveat}".strip()}


def _run_neb(start, end, adaptor, factory, n_images, fmax, steps, climb,
             spring, interpolation="idpp"):
    """One NEB. Returns the energies along the band and whether it converged."""
    from ase.mep.neb import NEB
    from ase.optimize import BFGS

    if len(start) != len(end):
        raise ValueError(f"endpoints have {len(start)} and {len(end)} atoms; "
                         f"an NEB needs a one-to-one correspondence")

    first = adaptor.get_atoms(start)
    last = adaptor.get_atoms(end)
    images = [first] + [first.copy() for _ in range(n_images - 2)] + [last]
    for image in images:
        image.calc = factory()

    # 'improvedtangent' rather than ASE's default 'aseneb'. ASE's own warning
    # calls that default "an unpublished, custom implementation that is not
    # recommended as it frequently results in very poor bands" — which is not
    # something to leave on and hope for. The improved-tangent formulation is
    # Henkelman and Jonsson (2000), doi:10.1063/1.1323224, and it is what the
    # barriers quoted in the documentation were computed with.
    try:
        band = NEB(images, k=spring, climb=climb, method="improvedtangent")
    except TypeError:                   # older ASE without the keyword
        band = NEB(images, k=spring, climb=climb)
    try:
        band.interpolate(method=interpolation)
    except Exception:
        band.interpolate()          # older ASE, or IDPP failing to converge
    optimiser = BFGS(band, logfile=None)
    converged = bool(optimiser.run(fmax=fmax, steps=steps))

    return np.array([image.get_potential_energy() for image in images]), converged


@register_function(
    aliases=["hop endpoints", "vacancy hop", "make endpoints", "migration "
             "path", "enumerate hops", "build neb endpoints"],
    category="neb",
    description="Build initial and final structures for a vacancy-mediated hop "
                "of one species, so a barrier can be computed without hand-"
                "editing two files per material.",
    requires={"structures": ["{source}"]},
    produces={"structures": ["{key_added}_initial", "{key_added}_final"],
              "obs": ["hop_distance", "hop_species"]},
    examples=["mv.neb.hop_endpoints(md, species='Cu')",
              "mv.neb.hop_endpoints(md, species='Li', supercell=(2, 2, 2))"],
    related=["mv.neb.barrier", "mv.pp.defects"],
    notes="Picks the shortest hop between two sites of the same species with a "
          "vacancy on the destination, which is the elementary event for "
          "vacancy-mediated diffusion. Interstitial and concerted mechanisms "
          "are different problems and are not built here.\n\n"
          "Relax both endpoints before running the NEB. A barrier measured "
          "from unrelaxed endpoints includes the relaxation energy and is not "
          "a barrier.",
)
def hop_endpoints(md: AnnData, species: str, source: str = "input",
                  supercell=(2, 2, 2), key_added: str = "hop") -> None:
    """Vacancy-mediated hop endpoints for one species."""
    from ._core import deposit_structures

    initial, final, distances, failed = [], [], [], 0

    for structure in structures(md, source):
        try:
            start, end, distance = _build_hop(structure, species, supercell)
        except Exception:
            failed += 1
            initial.append(structure)
            final.append(structure)
            distances.append(np.nan)
            continue
        initial.append(start)
        final.append(end)
        distances.append(distance)

    deposit_structures(md, f"{key_added}_initial", initial)
    deposit_structures(md, f"{key_added}_final", final)
    md.obs["hop_distance"] = distances
    md.obs["hop_species"] = [species] * md.n_obs
    md.uns.setdefault("hops", {})[key_added] = {
        "species": species, "supercell": list(supercell), "n_failed": failed,
        "mechanism": "vacancy-mediated, shortest same-species hop",
    }
    record(md, "neb.hop_endpoints", species=species, source=source,
           supercell=list(supercell), key_added=key_added)


def _build_hop(structure, species: str, supercell):
    """A vacancy at one site, and the same cell with a neighbour moved into it.

    The destination is the **nearest periodic image** of the vacancy, not its
    coordinates as stored. Those are different whenever the shortest hop crosses
    a cell boundary, and using the stored ones sends the atom the long way
    around: in a 2x2x2 fcc copper cell that turns a 2.55 angstrom hop into a
    7.66 angstrom traverse, and the NEB then measures the cost of dragging an
    atom through the lattice rather than a migration barrier.
    """
    cell = structure.copy()
    cell.make_supercell(list(supercell))

    sites = [i for i, site in enumerate(cell)
             if str(site.specie.symbol) == species]
    if len(sites) < 2:
        raise ValueError(f"{species} appears on {len(sites)} site(s) in the "
                         f"supercell; a hop needs at least two")

    vacancy = sites[0]
    neighbours = sorted(
        ((cell.get_distance(vacancy, j), j) for j in sites[1:]),
        key=lambda pair: pair[0])
    distance, mover = neighbours[0]

    # The image of the vacancy that is actually closest to the moving atom.
    _, image = cell.lattice.get_distance_and_image(cell[mover].frac_coords,
                                                   cell[vacancy].frac_coords)
    target = cell[vacancy].frac_coords + image

    start = cell.copy()
    start.remove_sites([vacancy])

    end = cell.copy()
    end.replace(mover, species, coords=target, coords_are_cartesian=False)
    end.remove_sites([vacancy])

    travelled = float(np.linalg.norm(
        end.cart_coords[_shift(mover, vacancy)] - start.cart_coords[
            _shift(mover, vacancy)]))
    if not np.isclose(travelled, distance, atol=0.05):
        raise ValueError(
            f"the moving atom travels {travelled:.2f} A but the hop is "
            f"{distance:.2f} A; the endpoints do not describe the intended "
            f"jump")

    return start, end, float(distance)


def _shift(index: int, removed: int) -> int:
    """An atom's index after an earlier site was removed."""
    return index - 1 if index > removed else index


__all__ = ["barrier", "hop_endpoints"]

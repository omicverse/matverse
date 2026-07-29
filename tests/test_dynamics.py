"""Molecular dynamics, migration barriers, surfaces and adsorption.

The barrier and surface-energy tests check against **published numbers and
orderings** rather than against stored output. EMT is a crude potential and gets
magnitudes wrong by a factor of about two on surfaces, so the tests assert what
EMT can be held to — the ordering of facets, the symmetry of a symmetric hop,
and a vacancy barrier that lands near the literature value — and say which is
which.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv


def _fcc(symbol: str, a: float):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [symbol] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


@pytest.fixture
def copper():
    return mv.data.from_structures([_fcc("Cu", 3.61)])


@pytest.fixture
def metals():
    return mv.data.from_structures([_fcc("Cu", 3.61), _fcc("Al", 4.05)])


class TestMolecularDynamics:
    @pytest.fixture
    def big(self):
        """A 32-atom cell. Four atoms cannot hold a temperature."""
        cell = _fcc("Cu", 3.61)
        cell.make_supercell([2, 2, 2])
        return mv.data.from_structures([cell])

    def test_the_temperature_trace_is_kept_not_just_its_mean(self, big):
        """A mean cannot distinguish a run that equilibrated from one still
        drifting at the last step, and that is the difference between a result
        and an artefact."""
        mv.md.run(big, level="emt", temperature=300.0, steps=400,
                  sample_every=20)
        trace = big.obsm["md_temperature_trace_emt"]
        assert trace.shape == (1, 20)
        assert float(trace[0].mean()) == pytest.approx(
            float(big.obs["md_temperature_emt"].iloc[0]), rel=1e-9)

    def test_a_second_run_replaces_an_incomparable_trace(self, big):
        """The trace axis is not chosen by the caller — its length falls out of
        steps and sample_every — so a rerun of different length produces a
        curve that cannot share the stored axis. The stale one goes rather than
        the deposit failing."""
        mv.md.run(big, level="emt", temperature=300.0, steps=400,
                  sample_every=20)
        mv.md.run(big, level="emt", temperature=300.0, steps=200,
                  sample_every=20)
        assert big.obsm["md_temperature_trace_emt"].shape == (1, 10)
        assert len(mv.grid_of(big, "md_temperature_trace")) == 10

    def test_a_sweep_after_a_run_is_not_blocked_by_the_trace(self, big):
        mv.md.run(big, level="emt", temperature=300.0, steps=400,
                  sample_every=20)
        mv.md.sweep(big, level="emt", temperatures=(300.0, 500.0), steps=200,
                    sample_every=20)
        assert big.obsm["md_volume_emt"].shape == (1, 2)

    def test_a_run_deposits_its_observables(self, big):
        mv.md.run(big, level="emt", temperature=300.0, steps=300,
                  sample_every=20)
        for column in ("md_energy_emt", "md_temperature_emt", "msd_emt",
                       "diffusivity_emt", "md_volume_emt"):
            assert column in big.obs
        assert "md_emt" in mv.variants(big)

    def test_the_thermostat_reaches_its_target(self, big):
        mv.md.run(big, level="emt", temperature=300.0, steps=400,
                  sample_every=20)
        achieved = float(big.obs["md_temperature_emt"].iloc[0])
        assert achieved == pytest.approx(300.0, rel=0.25)

    def test_it_says_so_when_the_thermostat_has_not_equilibrated(self, copper):
        """The achieved temperature is always recorded; this makes not noticing
        it harder."""
        with pytest.warns(UserWarning, match="requested"):
            mv.md.run(copper, level="emt", temperature=300.0, steps=200,
                      equilibration=20, friction=0.001, sample_every=20)

    def test_motion_grows_with_temperature(self, big):
        mv.md.run(big, level="emt", temperature=300.0, steps=300,
                  sample_every=20, key_added="cold")
        mv.md.run(big, level="emt", temperature=1500.0, steps=300,
                  sample_every=20, key_added="hot")
        assert float(big.obs["msd_hot"].iloc[0]) > \
            float(big.obs["msd_cold"].iloc[0])

    def test_a_solid_does_not_diffuse(self, big):
        """A perfect crystal below melting has no diffusion pathway, so the
        Einstein fit must return zero rather than a small positive number."""
        mv.md.run(big, level="emt", temperature=300.0, steps=400,
                  sample_every=20)
        assert float(big.obs["diffusivity_emt"].iloc[0]) == \
            pytest.approx(0.0, abs=1e-8)

    def test_per_element_diffusivity_is_a_layer(self, big):
        mv.md.run(big, level="emt", temperature=300.0, steps=200,
                  sample_every=20)
        layer = big.layers["diffusivity_emt"]
        assert layer.shape == (big.n_obs, big.n_vars)

    def test_an_unknown_ensemble_is_refused(self, big):
        with pytest.raises(ValueError, match="'nvt' or 'npt'"):
            mv.md.run(big, level="emt", ensemble="nve")

    def test_conductivity_needs_a_diffusivity_layer(self, copper):
        with pytest.raises(ValueError, match="mv.md.run"):
            mv.md.conductivity(copper, species="Cu", level="emt")

    def test_conductivity_of_a_non_diffusing_solid_is_zero(self, big):
        mv.md.run(big, level="emt", temperature=300.0, steps=300,
                  sample_every=20)
        mv.md.conductivity(big, species="Cu", charge=1.0, level="emt")
        assert float(big.obs["conductivity_Cu_emt"].iloc[0]) == \
            pytest.approx(0.0, abs=1e-6)

    def test_an_unknown_species_is_named(self, big):
        mv.md.run(big, level="emt", temperature=300.0, steps=200,
                  sample_every=20)
        with pytest.raises(ValueError, match="not on the element axis"):
            mv.md.conductivity(big, species="Li", level="emt")


class TestBatchedEngine:
    """The GPU path, skipped where its backend is absent.

    torch-sim-atomistic requires Python >= 3.11 while matverse's own floor is
    3.10, so whether these run is an environment decision rather than a
    dependency one. `mv.md.batched_available()` says which situation you are in.
    """

    @pytest.fixture
    def model(self):
        torch_sim = pytest.importorskip("torch_sim")           # noqa: F841
        from torch_sim.models.lennard_jones import LennardJonesModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mv.md.register_batched("lj-batched",
                               lambda: LennardJonesModel(device=device),
                               method="Lennard-Jones (TorchSim)",
                               license="MIT")
        return device

    @pytest.fixture
    def batch(self):
        cell = _fcc("Cu", 3.61)
        cell.make_supercell([2, 2, 2])
        return mv.data.from_structures([cell.copy() for _ in range(4)])

    def test_availability_reports_the_reason_when_absent(self):
        report = mv.md.batched_available()
        assert "torch_sim" in report
        if not report["torch_sim"]:
            assert "3.11" in report["install"]

    def test_a_batch_integrates_in_one_call(self, model, batch):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.md.run(batch, level="lj-batched", temperature=300.0, steps=100,
                      timestep=0.002)
        assert "md_energy_lj-batched" in batch.obs
        assert "md_lj-batched" in mv.variants(batch)
        assert np.isfinite(
            batch.obs["md_energy_lj-batched"].to_numpy(dtype=float)).all()

    def test_the_engine_is_recorded_on_the_level(self, model, batch):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.md.run(batch, level="lj-batched", temperature=300.0, steps=50,
                      timestep=0.002)
        info = mv.level_info(batch, "lj-batched")
        assert info["engine"] == "torchsim"
        assert "no MSD or diffusivity" in info["note"]

    def test_the_thermostat_check_applies_to_this_path_too(self, model, batch):
        """Default Lennard-Jones parameters are argon-like and wrong for a
        copper lattice, so the system heats itself. The point is that matverse
        says so rather than reporting the temperature that was requested."""
        with pytest.warns(UserWarning, match="requested"):
            mv.md.run(batch, level="lj-batched", temperature=300.0, steps=100,
                      timestep=0.002)


class TestTemperatureSweep:
    @pytest.fixture
    def big(self):
        cell = _fcc("Cu", 3.61)
        cell.make_supercell([2, 2, 2])
        return mv.data.from_structures([cell])

    def test_a_sweep_uses_the_condition_axis(self, big):
        """Property-versus-temperature is the same mechanism as a diffraction
        pattern: one grid, one obsm block per quantity."""
        mv.md.sweep(big, level="emt", temperatures=(300.0, 600.0),
                    steps=150, sample_every=25)
        grid = mv.grid_of(big, "md_volume")
        assert list(grid) == [300.0, 600.0]
        assert big.obsm["md_volume_emt"].shape == (big.n_obs, 2)
        assert big.uns["grids"]["md_volume"]["unit"] == "K"

    def test_thermal_expansion_is_positive(self, big):
        mv.md.sweep(big, level="emt", temperatures=(300.0, 900.0),
                    steps=200, ensemble="npt", sample_every=25)
        alpha = float(big.obs["thermal_expansion_emt"].iloc[0])
        assert np.isfinite(alpha) and alpha > 0

    def test_activation_energy_is_nan_when_nothing_diffused(self, big):
        """A slope through numerical noise is not an activation energy, and
        reporting one is worse than reporting nothing."""
        mv.md.sweep(big, level="emt", temperatures=(300.0, 500.0),
                    steps=150, sample_every=25)
        assert np.isnan(float(big.obs["activation_energy_emt"].iloc[0]))


class TestMeltQuench:
    def test_it_produces_an_amorphous_variant(self, copper):
        mv.md.melt_quench(copper, level="emt", melt_temperature=2500.0,
                          melt_steps=150, quench_steps=150,
                          equilibrate_steps=100, supercell=(2, 2, 2))
        assert "amorphous_emt" in mv.variants(copper)
        assert np.isfinite(float(copper.obs["amorphous_density_emt"].iloc[0]))

    def test_the_protocol_is_recorded(self, copper):
        """A 2026 study found every one of eight universal potentials produced
        catastrophically under-dense structures under a naive NPT quench. Which
        protocol ran is therefore part of the result."""
        mv.md.melt_quench(copper, level="emt", melt_steps=100,
                          quench_steps=100, equilibrate_steps=50)
        assert "NVT quench" in mv.level_info(copper, "emt")["protocol"]

    def test_the_density_ratio_exposes_a_failed_quench(self, copper):
        mv.md.melt_quench(copper, level="emt", melt_temperature=2500.0,
                          melt_steps=150, quench_steps=150,
                          equilibrate_steps=100, supercell=(2, 2, 2))
        ratio = float(copper.obs["amorphous_density_ratio_emt"].iloc[0])
        assert np.isfinite(ratio) and ratio > 0


class TestMigrationBarriers:
    @pytest.fixture
    def hopping(self, copper):
        mv.neb.hop_endpoints(copper, species="Cu", supercell=(2, 2, 2))
        mv.calc.relax(copper, level="emt", source="hop_initial",
                      key_added="start", fmax=0.05, steps=80)
        mv.calc.relax(copper, level="emt", source="hop_final",
                      key_added="end", fmax=0.05, steps=80)
        return copper

    def test_endpoints_move_exactly_one_atom_by_the_hop_distance(self, copper):
        """The destination must be the nearest periodic *image* of the vacancy.

        Using the stored coordinates sends the atom the long way round the cell
        — 7.66 angstrom instead of 2.55 in this case — and the NEB then measures
        the cost of dragging an atom through the lattice.
        """
        mv.neb.hop_endpoints(copper, species="Cu", supercell=(2, 2, 2))
        start = mv.structures(copper, "hop_initial")[0]
        end = mv.structures(copper, "hop_final")[0]
        moved = np.linalg.norm(end.cart_coords - start.cart_coords, axis=1)

        assert int((moved > 0.1).sum()) == 1
        assert moved.max() == pytest.approx(
            float(copper.obs["hop_distance"].iloc[0]), abs=0.05)

    def test_the_barrier_lands_near_the_literature_value(self, hopping):
        """Vacancy migration in fcc copper is about 0.70 eV."""
        mv.neb.barrier(hopping, initial="start", final="end", level="emt",
                       n_images=7, fmax=0.05, steps=200)
        assert float(hopping.obs["barrier_emt"].iloc[0]) == \
            pytest.approx(0.70, abs=0.20)
        assert bool(hopping.obs["neb_converged_emt"].iloc[0])

    def test_a_symmetric_hop_is_symmetric(self, hopping):
        mv.neb.barrier(hopping, initial="start", final="end", level="emt",
                       n_images=7, fmax=0.05, steps=200)
        forward = float(hopping.obs["barrier_emt"].iloc[0])
        reverse = float(hopping.obs["barrier_reverse_emt"].iloc[0])
        assert forward == pytest.approx(reverse, abs=0.02)
        assert float(hopping.obs["reaction_energy_emt"].iloc[0]) == \
            pytest.approx(0.0, abs=0.02)

    def test_the_profile_rises_and_falls(self, hopping):
        mv.neb.barrier(hopping, initial="start", final="end", level="emt",
                       n_images=7, fmax=0.05, steps=200)
        profile = hopping.obsm["neb_profile_emt"][0]
        assert profile[0] == pytest.approx(0.0, abs=1e-6)
        assert profile[-1] == pytest.approx(0.0, abs=0.02)
        assert profile.argmax() not in (0, len(profile) - 1)
        assert list(mv.grid_of(hopping, "neb_profile"))[0] == 0.0

    def test_the_surrogate_caveat_is_recorded(self, hopping):
        mv.neb.barrier(hopping, initial="start", final="end", level="emt",
                       n_images=5, steps=60)
        note = mv.level_info(hopping, "emt")["note"]
        assert "soften" in note
        assert "effective-medium" in note   # the calculator's own note survives

    def test_too_few_images_is_refused(self, hopping):
        with pytest.raises(ValueError, match="at least 3"):
            mv.neb.barrier(hopping, initial="start", final="end", n_images=2)

    def test_a_species_with_one_site_cannot_hop(self, metals):
        mv.neb.hop_endpoints(metals, species="Cu", supercell=(1, 1, 1))
        # Aluminium has no copper to move, so its hop is recorded as absent.
        assert np.isnan(float(metals.obs["hop_distance"].iloc[1]))


class TestSurfaces:
    @pytest.fixture
    def facets(self, metals):
        mv.pp.describe(metals)
        mv.calc.energy(metals, level="emt")
        cut = mv.surf.slabs(metals, max_index=1, min_slab=8.0,
                            min_vacuum=10.0)
        mv.calc.energy(cut, level="emt")
        return cut

    def test_slabs_make_more_rows_than_they_consume(self, metals, facets):
        assert facets.n_obs > metals.n_obs
        assert set(facets.obs["parent"]) <= set(map(str, metals.obs_names))
        assert set(facets.obs["miller"]) >= {"1_0_0", "1_1_0", "1_1_1"}

    def test_surface_energies_are_ordered_correctly(self, metals, facets):
        """EMT gets the magnitude wrong by roughly a factor of two, and the
        ordering right. Literature for copper: (111) < (100) < (110)."""
        mv.surf.surface_energy(facets, bulk=metals, level="emt")
        frame = facets.obs[facets.obs["parent"] == "0"]
        by_facet = frame.groupby("miller", observed=True)[
            "surface_energy_emt"].min()
        assert by_facet["1_1_1"] < by_facet["1_0_0"] < by_facet["1_1_0"]

    def test_surface_energies_are_positive(self, metals, facets):
        mv.surf.surface_energy(facets, bulk=metals, level="emt")
        gamma = facets.obs["surface_energy_emt"].to_numpy(dtype=float)
        assert (gamma[np.isfinite(gamma)] > 0).all()

    def test_surface_energy_requires_the_bulk_reference(self, metals, facets):
        bare = mv.data.from_structures(mv.structures(metals))
        with pytest.raises(ValueError, match="mv.calc.energy"):
            mv.surf.surface_energy(facets, bulk=bare, level="emt")

def _has_diffusion_addon() -> bool:
    try:
        import pymatgen.analysis.diffusion.aimd.rdf  # noqa: F401
    except ImportError:
        return False
    return True


class TestOccupancy:
    """Localised, floppy and liquid, in that order — and the sampling trap.

    The metric is a histogram, so it is only about the material when there are
    enough samples to fill the grid. These pin the ordering and then pin the
    artefact, because the artefact is what would make the number wrong without
    making it look wrong.
    """

    @pytest.fixture
    def crystal(self):
        from pymatgen.core import Lattice, Structure
        fcc = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
        st = Structure(Lattice.cubic(5.0), ["Li"] * 4, fcc)
        return mv.data.from_structures([st]), np.array(fcc)

    def test_delocalisation_is_ordered(self, crystal):
        """Same cell, same grid, same run length — only the amplitude
        changes, so the comparison is the one this quantity supports."""
        md, base = crystal
        rng = np.random.default_rng(0)
        runs = {
            "frozen": base[None].repeat(400, axis=0),
            "tight": (base[None] + rng.normal(0, 0.01, (400, 4, 3))) % 1.0,
            "floppy": (base[None] + rng.normal(0, 0.05, (400, 4, 3))) % 1.0,
        }
        for label, frames in runs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mv.md.occupancy(md, frames, species="Li", bins=8,
                                key_added=label)
        values = [float(md.obs[f"occupied_fraction_{k}"].iloc[0])
                  for k in ("frozen", "tight", "floppy")]
        assert values[0] < values[1] < values[2], values
        entropies = [float(md.obs[f"occupancy_entropy_{k}"].iloc[0])
                     for k in ("frozen", "tight", "floppy")]
        assert entropies[0] < entropies[1] < entropies[2], entropies

    def test_a_frozen_crystal_puts_each_ion_in_one_voxel(self, crystal):
        md, base = crystal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mv.md.occupancy(md, base[None].repeat(400, axis=0), species="Li",
                            bins=8, key_added="frozen")
        # four atoms, four voxels, equal weight
        assert float(md.obs["occupancy_peak_frozen"].iloc[0]) == \
            pytest.approx(0.25)

    def test_a_well_sampled_uniform_run_approaches_its_limit(self, crystal):
        """Covering 90% of a uniform probability takes about 87% of the
        voxels, not 100% — the least-visited tail is what is left out."""
        md, _ = crystal
        rng = np.random.default_rng(0)
        mv.md.occupancy(md, rng.random((4000, 4, 3)), species="Li", bins=8,
                        key_added="liquid")
        assert float(md.obs["occupied_fraction_liquid"].iloc[0]) == \
            pytest.approx(0.87, abs=0.05)
        assert float(md.obs["occupancy_entropy_liquid"].iloc[0]) > 0.99

    def test_undersampling_is_warned_about_not_reported_as_order(self, crystal):
        """The trap. The same uniform distribution reads 0.87 on a grid it can
        fill and near zero on one it cannot, and nothing about the number says
        which happened — so the function says it."""
        md, _ = crystal
        rng = np.random.default_rng(0)
        sparse = rng.random((100, 4, 3))
        with pytest.warns(UserWarning, match="per voxel"):
            mv.md.occupancy(md, sparse, species="Li", bins=24,
                            key_added="sparse")
        assert float(md.obs["occupied_fraction_sparse"].iloc[0]) < 0.2
        assert md.uns["occupancy"]["sparse"]["samples_per_voxel"][0] < 5.0

    def test_enough_sampling_passes_quietly(self, crystal):
        md, _ = crystal
        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mv.md.occupancy(md, rng.random((4000, 4, 3)), species="Li",
                            bins=8, key_added="ok")

    def test_a_bad_coverage_is_refused(self, crystal):
        md, base = crystal
        with pytest.raises(ValueError, match="coverage"):
            mv.md.occupancy(md, base[None].repeat(10, axis=0), coverage=1.5)

    def test_a_bad_shape_is_refused(self, crystal):
        md, base = crystal
        with pytest.raises(ValueError, match="fractional coordinates"):
            mv.md.occupancy(md, base)

    def test_an_absent_species_is_recorded_not_guessed(self, crystal):
        md, base = crystal
        mv.md.occupancy(md, base[None].repeat(10, axis=0), species="Na",
                        key_added="none")
        assert np.isnan(float(md.obs["occupied_fraction_none"].iloc[0]))


class TestVanHove:
    """Both parts are checked against identities, not against stored output.

    The self part is a probability density over displacement magnitude, so it
    integrates to one whatever the trajectory. The distinct part is normalised
    like a radial distribution function, so integrating 4 pi r^2 rho G_d out to
    R has to give the number of neighbours actually within R. Those two pin the
    normalisation, which is the only part of this that is easy to get wrong and
    hard to notice.
    """

    A = 4.2
    SIGMA = 0.01

    @pytest.fixture
    def salt(self):
        from pymatgen.core import Lattice, Structure
        cell = Structure(Lattice.cubic(self.A), ["Li", "Cl", "Cl", "Cl"],
                         [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        cell.make_supercell([3, 3, 3])
        md = mv.data.from_structures([cell])
        return md, cell, np.array(cell.frac_coords)

    def test_the_self_part_is_a_probability_density(self, salt):
        md, _, base = salt
        mv.md.van_hove(md, base[None].repeat(3, axis=0), dt=0, r_max=8.0,
                       n_grid=401, sigma=0.05, level="static")
        r = mv.grid_of(md, "van_hove_self")
        assert np.trapezoid(md.obsm["van_hove_self_static"][0], r) == \
            pytest.approx(1.0, rel=1e-3)

    def test_it_still_integrates_to_one_once_things_move(self, salt):
        md, cell, base = salt
        rng = np.random.default_rng(0)
        frames = (base[None]
                  + rng.normal(0, self.SIGMA, (20, len(cell), 3))) % 1.0
        mv.md.van_hove(md, frames, dt=5, r_max=8.0, n_grid=401, sigma=0.05)
        r = mv.grid_of(md, "van_hove_self")
        assert np.trapezoid(md.obsm["van_hove_self_md"][0], r) == \
            pytest.approx(1.0, rel=1e-3)

    def test_a_still_crystal_has_not_moved(self, salt):
        md, _, base = salt
        mv.md.van_hove(md, base[None].repeat(3, axis=0), dt=0, r_max=8.0,
                       n_grid=401, sigma=0.05, level="static")
        assert float(md.obs["van_hove_rms_static"].iloc[0]) == \
            pytest.approx(0.0, abs=1e-9)

    def test_the_rms_displacement_is_what_was_injected(self, salt):
        """Two frames each carry independent Gaussian noise of sigma per
        fractional component, so their difference has sigma*sqrt(2), and the
        RMS over three components in a cubic cell of side a is
        a*sigma*sqrt(2)*sqrt(3)."""
        md, cell, base = salt
        rng = np.random.default_rng(0)
        frames = (base[None]
                  + rng.normal(0, self.SIGMA, (20, len(cell), 3))) % 1.0
        mv.md.van_hove(md, frames, dt=5, r_max=8.0)
        expected = self.A * 3 * self.SIGMA * np.sqrt(2) * np.sqrt(3)
        assert float(md.obs["van_hove_rms_md"].iloc[0]) == \
            pytest.approx(expected, rel=0.05)

    def test_the_distinct_part_counts_the_right_neighbours(self, salt):
        """The identity that fixes the normalisation. Integrating the distinct
        part against the shell volume must return the neighbour count, and it
        must do so at more than one radius or a wrong power of r would pass."""
        md, cell, base = salt
        mv.md.van_hove(md, base[None].repeat(3, axis=0), dt=0, r_max=8.0,
                       n_grid=401, sigma=0.05, level="static")
        r = mv.grid_of(md, "van_hove_distinct")
        g = md.obsm["van_hove_distinct_static"][0]
        rho = len(cell) / cell.lattice.volume
        for radius in (3.5, 5.0, 6.5):
            inside = r <= radius
            integral = np.trapezoid(
                4 * np.pi * r[inside] ** 2 * rho * g[inside], r[inside])
            actual = np.mean([len(cell.get_neighbors(site, radius))
                              for site in cell])
            assert integral == pytest.approx(actual, rel=0.05), \
                f"neighbour count wrong at {radius} A"

    def test_a_hop_shows_up_as_a_second_feature(self, salt):
        """The shape a diffusivity throws away. Most ions stay put and a few
        move a lattice spacing, so the self part keeps its peak at zero and
        grows a bump at the jump distance — which is what obs['van_hove_peak']
        reports."""
        md, cell, base = salt
        rng = np.random.default_rng(0)
        frames = (base[None]
                  + rng.normal(0, self.SIGMA, (20, len(cell), 3))) % 1.0
        # move a handful of ions by one nearest-neighbour vector part way
        jump = np.array([0.5, 0.5, 0.0]) / 3.0        # a/2 in the 3x3x3 cell
        hopped = frames.copy()
        hopped[10:, :6, :] = (frames[10:, :6, :] + jump) % 1.0
        mv.md.van_hove(md, hopped, dt=15, r_max=8.0, n_grid=401, sigma=0.1,
                       level="hop")
        assert float(md.obs["van_hove_jump_hop"].iloc[0]) == \
            pytest.approx(self.A * np.sqrt(0.5), rel=0.1), \
            "expected a feature at the nearest-neighbour jump distance"
        # and the tallest feature is still the vibration, not the jump
        assert float(md.obs["van_hove_peak_hop"].iloc[0]) < 1.0

    def test_a_rattling_solid_reports_no_jump(self, salt):
        """The negative half of the previous test, and the one that makes it
        mean something: if every trajectory produced a jump distance, finding
        one would say nothing."""
        md, cell, base = salt
        rng = np.random.default_rng(0)
        frames = (base[None]
                  + rng.normal(0, self.SIGMA, (20, len(cell), 3))) % 1.0
        mv.md.van_hove(md, frames, dt=15, r_max=8.0, n_grid=401, sigma=0.1,
                       level="still")
        assert np.isnan(float(md.obs["van_hove_jump_still"].iloc[0]))
        assert np.isfinite(float(md.obs["van_hove_peak_still"].iloc[0]))

    def test_the_most_probable_displacement_is_not_zero(self, salt):
        """A shell at radius r has area 4 pi r^2, so even for a Gaussian
        centred on the origin the most likely *distance* is finite — the mode
        of a Maxwell distribution, sqrt(2) times the one-dimensional width.
        Reporting zero here would mean the r^2 weighting had been dropped."""
        md, cell, base = salt
        rng = np.random.default_rng(0)
        frames = (base[None]
                  + rng.normal(0, self.SIGMA, (20, len(cell), 3))) % 1.0
        mv.md.van_hove(md, frames, dt=5, r_max=8.0, n_grid=401, sigma=0.02,
                       level="vib")
        one_d = self.A * 3 * self.SIGMA * np.sqrt(2)     # per component
        assert float(md.obs["van_hove_peak_vib"].iloc[0]) == \
            pytest.approx(np.sqrt(2) * one_d, rel=0.25)

    def test_a_bad_trajectory_shape_is_refused(self, salt):
        md, _, base = salt
        with pytest.raises(ValueError, match="fractional coordinates"):
            mv.md.van_hove(md, base, dt=0)

    def test_an_impossible_interval_is_refused(self, salt):
        md, _, base = salt
        with pytest.raises(ValueError, match="dt must be between"):
            mv.md.van_hove(md, base[None].repeat(3, axis=0), dt=9)

    def test_a_mismatched_cell_is_refused(self, salt):
        md, cell, base = salt
        with pytest.raises(ValueError, match="must be the same"):
            mv.md.van_hove(md, base[None, :10, :].repeat(3, axis=0), dt=0)


@pytest.mark.skipif(not _has_diffusion_addon(),
                    reason="pymatgen-analysis-diffusion is an optional extra")
class TestTrajectorySites:
    """Separating a rattle from a hop.

    Both raise the mean-squared displacement, and only one of them is
    diffusion. These use synthetic trajectories rather than a real run so the
    right answer is known exactly: the sites are put where they are found, and
    the hops are counted before they are measured.
    """

    A = 5.0
    SIGMA = 0.01          # fractional noise per component

    @staticmethod
    def _cell(a):
        from pymatgen.core import Lattice, Structure
        fcc = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
        return Structure(Lattice.cubic(a), ["Li"] * 4, fcc), np.array(fcc)

    @pytest.fixture
    def vibrating(self):
        """Four ions, each rattling about its own site, none leaving."""
        structure, base = self._cell(self.A)
        rng = np.random.RandomState(0)
        frames = (base[None] + rng.normal(0, self.SIGMA, (40, 4, 3))) % 1.0
        md = mv.data.from_structures([structure])
        return md, frames, base

    def test_the_sites_are_recovered(self, vibrating):
        md, frames, _ = vibrating
        mv.md.sites(md, frames, species="Li")
        assert int(md.obs["md_sites_Li_md"].iloc[0]) == 4

    def test_a_vibrating_solid_visits_one_site_each(self, vibrating):
        md, frames, _ = vibrating
        mv.md.sites(md, frames, species="Li")
        assert float(md.obs["md_site_visits_Li_md"].iloc[0]) == \
            pytest.approx(1.0)

    def test_the_spread_is_the_amplitude_that_was_put_in(self, vibrating):
        """Gaussian noise of sigma per fractional component, in a cubic cell of
        side a, has RMS distance a*sigma*sqrt(3) from the centre. Getting this
        back in angstroms is what makes the number comparable to a real thermal
        amplitude rather than a unitless cluster score."""
        md, frames, _ = vibrating
        mv.md.sites(md, frames, species="Li")
        expected = self.A * self.SIGMA * np.sqrt(3)
        assert float(md.obs["md_site_spread_Li_md"].iloc[0]) == \
            pytest.approx(expected, rel=0.15)

    def test_a_hop_is_counted_and_a_rattle_is_not(self, vibrating):
        """One atom of four moves to a second site half way through, so the
        mean number of sites visited per atom must be exactly (2+1+1+1)/4."""
        md, frames, base = vibrating
        rng = np.random.RandomState(1)
        hopped = frames.copy()
        hopped[20:, 0, :] = base[1] + rng.normal(0, self.SIGMA, (20, 3))
        mv.md.sites(md, hopped % 1.0, species="Li", key_added="hop")
        mv.md.sites(md, frames, species="Li", key_added="still")
        assert float(md.obs["md_site_visits_hop"].iloc[0]) == \
            pytest.approx(1.25)
        assert float(md.obs["md_site_visits_still"].iloc[0]) == \
            pytest.approx(1.0)

    def test_the_spread_does_not_notice_the_hop(self, vibrating):
        """The point of reporting both. A hop moves an ion a long way and
        leaves the vibration amplitude alone, which is precisely what an MSD
        cannot tell you."""
        md, frames, base = vibrating
        rng = np.random.RandomState(1)
        hopped = frames.copy()
        hopped[20:, 0, :] = base[1] + rng.normal(0, self.SIGMA, (20, 3))
        mv.md.sites(md, hopped % 1.0, species="Li", key_added="hop")
        mv.md.sites(md, frames, species="Li", key_added="still")
        assert float(md.obs["md_site_spread_hop"].iloc[0]) == \
            pytest.approx(float(md.obs["md_site_spread_still"].iloc[0]),
                          rel=0.1)

    def test_the_same_trajectory_gives_the_same_answer(self, vibrating):
        """KmeansPBC seeds itself with an unseeded random.sample, so the
        default path returns a different clustering on every call. This one
        caught it: the same input gave 1.0 and 1.25 on different runs of the
        suite. Centroids are seeded at the crystallographic sites instead."""
        md, frames, _ = vibrating
        answers = set()
        for trial in range(6):
            fresh = mv.data.from_structures(mv.structures(md, "input"))
            mv.md.sites(fresh, frames, species="Li")
            answers.add((int(fresh.obs["md_sites_Li_md"].iloc[0]),
                         round(float(fresh.obs["md_site_visits_Li_md"].iloc[0]), 6),
                         round(float(fresh.obs["md_site_spread_Li_md"].iloc[0]), 6)))
        assert len(answers) == 1, f"clustering is not deterministic: {answers}"

    def test_a_mismatched_trajectory_is_refused(self, vibrating):
        md, frames, _ = vibrating
        with pytest.raises(ValueError, match="must be the same"):
            mv.md.sites(md, frames[:, :3, :], species="Li")

    def test_a_trajectory_of_the_wrong_shape_is_refused(self, vibrating):
        md, frames, _ = vibrating
        with pytest.raises(ValueError, match="fractional coordinates"):
            mv.md.sites(md, frames[0], species="Li")

    def test_an_absent_species_is_recorded_not_guessed(self, vibrating):
        md, frames, _ = vibrating
        mv.md.sites(md, frames, species="Na")
        assert int(md.obs["md_sites_Na_md"].iloc[0]) == 0
        assert np.isnan(float(md.obs["md_site_visits_Na_md"].iloc[0]))


@pytest.mark.skipif(not _has_diffusion_addon(),
                    reason="pymatgen-analysis-diffusion is an optional extra")
class TestPercolation:
    """Checked against three structures whose answers are textbook.

    Spinel LiMn2O4 is the archetypal three-dimensional lithium conductor and
    layered LiCoO2 the archetypal two-dimensional one, and the difference is
    connectivity rather than barrier height. If this function cannot separate
    those two it is measuring nothing.
    """

    A_SPINEL = 8.24
    A_LAYERED = 2.82
    A_BCC = 3.51

    @pytest.fixture
    def conductors(self):
        from pymatgen.core import Lattice, Structure
        # Fd-3m in pymatgen is origin choice 1, where the tetrahedral 8a site
        # is at the origin rather than at (1/8, 1/8, 1/8). Putting Li at
        # (1/8, 1/8, 1/8) here silently lands it on a 16-fold site and builds
        # Li2MnO4, which is not spinel and not what this test claims to check.
        spinel = Structure.from_spacegroup(
            "Fd-3m", Lattice.cubic(self.A_SPINEL), ["Li", "Mn", "O"],
            [[0, 0, 0], [0.625] * 3, [0.3875] * 3])
        layered = Structure.from_spacegroup(
            "R-3m", Lattice.hexagonal(self.A_LAYERED, 14.05),
            ["Li", "Co", "O"], [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.2395]])
        bcc = Structure(Lattice.cubic(self.A_BCC), ["Li", "Li"],
                        [[0, 0, 0], [.5, .5, .5]])
        md = mv.data.from_structures([spinel, layered, bcc])
        md.obs_names = ["spinel", "layered", "bcc"]
        return md

    def test_the_fixture_really_is_spinel(self, conductors):
        """The premise. Li tetrahedral, Mn octahedral, LiMn2O4."""
        spinel = mv.structures(conductors, "input")[0]
        assert spinel.composition.reduced_formula == "LiMn2O4"
        li = next(s for s in spinel if s.specie.symbol == "Li")
        mn = next(s for s in spinel if s.specie.symbol == "Mn")
        assert sum(1 for n in spinel.get_neighbors(li, 2.2)
                   if n.specie.symbol == "O") == 4
        assert sum(1 for n in spinel.get_neighbors(mn, 2.2)
                   if n.specie.symbol == "O") == 6

    def test_spinel_is_three_dimensional_and_layered_is_not(self, conductors):
        """The result the module exists to produce. Both have a percolating
        lithium network at this cutoff; only one of them percolates in three
        directions."""
        mv.neb.percolation(conductors, species="Li", cutoff=3.6)
        dim = conductors.obs["percolation_dimensionality_Li"]
        assert dim["spinel"] == 3
        assert dim["layered"] == 2
        assert dim["bcc"] == 3

    def test_the_threshold_is_the_nearest_neighbour_distance(self, conductors):
        """Each of these networks connects as soon as nearest neighbours do, so
        the bottleneck is a distance that can be written down in closed form:
        a*sqrt(3)/4 for the diamond-like 8a sublattice of spinel, the hexagonal
        a for in-plane lithium in a layered oxide, a*sqrt(3)/2 for bcc."""
        mv.neb.percolation(conductors, species="Li", cutoff=3.6)
        threshold = conductors.obs["percolation_threshold_Li"]
        assert threshold["spinel"] == pytest.approx(
            self.A_SPINEL * np.sqrt(3) / 4, rel=1e-4)
        assert threshold["layered"] == pytest.approx(self.A_LAYERED, rel=1e-4)
        assert threshold["bcc"] == pytest.approx(
            self.A_BCC * np.sqrt(3) / 2, rel=1e-4)

    def test_the_threshold_does_not_depend_on_the_cutoff(self, conductors):
        """It is a property of the structure. The cutoff is a question asked of
        it, and must not leak into the answer."""
        mv.neb.percolation(conductors, species="Li", cutoff=3.0,
                           key_added="tight")
        mv.neb.percolation(conductors, species="Li", cutoff=6.0,
                           key_added="loose")
        assert np.allclose(conductors.obs["percolation_threshold_tight"],
                           conductors.obs["percolation_threshold_loose"],
                           rtol=1e-9)

    def test_nothing_percolates_below_its_threshold(self, conductors):
        mv.neb.percolation(conductors, species="Li", cutoff=2.5)
        dim = conductors.obs["percolation_dimensionality_Li"].to_numpy()
        threshold = conductors.obs["percolation_threshold_Li"].to_numpy()
        assert (threshold > 2.5).all(), "fixture no longer tests what it says"
        assert (dim == 0).all()

    def test_dimensionality_never_falls_as_the_cutoff_grows(self, conductors):
        """Adding longer hops adds edges, and an edge cannot disconnect a
        graph. A non-monotonic answer would mean the search is missing paths."""
        previous = None
        for cutoff in (2.5, 3.0, 3.6, 4.5, 6.0):
            mv.neb.percolation(conductors, species="Li", cutoff=cutoff,
                               key_added=f"c{cutoff}")
            current = conductors.obs[
                f"percolation_dimensionality_c{cutoff}"].to_numpy()
            if previous is not None:
                assert (current >= previous).all(), f"fell at cutoff {cutoff}"
            previous = current

    def test_a_long_enough_hop_bridges_the_layers(self, conductors):
        """The two-dimensionality of a layered oxide is a statement about hop
        length, not a fact about the material, and the function must not
        pretend otherwise."""
        mv.neb.percolation(conductors, species="Li", cutoff=5.0)
        assert conductors.obs["percolation_dimensionality_Li"]["layered"] == 3

    def test_a_material_without_the_mobile_species_is_not_a_conductor(self):
        from pymatgen.core import Lattice, Structure
        nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64),
                                         ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
        md = mv.data.from_structures([nacl])
        mv.neb.percolation(md, species="Li", cutoff=4.0)
        assert int(md.obs["percolation_sites_Li"].iloc[0]) == 0
        assert int(md.obs["percolation_dimensionality_Li"].iloc[0]) == 0
        assert np.isnan(float(md.obs["percolation_threshold_Li"].iloc[0]))

    def test_isolated_ions_do_not_percolate(self):
        """One lithium in a large cell has neighbours only in other cells, and
        at a short cutoff it reaches none of them. This separates 'connected to
        something' from 'gets out of the cell', which is the distinction the
        whole function rests on."""
        from pymatgen.core import Lattice, Structure
        lonely = Structure(Lattice.cubic(12.0), ["Li"], [[0, 0, 0]])
        md = mv.data.from_structures([lonely])
        mv.neb.percolation(md, species="Li", cutoff=4.0)
        assert int(md.obs["percolation_dimensionality_Li"].iloc[0]) == 0
        assert float(md.obs["percolation_threshold_Li"].iloc[0]) == \
            pytest.approx(12.0, rel=1e-6)


class TestOffStoichiometricSurfaces:
    """A symmetrized slab of an ordered alloy is usually not the bulk formula.

    ``E_slab - N * e_bulk`` then leaves the energy of the surplus atoms in the
    number, which is not a surface energy and is not even close to one. These
    pin that the plain function refuses rather than answers, and that the
    chemical-potential version answers correctly.
    """

    @pytest.fixture
    def cu3au(self):
        from pymatgen.core import Lattice, Structure
        bulk = Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(3.75), ["Au", "Cu"],
            [[0, 0, 0], [0.5, 0.5, 0]])
        md = mv.data.from_structures([bulk])
        md.obs_names = ["Cu3Au"]
        mv.calc.energy(md, level="emt")
        return md

    @pytest.fixture
    def reservoirs(self):
        from pymatgen.core import Lattice, Structure
        fcc = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
        refs = mv.data.from_structures(
            [Structure(Lattice.cubic(a), [e] * 4, fcc)
             for e, a in (("Cu", 3.61), ("Au", 4.08))])
        refs.obs_names = ["Cu", "Au"]
        mv.calc.energy(refs, level="emt")
        return refs

    @pytest.fixture
    def symmetrized(self, cu3au):
        cut = mv.surf.slabs(cu3au, max_index=1, min_slab=8.0, min_vacuum=10.0,
                            symmetrize=True)
        mv.calc.energy(cut, level="emt")
        return cut

    def test_symmetrize_really_does_break_stoichiometry(self, symmetrized):
        """The premise of the rest of this class. If pymatgen ever starts
        preserving composition under symmetrize, these tests go quiet and this
        one says why."""
        off = [s for s in mv.structures(symmetrized, "input")
               if abs(s.composition["Cu"] / s.composition["Au"] - 3.0) > 1e-6]
        assert off, "expected symmetrize=True to delete sites off-stoichiometry"

    def test_plain_surface_energy_refuses_off_stoichiometric_slabs(
            self, cu3au, symmetrized):
        with pytest.warns(UserWarning, match="surface_energy_chempot"):
            mv.surf.surface_energy(symmetrized, bulk=cu3au, level="emt")
        off = symmetrized.obs["surface_energy_emt_off_stoichiometry"].to_numpy(bool)
        gamma = symmetrized.obs["surface_energy_emt"].to_numpy(dtype=float)
        assert off.any() and np.isnan(gamma[off]).all()
        assert np.isfinite(gamma[~off]).all()

    def test_chempot_version_agrees_where_both_are_defined(
            self, cu3au, symmetrized, reservoirs):
        """The stoichiometric slab has one surface energy, and two independent
        routes to it must land on the same number."""
        with pytest.warns(UserWarning):
            mv.surf.surface_energy(symmetrized, bulk=cu3au, level="emt")
        plain = symmetrized.obs["surface_energy_emt"].to_numpy(dtype=float)
        mv.surf.surface_energy_chempot(symmetrized, bulk=cu3au,
                                       refs=reservoirs, level="emt",
                                       key_added="gamma_mu")
        viac = symmetrized.obs["gamma_mu"].to_numpy(dtype=float)
        both = np.isfinite(plain)
        assert both.any()
        assert np.allclose(plain[both], viac[both], rtol=1e-9)
        assert np.isfinite(viac).all(), "chempot route should answer everywhere"

    def test_opposite_terminations_carry_opposite_excess(
            self, cu3au, symmetrized, reservoirs):
        """Cutting one plane two ways makes one face Au-rich and the other
        Au-poor by the same amount; the excesses must sum to zero."""
        mv.surf.surface_energy_chempot(symmetrized, bulk=cu3au,
                                       refs=reservoirs, level="emt")
        frame = symmetrized.obs
        excess = frame["surface_excess_Au_emt"].to_numpy(dtype=float)
        for miller in frame["miller"].unique():
            pair = excess[(frame["miller"] == miller).to_numpy()]
            pair = pair[np.isfinite(pair)]
            if len(pair) == 2:
                assert abs(pair.sum()) < 1e-9, f"{miller} excesses do not cancel"

    def test_the_facet_ordering_depends_on_the_chemical_potential(
            self, cu3au, symmetrized, reservoirs):
        """This is the reason the module exists. At the Au-rich end the
        Au-rich termination is cheaper; drive the reservoir down and the
        ordering inverts. A single 'the surface energy' would hide it."""
        mv.surf.surface_energy_chempot(symmetrized, bulk=cu3au,
                                       refs=reservoirs, level="emt",
                                       key_added="rich")
        mv.surf.surface_energy_chempot(symmetrized, bulk=cu3au,
                                       refs=reservoirs, level="emt",
                                       chempot={"Au": -0.5}, key_added="poor")
        frame = symmetrized.obs
        pair = frame[np.isfinite(frame["surface_excess_Au_emt"])
                     & (frame["miller"] == "1_0_0")]
        assert len(pair) == 2
        rich = pair["rich"].to_numpy(dtype=float)
        poor = pair["poor"].to_numpy(dtype=float)
        assert np.argmin(rich) != np.argmin(poor), \
            "expected the cheaper (100) termination to change with delta-mu"

    def test_gamma_is_linear_in_the_chemical_potential(
            self, cu3au, symmetrized, reservoirs):
        """gamma(dmu) = gamma_0 - Gamma * dmu, so the deposited excess must
        reproduce the shift without recomputing an energy."""
        mv.surf.surface_energy_chempot(symmetrized, bulk=cu3au,
                                       refs=reservoirs, level="emt",
                                       key_added="at_zero")
        mv.surf.surface_energy_chempot(symmetrized, bulk=cu3au,
                                       refs=reservoirs, level="emt",
                                       chempot={"Au": -0.37}, key_added="at_mu")
        frame = symmetrized.obs
        excess = frame["surface_excess_Au_emt"].to_numpy(dtype=float)
        moved = np.isfinite(excess)
        predicted = (frame["at_zero"].to_numpy(dtype=float)[moved]
                     - excess[moved] * -0.37)
        assert np.allclose(predicted, frame["at_mu"].to_numpy(dtype=float)[moved],
                           rtol=1e-9)

    def test_reservoirs_must_be_elemental(self, cu3au, symmetrized):
        with pytest.raises(ValueError, match="not an element"):
            mv.surf.surface_energy_chempot(symmetrized, bulk=cu3au,
                                           refs=cu3au, level="emt")


class TestSurfacesContinued:
    @pytest.fixture
    def facets(self, metals):
        mv.pp.describe(metals)
        mv.calc.energy(metals, level="emt")
        cut = mv.surf.slabs(metals, max_index=1, min_slab=8.0,
                            min_vacuum=10.0)
        mv.calc.energy(cut, level="emt")
        return cut

    def test_wulff_expresses_the_low_energy_facets(self, metals, facets):
        mv.surf.surface_energy(facets, bulk=metals, level="emt")
        mv.surf.wulff(facets, bulk=metals, level="emt")

        assert np.isfinite(metals.obs["wulff_effective_radius_emt"]).all()
        fractions = facets.obs["wulff_area_fraction_emt"].to_numpy(dtype=float)
        assert np.nansum(fractions) > 0
        assert "1_1_1" in metals.uns["wulff"]["emt"]["0"]["expressed"]


class TestAdsorption:
    @pytest.fixture
    def clean(self, copper):
        cut = mv.surf.slabs(copper, miller=(1, 1, 1), min_slab=8.0,
                            min_vacuum=12.0)
        mv.calc.energy(cut, level="emt")
        return cut

    def test_sites_are_enumerated_not_guessed(self, clean):
        """Which site binds most strongly is the question, so several are
        placed and all are relaxed — the AdsorbML protocol."""
        configs = mv.surf.adsorption_sites(clean, "O", height=1.8)
        assert configs.n_obs > 1
        assert set(configs.obs["site_kind"]) & {"ontop", "bridge", "hollow"}

    def test_hollow_binds_more_strongly_than_atop(self, clean):
        """The standard result for oxygen on a close-packed metal surface."""
        configs = mv.surf.adsorption_sites(clean, "O", height=1.8)
        mv.calc.relax(configs, level="emt", fmax=0.1, steps=40)
        mv.surf.adsorption_energy(configs, clean=clean, reference=0.0,
                                  level="emt")

        by_kind = configs.obs.groupby("site_kind", observed=True)[
            "adsorption_energy_emt"].min()
        assert by_kind["hollow"] < by_kind["ontop"]

    def test_one_best_site_per_slab(self, clean):
        configs = mv.surf.adsorption_sites(clean, "O", height=1.8)
        mv.calc.relax(configs, level="emt", fmax=0.2, steps=20)
        mv.surf.adsorption_energy(configs, clean=clean, reference=0.0,
                                  level="emt")
        assert int(configs.obs["is_best_site_emt"].sum()) == clean.n_obs

    def test_the_reference_convention_is_recorded(self, clean):
        """A hydrogen adsorption energy against half an H2 and one against an
        isolated H atom differ by about 2.3 eV, and both are in use."""
        configs = mv.surf.adsorption_sites(clean, "O", height=1.8)
        mv.calc.relax(configs, level="emt", fmax=0.2, steps=20)
        mv.surf.adsorption_energy(configs, clean=clean, reference=-1.5,
                                  level="emt")
        recorded = configs.uns["adsorption_energy"]["emt"]
        assert recorded["reference"] == -1.5
        assert "E(slab)" in recorded["convention"]

    def test_it_needs_energies_on_both_objects(self, clean):
        configs = mv.surf.adsorption_sites(clean, "O", height=1.8)
        with pytest.raises(ValueError, match="mv.calc.relax"):
            mv.surf.adsorption_energy(configs, clean=clean, reference=0.0,
                                      level="emt")


class TestRelaxNaming:
    def test_two_variants_can_be_relaxed_without_collision(self, copper):
        """mv.calc.relax writes relaxed_<level> by default, which is the same
        name every time. Anything needing two relaxed geometries — an NEB, a
        slab against its bulk — must be able to name them apart."""
        mv.pp.supercell(copper, [2, 1, 1], name="big")
        mv.calc.relax(copper, level="emt", source="input", key_added="a",
                      fmax=0.2, steps=20)
        mv.calc.relax(copper, level="emt", source="big", key_added="b",
                      fmax=0.2, steps=20)

        assert {"a", "b"} <= set(mv.variants(copper))
        assert len(mv.structures(copper, "a")[0]) != \
            len(mv.structures(copper, "b")[0])


@pytest.mark.skipif(not _has_diffusion_addon(),
                    reason="pymatgen-analysis-diffusion is an optional extra")
class TestTrajectoryRDF:
    """Averaging over frames, and a coordination number that means what it says."""

    @staticmethod
    def _cell():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(4.2), ["Li", "Cl", "Cl", "Cl"],
                         [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])

    @pytest.fixture(scope="class")
    def averaged(self):
        cell = self._cell()
        md = mv.data.from_structures([cell])
        mv.pp.describe(md)
        frames = np.tile(np.array(cell.frac_coords), (40, 1, 1))
        frames = frames + np.random.default_rng(0).normal(0, 0.01,
                                                          frames.shape)
        mv.md.rdf(md, frames, species="Li", r_max=8.0)
        return md

    def test_the_first_peak_is_the_nearest_neighbour_distance(self, averaged):
        expected = self._cell().get_distance(0, 1)
        assert averaged.obs["first_shell_md"].iloc[0] == pytest.approx(
            expected, abs=0.1)

    def test_the_coordination_number_counts_neighbours(self, averaged):
        """Twelve, from counting. pymatgen's own coordination_number returns
        4.0 for this cell — the count per reference index, spread over three
        reference sites — so the integral is taken from its definition here."""
        true_count = len(self._cell().get_neighbors(self._cell()[0], 3.1))
        assert true_count == 12
        assert averaged.obs["first_shell_coordination_md"].iloc[0] == \
            pytest.approx(12.0, rel=0.1)

    def test_the_curve_lands_on_the_grid_convention(self, averaged):
        grid = mv.grid_of(averaged, "rdf_md")
        assert averaged.obsm["rdf_md_md"].shape == (1, grid.size)
        assert averaged.obsm["coordination_md_md"].shape == (1, grid.size)

    def test_one_trajectory_belongs_to_one_structure(self):
        cell = self._cell()
        md = mv.data.from_structures([cell, cell])
        mv.pp.describe(md)
        frames = np.tile(np.array(cell.frac_coords), (5, 1, 1))
        with pytest.raises(ValueError, match="one trajectory belongs"):
            mv.md.rdf(md, frames, species="Li")

    def test_a_mismatched_trajectory_is_refused(self):
        md = mv.data.from_structures([self._cell()])
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="same cell in the same order"):
            mv.md.rdf(md, np.zeros((5, 9, 3)), species="Li")

    def test_an_absent_species_is_named(self):
        md = mv.data.from_structures([self._cell()])
        mv.pp.describe(md)
        frames = np.tile(np.array(self._cell().frac_coords), (5, 1, 1))
        with pytest.raises(ValueError, match="no 'Na' in this structure"):
            mv.md.rdf(md, frames, species="Na")

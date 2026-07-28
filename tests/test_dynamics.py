"""Molecular dynamics, migration barriers, surfaces and adsorption.

The barrier and surface-energy tests check against **published numbers and
orderings** rather than against stored output. EMT is a crude potential and gets
magnitudes wrong by a factor of about two on surfaces, so the tests assert what
EMT can be held to — the ordering of facets, the symmetry of a symmetric hop,
and a vacancy barrier that lands near the literature value — and say which is
which.
"""

from __future__ import annotations

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

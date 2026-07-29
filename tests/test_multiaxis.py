"""Grid-shaped results, the sites axis, and cross-database harmonisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matverse as mv


class TestGrids:
    def test_xrd_lands_on_a_shared_grid(self, md):
        mv.prop.xrd(md, two_theta=(10, 60), step=0.05)
        grid = mv.grid_of(md, "xrd")
        assert md.obsm["xrd_calc"].shape == (6, len(grid))
        assert md.uns["grids"]["xrd"]["unit"] == "degrees 2theta"

    def test_patterns_differ_between_materials(self, md):
        mv.prop.xrd(md, two_theta=(10, 60), step=0.05)
        X = md.obsm["xrd_calc"]
        assert not np.allclose(X[0], X[1])
        assert X.max() == pytest.approx(100.0)      # normalised to the top peak

    def test_a_level_cannot_change_the_grid_underneath_another(self, md):
        mv.prop.xrd(md, two_theta=(10, 60), step=0.05)
        with pytest.raises(ValueError, match="must share a grid"):
            mv.prop.xrd(md, two_theta=(10, 90), step=0.05, level="other")

    def test_grid_results_are_rejected_when_misaligned(self, md):
        from matverse._core import deposit_grid
        with pytest.raises(ValueError, match="aligned to the material axis"):
            deposit_grid(md, "junk", "calc", np.zeros((3, 5)), np.arange(5))

    def test_rdf_separates_polymorphs_that_composition_cannot(self, structures):
        """Two cells of the same composition, different lattice parameter."""
        from pymatgen.core import Lattice, Structure
        a = Structure(Lattice.cubic(3.6), ["Cu"] * 4,
                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        b = Structure(Lattice.cubic(4.2), ["Cu"] * 4,
                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        md = mv.data.from_structures([a, b])
        assert np.allclose(md.X.toarray()[0], md.X.toarray()[1])   # same chemistry
        mv.prop.rdf(md, r_max=8.0)
        assert not np.allclose(md.obsm["rdf_calc"][0], md.obsm["rdf_calc"][1])

    def test_compare_grids_needs_both_levels(self, md):
        mv.prop.xrd(md, two_theta=(10, 40), step=0.1)
        with pytest.raises(ValueError, match="absent"):
            mv.prop.compare_grids(md, "xrd", "calc", "experiment")


class TestExperimentAsALevel:
    def test_a_measurement_is_just_another_level(self, md):
        mv.calc.energy(md, level="emt")
        mv.exp.measure(md, "energy_per_atom", np.zeros(6), instrument="bomb calorimeter")
        info = mv.level_info(md, "experiment")
        assert info["kind"] == "experiment"
        assert info["instrument"] == "bomb calorimeter"

        table = mv.compare_levels(md, "energy_per_atom")
        assert {"emt", "experiment"} <= set(table.columns)

    def test_attached_spectra_share_the_computed_grid(self, md):
        """A different instrument step is resampled onto the existing grid."""
        mv.prop.xrd(md, two_theta=(10, 60), step=0.05, fwhm=0.3)
        grid = mv.grid_of(md, "xrd")
        instrument_grid = np.arange(10.0, 60.0, 0.04)      # comparable resolution
        measured = np.vstack([np.interp(instrument_grid, grid, row)
                              for row in md.obsm["xrd_calc"]])
        mv.exp.attach(md, "xrd", measured, instrument_grid,
                      instrument="Bruker D8")
        assert md.obsm["xrd_experiment"].shape == md.obsm["xrd_calc"].shape

        mv.prop.compare_grids(md, "xrd", "calc", "experiment")
        cosine = md.obs["xrd_cosine_calc_vs_experiment"].to_numpy(dtype=float)
        assert (cosine > 0.98).all()

    def test_resampling_below_the_peak_width_loses_peaks(self, md):
        """A caveat rather than a bug, pinned so nobody 'fixes' it.

        Diffraction peaks are narrow. Resampling onto a grid coarser than the
        peak width throws them away, and no later interpolation brings them
        back — so a measurement must be attached at its own resolution or
        better, never downsampled first.
        """
        mv.prop.xrd(md, two_theta=(10, 60), step=0.05, fwhm=0.1)
        grid = mv.grid_of(md, "xrd")
        too_coarse = np.linspace(10, 60, 200)              # 0.25 deg > 0.1 fwhm
        measured = np.vstack([np.interp(too_coarse, grid, row)
                              for row in md.obsm["xrd_calc"]])
        mv.exp.attach(md, "xrd", measured, too_coarse)
        mv.prop.compare_grids(md, "xrd", "calc", "experiment")
        cosine = md.obs["xrd_cosine_calc_vs_experiment"].to_numpy(dtype=float)
        assert (cosine < 0.9).all()

    def test_a_narrower_measurement_still_compares(self, md):
        """One undefined point must not take the whole comparison with it."""
        mv.prop.xrd(md, two_theta=(10, 60), step=0.05)
        grid = mv.grid_of(md, "xrd")
        narrow = np.linspace(20, 50, 300)           # covers part of the range
        measured = np.vstack([np.interp(narrow, grid, row)
                              for row in md.obsm["xrd_calc"]])
        mv.exp.attach(md, "xrd", measured, narrow)

        assert np.isnan(md.obsm["xrd_experiment"]).any()
        mv.prop.compare_grids(md, "xrd", "calc", "experiment")
        cosine = md.obs["xrd_cosine_calc_vs_experiment"].to_numpy(dtype=float)
        overlap = md.obs["xrd_overlap_calc_vs_experiment"].to_numpy(dtype=float)
        assert np.isfinite(cosine).all()
        assert (overlap < len(grid)).all() and (overlap > 0).all()

    def test_attach_rejects_the_wrong_number_of_curves(self, md):
        mv.prop.xrd(md, two_theta=(10, 40), step=0.1)
        with pytest.raises(ValueError, match="one row per material"):
            mv.exp.attach(md, "xrd", np.zeros((2, 5)), np.arange(5))

    def test_match_xrd_identifies_the_right_candidate(self, md):
        mv.prop.xrd(md, two_theta=(10, 60), step=0.05)
        grid = mv.grid_of(md, "xrd")
        target = 2                                   # feed row 2 back in
        mv.exp.match_xrd(md, md.obsm["xrd_calc"][target], grid)

        assert md.uns["xrd_match"]["best"] == str(md.obs_names[target])
        assert md.obs["xrd_match"].iloc[target] == pytest.approx(1.0, abs=1e-6)
        assert md.obs["xrd_match_rank"].iloc[target] == 1.0

    def test_match_xrd_says_what_it_scored_against(self, md):
        mv.prop.xrd(md, two_theta=(10, 40), step=0.1)
        mv.exp.match_xrd(md, np.ones(50), np.linspace(10, 40, 50))
        assert md.uns["xrd_match"]["scored_against"] == \
            "this object's candidates only"

    def test_match_xrd_needs_patterns_first(self, md):
        with pytest.raises(ValueError, match="mv.prop.xrd"):
            mv.exp.match_xrd(md, np.ones(10), np.linspace(10, 40, 10))


class TestSitesAxis:
    def test_one_row_per_atom(self, md, structures):
        sites = mv.multi.sites(md)
        assert sites.n_obs == sum(len(s) for s in structures)
        assert list(sites.obs.columns[:4]) == ["material", "material_index",
                                               "site_index", "element"]

    def test_sites_carry_the_same_element_axis(self, md):
        sites = mv.multi.sites(md)
        assert set(sites.var_names) == set(md.var_names)
        assert sites.var.loc["Cu", "Z"] == pytest.approx(29.0)

    def test_X_is_the_one_hot_element(self, md):
        sites = mv.multi.sites(md)
        counts = np.asarray(sites.X.sum(axis=1)).ravel()
        assert np.allclose(counts, 1.0)              # exactly one element per atom

    def test_coordinates_are_kept(self, md):
        sites = mv.multi.sites(md)
        assert sites.obsm["X_frac"].shape == (sites.n_obs, 3)
        assert sites.obsm["X_cart"].shape == (sites.n_obs, 3)

    def test_forces_land_on_the_sites_axis(self, md):
        mv.pp.rattle(md, stdev=0.05, seed=0)
        sites = mv.multi.sites(md, source="rattled")
        mv.calc.forces(md, sites, level="emt", source="rattled")

        assert sites.obsm["forces_emt"].shape == (sites.n_obs, 3)
        assert (sites.obs["force_magnitude_emt"] > 0).any()
        assert mv.level_info(sites, "emt")["method"] == "EMT"

    def test_forces_refuse_a_mismatched_source(self, md):
        mv.pp.rattle(md, stdev=0.05, seed=0)
        sites = mv.multi.sites(md, source="input")
        with pytest.raises(ValueError, match="would not line up"):
            mv.calc.forces(md, sites, level="emt", source="rattled")

    def test_forces_refuse_a_materials_object(self, md):
        with pytest.raises(ValueError, match="must be a sites object"):
            mv.calc.forces(md, md, level="emt")

    def test_aggregate_bridges_back_to_materials(self, md):
        mv.pp.rattle(md, stdev=0.05, seed=0)
        sites = mv.multi.sites(md, source="rattled")
        mv.calc.forces(md, sites, level="emt", source="rattled")
        mv.multi.aggregate(sites, md, "force_magnitude_emt", how="max")

        assert md.obs["force_magnitude_emt_max"].shape == (6,)
        per_material = sites.obs.groupby("material_index", observed=True)[
            "force_magnitude_emt"].max()
        assert np.allclose(md.obs["force_magnitude_emt_max"].to_numpy(),
                           per_material.to_numpy())

    def test_aggregate_rejects_an_unknown_reducer(self, md):
        sites = mv.multi.sites(md)
        sites.obs["value"] = 1.0
        with pytest.raises(ValueError, match="unknown how"):
            mv.multi.aggregate(sites, md, "value", how="median-ish")

    def test_element_enrichment_works_on_the_sites_axis_too(self, md):
        """The dividend of giving both axes the same element var."""
        sites = mv.multi.sites(md)
        sites.obs["heavy"] = ["yes" if e in ("Cu", "Ni") else "no"
                              for e in sites.obs["element"]]
        mv.tl.rank_elements_groups(sites, "heavy")
        top = sites.uns["rank_elements_groups"]["no"].iloc[0]
        assert top["element"] == "Al"

    def test_mudata_assembly_is_optional_and_works(self, md):
        sites = mv.multi.sites(md)
        mdata = mv.multi.to_mudata(md, sites)
        assert set(mdata.mod) == {"materials", "sites"}
        assert mdata["sites"].n_obs == sites.n_obs


class TestHarmonize:
    @staticmethod
    def _two_databases(offsets):
        """The same four materials in two databases, one carrying an offset."""
        from pymatgen.core import Lattice, Structure

        def fcc(sym, a):
            return Structure(Lattice.cubic(a), [sym] * 4,
                             [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

        def l12(host, guest, a):
            return Structure(Lattice.cubic(a), [guest, host, host, host],
                             [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

        base = [fcc("Al", 4.05), fcc("Cu", 3.61),
                l12("Al", "Cu", 3.90), l12("Cu", "Al", 3.70)]
        reference_energy = np.array([-3.0, -3.5, -3.2, -3.4])
        fractions = np.array([[1.0, 0.0], [0.0, 1.0], [.75, .25], [.25, .75]])
        shifted = reference_energy + fractions @ np.array(offsets)

        obs = pd.DataFrame({
            "database": ["mp"] * 4 + ["oqmd"] * 4,
            "energy_per_atom_dft": np.concatenate([reference_energy, shifted]),
        })
        return mv.data.from_structures(base + [s.copy() for s in base], obs=obs)

    def test_recovers_a_known_per_element_offset(self):
        md = self._two_databases([0.30, -0.20])
        mv.pp.harmonize(md, batch_key="database",
                        energy_key="energy_per_atom_dft", reference="mp")

        fitted = md.uns["harmonize"]["offsets"]["oqmd"]
        order = list(md.uns["harmonize"]["elements"])
        assert fitted[order.index("Al")] == pytest.approx(0.30, abs=1e-6)
        assert fitted[order.index("Cu")] == pytest.approx(-0.20, abs=1e-6)

    def test_correction_removes_the_disagreement(self):
        md = self._two_databases([0.30, -0.20])
        mv.pp.harmonize(md, batch_key="database",
                        energy_key="energy_per_atom_dft", reference="mp")
        corrected = md.obs["energy_per_atom_dft_harmonized"].to_numpy(dtype=float)
        assert np.allclose(corrected[:4], corrected[4:], atol=1e-8)

    def test_reports_how_much_it_explained(self):
        md = self._two_databases([0.30, -0.20])
        mv.pp.harmonize(md, batch_key="database",
                        energy_key="energy_per_atom_dft", reference="mp")
        diagnostics = md.uns["harmonize"]["diagnostics"]["oqmd"]
        assert diagnostics["n_anchors"] == 4
        assert diagnostics["rmse_after"] < diagnostics["rmse_before"]
        assert diagnostics["rmse_after"] == pytest.approx(0.0, abs=1e-8)

    def test_leaves_the_reference_alone(self):
        md = self._two_databases([0.30, -0.20])
        original = md.obs["energy_per_atom_dft"].to_numpy(dtype=float).copy()
        mv.pp.harmonize(md, batch_key="database",
                        energy_key="energy_per_atom_dft", reference="mp")
        corrected = md.obs["energy_per_atom_dft_harmonized"].to_numpy(dtype=float)
        assert np.allclose(corrected[:4], original[:4])

    def test_a_non_compositional_difference_survives(self):
        """The honest limit: an offset linear in composition absorbs only the
        compositional part of a disagreement."""
        md = self._two_databases([0.0, 0.0])
        energies = md.obs["energy_per_atom_dft"].to_numpy(dtype=float).copy()
        energies[4] += 0.5                       # one structure disagrees, alone
        energies[5] -= 0.5
        md.obs["energy_per_atom_dft"] = energies

        mv.pp.harmonize(md, batch_key="database",
                        energy_key="energy_per_atom_dft", reference="mp")
        assert md.uns["harmonize"]["diagnostics"]["oqmd"]["rmse_after"] > 0.05

    def test_warns_when_nothing_is_shared(self, structures):
        obs = pd.DataFrame({
            "database": ["mp"] * 3 + ["oqmd"] * 3,
            "energy_per_atom_dft": np.arange(6, dtype=float),
        })
        md = mv.data.from_structures(structures, obs=obs)
        # Every formula is distinct, so no composition anchors the two sets.
        with pytest.warns(UserWarning, match="needs overlap"):
            mv.pp.harmonize(md, batch_key="database",
                            energy_key="energy_per_atom_dft", reference="mp")
        assert md.uns["harmonize"]["n_anchor_groups"] == 0

    def test_needs_more_than_one_database(self, md):
        md.obs["database"] = "mp"
        md.obs["e"] = np.zeros(6)
        with pytest.raises(ValueError, match="nothing to harmonise"):
            mv.pp.harmonize(md, batch_key="database", energy_key="e")


class TestGenerativeValidation:
    @staticmethod
    def _fcc(symbol, a):
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(a), [symbol] * 4,
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    @pytest.fixture
    def generated(self):
        from pymatgen.core import Lattice, Structure
        broken = Structure(Lattice.cubic(4.0), ["Pd", "Pd"],
                           [[0, 0, 0], [0.01, 0, 0]])
        return mv.data.from_structures([
            self._fcc("Al", 4.05),                     # duplicates a known phase
            self._fcc("Ni", 3.52),                     # novel
            self._fcc("Ni", 3.52),                     # repeat of the previous
            broken,                                    # invalid
        ])

    @pytest.fixture
    def known(self):
        return mv.data.from_structures([self._fcc("Al", 4.05),
                                        self._fcc("Cu", 3.61)])

    def test_scores_each_axis_separately(self, generated, known):
        mv.gen.validate(generated, reference=known)
        assert list(generated.obs["gen_valid"]) == [True, True, True, False]
        assert list(generated.obs["gen_unique"]) == [True, True, False, True]
        assert list(generated.obs["gen_novel"]) == [False, True, True, True]

    def test_rates_match_the_flags(self, generated, known):
        mv.gen.validate(generated, reference=known)
        rates = generated.uns["gen_validate"]["rates"]
        assert rates["valid"] == pytest.approx(0.75)
        assert rates["unique"] == pytest.approx(0.75)
        assert rates["novel"] == pytest.approx(0.75)

    def test_stability_is_not_assessed_rather_than_zero(self, generated, known):
        mv.gen.validate(generated, reference=known)
        report = generated.uns["gen_validate"]
        assert report["rates"]["stable"] is None
        assert report["rates"]["sun"] is None
        assert "stability" in report["not_assessed"]

    def test_a_closed_hull_cannot_certify_stability(self, generated, known):
        mv.calc.energy(generated, level="emt")
        with pytest.warns(UserWarning):
            mv.thermo.hull(generated, level="emt")
        mv.gen.validate(generated, reference=known, level="emt")
        assert "own compositions only" in \
            generated.uns["gen_validate"]["not_assessed"]["stability"]

    def test_sun_is_reported_once_the_hull_is_referenced(self, generated, known):
        mv.calc.energy(generated, level="emt")
        mv.calc.energy(known, level="emt")
        mv.thermo.hull(generated, level="emt", references=known)
        mv.gen.validate(generated, reference=known, level="emt",
                        metastability_threshold=5.0)
        report = generated.uns["gen_validate"]
        assert report["not_assessed"] == {}
        assert report["rates"]["msun"] is not None
        assert 0.0 <= report["rates"]["msun"] <= 1.0

    def test_definitions_are_recorded_with_the_rates(self, generated, known):
        mv.gen.validate(generated, reference=known, stability_threshold=0.03)
        definitions = generated.uns["gen_validate"]["definitions"]
        assert definitions["stability_threshold"] == 0.03
        assert definitions["reference"] == "2 structures"
        assert "LeMat-GenBench" in definitions["metric_family"]

    def test_novelty_needs_a_reference(self, generated):
        mv.gen.validate(generated)
        assert generated.uns["gen_validate"]["rates"]["novel"] is None
        assert "novelty" in generated.uns["gen_validate"]["not_assessed"]


class TestSubstitution:
    def test_enumerates_replacements(self, md):
        candidates = mv.gen.substitute(md, {"Al": ["Ga", "In"]})
        assert set(candidates.obs["substitution"]) == {"Al->Ga", "Al->In"}
        assert "Ga" in set(candidates.var_names)

    def test_parent_is_recorded(self, md):
        candidates = mv.gen.substitute(md, {"Cu": ["Ag"]})
        assert set(candidates.obs["parent"]) <= set(md.obs_names)

    def test_keeping_parents_includes_them(self, md):
        candidates = mv.gen.substitute(md, {"Al": ["Ga"]}, keep_parents=True)
        assert candidates.obs["is_parent"].sum() == md.n_obs

    def test_nothing_to_substitute_says_so(self, md):
        with pytest.raises(ValueError, match="no candidate survived"):
            mv.gen.substitute(md, {"Xe": ["Kr"]})


class TestExperimentalHull:
    """A hull from measured enthalpies, checked against the Fe-O system.

    The numbers are NIST-JANAF standard formation enthalpies at 298 K, and the
    answer is a known piece of metallurgy: hematite and magnetite are stable,
    and wustite is not — FeO disproportionates to iron and magnetite below
    about 570 C. A hull that puts all three on it has got the units or the
    references wrong.
    """

    #: kJ/mol at 298 K, NIST-JANAF.
    DHF = {"Fe2O3": -824.2, "Fe3O4": -1118.4, "FeO": -272.0, "Fe": 0.0}
    EV_PER_KJ_MOL = 1.0 / 96.48533212331

    @staticmethod
    def _cell(formula):
        from pymatgen.core import Composition, Lattice, Structure
        composition = Composition(formula)
        symbols = [str(element) for element in composition.elements
                   for _ in range(int(composition[element]))]
        n = len(symbols)
        return Structure(Lattice.cubic(10.0), symbols,
                         [[i / n, 0, 0] for i in range(n)])

    @pytest.fixture
    def iron_oxides(self):
        names = list(self.DHF)
        md = mv.data.from_structures([self._cell(f) for f in names])
        md.obs_names = names
        mv.exp.measure(md, "dHf", [self.DHF[f] for f in names],
                       level="janaf", instrument="NIST-JANAF")
        return md

    @staticmethod
    def _column(md):
        return next(c for c in md.obs.columns if c.startswith("dHf"))

    def test_the_conversion_is_per_atom_of_the_formula(self, iron_oxides):
        """-824.2 kJ/mol of Fe2O3 is -8.542 eV per formula unit, and Fe2O3 has
        five atoms, so -1.7084 eV/atom. Two divisions, both easy to skip."""
        mv.exp.formation_hull(iron_oxides, self._column(iron_oxides),
                              unit="kJ/mol", level="janaf")
        expected = self.DHF["Fe2O3"] * self.EV_PER_KJ_MOL / 5.0
        assert float(iron_oxides.obs["formation_energy_janaf"]["Fe2O3"]) == \
            pytest.approx(expected, rel=1e-9)
        assert expected == pytest.approx(-1.7084, abs=1e-4)

    def test_hematite_and_magnetite_are_stable(self, iron_oxides):
        mv.exp.formation_hull(iron_oxides, self._column(iron_oxides),
                              unit="kJ/mol", level="janaf")
        stable = iron_oxides.obs["is_stable_janaf"]
        assert bool(stable["Fe2O3"]) and bool(stable["Fe3O4"])

    def test_wustite_comes_out_metastable(self, iron_oxides):
        """The result worth having, and it is not an artefact: the tie line
        from Fe to Fe3O4 at x_O = 1/2 sits at -1.4489 eV/atom and FeO is at
        -1.4095, so it is 0.0394 above. Computed here from the tie line rather
        than pasted, so a change in either number has to move both."""
        mv.exp.formation_hull(iron_oxides, self._column(iron_oxides),
                              unit="kJ/mol", level="janaf")
        magnetite = self.DHF["Fe3O4"] * self.EV_PER_KJ_MOL / 7.0
        on_tie_line = magnetite * (0.5 / (4.0 / 7.0))
        feo = self.DHF["FeO"] * self.EV_PER_KJ_MOL / 2.0
        assert float(iron_oxides.obs["e_above_hull_janaf"]["FeO"]) == \
            pytest.approx(feo - on_tie_line, abs=1e-6)
        assert not bool(iron_oxides.obs["is_stable_janaf"]["FeO"])

    def test_the_oxygen_reference_is_supplied_without_a_row(self, iron_oxides):
        """An oxide hull needs its O2 corner, and nobody has a row for oxygen
        gas. pymatgen's ExpEntry cannot hold one at all — it rejects any phase
        marked gas or liquid — which is a large part of why this is not a
        wrapper."""
        mv.exp.formation_hull(iron_oxides, self._column(iron_oxides),
                              unit="kJ/mol", level="janaf")
        assert "O2" in iron_oxides.uns["experimental_hull"]["janaf"]["stable"]
        assert "O" not in list(iron_oxides.obs_names)

    def test_the_unit_is_required_and_checked(self, iron_oxides):
        with pytest.raises(ValueError, match="unit must be one of"):
            mv.exp.formation_hull(iron_oxides, self._column(iron_oxides),
                                  unit="joules")

    def test_quoting_the_same_data_two_ways_agrees(self, iron_oxides):
        """The unit argument has to actually do something. Feeding eV/atom
        directly must reproduce the kJ/mol route exactly — this is the check
        that a silently-ignored unit would fail, which is the failure mode
        ExpEntry has."""
        column = self._column(iron_oxides)
        mv.exp.formation_hull(iron_oxides, column, unit="kJ/mol",
                              level="from_kj")
        iron_oxides.obs["dHf_ev"] = \
            iron_oxides.obs["formation_energy_from_kj"]
        mv.exp.formation_hull(iron_oxides, "dHf_ev", unit="eV/atom",
                              level="from_ev")
        assert np.allclose(iron_oxides.obs["e_above_hull_from_kj"],
                           iron_oxides.obs["e_above_hull_from_ev"], atol=1e-9)

    def test_treating_kj_as_ev_would_be_wrong_by_96(self, iron_oxides):
        """Documents the failure this function exists to prevent. Handing the
        table's numbers over as if they were eV — which is exactly what
        ExpEntry does — leaves a hull whose energies are off by the Faraday
        constant, and it does not complain."""
        column = self._column(iron_oxides)
        mv.exp.formation_hull(iron_oxides, column, unit="kJ/mol",
                              level="right")
        mv.exp.formation_hull(iron_oxides, column, unit="eV", level="wrong")
        ratio = (float(iron_oxides.obs["formation_energy_wrong"]["Fe2O3"])
                 / float(iron_oxides.obs["formation_energy_right"]["Fe2O3"]))
        assert ratio == pytest.approx(96.485, rel=1e-3)

    def test_a_missing_column_says_how_to_make_one(self, iron_oxides):
        with pytest.raises(ValueError, match="mv.exp.measure"):
            mv.exp.formation_hull(iron_oxides, "not_a_column", unit="kJ/mol")

    def test_the_level_records_where_the_numbers_came_from(self, iron_oxides):
        mv.exp.formation_hull(iron_oxides, self._column(iron_oxides),
                              unit="kJ/mol", level="janaf")
        assert "janaf" in mv.levels_used(iron_oxides)
        assert iron_oxides.uns["levels"]["janaf"]["kind"] == "experiment"

"""Local and coordination environments.

The point of ``mv.env`` is the one question a composition vector provably cannot
answer, so the tests check chemistry rather than plumbing: olivine LiFePO4 has
octahedral Li and Fe and a rigid tetrahedral phosphate, and if the namespace
ever stops saying so it is broken regardless of what it returns.
"""

from __future__ import annotations

import numpy as np
import pytest

import matverse as mv


@pytest.fixture(scope="module")
def olivine():
    md = mv.datasets.load("battery_cathodes")[:1].copy()
    mv.pp.describe(md)
    return md


@pytest.fixture(scope="module")
def sited(olivine):
    sites = mv.multi.sites(olivine)
    mv.env.coordination(olivine, sites)
    return sites


class TestCoordination:
    def test_the_numbers_are_the_published_ones(self, sited):
        """LiFePO4: LiO6 and FeO6 octahedra, PO4 tetrahedra. Any near-neighbour
        algorithm that does not recover this is not usable."""
        by_element = sited.obs.groupby("element", observed=True)[
            "coordination_number"].mean()
        assert by_element["Li"] == pytest.approx(6.0)
        assert by_element["Fe"] == pytest.approx(6.0)
        assert by_element["P"] == pytest.approx(4.0)

    def test_the_strategy_is_recorded_with_the_number(self, sited):
        """CrystalNN and MinimumDistanceNN routinely disagree by one or two on
        the same site, so a coordination number without its strategy is not
        reproducible."""
        assert set(sited.obs["coordination_strategy"]) == {"crystalnn"}
        assert sited.uns["coordination"]["strategy"] == "crystalnn"
        assert sited.uns["coordination"]["description"]

    def test_a_second_strategy_gives_its_own_answer(self, olivine):
        sites = mv.multi.sites(olivine)
        mv.env.coordination(olivine, sites, strategy="minimum_distance")
        assert set(sites.obs["coordination_strategy"]) == {"minimum_distance"}
        assert np.isfinite(
            sites.obs["coordination_number"].to_numpy(dtype=float)).any()

    def test_an_unknown_strategy_lists_the_options(self, olivine):
        sites = mv.multi.sites(olivine)
        with pytest.raises(ValueError, match="unknown strategy"):
            mv.env.coordination(olivine, sites, strategy="telepathy")

    def test_it_refuses_the_wrong_axis(self, olivine):
        """Coordination is one number per atom, so it needs the atom axis."""
        with pytest.raises(ValueError, match="not a sites object"):
            mv.env.coordination(olivine, olivine)

    def test_neighbour_distances_are_physical(self, sited):
        distances = sited.obs["mean_neighbour_distance"].to_numpy(dtype=float)
        finite = distances[np.isfinite(distances)]
        assert finite.size
        assert (finite > 1.0).all() and (finite < 4.0).all()


class TestChemenv:
    @pytest.fixture(scope="class")
    def classified(self, olivine):
        sites = mv.multi.sites(olivine)
        mv.env.chemenv(olivine, sites)
        return sites

    def test_octahedra_and_tetrahedra_are_named(self, classified):
        frame = classified.obs
        assert set(frame.loc[frame["element"] == "Li", "environment"]) == {"O:6"}
        assert set(frame.loc[frame["element"] == "Fe", "environment"]) == {"O:6"}
        assert set(frame.loc[frame["element"] == "P", "environment"]) == {"T:4"}

    def test_the_phosphate_is_more_regular_than_the_metal_octahedra(
            self, classified):
        """The structural story of an olivine cathode: a rigid PO4 tetrahedron
        in a framework of distorted metal octahedra. The continuous symmetry
        measure says so, and a coordination number alone cannot."""
        frame = classified.obs
        phosphate = frame.loc[frame["element"] == "P",
                              "environment_csm"].mean()
        iron = frame.loc[frame["element"] == "Fe", "environment_csm"].mean()
        assert phosphate < 0.5
        assert iron > 1.5
        assert phosphate < iron

    def test_the_symmetry_measure_is_kept_not_just_the_name(self, classified):
        """A coordination number of 6 can be octahedral, trigonal prismatic or
        a badly distorted octahedron. Reporting the name without the measure
        would make those indistinguishable."""
        csm = classified.obs["environment_csm"].to_numpy(dtype=float)
        assert np.isfinite(csm).any()
        assert (csm[np.isfinite(csm)] >= 0).all()

    def test_the_settings_are_recorded(self, classified):
        assert classified.uns["chemenv"]["max_csm"] == 8.0
        assert "octahedral" in classified.uns["chemenv"]["note"]


class TestBonds:
    def test_the_network_lands_in_obsp(self, olivine):
        sites = mv.multi.sites(olivine)
        mv.env.bonds(olivine, sites)
        assert sites.obsp["bonds"].shape == (sites.n_obs, sites.n_obs)
        assert sites.obsp["bonds"].nnz > 0
        assert sites.obsp["bond_distances"].nnz == sites.obsp["bonds"].nnz

    def test_bonds_do_not_cross_materials(self):
        """Two atoms in different structures are never bonded, so the matrix is
        block diagonal — otherwise a graph algorithm would walk from one
        material into another."""
        md = mv.datasets.load("battery_cathodes")
        mv.pp.describe(md)
        sites = mv.multi.sites(md)
        mv.env.bonds(md, sites)

        index = sites.obs["material_index"].to_numpy(dtype=int)
        rows, cols = sites.obsp["bonds"].nonzero()
        assert (index[rows] == index[cols]).all()

    def test_degree_matches_the_coordination_number(self, olivine):
        sites = mv.multi.sites(olivine)
        mv.env.coordination(olivine, sites)
        mv.env.bonds(olivine, sites)

        degree = np.asarray(sites.obsp["bonds"].sum(axis=1)).ravel()
        expected = sites.obs["coordination_number"].to_numpy(dtype=float)
        assert degree == pytest.approx(expected)


class TestSummarise:
    def test_it_reaches_the_material_axis(self, olivine, sited):
        mv.env.summarise(sited, olivine)
        for column in ("mean_coordination", "min_coordination",
                       "max_coordination", "coordination_spread"):
            assert column in olivine.obs
        assert np.isfinite(
            olivine.obs["mean_coordination"].to_numpy(dtype=float)).all()

    def test_a_screen_can_reach_it(self, olivine, sited):
        mv.env.summarise(sited, olivine)
        mv.screen.filter(olivine, max_coordination__ge=6.0, name="has_octahedra")
        assert bool(olivine.obs["has_octahedra"].iloc[0])

    def test_it_needs_coordination_first(self, olivine):
        bare = mv.multi.sites(olivine)
        with pytest.raises(ValueError, match="mv.env.coordination"):
            mv.env.summarise(bare, olivine)


class TestDimensionality:
    """The same bond graph, asked a global question instead of a local one.

    The four cases have textbook answers, which is the point of choosing them:
    graphite and MoS2 are the archetypal layered materials, fcc copper is a
    framework, and iodine at a wide separation is molecular.
    """

    @staticmethod
    def _graphite():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.hexagonal(2.46, 6.70), ["C"] * 4,
                         [[0, 0, 0], [1 / 3, 2 / 3, 0],
                          [0, 0, 0.5], [2 / 3, 1 / 3, 0.5]])

    @staticmethod
    def _mos2():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.hexagonal(3.16, 12.3),
                         ["Mo", "Mo", "S", "S", "S", "S"],
                         [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75],
                          [2 / 3, 1 / 3, 0.371], [1 / 3, 2 / 3, 0.629],
                          [1 / 3, 2 / 3, 0.871], [2 / 3, 1 / 3, 0.129]])

    @staticmethod
    def _copper():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(3.61), ["Cu"] * 4,
                         [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])

    @staticmethod
    def _iodine():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.cubic(12.0), ["I", "I"],
                         [[0, 0, 0], [0.222, 0, 0]])

    @pytest.fixture(scope="class")
    def mixed(self):
        md = mv.data.from_structures(
            [self._graphite(), self._mos2(), self._copper(), self._iodine()])
        mv.pp.describe(md)
        mv.prop.dimensionality(md)
        return md

    def test_the_textbook_answers(self, mixed):
        assert list(mixed.obs["dimensionality"]) == [2, 2, 3, 0]

    def test_layers_per_cell_are_counted(self, mixed):
        """Graphite and MoS2 both have two layers in the conventional cell; a
        framework has one component by construction."""
        assert list(mixed.obs["n_components"]) == [2, 2, 1, 1]

    def test_a_screen_can_ask_for_exfoliable_candidates(self, mixed):
        mv.screen.filter(mixed, dimensionality__eq=2, name="exfoliable")
        assert list(mixed.obs["exfoliable"]) == [True, True, False, False]

    def test_the_strategy_is_recorded_next_to_the_answer(self, mixed):
        """A dimensionality without its near-neighbour algorithm is not
        reproducible, for the same reason a coordination number is not."""
        assert set(mixed.obs["dimensionality_strategy"]) == {"crystalnn"}
        assert mixed.uns["dimensionality"]["strategy"] == "crystalnn"

    def test_it_is_not_derivable_from_composition(self, mixed):
        """Two materials can share X and differ here, which is the argument
        for computing it at all. Graphite and diamond are the same element."""
        from pymatgen.core import Lattice, Structure
        diamond = Structure(Lattice.cubic(3.567), ["C"] * 8,
                            [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                             [.25, .25, .25], [.25, .75, .75],
                             [.75, .25, .75], [.75, .75, .25]])
        md = mv.data.from_structures([self._graphite(), diamond])
        mv.prop.dimensionality(md)
        assert list(md.obs["dimensionality"]) == [2, 3]
        assert (md.X[0].toarray() > 0).sum() == (md.X[1].toarray() > 0).sum()

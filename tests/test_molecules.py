"""Molecules on the same substrate as crystals.

The design note used to say molecules were out of scope "by construction".
These tests exist because that was wrong: the composition axis does not care
whether a formula unit repeats, and only the decoder assumed a lattice.

Point groups have textbook answers, so the tests assert chemistry — C2v for
water, Td for methane, and the selection rule that follows from Td, which is
that methane cannot carry a dipole no matter how it is distorted within its
symmetry.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")


def _water():
    from pymatgen.core import Molecule
    return Molecule(["O", "H", "H"],
                    [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


def _methane():
    from pymatgen.core import Molecule
    return Molecule(["C", "H", "H", "H", "H"],
                    [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                     [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])


def _ammonia():
    from pymatgen.core import Molecule
    return Molecule(["N", "H", "H", "H"],
                    [[0, 0, 0.12], [0, 0.94, -0.27],
                     [0.81, -0.47, -0.27], [-0.81, -0.47, -0.27]])


def _ethanol():
    from pymatgen.core import Molecule
    return Molecule(
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [[-1.2, 0.2, 0], [0.0, -0.6, 0], [1.1, 0.3, 0],
         [-1.2, 0.9, 0.9], [-1.2, 0.9, -0.9], [-2.1, -0.4, 0],
         [0.0, -1.2, 0.9], [0.0, -1.2, -0.9], [1.9, -0.2, 0]])


@pytest.fixture(scope="module")
def molecules():
    md = mv.mol.from_molecules([_water(), _methane(), _ammonia(), _ethanol()])
    mv.pp.describe(md)
    return md


class TestSubstrate:
    def test_the_composition_axis_works_unchanged(self):
        """H2O contributes H:2, O:1 exactly as a crystal would — which is why
        molecules were never actually out of scope."""
        md = mv.mol.from_molecules([_water()])
        counts = dict(zip(md.var_names, md.X.toarray().ravel()))
        assert counts == {"H": 2.0, "O": 1.0}

    def test_a_molecule_survives_a_round_trip(self):
        from pymatgen.core import Molecule

        md = mv.mol.from_molecules([_methane()])
        back = mv.structures(md)[0]
        assert isinstance(back, Molecule)
        assert back.composition.reduced_formula == "H4C"

    def test_one_object_can_hold_both_kinds(self):
        crystal = mv.structures(mv.datasets.metals(["Cu"]))[0]
        md = mv.data.from_structures([_water(), crystal])
        mv.pp.describe(md)
        assert list(md.obs["is_periodic"]) == [False, True]
        assert np.isnan(md.obs["volume"].to_numpy(dtype=float)[0])
        assert np.isfinite(md.obs["volume"].to_numpy(dtype=float)[1])

    def test_molecular_weight_is_there_for_both(self, molecules):
        weights = molecules.obs["molecular_weight"].to_numpy(dtype=float)
        assert weights[0] == pytest.approx(18.015, abs=0.01)   # water

    def test_the_sites_axis_works(self, molecules):
        sites = mv.multi.sites(molecules)
        assert sites.n_obs == sum(molecules.obs["nsites"])
        assert np.isfinite(sites.obsm["X_cart"]).all()
        # A molecule has no lattice, so fractional coordinates are absent
        assert np.isnan(sites.obsm["X_frac"]).all()

    def test_a_periodic_dataset_still_has_fractional_coordinates(self):
        md = mv.datasets.metals(["Cu"])
        sites = mv.multi.sites(md)
        assert np.isfinite(sites.obsm["X_frac"]).all()


class TestPointGroup:
    def test_the_textbook_answers(self, molecules):
        mv.mol.point_group(molecules)
        groups = dict(zip(molecules.obs["formula"],
                          molecules.obs["point_group"]))
        assert groups["H2O"] == "C2v"
        assert groups["H4C"] == "Td"
        assert groups["H3N"] == "C3v"

    def test_the_order_matches_the_group(self, molecules):
        mv.mol.point_group(molecules)
        orders = dict(zip(molecules.obs["formula"],
                          molecules.obs["symmetry_order"]))
        assert orders["H2O"] == 4        # E, C2, 2 sigma_v
        assert orders["H4C"] == 24       # Td
        assert orders["H3N"] == 6        # E, 2C3, 3 sigma_v

    def test_symmetry_forbids_a_dipole_in_methane(self, molecules):
        """A hard selection rule, not a tendency: Td contains operations that
        map any candidate dipole onto its negative."""
        mv.mol.point_group(molecules)
        polar = dict(zip(molecules.obs["formula"],
                         molecules.obs["can_be_polar"]))
        assert polar["H2O"] is np.True_ or polar["H2O"]
        assert polar["H3N"]
        assert not polar["H4C"]

    def test_none_of_these_are_chiral(self, molecules):
        """All four have a mirror plane, so none has an enantiomer."""
        mv.mol.point_group(molecules)
        assert not molecules.obs["is_chiral"].any()

    def test_it_refuses_a_purely_periodic_dataset(self):
        md = mv.datasets.metals(["Cu"])
        with pytest.raises(ValueError, match="every row of this dataset is"):
            mv.mol.point_group(md)


class TestDescriptors:
    def test_ethanol_has_two_rotatable_bonds(self, molecules):
        """C-C and C-O. The C-H bonds are terminal and do not count."""
        mv.mol.descriptors(molecules)
        rotatable = dict(zip(molecules.obs["formula"],
                             molecules.obs["rotatable_bonds"]))
        assert rotatable["H6C2O"] == 2

    def test_bond_counts_are_right(self, molecules):
        mv.mol.descriptors(molecules)
        bonds = dict(zip(molecules.obs["formula"], molecules.obs["n_bonds"]))
        assert bonds["H2O"] == 2
        assert bonds["H4C"] == 4
        assert bonds["H3N"] == 3
        assert bonds["H6C2O"] == 8

    def test_nothing_here_has_a_ring(self, molecules):
        mv.mol.descriptors(molecules)
        assert (molecules.obs["n_rings"].to_numpy(dtype=float) == 0).all()

    def test_heavy_atoms_excludes_hydrogen(self, molecules):
        mv.mol.descriptors(molecules)
        heavy = dict(zip(molecules.obs["formula"],
                         molecules.obs["heavy_atoms"]))
        assert heavy["H2O"] == 1
        assert heavy["H6C2O"] == 3


class TestBonds:
    def test_the_graph_lands_in_obsp(self, molecules):
        sites = mv.multi.sites(molecules)
        mv.mol.bonds(molecules, sites)
        assert sites.obsp["bonds"].nnz > 0
        assert sites.uns["bonds"]["kind"] == "covalent"

    def test_it_lands_where_mv_env_bonds_lands(self, molecules):
        """A graph algorithm should not need to know which kind of material it
        was handed."""
        sites = mv.multi.sites(molecules)
        mv.mol.bonds(molecules, sites)
        assert set(sites.obsp) >= {"bonds", "bond_distances"}

    def test_bonds_do_not_cross_molecules(self, molecules):
        sites = mv.multi.sites(molecules)
        mv.mol.bonds(molecules, sites)
        index = sites.obs["material_index"].to_numpy(dtype=int)
        rows, cols = sites.obsp["bonds"].nonzero()
        assert (index[rows] == index[cols]).all()

    def test_a_missing_backend_names_the_install(self, molecules):
        sites = mv.multi.sites(molecules)
        try:
            import openbabel                             # noqa: F401
            pytest.skip("openbabel is installed here")
        except ImportError:
            pass
        with pytest.raises((ImportError, Exception)):
            mv.mol.bonds(molecules, sites, strategy="openbabel")


class TestFragments:
    @pytest.fixture(scope="class")
    def pieces(self):
        md = mv.mol.from_molecules([_ethanol()])
        mv.pp.describe(md)
        out = mv.mol.fragments(md)
        mv.pp.describe(out)
        return out

    def test_atoms_are_conserved_by_every_cut(self, pieces):
        """One cut splits nine atoms into two pieces that still total nine."""
        totals = pieces.obs.groupby("fragment_index")["fragment_size"].sum()
        assert (totals == 9).all()

    def test_breaking_the_c_c_bond_gives_the_expected_pieces(self, pieces):
        cut = pieces.obs[pieces.obs["broken_bond"] == "C0-C1"]
        assert set(cut["fragment_formula"]) == {"H3C1", "H3C1O1"}

    def test_the_unreduced_formula_is_recorded(self, pieces):
        """pymatgen's reduced formula applies the diatomic convention, so a
        single hydrogen atom reads 'H2' and a hydroxyl reads 'H2O2'. A fragment
        is a set of atoms, not a stoichiometry."""
        single = pieces.obs[pieces.obs["fragment_size"] == 1]
        assert not single.empty
        assert set(single["fragment_formula"]) == {"H1"}
        assert set(single["formula"]) == {"H2"}      # the misleading one

    def test_ring_bonds_are_not_cut(self):
        """Breaking one ring bond leaves the molecule connected, so it produces
        no fragmentation at all."""
        from pymatgen.core import Molecule

        angle = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ring = Molecule(["C"] * 6,
                        np.c_[1.4 * np.cos(angle), 1.4 * np.sin(angle),
                              np.zeros(6)])
        md = mv.mol.from_molecules([ring])
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="ring"):
            mv.mol.fragments(md)

    def test_rows_point_back_at_the_parent(self, pieces):
        assert set(pieces.obs["parent"]) == {"0"}


class TestMatch:
    def test_a_translated_copy_is_a_duplicate(self):
        from pymatgen.core import Molecule

        water = _water()
        moved = Molecule(["O", "H", "H"],
                         np.asarray(water.cart_coords) + 5.0)
        md = mv.mol.from_molecules([water, moved, _ethanol()])
        mv.pp.describe(md)
        mv.mol.match(md)

        assert list(md.obs["is_duplicate"]) == [False, True, False]
        assert md.uns["molecule_match"]["n_unique"] == 2

    def test_different_molecules_are_not_grouped(self, molecules):
        mv.mol.match(molecules)
        assert not molecules.obs["is_duplicate"].any()
        assert molecules.uns["molecule_match"]["n_unique"] == molecules.n_obs

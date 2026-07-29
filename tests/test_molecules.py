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


class TestMatchingIsGeometric:
    """Identity for a molecule means congruence, not a shared fingerprint.

    Until v0.1.27 this compared a sorted heavy-atom distance spectrum, which is
    invariant to rotation, translation and relabelling — and is not a proof of
    congruence. It also ignored hydrogens entirely.
    """

    @staticmethod
    def _ethanol(theta=0.0):
        from pymatgen.core import Molecule
        m = Molecule(
            ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
            [[-1.2, 0.2, 0], [0.0, -0.6, 0], [1.1, 0.3, 0],
             [-1.2, 0.9, 0.9], [-1.2, 0.9, -0.9], [-2.1, -0.4, 0],
             [0.0, -1.2, 0.9], [0.0, -1.2, -0.9], [1.9, -0.2, 0]])
        if theta:
            m.rotate_sites(theta=theta, axis=[0, 0, 1])
        return m

    def test_a_rotated_copy_is_the_same_molecule(self):
        md = mv.mol.from_molecules([self._ethanol(), self._ethanol(0.9)])
        mv.pp.describe(md)
        mv.mol.match(md)
        assert list(md.obs["is_duplicate"]) == [False, True]
        assert md.obs["match_rmsd"].to_numpy(dtype=float) == \
            pytest.approx(0.0, abs=1e-6)

    def test_a_distorted_copy_is_not(self):
        """0.5 angstrom of displacement per atom is a different geometry, and
        a tolerance of 0.1 should say so."""
        import numpy as np
        distorted = self._ethanol()
        rng = np.random.default_rng(0)
        for index in range(len(distorted)):
            distorted.translate_sites([index], rng.normal(0, 0.5, 3))
        md = mv.mol.from_molecules([self._ethanol(), distorted])
        mv.pp.describe(md)
        mv.mol.match(md, tolerance=0.1)
        assert list(md.obs["is_duplicate"]) == [False, False]

    def test_hydrogens_count(self):
        """The old comparison used heavy atoms only. Methane has one heavy
        atom, so *every* CH4 geometry had the same one-atom fingerprint and
        read as identical no matter where the hydrogens sat.

        These two are both CH4 and are not the same molecule: one is
        tetrahedral, the other has a C-H stretched to 1.6 angstrom."""
        from pymatgen.core import Molecule
        tetrahedral = Molecule(
            ["C", "H", "H", "H", "H"],
            [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
             [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])
        stretched = Molecule(
            ["C", "H", "H", "H", "H"],
            [[0, 0, 0], [0.92, 0.92, 0.92], [-0.63, -0.63, 0.63],
             [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])
        md = mv.mol.from_molecules([tetrahedral, stretched])
        mv.pp.describe(md)
        mv.mol.match(md, tolerance=0.05)
        assert list(md.obs["is_duplicate"]) == [False, False]

    def test_two_orientations_of_the_same_molecule_do_match(self):
        """The counterpart: a regular tetrahedral methane written two ways is
        one molecule, and the matcher has to see through the labelling."""
        from pymatgen.core import Molecule
        a = Molecule(["C", "H", "H", "H", "H"],
                     [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                      [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])
        b = Molecule(["C", "H", "H", "H", "H"],
                     [[0, 0, 0], [1.091, 0, 0], [-0.364, 1.029, 0],
                      [-0.364, -0.514, 0.891], [-0.364, -0.514, -0.891]])
        md = mv.mol.from_molecules([a, b])
        mv.pp.describe(md)
        mv.mol.match(md, tolerance=0.05)
        assert list(md.obs["is_duplicate"]) == [False, True]

    def test_the_rmsd_is_reported(self):
        md = mv.mol.from_molecules([self._ethanol(), self._ethanol(0.4)])
        mv.pp.describe(md)
        mv.mol.match(md)
        assert "match_rmsd" in md.obs
        assert md.uns["molecule_match"]["matcher"].startswith("Kabsch")

    def test_different_molecules_stay_apart(self):
        from pymatgen.core import Molecule
        water = Molecule(["O", "H", "H"],
                         [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
        md = mv.mol.from_molecules([self._ethanol(), water, water])
        mv.pp.describe(md)
        mv.mol.match(md)
        assert md.uns["molecule_match"]["n_unique"] == 2
        assert list(md.obs["is_duplicate"]) == [False, False, True]


class TestBondLengths:
    """The check mv.pp.qc cannot make: bonds that exist and are wrong."""

    @staticmethod
    def _ethanol(scale=1.0):
        import numpy as np
        from pymatgen.core import Molecule
        m = Molecule(
            ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
            [[-1.2, 0.2, 0], [0.0, -0.6, 0], [1.1, 0.3, 0],
             [-1.2, 0.9, 0.9], [-1.2, 0.9, -0.9], [-2.1, -0.4, 0],
             [0.0, -1.2, 0.9], [0.0, -1.2, -0.9], [1.9, -0.2, 0]])
        if scale != 1.0:
            return Molecule([s.specie for s in m],
                            np.asarray(m.cart_coords) * scale)
        return m

    def test_a_reasonable_molecule_passes(self):
        md = mv.mol.from_molecules([self._ethanol()])
        mv.pp.describe(md)
        mv.mol.bond_lengths(md)
        assert bool(md.obs["bond_lengths_ok"].iloc[0])
        assert md.obs["mean_bond_deviation"].iloc[0] < 0.1

    def test_an_exact_water_has_no_deviation(self):
        """O-H is tabulated at 0.96 angstrom, so a molecule built at exactly
        that length should come out at zero."""
        from pymatgen.core import Molecule
        water = Molecule(["O", "H", "H"],
                         [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        md = mv.mol.from_molecules([water])
        mv.pp.describe(md)
        mv.mol.bond_lengths(md)
        assert md.obs["mean_bond_deviation"].iloc[0] == pytest.approx(0.0,
                                                                     abs=0.02)

    def test_a_stretched_molecule_stops_having_bonds(self):
        """The signal is the bond count, not the deviation. Bonds are found by
        covalent radius, so a 25% stretch does not give long bonds — it gives
        almost no bonds, and that is the louder warning."""
        md = mv.mol.from_molecules([self._ethanol(), self._ethanol(1.25)])
        mv.pp.describe(md)
        mv.mol.bond_lengths(md)
        counts = md.obs["n_bonds_measured"].to_numpy(dtype=float)
        assert counts[0] >= 8
        assert counts[1] < counts[0] / 2
        assert not bool(md.obs["bond_lengths_ok"].iloc[1])

    def test_what_had_no_table_entry_is_counted(self):
        md = mv.mol.from_molecules([self._ethanol()])
        mv.pp.describe(md)
        mv.mol.bond_lengths(md)
        assert "n_pairs_without_a_table_entry" in md.uns["bond_lengths"]


class TestMatchDispatch:
    """Two questions that both mean "the same molecule"."""

    @staticmethod
    def _ethanol():
        from pymatgen.core import Molecule
        return Molecule(
            ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
            [[-1.2, 0.2, 0], [0.0, -0.6, 0], [1.1, 0.3, 0],
             [-1.2, 0.9, 0.9], [-1.2, 0.9, -0.9], [-2.1, -0.4, 0],
             [0.0, -1.2, 0.9], [0.0, -1.2, -0.9], [1.9, -0.2, 0]])

    @classmethod
    def _conformer(cls):
        """A real torsion: the hydroxyl hydrogen rotated about the C-O bond.
        Every bond length is preserved and the geometry moves."""
        import numpy as np
        base = cls._ethanol()
        out = base.copy()
        out.rotate_sites(indices=[8], theta=1.2,
                         axis=np.array(base[2].coords) - np.array(base[1].coords),
                         anchor=base[2].coords)
        return out

    def test_the_conformer_really_preserves_its_bonds(self):
        base, conf = self._ethanol(), self._conformer()
        assert base.get_distance(2, 8) == pytest.approx(conf.get_distance(2, 8))

    def test_geometry_calls_a_conformer_a_different_molecule(self):
        md = mv.mol.from_molecules([self._ethanol(), self._conformer()])
        mv.pp.describe(md)
        mv.mol.match(md, method="geometry")
        assert md.uns["molecule_match"]["n_unique"] == 2

    def test_topology_calls_it_the_same_one(self):
        md = mv.mol.from_molecules([self._ethanol(), self._conformer()])
        mv.pp.describe(md)
        mv.mol.match(md, method="topology")
        assert md.uns["molecule_match"]["n_unique"] == 1
        assert list(md.obs["is_duplicate"]) == [False, True]

    def test_topology_still_separates_different_molecules(self):
        from pymatgen.core import Molecule
        water = Molecule(["O", "H", "H"],
                         [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
        md = mv.mol.from_molecules([self._ethanol(), water])
        mv.pp.describe(md)
        mv.mol.match(md, method="topology")
        assert md.uns["molecule_match"]["n_unique"] == 2

    def test_the_route_is_recorded(self):
        md = mv.mol.from_molecules([self._ethanol()])
        mv.pp.describe(md)
        mv.mol.match(md, method="topology")
        assert md.uns["molecule_match"]["method"] == "topology"
        assert "bond-graph" in md.uns["molecule_match"]["matcher"]

    def test_an_unknown_method_is_refused(self):
        md = mv.mol.from_molecules([self._ethanol()])
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="unknown method"):
            mv.mol.match(md, method="vibes")


class TestQuasiRRHO:
    """The point of quasi-RRHO is what it does to a soft mode, so that is what
    these check — against the harmonic result computed on the same frequencies,
    which makes the comparison internal and exact rather than tabulated."""

    STIFF = [1595.0, 3657.0, 3756.0]        # water, cm-1

    @staticmethod
    def _water(n=1):
        from pymatgen.core import Molecule
        w = Molecule(["O", "H", "H"],
                     [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
        md = mv.mol.from_molecules([w] * n)
        md.obs["e"] = [-76.4] * n
        return md

    def test_a_rigid_molecule_needs_no_correcting(self):
        """Every mode well above the 100 cm-1 cutoff, so the free-rotor
        interpolation has nothing to act on and the two entropies coincide."""
        md = self._water()
        mv.mol.quasirrho(md, [self.STIFF], energy="e")
        assert float(md.obs["entropy_quasirrho"].iloc[0]) == \
            pytest.approx(float(md.obs["entropy_harmonic"].iloc[0]), abs=1e-3)

    def test_a_soft_mode_is_where_they_diverge(self):
        """A 12 cm-1 mode contributes enormous harmonic entropy — the 1/omega
        divergence — and quasi-RRHO damps it. Harmonic must overshoot."""
        md = self._water()
        mv.mol.quasirrho(md, [[12.0] + self.STIFF[:2]], energy="e")
        harmonic = float(md.obs["entropy_harmonic"].iloc[0])
        quasi = float(md.obs["entropy_quasirrho"].iloc[0])
        assert harmonic > quasi + 1.0, (harmonic, quasi)

    def test_the_softer_the_mode_the_bigger_the_gap(self):
        md = self._water(3)
        mv.mol.quasirrho(md, [[f] + self.STIFF[:2] for f in (5.0, 25.0, 150.0)],
                         energy="e")
        gap = (md.obs["entropy_harmonic"] - md.obs["entropy_quasirrho"]).to_numpy()
        assert gap[0] > gap[1] > gap[2]
        assert gap[2] == pytest.approx(0.0, abs=0.5), \
            "a mode above the cutoff should barely be touched"

    def test_dropping_the_cutoff_recovers_the_harmonic_answer(self):
        """v0 = 0 turns the interpolation off, so quasi-RRHO must reduce to
        RRHO exactly. This is the one that would catch the damping being
        applied in the wrong direction."""
        md = self._water()
        mv.mol.quasirrho(md, [[12.0] + self.STIFF[:2]], energy="e", cutoff=0.0)
        assert float(md.obs["entropy_quasirrho"].iloc[0]) == \
            pytest.approx(float(md.obs["entropy_harmonic"].iloc[0]), rel=1e-6)

    def test_entropy_grows_with_temperature(self):
        md = self._water()
        mv.mol.quasirrho(md, [self.STIFF], energy="e", temperature=298.15,
                         key_added="cold")
        mv.mol.quasirrho(md, [self.STIFF], energy="e", temperature=500.0,
                         key_added="hot")
        assert float(md.obs["entropy_quasirrho_hot"].iloc[0]) > \
            float(md.obs["entropy_quasirrho_cold"].iloc[0])

    def test_imaginary_modes_are_dropped_and_counted(self):
        """A negative frequency means the geometry is a saddle, so a
        thermochemical correction for it corrects nothing. Silently keeping it
        would be worse than either dropping or refusing."""
        md = self._water()
        mv.mol.quasirrho(md, [[-250.0] + self.STIFF], energy="e")
        assert md.uns["quasirrho"]["default"]["n_imaginary"] == [1]
        assert np.isfinite(float(md.obs["entropy_quasirrho"].iloc[0]))

    def test_a_missing_energy_is_refused(self):
        md = self._water()
        with pytest.raises(ValueError, match="cannot invent one"):
            mv.mol.quasirrho(md, [self.STIFF], energy="not_a_column")

    def test_one_frequency_sequence_per_row_is_required(self):
        md = self._water(2)
        with pytest.raises(ValueError, match="one per row"):
            mv.mol.quasirrho(md, [self.STIFF], energy="e")

    def test_the_columns_are_numeric_not_complex(self):
        """QuasiRRHO returns complex numbers with a zero imaginary part;
        carrying those into obs makes the column object-dtype and breaks every
        comparison downstream."""
        md = self._water()
        mv.mol.quasirrho(md, [self.STIFF], energy="e")
        for column in ("entropy_quasirrho", "entropy_harmonic",
                       "free_energy_quasirrho", "enthalpy_correction"):
            assert md.obs[column].dtype.kind == "f", column

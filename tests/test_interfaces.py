"""Interfaces: lattice matching, contact reactions, and building the cell.

Two materials that are each stable can destroy each other on contact, and two
that are compatible may still be unable to grow on one another. The tests check
both failures separately, because they are separate physics.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")


def _l12(host, guest, a):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


@pytest.fixture(scope="module")
def metals():
    md = mv.datasets.metals(["Cu", "Al", "Ni"])
    mv.pp.describe(md)
    mv.calc.energy(md, level="emt")
    return md


@pytest.fixture(scope="module")
def pairs(metals):
    return mv.iface.match(metals, max_area=120.0)


class TestMatch:
    def test_every_ordered_pair_is_evaluated(self, pairs, metals):
        """Film on substrate is not the same as substrate on film, so the
        pairing is ordered: n(n-1) rows, not n(n-1)/2."""
        n = metals.n_obs
        assert pairs.n_obs == n * (n - 1)
        assert (pairs.obs["film"] != pairs.obs["substrate"]).all()

    def test_it_is_its_own_axis(self, pairs):
        assert pairs.uns[mv.iface.AXIS_KEY] == "interface_pairs"

    def test_pairs_are_labelled_by_name_not_row_number(self, pairs):
        assert set(pairs.obs["film"]) == {"Cu", "Al", "Ni"}

    def test_the_row_index_is_kept_for_resolution(self, pairs, metals):
        """Readable labels are for people; the index is what resolves back to a
        row without a lookup."""
        films = pairs.obs["film_index"].to_numpy(dtype=int)
        assert films.min() >= 0 and films.max() < metals.n_obs

    def test_fcc_metals_match_each_other_easily(self, pairs):
        """Cu, Al and Ni are all fcc with similar parameters, so every pairing
        should find a low-strain supercell. A method that cannot match these
        cannot match anything."""
        strain = pairs.obs["von_mises_strain"].to_numpy(dtype=float)
        assert np.isfinite(strain).all()
        assert (strain < 0.05).all()
        assert pairs.obs["epitaxial"].all()

    def test_the_strain_threshold_is_what_decides_epitaxy(self, metals):
        tight = mv.iface.match(metals, max_area=120.0, max_strain=0.001)
        assert not tight.obs["epitaxial"].all()

    def test_the_algorithm_is_recorded(self, pairs):
        assert "Zur and McGill" in pairs.uns["match"]["algorithm"]
        assert pairs.uns["match"]["max_area"] == 120.0

    def test_a_single_material_cannot_pair(self):
        one = mv.datasets.metals(["Cu"])
        mv.pp.describe(one)
        with pytest.raises(ValueError, match="at least two materials"):
            mv.iface.match(one, max_area=80.0)


class TestReactivity:
    def test_elements_with_no_compound_between_them_do_not_react(self, pairs,
                                                                 metals):
        """With only elemental phases in the diagram there is nothing to react
        into, so zero is the right answer rather than a missing one."""
        mv.iface.reactivity(pairs, metals, level="emt")
        energies = pairs.obs["reaction_energy_emt"].to_numpy(dtype=float)
        assert np.isfinite(energies).all()
        assert not pairs.obs["reacts_emt"].any()

    def test_a_stable_compound_makes_the_contact_reactive(self):
        """Al and Ni are each stable and react to make Al3Ni, which is a real
        phase. The energy must come out at the formation energy it was given."""
        elemental = mv.datasets.metals(["Al", "Ni"])
        md = mv.data.from_structures(
            mv.structures(elemental) + [_l12("Al", "Ni", 3.78)])
        mv.pp.describe(md)
        md.obs["energy_per_atom_lit"] = [0.0, 0.0, -0.45]
        mv.set_level(md, "lit", kind="dft", method="literature")

        pairs = mv.iface.match(md, max_area=120.0)
        mv.iface.reactivity(pairs, md, level="lit")

        contact = pairs.obs[(pairs.obs["film"] == "0")
                            & (pairs.obs["substrate"] == "1")]
        if contact.empty:                       # labels fall back to obs_names
            contact = pairs.obs.iloc[[0]]
        assert float(contact["reaction_energy_lit"].iloc[0]) == pytest.approx(
            -0.45, abs=1e-6)
        assert bool(contact["reacts_lit"].iloc[0])
        assert "Al3Ni" in str(contact["reaction_lit"].iloc[0])

    def test_it_needs_energies_first(self, pairs, metals):
        bare = mv.data.from_structures(mv.structures(metals))
        mv.pp.describe(bare)
        with pytest.raises(ValueError, match="mv.calc.relax"):
            mv.iface.reactivity(pairs, bare, level="nothing")

    def test_it_refuses_the_wrong_axis(self, metals):
        with pytest.raises(ValueError, match="not a pairs object"):
            mv.iface.reactivity(metals, metals, level="emt")

    def test_the_closed_system_is_recorded(self, pairs, metals):
        """'Inert' against three elements is a much weaker claim than inert,
        and the object has to say which one it is making."""
        mv.iface.reactivity(pairs, metals, level="emt")
        assert pairs.uns["reactivity"]["closed_system"] is True
        assert "this dataset" in pairs.uns["reactivity"]["note"]


class TestBuild:
    def test_it_returns_an_ordinary_materials_object(self, metals):
        interfaces = mv.iface.build(metals, film="Cu", substrate="Al",
                                    film_miller=(1, 1, 1),
                                    substrate_miller=(1, 1, 1))
        assert interfaces.n_obs >= 1
        mv.pp.describe(interfaces)
        assert "formula" in interfaces.obs
        assert set(interfaces.obs["film"]) == {"Cu"}
        assert set(interfaces.obs["substrate"]) == {"Al"}

    def test_the_cell_contains_both_materials(self, metals):
        interfaces = mv.iface.build(metals, film="Cu", substrate="Al")
        elements = {str(s) for s in mv.structures(interfaces)[0].species}
        assert {"Cu", "Al"} <= elements

    def test_terminations_are_enumerated_not_guessed(self, metals):
        """Cutting the same orientation at a different plane gives a different
        interface with a different energy."""
        interfaces = mv.iface.build(metals, film="Cu", substrate="Al")
        assert interfaces.obs["termination"].notna().all()
        assert interfaces.uns["interface"]["n_terminations"] >= 1

    def test_a_row_can_be_named_or_indexed(self, metals):
        by_name = mv.iface.build(metals, film="Cu", substrate="Al")
        by_index = mv.iface.build(metals, film=0, substrate=1)
        assert by_name.n_obs == by_index.n_obs

    def test_an_unknown_material_lists_the_rows(self, metals):
        with pytest.raises(KeyError, match="not a row"):
            mv.iface.build(metals, film="Vibranium", substrate="Al")

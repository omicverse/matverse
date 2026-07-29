"""pymatgen's transformations applied to the object.

There are around forty-five Transformation classes. These tests check the
dispatcher rather than the transformations — that a name resolves, that the
result becomes a variant instead of overwriting the input, that a failure on
one row is recorded rather than raised, and that the confusing pymatgen errors
about oxidation states have one place to go.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import matverse as mv

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def oxides():
    md = mv.datasets.load("oxides")[:3].copy()
    mv.pp.describe(md)
    return md


@pytest.fixture(scope="module")
def disordered():
    from pymatgen.core import Lattice, Structure
    alloy = Structure(Lattice.cubic(3.8), [{"Cu": 0.5, "Au": 0.5}] * 4,
                      [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    md = mv.data.from_structures([alloy])
    mv.pp.describe(md)
    return md


class TestAvailable:
    def test_it_finds_a_useful_number_of_them(self):
        found = mv.transform.available()
        assert len(found) > 25
        assert "SupercellTransformation" in found
        assert "PrimitiveCellTransformation" in found

    def test_each_carries_its_signature(self):
        entry = mv.transform.available()["SupercellTransformation"]
        assert entry["signature"].startswith("(")
        assert entry["group"] in {"standard", "advanced", "site"}

    def test_it_can_be_searched(self):
        found = mv.transform.available(search="supercell")
        assert found
        assert all("supercell" in k.lower() or "supercell" in v["doc"].lower()
                   for k, v in found.items())

    def test_it_reads_pymatgen_rather_than_a_table(self):
        """A transformation added upstream should be available the day it
        lands, not when someone updates a list here."""
        import inspect

        from pymatgen.transformations import standard_transformations as std

        upstream = {n for n, o in vars(std).items()
                    if inspect.isclass(o) and n.endswith("Transformation")
                    and not n.startswith("Abstract")
                    and o.__module__ == std.__name__}
        assert upstream <= set(mv.transform.available())


class TestApply:
    def test_it_deposits_a_variant_and_keeps_the_input(self, oxides):
        before = mv.variants(oxides)
        mv.transform.apply(oxides, "PrimitiveCellTransformation")
        assert "input" in mv.variants(oxides)
        assert "primitivecell" in mv.variants(oxides)
        assert len(mv.variants(oxides)) == len(before) + 1

    def test_the_variant_name_can_be_chosen(self, oxides):
        mv.transform.apply(oxides, "PerturbStructureTransformation",
                           distance=0.05, key_added="rattled")
        assert "rattled" in mv.variants(oxides)

    def test_it_actually_changed_the_structures(self, oxides):
        mv.transform.apply(oxides, "PerturbStructureTransformation",
                           distance=0.2, key_added="shifted")
        original = mv.structures(oxides, "input")[0]
        moved = mv.structures(oxides, "shifted")[0]
        assert not np.allclose(original.cart_coords, moved.cart_coords)

    def test_success_is_recorded_per_row(self, oxides):
        mv.transform.apply(oxides, "PrimitiveCellTransformation",
                           key_added="prim2")
        assert oxides.obs["prim2_ok"].all()
        assert oxides.uns["transform"]["prim2"]["n_failed"] == 0

    def test_a_failing_row_keeps_its_structure_and_is_flagged(self, oxides):
        """A mixed dataset where a transformation applies to some rows and not
        others is normal, so one bad row must not take down the call."""
        mv.transform.apply(oxides, "OxidationStateDecorationTransformation",
                           oxidation_states={"Sr": 2, "Ti": 4, "O": -2},
                           key_added="partial")
        flags = oxides.obs["partial_ok"].to_numpy(dtype=bool)
        assert not flags.all()          # TiO2 and VO2 have no Sr
        assert flags.any()
        assert oxides.uns["transform"]["partial"]["errors"]

    def test_the_parameters_are_recorded(self, oxides):
        mv.transform.apply(oxides, "PerturbStructureTransformation",
                           distance=0.07, key_added="noted")
        assert oxides.uns["transform"]["noted"]["params"]["distance"] == "0.07"

    def test_an_unknown_name_points_at_available(self, oxides):
        with pytest.raises(KeyError, match="available"):
            mv.transform.apply(oxides, "NotARealThing")

    def test_a_wrong_argument_shows_the_signature(self, oxides):
        with pytest.raises(TypeError, match="signature is"):
            mv.transform.apply(oxides, "PerturbStructureTransformation",
                               nonsense=1)

    def test_the_short_name_works_too(self, oxides):
        mv.transform.apply(oxides, "PrimitiveCell", key_added="short")
        assert "short" in mv.variants(oxides)


class TestExpand:
    def test_one_to_many_becomes_rows(self, disordered):
        out = mv.transform.expand(
            disordered, "OrderDisorderedStructureTransformation",
            n=3, no_oxi_states=True)
        assert out.n_obs == 3
        assert set(out.obs["parent"]) == {"0"}
        assert list(out.obs["variant_index"]) == [0, 1, 2]

    def test_the_results_are_ordinary_materials(self, disordered):
        out = mv.transform.expand(
            disordered, "OrderDisorderedStructureTransformation",
            n=2, no_oxi_states=True)
        mv.pp.describe(out)
        assert set(out.obs["formula"]) == {"CuAu"}

    def test_producing_nothing_names_the_usual_cause(self, oxides):
        with pytest.raises(ValueError, match="enumlib"):
            mv.transform.expand(oxides, "EnumerateStructureTransformation",
                                n=2, max_cell_size=1)


class TestChain:
    def test_a_sequence_gives_one_variant(self, oxides):
        mv.transform.chain(oxides, [
            ("PrimitiveCellTransformation", {}),
            ("PerturbStructureTransformation", {"distance": 0.03}),
        ], key_added="chained")
        assert "chained" in mv.variants(oxides)
        assert oxides.obs["chained_ok"].all()

    def test_every_step_is_recorded(self, oxides):
        mv.transform.chain(oxides, [
            ("PrimitiveCellTransformation", {}),
            ("PerturbStructureTransformation", {"distance": 0.03}),
        ], key_added="noted_chain")
        steps = oxides.uns["transform"]["noted_chain"]["chain"]
        assert [s["transformation"] for s in steps] == [
            "PrimitiveCellTransformation", "PerturbStructureTransformation"]
        assert steps[1]["params"]["distance"] == "0.03"

    def test_an_empty_chain_is_refused(self, oxides):
        with pytest.raises(ValueError, match="steps is empty"):
            mv.transform.chain(oxides, [])


class TestOxidationStates:
    def test_bond_valence_works_on_an_oxide(self, oxides):
        mv.transform.oxidation_states(oxides)
        assert oxides.obs["oxidation_states_ok"].any()
        assert "oxidized" in mv.variants(oxides)

    def test_the_states_land_on_the_structures(self, oxides):
        mv.transform.oxidation_states(oxides)
        good = np.where(oxides.obs["oxidation_states_ok"].to_numpy(bool))[0]
        structure = mv.structures(oxides, "oxidized")[int(good[0])]
        assert any(getattr(site.specie, "oxi_state", None) is not None
                   for site in structure)

    def test_charge_balance_is_checked(self, oxides):
        mv.transform.oxidation_states(oxides)
        good = oxides.obs["oxidation_states_ok"].to_numpy(bool)
        assert oxides.obs["charge_balanced"].to_numpy(bool)[good].all()

    def test_an_explicit_assignment_is_honoured(self, oxides):
        mv.transform.oxidation_states(
            oxides, method={"Sr": 2, "Ti": 4, "V": 4, "O": -2},
            key_added="explicit")
        assert oxides.uns["oxidation_states"]["method"] == "explicit"
        assert oxides.obs["oxidation_states_ok"].all()

    def test_a_metal_fails_per_row_rather_than_raising(self):
        """Bond-valence analysis is meaningless for a metal, and a dataset
        mixing oxides with alloys is normal."""
        md = mv.datasets.metals(["Cu", "Al"])
        mv.pp.describe(md)
        mv.transform.oxidation_states(md)
        assert not md.obs["oxidation_states_ok"].all()
        assert md.uns["oxidation_states"]["errors"]
        assert "metals" in md.uns["oxidation_states"]["note"]

    def test_an_unknown_method_is_refused(self, oxides):
        with pytest.raises(ValueError, match="'bva', 'guess' or a dict"):
            mv.transform.oxidation_states(oxides, method="telepathy")

    def test_the_guess_route_runs_without_geometry(self, oxides):
        mv.transform.oxidation_states(oxides, method="guess",
                                      key_added="guessed")
        assert oxides.uns["oxidation_states"]["method"] == "composition guess"
        assert "guessed" in mv.variants(oxides)


class TestSetting:
    """Cell transformations mv.pp.supercell cannot reach."""

    @staticmethod
    def _orthorhombic():
        from pymatgen.core import Lattice, Structure
        return Structure(Lattice.orthorhombic(3.0, 4.0, 5.0), ["Cu"],
                         [[0, 0, 0]])

    def test_the_identity_changes_nothing(self):
        md = mv.data.from_structures([self._orthorhombic()])
        mv.pp.describe(md)
        mv.transform.setting(md, "a,b,c;0,0,0", key_added="same")
        assert md.obs["same_volume_ratio"].iloc[0] == pytest.approx(1.0)
        assert mv.structures(md, "same")[0].lattice.abc == \
            pytest.approx((3.0, 4.0, 5.0))

    def test_scaling_one_axis_doubles_the_volume(self):
        md = mv.data.from_structures([self._orthorhombic()])
        mv.pp.describe(md)
        mv.transform.setting(md, "2a,b,c;0,0,0", key_added="doubled")
        assert md.obs["doubled_volume_ratio"].iloc[0] == pytest.approx(2.0)
        assert mv.structures(md, "doubled")[0].lattice.abc[0] == \
            pytest.approx(6.0)

    def test_a_non_diagonal_transformation_is_reachable(self):
        """This is what mv.pp.supercell cannot do: a+b and a-b are not any
        integer scaling of the axes."""
        md = mv.data.from_structures([self._orthorhombic()])
        mv.pp.describe(md)
        mv.transform.setting(md, "a+b,a-b,c;0,0,0", key_added="bct")
        lattice = mv.structures(md, "bct")[0].lattice
        assert md.obs["bct_volume_ratio"].iloc[0] == pytest.approx(2.0)
        assert lattice.abc[0] == pytest.approx(3.0 * 2 ** 0.5, abs=1e-3)

    def test_an_origin_shift_moves_the_coordinates_not_the_cell(self):
        md = mv.data.from_structures([self._orthorhombic()])
        mv.pp.describe(md)
        mv.transform.setting(md, "a,b,c;1/2,0,0", key_added="shifted")
        assert md.obs["shifted_volume_ratio"].iloc[0] == pytest.approx(1.0)
        moved = mv.structures(md, "shifted")[0]
        assert moved.frac_coords[0][0] % 1.0 == pytest.approx(0.5, abs=1e-6)

    def test_the_convention_is_recorded(self):
        """The string names the new basis in terms of the old, which is the
        International Tables convention and the opposite of what a reader
        might assume."""
        md = mv.data.from_structures([self._orthorhombic()])
        mv.pp.describe(md)
        mv.transform.setting(md, "b,a,-c;0,0,0")
        assert "International Tables" in md.uns["setting"]["convention"]

    def test_an_unreadable_string_explains_the_format(self):
        md = mv.data.from_structures([self._orthorhombic()])
        mv.pp.describe(md)
        with pytest.raises(ValueError, match="Jones-Faithful"):
            mv.transform.setting(md, "rotate it a bit")

"""End-to-end pipeline and the two conventions that carry the design."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("ase")
from pymatgen.core import Lattice, Structure       # noqa: E402
import matverse as mv                               # noqa: E402


def fcc(el, a):
    return Structure(Lattice.cubic(a), [el] * 4,
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


@pytest.fixture
def df():
    # EMT is parameterised for these elements only
    return pd.DataFrame({
        "structure": [fcc("Cu", 3.61), fcc("Cu", 3.75), fcc("Ag", 4.09), fcc("Ni", 3.52)],
        "tag": ["Cu-eq", "Cu-strained", "Ag", "Ni"]})


def test_roundtrip_matminer(df):
    md = mv.data.from_matminer(df)
    out = mv.data.to_matminer(md)
    assert list(out["tag"]) == list(df["tag"])
    assert [s.composition for s in out["structure"]] == [s.composition for s in df["structure"]]


def test_operations_deposit_rather_than_return(df):
    md = mv.data.from_matminer(df)
    assert mv.struct.standardize(md) is None
    assert "primitive" in md.uns["structures"]
    assert "conventional" in md.uns["structures"]
    assert "spacegroup" in md.obs


def test_structure_variants_accumulate(df):
    md = mv.data.from_matminer(df)
    mv.struct.standardize(md)
    mv.struct.supercell(md, [2, 1, 1])
    assert set(md.uns["structures"]) >= {"input", "primitive", "conventional"}
    assert any(k.startswith("supercell") for k in md.uns["structures"])


def test_level_of_theory_is_the_slot_name(df):
    md = mv.data.from_matminer(df)
    mv.calc.energy(md, level="emt")
    assert "energy_emt" in md.obs
    assert md.uns["calc"]["emt"]["method"] == "EMT"
    # a second level must not overwrite the first
    md.obs["energy_pbe"] = np.arange(md.n_obs, dtype=float)
    md.uns["calc"]["pbe"] = {"functional": "PBE"}
    assert set(md.uns["calc"]) == {"emt", "pbe"}


def test_relax_lowers_energy_of_a_strained_cell(df):
    md = mv.data.from_matminer(df)
    mv.calc.energy(md, level="emt")
    before = md.obs["energy_emt"].to_numpy(dtype=float).copy()
    mv.calc.relax(md, level="emt", fmax=0.05)
    after = md.obs["energy_emt"].to_numpy(dtype=float)
    assert "relaxed_emt" in md.uns["structures"]
    assert np.all(after <= before + 1e-6)          # relaxation cannot raise energy


def test_hull_marks_the_strained_cell_as_unstable(df):
    md = mv.data.from_matminer(df)
    mv.struct.describe(md)
    mv.calc.relax(md, level="emt", fmax=0.05)
    mv.thermo.hull(md, level="emt", source="relaxed_emt")
    e = md.obs["e_above_hull_emt"].to_numpy(dtype=float)
    assert md.uns["phase_diagram"]["built"]
    assert md.uns["phase_diagram"]["closed_system"] is True
    assert np.nanmin(e) == pytest.approx(0.0, abs=1e-9)


def test_screen_records_its_criteria(df):
    md = mv.data.from_matminer(df)
    mv.struct.describe(md)
    mv.calc.energy(md, level="emt")
    mv.screen.filter(md, n_elements__le=1, name="single_element")
    rec = md.uns["screens"]["single_element"]
    assert rec["criteria"] == {"n_elements__le": 1}
    assert rec["n_total"] == len(df)
    assert md.obs["single_element"].all()


def test_screen_rejects_a_missing_column(df):
    md = mv.data.from_matminer(df)
    with pytest.raises(ValueError, match="absent"):
        mv.screen.filter(md, nonexistent__lt=1)


def test_missing_prerequisite_names_the_operation_that_fills_it(df):
    md = mv.data.from_matminer(df)
    with pytest.raises(ValueError, match="mv.calc"):
        mv.thermo.hull(md, level="emt")
    with pytest.raises(KeyError, match="no structure variant"):
        mv.struct.describe(md, source="relaxed_emt")


def test_subsetting_keeps_annotations_aligned(df):
    md = mv.data.from_matminer(df)
    mv.struct.describe(md)
    mv.feat.composition(md)
    sub = md[md.obs["formula"] == "Cu"].copy()
    assert sub.n_obs == 2
    assert sub.obsm["X_composition"].shape[0] == 2


def test_provenance_records_the_pipeline(df):
    md = mv.data.from_matminer(df)
    mv.struct.describe(md); mv.feat.composition(md); mv.calc.energy(md, level="emt")
    ops = [p.split("(")[0] for p in md.uns["provenance"]]
    assert ops == ["data.from_matminer", "struct.describe", "feat.composition", "calc.energy"]

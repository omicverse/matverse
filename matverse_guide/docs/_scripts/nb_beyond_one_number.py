"""Cells for tutorials/beyond_one_number.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Beyond one number per material

The screening tutorial produced scalars: an energy, a distance above the hull, a
pass/fail. Plenty of results are not scalars. A diffraction pattern is a curve.
A force is one vector per atom, and the number of atoms differs from material to
material. A measurement is a number somebody took off an instrument rather than
out of a calculator.

This tutorial covers where each of those lives, and why."""),

    ("code", """\
import matverse as mv
import numpy as np
import pandas as pd

mv.pl.set_style()"""),

    ("markdown", """\
## Loading a dataset

The same Al–Ni–Cu library as [Screening, end to end](screening.ipynb): three
elemental fcc metals at their published lattice parameters, four hypothetical
intermetallics."""),

    ("code", """\
from pymatgen.core import Lattice, Structure


def l12(host, guest, a):
    return Structure(Lattice.cubic(a), [guest, host, host, host],
                     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


elemental = mv.datasets.metals(["Al", "Cu", "Ni"])
md = mv.data.from_structures(mv.structures(elemental) + [
    l12("Al", "Cu", 3.90), l12("Cu", "Al", 3.70), l12("Al", "Ni", 3.78),
    Structure(Lattice.cubic(2.89), ["Al", "Ni"], [[0, 0, 0], [.5, .5, .5]]),
])
mv.pp.describe(md)

md.obs[["formula", "nsites", "density"]].round(3)"""),

    ("markdown", """\
## Curves live in `obsm` on a shared grid

pymatgen returns a peak list, and peak lists cannot be compared across materials
because no two share the same peak positions. Broadening each reflection onto a
common grid is what turns a set of patterns into a matrix — and a matrix is
something you can take a distance on, cluster, or subtract a measurement from."""),

    ("code", """\
mv.prop.xrd(md, two_theta=(10, 90), step=0.02, fwhm=0.1)

md.obsm["xrd_calc"].shape, mv.grid_of(md, "xrd").shape"""),

    ("markdown", """\
The block is named `'<quantity>_<level>'`, exactly like a scalar:

```
obs['energy_pbe']        a number per material, level in the suffix
obsm['xrd_pbe']          a curve per material, level in the suffix
obsm['xrd_experiment']   the same curve, measured
```

```{note}
This is one convention, not two. An earlier draft of the design had curves using
an AnnData `layer` for the level of theory while scalars used a name suffix, and
listed the split as a wart. Putting curves in `obsm` removed it — and removed
the need for MuData along with it, since a `materials × grid` matrix is aligned
to the material axis and that is what `obsm` is for.
```

The grid itself is stored once, in `uns`, because it is the one thing here that
is **not** aligned to the material axis:"""),

    ("code", """\
md.uns["grids"]["xrd"]"""),

    ("code", """\
ax = mv.pl.spectra(md, "xrd", rows=[0, 3, 4], offset=120)
ax.set_title("Al, Al₃Cu, AlCu₃ — ordering adds superlattice reflections")"""),

    ("markdown", """\
`mv.prop.rdf` is the other curve that ships, and it earns its place by doing
something composition cannot."""),

    ("code", """\
mv.prop.rdf(md, r_max=8.0)

ax = mv.pl.spectra(md, "rdf", rows=[0, 6], offset=6)
ax.set_title("fcc Al against B2 AlNi")"""),

    ("markdown", """\
Two polymorphs have the same composition, so `X` cannot tell them apart and
neither can any composition descriptor. A radial distribution function can, and
unlike SOAP it needs no extra dependency.

## Per-atom results get their own axis

Forces are ragged. This library has 4, 4, 4, 4, 4, 4 and 2 atoms — 26 force
vectors, and no way to put them in a column of a seven-row table.

First, something to compute forces *on*. Every structure above is a perfect
cubic cell, and by symmetry the force on every atom in one is exactly zero — a
force example needs displaced atoms. `mv.pp.rattle` is how you make them, and
it is also how force-training sets for machine-learned potentials are built."""),

    ("code", """\
mv.pp.rattle(md, stdev=0.05, seed=0)
mv.variants(md)"""),

    ("code", """\
sites = mv.multi.sites(md, source="rattled")
sites"""),

    ("code", """\
sites.obs.head(8)"""),

    ("markdown", """\
One row per atom, with `obs['material']` pointing back at the parent. Per-atom
results are a matrix again."""),

    ("code", """\
mv.calc.forces(md, sites, level="emt", source="rattled")

sites.obsm["forces_emt"].shape"""),

    ("code", """\
sites.obs[["material", "element", "force_magnitude_emt"]].head(8).round(4)"""),

    ("code", """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 3.6))
for k, element in enumerate(sites.var_names):
    values = sites.obs.loc[sites.obs["element"] == element,
                           "force_magnitude_emt"]
    ax.scatter(np.full(len(values), k) + np.linspace(-0.12, 0.12, len(values)),
               values, s=22, alpha=0.8)
ax.set_xticks(range(sites.n_vars), list(sites.var_names))
ax.set_ylabel("|F| (eV/Å)")
ax.set_title("one point per atom, 26 in total")"""),

    ("markdown", """\
Twenty-six points where the material axis has room for seven — which is the
whole reason the sites axis exists.

`mv.calc.forces` takes **both** objects, and the signature is honest about why:
forces need the structures, which live on the material axis, and produce one row
per atom, which lives on the sites axis.

```{warning}
The sites object is a snapshot of one structure variant. If you build it from
`'input'` and then ask for forces on `'relaxed_emt'`, the atoms will not line up
— so matverse refuses rather than silently misaligning them. Rebuild with
`mv.multi.sites(md, source='relaxed_emt')`.
```

### Getting back to the material axis

A per-site result is not screenable until it is summarised."""),

    ("code", """\
mv.multi.aggregate(sites, md, "force_magnitude_emt", how="max")

md.obs[["formula", "force_magnitude_emt_max"]].round(4)"""),

    ("markdown", """\
The largest force in each cell, on the material axis, where a screen can reach
it. A residual force is a useful convergence criterion: the cells that respond
least to being rattled are the ones sitting in the stiffest local minimum."""),

    ("code", """\
mv.screen.filter(md, force_magnitude_emt_max__lt=0.8, name="soft_response")
md.obs[["formula", "force_magnitude_emt_max", "soft_response"]].round(4)"""),

    ("markdown", """\
The detail stays on the sites object where it can still be inspected; the
summary lands on the materials object where a screen can reach it.

### The element axis is shared

`sites.X` is the one-hot element indicator, so `sites.var` is the same periodic
table the parent carries. Everything written for element-level questions works
on atoms without modification."""),

    ("code", """\
forces = sites.obs["force_magnitude_emt"].to_numpy(dtype=float)
sites.obs["strained"] = forces > np.median(forces)
mv.tl.rank_elements_groups(sites, "strained")

sites.uns["rank_elements_groups"]["True"][
    ["element", "n_in_group", "frac_in_group", "odds_ratio", "pval"]]"""),

    ("markdown", """\
That is `rank_genes_groups` answering "which elements sit in the highest-force
environments", and it needed no new function.

### One object, if you want one

```python
mdata = mv.multi.to_mudata(md, sites)     # needs matverse[multi]
```

Optional throughout. matverse's operations take `AnnData`, and the sites object
is useful without ever being assembled."""),

    ("code", """\
mv.prop.tem(md, r_max=1.2, step=0.02)
md.obs[["formula", "tem_n_reflections_calc", "tem_strongest_calc",
        "tem_zone_axis"]]"""),

    ("markdown", """\
A TEM pattern is spots on a plane, and a plane of spots does not compare across
materials — two crystals on the same zone axis put different spots in different
places. What goes into `obsm` is the **ring profile**: intensity against the
magnitude of the scattering vector, which is what a polycrystalline
selected-area pattern looks like and what can share an axis with an X-ray or
neutron pattern.

`obs` keeps the two facts the reduction loses — how many reflections were
excited, and the strongest one's Miller indices — and the **zone axis**, without
which the pattern means nothing.

```{note}
Those columns show the F-centring rule rather than assert it. A face-centred
lattice allows only all-even or all-odd `hkl`, one reflection in four, so an fcc
metal keeps a quarter of what a simple cubic cell does — and (010), extinct in
fcc, is the brightest spot in simple cubic.
```"""),

    ("markdown", """\
## A result computed somewhere else

Everything so far, matverse computed. Plenty of the results a screen wants it
cannot compute at all: a chemical shielding tensor, a piezoelectric tensor and
a dielectric function all come out of a DFT code, and no cheap potential
produces any of them.

That is not a reason for the object to have nothing to say about them. The step
*after* the calculation — reduce a tensor to the parameters a spectrum is
described by, check it against the crystal symmetry, turn a dielectric function
into an efficiency — is arithmetic with conventions in it, and conventions are
exactly what gets a result quoted wrongly.

So these functions take the computed quantity as an **argument**, the same way
`mv.elec.bands` takes band structures and `mv.exp.attach` takes measured curves.

### A tensor per atom

NMR shielding is per-atom, so it lands on the sites axis. Here is a tensor with
principal values 10, 20 and 60 — round numbers, so every convention has a hand
answer."""),

    ("code", """\
shieldings = np.tile(np.diag([10.0, 20.0, 60.0]), (sites.n_obs, 1, 1))
mv.prop.nmr(md, sites, shieldings, level="pbe")

sites.obs[["material", "element", "shielding_iso_pbe",
           "shielding_anisotropy_pbe", "shielding_asymmetry_pbe",
           "shielding_span_pbe", "shielding_skew_pbe"]].head(4).round(3)"""),

    ("markdown", """\
`sigma_iso` is 30, the trace over three. `zeta` is 30, which is
`sigma_33 - sigma_iso`. `eta` is 1/3, the span is 50 and the skew −0.6.

Two conventions are reported because both are in use and **they disagree about
what "anisotropy" means**: Haeberlen's `zeta` is `sigma_33 - sigma_iso`, while
the reduced anisotropy most spectrometer software prints is 3/2 of it. Span and
skew are the Herzfeld-Berger pair a sideband analysis returns. A single column
called `anisotropy` would be wrong for half its readers.

These are **shieldings**, not shifts — a shift is a shielding referenced to a
standard compound, and it runs the other way in sign.

The electric field gradient is the other half of a solid-state NMR experiment,
and it sets the lineshape of every quadrupolar nucleus."""),

    ("code", """\
gradients = np.tile(np.diag([-1.0, -2.0, 3.0]), (sites.n_obs, 1, 1))
mv.prop.efg(md, sites, gradients, level="pbe")

sites.obs[["element", "efg_vzz_pbe", "efg_asymmetry_pbe",
           "efg_coupling_pbe"]].head(4).round(3)"""),

    ("markdown", """\
The coupling constant is the only one of the three that needs to know which
*nucleus* it is looking at — a quadrupole moment is a property of the isotope,
not of the calculation — so it is looked up from the element on the site.

### A tensor per material, checked against symmetry

α-quartz is the piezoelectric everyone learns first. Its measured constants are
d₁₁ = 2.3 pC/N and d₁₄ = −0.67 pC/N, and its point group 32 fixes the rest of
the matrix."""),

    ("code", """\
from pymatgen.core import Lattice, Structure

quartz = mv.data.from_structures([Structure.from_spacegroup(
    "P3121", Lattice.hexagonal(4.913, 5.405), ["Si", "O"],
    [[0.4697, 0.0, 0.0], [0.4135, 0.2669, 0.1191]])])
mv.pp.describe(quartz)

d11, d14 = 2.3, -0.67
voigt = np.array([[[d11, -d11, 0, d14, 0, 0],
                   [0, 0, 0, 0, -d14, -2 * d11],
                   [0, 0, 0, 0, 0, 0]]])

mv.prop.piezoelectric(quartz, voigt, level="exp")
quartz.obs[["formula", "piezo_max_longitudinal_exp",
            "piezo_symmetry_valid_exp"]].round(3)"""),

    ("markdown", """\
**2.30 pC/N**, recovered from the tensor without being told where to look. For
class 32 the longitudinal response in the basal plane goes as `d11 cos(3θ)`, so
its maximum over all directions *is* d₁₁ — which is why the maximum is the
screening number rather than any single component. Components depend on how
somebody oriented the cell; the maximum does not.

`piezo_symmetry_valid` is the column to read first. Put the same tensor on a
centrosymmetric crystal and it fails, because inversion symmetry forbids
piezoelectricity outright:"""),

    ("code", """\
copper = mv.datasets.metals(["Cu"])
mv.pp.describe(copper)
mv.prop.piezoelectric(copper, voigt, level="bogus")

bool(copper.obs["piezo_symmetry_valid_bogus"].iloc[0])"""),

    ("markdown", """\
`False` — and a non-zero piezoelectric tensor on an fcc metal is an error in the
calculation, or a tensor paired with the wrong structure, rather than a
discovery.

### A spectrum per material, and what it is worth

A dielectric function is a curve, so it goes where curves go. The absorption
coefficient is **derived** from it rather than stored alongside it, because
`α = 2Ek/ħc` is a definition and a spectrum that has been through it twice
cannot be reconstructed."""),

    ("code", """\
energies = np.linspace(0.3, 4.0, 800)
gaps = [1.34, 0.90, 2.50, 1.34, 1.34, 1.34, 1.34]

# A model absorber: a step edge at the gap. The last four share a gap and
# differ only in how strongly they absorb above it.
strength = [6.0, 6.0, 6.0, 6.0, 0.5, 0.05, 0.005]
eps1 = np.tile(4.0, (md.n_obs, energies.size))
eps2 = np.stack([np.where(energies >= g, s, 0.0)
                 for g, s in zip(gaps, strength)])

mv.prop.dielectric(md, energies, eps1, eps2, level="pbe")
md.obs["band_gap_pbe"] = gaps
md.obsm["absorption_pbe"].shape"""),

    ("code", """\
mv.prop.slme(md, level="pbe", thickness=5e-7)

md.obs[["band_gap_pbe", "slme_pbe", "sq_limit_pbe"]].round(2)"""),

    ("markdown", """\
Read the first three rows against the last four.

Rows 0–2 differ only in **gap**, and `sq_limit` traces the Shockley-Queisser
curve: 1.34 eV beats both 0.90 and 2.50, which is why 1.34 eV is quoted as the
optimum for a single junction. Row 0 lands near 33%, the textbook limit — a
model absorber with a step edge and no indirect gap *is* the Shockley-Queisser
idealisation, so reproducing it is the calibration rather than a result.

Rows 3–6 all have the **same gap** and therefore the same `sq_limit`, and
wildly different `slme`. That spread is the whole reason this function exists.
A screen ranked on band gap cannot tell those four apart — by that measure they
are one candidate. They are not one candidate.

```{note}
`slme` is a **percentage**, and the unit is recorded rather than left to the
reader; `mv.utils.check_units(md)` will say so. pymatgen's own `slme()` returns
the same number with no unit in its docstring, which is precisely the kind of
thing that becomes a factor of 100 three functions downstream.
```

An indirect gap costs efficiency on top of that, through the radiative fraction
`exp(-(E_direct - E_indirect)/kT)`. Left unset, the direct gap is used for both,
which is the optimistic Shockley-Queisser assumption:"""),

    ("code", """\
md.obs["gap_indirect_pbe"] = [g - 0.3 for g in gaps]
mv.prop.slme(md, level="pbe", thickness=5e-7,
             indirect_key="gap_indirect_pbe")

md.obs[["band_gap_pbe", "gap_indirect_pbe", "slme_pbe"]].head(3).round(2)"""),

    ("markdown", """\
That is the penalty silicon pays, and the reason a direct-gap absorber a tenth
as thick can outperform it.

## Experiment is a level of theory

This needed no new machinery, which is the argument for having typed the level
of theory in the first place. Two calculated levels and one measured one, on the
same quantity:"""),

    ("code", """\
mv.calc.energy(md, level="emt")
md.obs["energy_per_atom_shifted"] = \\
    md.obs["energy_per_atom_emt"].to_numpy(dtype=float) - 0.05
mv.set_level(md, "shifted", kind="classical", method="EMT with an offset")

measured = md.obs["energy_per_atom_emt"].to_numpy(dtype=float) + 0.02
mv.exp.measure(md, "energy_per_atom", measured, instrument="calorimeter")

mv.compare_levels(md, "energy_per_atom").round(4)"""),

    ("markdown", """\
Three numbers, three levels, one table, and nobody had to decide which one is
"the" energy. `uns['levels']['experiment']` records the instrument where
`uns['levels']['emt']` records the method:"""),

    ("code", """\
mv.level_info(md, "experiment")"""),

    ("code", """\
ax = mv.pl.parity(md, "energy_per_atom", "emt", "experiment")"""),

    ("markdown", """\
`mv.pl.parity` annotates the error and — the useful bit — prints a warning on
the plot when the two levels reproduce different methods. A parity plot silently
comparing PBE against r2SCAN looks exactly like one comparing PBE against PBE.

### Phase identification

The compelling case is a measured pattern against a candidate library. Take the
Al₃Cu pattern, add noise and a sloping background, and hand it back as though it
came off an instrument."""),

    ("code", """\
rng = np.random.default_rng(0)
two_theta = mv.grid_of(md, "xrd")
observed = (md.obsm["xrd_calc"][3]
            + rng.normal(0, 0.5, two_theta.size)      # counting noise
            + 0.02 * two_theta)                        # sloping background

mv.exp.match_xrd(md, observed, two_theta)
md.obs[["formula", "xrd_match", "xrd_match_rank"]].sort_values("xrd_match_rank")"""),

    ("code", """\
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(two_theta, observed, linewidth=0.7, label="observed")
ax.plot(two_theta, md.obsm["xrd_calc"][3] - 60, linewidth=0.7,
        label="Al₃Cu, calculated (offset)")
ax.set_xlabel("2θ (degrees)")
ax.set_ylabel("intensity")
ax.set_title("the match, drawn")
ax.legend()"""),

    ("markdown", """\
Both patterns are baseline-shifted and unit-normalised before the dot product,
so the score reflects peak positions and relative heights rather than exposure
time — which is why the sloping background above did not move the answer."""),

    ("code", """\
md.uns["xrd_match"]["best"], md.uns["xrd_match"]["scored_against"]"""),

    ("markdown", """\
```{warning}
`match_xrd` scores against the candidates in this object and nothing else, and
records that fact in `uns['xrd_match']['scored_against']`. A high score means
"the best of what you gave it", not "identified" — the true phase can be absent
from your library entirely.
```

### Attaching a full measured curve

A measurement usually covers a narrower range than the calculation, at its own
resolution. Simulate that: 20–70° on a coarser step."""),

    ("code", """\
instrument_grid = np.arange(20.0, 70.0, 0.05)
patterns = np.vstack([
    np.interp(instrument_grid, two_theta, row) + rng.normal(0, 0.3,
                                                            instrument_grid.size)
    for row in md.obsm["xrd_calc"]])

mv.exp.attach(md, "xrd", patterns, instrument_grid, instrument="Bruker D8")
mv.prop.compare_grids(md, "xrd", "calc", "experiment")

md.obs[["formula", "xrd_cosine_calc_vs_experiment",
        "xrd_overlap_calc_vs_experiment"]].round(3)"""),

    ("markdown", """\
Measurements are resampled onto the existing grid, because two curves on
different grids cannot be subtracted. Points outside the measured range become
NaN rather than zero, and the comparison runs over the overlap — `overlap` is
the **number of grid points** both curves cover. 2497 of the calculation's 4000
is the 20–70° the instrument saw, out of the 10–90° that was computed.

```{warning}
Attach a measurement at its own resolution or better. Diffraction peaks are
narrow, and resampling onto a grid coarser than the peak width discards them
permanently — no later interpolation brings them back. There is a test in the
suite pinning this behaviour so nobody "fixes" it.
```

## Reconciling databases

If your candidates came from more than one source, their energies are not
directly comparable. Materials Project, OQMD and Alexandria differ in
pseudopotentials, cutoffs and correction schemes, and the resulting offsets have
compositional structure — which makes them a batch effect.

Simulate one. Take every material **twice**, label one copy `mp` and the other
`oqmd`, and shift the `oqmd` copy by a per-element offset. The overlap matters:
harmonisation is a fit, and a fit needs compositions both databases contain."""),

    ("code", """\
both = mv.data.from_structures(mv.structures(md) * 2)
mv.pp.describe(both)
mv.pp.normalize_composition(both)
mv.calc.energy(both, level="emt")

offsets = np.array([{"Al": 0.10, "Ni": -0.06, "Cu": 0.03}[e]
                    for e in both.var_names])
shift = np.asarray(both.layers["fraction"].todense()) @ offsets

both.obs["database"] = ["mp"] * md.n_obs + ["oqmd"] * md.n_obs
both.obs["energy_per_atom_dft"] = (
    both.obs["energy_per_atom_emt"].to_numpy(dtype=float)
    + np.where(both.obs["database"] == "oqmd", shift, 0.0))

both.obs[["formula", "database", "energy_per_atom_dft"]].round(4)"""),

    ("code", """\
mv.pp.harmonize(both, batch_key="database",
                energy_key="energy_per_atom_dft", reference="mp")

both.uns["harmonize"]["diagnostics"]["oqmd"]"""),

    ("markdown", """\
The fitted offsets should recover the ones put in — 0.10, −0.06 and 0.03:"""),

    ("code", """\
pd.Series(both.uns["harmonize"]["offsets"]["oqmd"],
          index=both.var_names, name="fitted offset").round(3)"""),

    ("code", """\
n = md.n_obs
raw = both.obs["energy_per_atom_dft"].to_numpy(dtype=float)
fixed = both.obs["energy_per_atom_dft_harmonized"].to_numpy(dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
for ax, values, title in ((axes[0], raw, "before"), (axes[1], fixed, "after")):
    ax.scatter(values[:n], values[n:], s=36)
    lo, hi = values.min() - 0.02, values.max() + 0.02
    ax.plot([lo, hi], [lo, hi], "--", linewidth=0.8, color="#888")
    ax.set_xlabel("mp"); ax.set_title(title)
axes[0].set_ylabel("oqmd")
fig.tight_layout()"""),

    ("markdown", """\
Same materials, two databases, plotted against each other. Before, they scatter
off the diagonal in a way that depends on composition; after, they land on it.

The model is the one the field already uses by hand — a per-element reference
offset — fitted by least squares on the compositions two databases share.

```{warning}
It cannot repair a disagreement that is not linear in composition. Two databases
that differ because one relaxed with a different functional will differ
structure by structure, and a compositional offset absorbs only the average of
that. Read `rmse_after`: it is what is left.

`harmonize` also warns and does nothing when the databases share no composition,
because with no anchors there is nothing to fit.
```

## Scoring generated candidates

Substitution within a known structure type is the cheap baseline every
generative model should be measured against — and is what several were found to
be doing implicitly."""),

    ("code", """\
generated = mv.gen.substitute(md, {"Al": ["Ga", "In"]}, charge_balanced=False)
mv.pp.describe(generated)

generated.obs[["formula", "parent", "substitution"]].head(8)"""),

    ("code", """\
mv.gen.validate(generated, reference=md)

generated.uns["gen_validate"]["rates"]"""),

    ("markdown", """\
Validity, uniqueness, novelty and stability use LeMat-GenBench's definitions
rather than a variant, and every parameter is recorded next to the rates:"""),

    ("code", """\
generated.uns["gen_validate"]["definitions"]"""),

    ("markdown", """\
That matters more than it looks. Until those definitions were pinned, the same
metric name meant different things depending on which reference set, stability
threshold and matching tolerance a paper used — so the numbers were not
comparable even when the names were identical.

```{warning}
Stability is reported as **not assessed** rather than zero when no level is
given or when the hull was built over the dataset's own compositions. A closed
hull cannot say whether anything is stable.
```"""),

    ("code", """\
generated.uns["gen_validate"]["not_assessed"]"""),

    ("markdown", """\
Novelty means "absent from the reference set you named", which is a weaker claim
than it sounds. A 2026 stress test found that neither MatterGen nor DiffCSP++
recovered the experimentally observed structures of the newly synthesised
GdNiSn₄ and LuNiSn₄, despite both being built from known motifs — current models
recombine compositions within known structural families. A high novelty rate
measured against a database does not contradict that.

### Where the candidates come from in the first place

`mv.gen.substitute` enumerates the swaps you name. `mv.gen.predict_substitutions`
ranks the swaps you did not think of, from the data-mined ionic substitution
model of Hautier et al. — how often two species replace one another across the
ICSD, rather than whether their radii match.

It needs **oxidation states**, because the model is defined over ionic species:
Fe²⁺ and Fe³⁺ substitute differently and that distinction is the whole point."""),

    ("code", """\
cathode = mv.datasets.load("battery_cathodes")[:1].copy()
mv.pp.describe(cathode)
mv.transform.oxidation_states(cathode)

proposed = mv.gen.predict_substitutions(cathode, source="oxidized", n=8)
mv.pp.describe(proposed)
proposed.obs[["parent", "substitution", "substitution_probability",
              "formula"]].round(5)"""),

    ("markdown", """\
From LiFePO₄ it reaches **LiVPO₄** and **LiTiPO₄** — both real olivine
analogues — without being told anything about olivines. The probability is a
prior from what has been made before; it says nothing about whether the
substitution is stable in this structure, which is what the hull is for.

The result is an ordinary materials object, so the whole pipeline runs on it:
`mv.pp.qc`, `mv.calc.relax`, `mv.thermo.hull`.

Doping is the same statistics asked a narrower question:"""),

    ("code", """\
mv.gen.predict_dopants(cathode, source="oxidized", n=5)
cathode.obs[["name", "n_type_dopant", "n_type_probability",
             "p_type_dopant", "p_type_probability"]].round(4)"""),

    ("markdown", """\
Fluorine comes out as the n-type choice, which is the doping strategy LiFePO₄ is
actually treated with. n-type versus p-type here is arithmetic on oxidation
states — the dopant carries more charge than the site it replaces, or less — not
a calculation of where the level lands. Whether it is shallow, soluble or
compensated is `mv.thermo.defect_formation`'s question.

A candidate built by substitution keeps the cell it was built from, which can be
several percent from where the new composition wants to sit. Starting a
relaxation there costs steps and can land in a different minimum:"""),

    ("code", """\
mv.pp.predict_volume(proposed)
proposed.obs[["formula", "volume", "predicted_volume", "volume_scale"]].round(3)"""),

    ("markdown", """\
No calculator involved — the prediction comes from tabulated bond lengths — and
`volume_scale` records how far the cell moved, so a suspicious rescaling is
visible rather than silent.

### The inverse question

`mv.gen.predict_substitutions` starts from a structure and asks what could be
swapped into it. `mv.gen.predict_hosts` starts from a **composition** and asks
which of the structures you already have could hold it — which is the more
useful direction when you know what you want."""),

    ("code", """\
hosts = mv.gen.predict_hosts(cathode, ["Na+", "Mn2+", "P5+", "O2-"],
                             source="oxidized")
mv.pp.describe(hosts)
hosts.obs[["parent", "target", "host_probability", "formula"]].round(5)"""),

    ("markdown", """\
From LiFePO₄ alone it builds **NaMnPO₄** — a real sodium-ion cathode — because
the model has seen Li→Na and Fe→Mn often enough in the ICSD.

The library you pass *is* the search space, so a database export finds far more
than three structures will.

```{note}
The species count has to match. A four-species target only considers
four-species hosts, because the model substitutes one for one and never changes
how many there are. Ask for a two-species target against this library and it
says so rather than returning an empty list — "no host found... it only
considers hosts with 2 distinct species; this library has [4]".
```

```{seealso}
[Models and campaigns](models_and_campaigns.ipynb) covers the other half:
predicting what you have not computed, and choosing what to compute next.
```"""),
]

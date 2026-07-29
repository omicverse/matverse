# Release notes

## v0.1.30

**87 of 136 in-scope pymatgen modules, 64.0%.** 31 open gaps, 18 blocked.

### `mv.pp.symmetry`

`mv.pp.describe` reports the space group symbol. This reports what the symbol
implies and what a screen can filter on: crystal system, crystallographic point
group, how many operations the group contains, the Wyckoff positions the atoms
occupy, and the order of the site symmetry group at the least and most symmetric
site.

SrTiO₃ comes back cubic, m-3m, 48 operations, Wyckoff **1a, 1b, 3c** — the
standard perovskite assignment, arrived at from the coordinates rather than from
the label. fcc copper has one distinct site; LiFePO₄ has eight.

**Wyckoff count is the number worth having.** It says how many *distinct* sites
a structure has, which decides how much work everything downstream is: how many
vacancies `mv.pp.defects` enumerates, how many NMR environments to expect, how
many independent parameters a refinement has. The defect tutorial gets six
inequivalent vacancies from a 28-atom LiFePO₄ cell rather than twenty-eight, and
that is this number. A test asserts the two agree.

The alias collision check fired a fifth time: `mv.mol.point_group` owns "point
group", where for a molecule it is the whole symmetry answer. For a crystal it
is one fact among several, so this is reached as "crystallographic point group".

## v0.1.29

**84 of 136 in-scope pymatgen modules, 61.8%** — and this release is mostly
bookkeeping, so the number needs unpacking before it is believed.

### A gap that cannot be closed is still a gap

The coverage map said which modules were uncovered. It could not say *why*, and
the difference between "nobody has done it" and "this needs a DFT run matverse
does not perform" is the difference between a backlog and a wish.

`BLOCKED` now records the eighteen with an external blocker — the Freysoldt and
Kumagai charge corrections need the electrostatic potential from a real DFT run,
XPS needs a projected density of states rather than a total one, three molecular
modules need openbabel, BoltzTraP needs a binary that does not build here.
**They stay in scope and uncovered.** What they gain is a reason.

That leaves **34 open gaps** out of 136 in-scope modules.

### Two reclassifications, and one of them was wrong

`phonon.*`, `core.trajectory` and `core.units` moved to NATIVE: matverse builds
the dynamical matrix from ASE displacements, keeps its own trajectory
statistics, and records units on the object. Those are the NATIVE definition.

Fifteen `core` and `symmetry` modules were then moved to INTERNAL as "data types
rather than capabilities", which took coverage from 53.5% to 65.4% in one step.
That was too aggressive, and checking each one the way this branch has been
checking pymatgen showed it: `core.bonds` has `get_bond_order` and `is_bonded`,
`core.molecular_orbitals` has `obtain_band_edges`, `symmetry.groups` has the
symmetry operations and crystal system. Those are real API and belong in the gap
list.

Only six are genuinely data types or enums — `Site`, `SymmOp`, `Spectrum`, the
two XC functional enums, and the package re-export. `core.interface` turned out
to be neither: `CoherentInterfaceBuilder.get_interfaces` yields `Interface`
objects that `mv.iface.build` stores, so it is transitive use.

The corrected figure is 61.8%, not 65.4%. The three-point difference is the
whole point of writing this down.

## v0.1.28

**76 of 142 in-scope pymatgen modules, 53.5%.**

### Three modules matverse was already using without an import

`SubstrateAnalyzer` runs the Zur-McGill lattice search out of
`analysis.interfaces.zsl`; `HighSymmKpath` is the front door to
`symmetry.kpath`; `PiezoTensor` subclasses `core.tensors.Tensor`, so the
symmetry check and IEEE conversion in `mv.prop.piezoelectric` are that module's
code. All three are used, none is imported, and each is recorded in `TRANSITIVE`
with the object that hands it over.

### One thing deliberately not added

`analysis.chempot_diagram` builds the full chemical potential domain per phase,
where `mv.thermo.chempot_limits` gives the window for one target. It is not
added, and the reason is shape rather than difficulty: a domain is a polytope,
not a scalar, so there is no obvious `obs` column for it — and EMT finds no
stable Al-Ni compound, so there is no worked example on a calculator matverse
ships to check the deposit against.

Adding it would have moved the number by one. That is not a reason.

## v0.1.27

**73 of 142 in-scope pymatgen modules, 51.4%.**

### `mv.mol.match` compared a fingerprint, not the molecules

It grouped molecules by the sorted list of pairwise distances between their
**heavy atoms**. That is invariant to rotation, translation and relabelling —
and it is not a proof of congruence. Two different geometries can share a
distance spectrum, and hydrogens were ignored entirely, so every CH₄ had the
same one-atom fingerprint no matter where its hydrogens sat.

It now superposes: Kabsch for the rotation, Hungarian assignment over the atom
labels, and a match when the best RMSD is within tolerance. `obs['match_rmsd']`
carries that RMSD in angstrom, so a tolerance can be chosen by looking rather
than guessed.

A rotated ethanol matches at RMSD 0. The same ethanol with 0.5 Å of noise per
atom does not, at a tolerance of 0.1. A stretched methane no longer reads as
tetrahedral methane.

pymatgen's default `MoleculeMatcher` needs openbabel, a C++ library rather than
a wheel. The ordering matchers used here need nothing beyond numpy and scipy.

### A layout split that reached the library code

`molecule_matcher` moved from `pymatgen.analysis` to `pymatgen.core` in 2026.5,
and matverse supports both sides of that move — so the import tries one and
falls back to the other. The coverage map already had `EQUIVALENT` for exactly
this; this is the first time the split reached a function rather than the
bookkeeping, and running the suite on both interpreters is what caught it.

## v0.1.26

**72 of 142 in-scope pymatgen modules, 50.7%.**

### `mv.env.chemenv` was returning empty results, silently

Building `mv.env.connectivity` on top of ChemEnv exposed a bug in the ChemEnv
wrapper that has been shipped since v0.1.15.

`LocalGeometryFinder` keeps state that `setup_structure` does not clear.
`mv.env.chemenv` built one and reused it across the dataset, so **every material
after the first got empty environments** — a blank string, not an error, not a
warning, not a count in `n_failed`.

Nothing caught it because **every test and every notebook cell used a
single-material dataset**. The olivine fixture is `[:1]`. The tutorial loads
`[:1]`. A bug that only appears from the second row onward was invisible to a
suite that never had a second row.

Both functions now build a fresh finder per structure, and
`TestOrderIndependence` pins the answer against dataset order rather than
against a stored value.

### `mv.env.connectivity`

`mv.prop.dimensionality` asks whether the *bonds* close in three directions.
This asks whether the **polyhedra** do, which is a different question and the
one an ion conductor turns on: a framework whose octahedra share corners in only
two directions cannot conduct in the third, however low the individual hop
barrier is — and the hop barrier is the expensive thing to compute.

Rocksalt and cubic perovskite both come back 3D-connected, which is what
edge-sharing and corner-sharing octahedral frameworks are.

The alias collision check fired for the fourth time on this branch:
`mv.env.bonds` already owned "connectivity" for the same idea one level down —
which *atoms* connect rather than which polyhedra — so this one is reached as
"polyhedral connectivity".

### A third kind of coverage claim

Two chemenv connectivity modules are reached through objects a third hands over,
never by an import: `ConnectivityFinder` returns a `StructureConnectivity`, and
the components it yields are `ConnectedComponent`s. The direct-import check
caught the overclaim, and the answer was to record *how* they are reached rather
than to weaken the check.

## v0.1.25

**69 of 142 in-scope pymatgen modules, 48.6%.**

### `mv.pp.prototype`

A space group says which symmetries a structure has. A prototype says which
structure it *is*, and the two are different questions: Fm-3m covers rocksalt,
face-centred cubic and half-Heusler alike, so a screen that groups by space
group puts them in one bin.

Matched against the AFLOW prototype encyclopedia, and the textbook symbols come
back: NaCl → B1 Halite, diamond → A4, GaAs → B3 Zincblende, fcc Cu → A1 Copper.

An unmatched structure gets an empty string rather than a guess. The library is
large but finite, and "not in AFLOW" is worth keeping distinct from "matched
something wrong".

### Three capabilities not wrapped, each with a tested reason

The coverage map now records *why* a gap is a gap, not only that it is one.

**`analysis.disorder`** — `get_warren_cowley_parameters` returns the same value
for every pair. On B2, where every nearest neighbour is unlike, the definition
requires α = −1 for the unlike pairs and **+1** for the like ones; it returns
−1.0 for all four. It also raises on a genuinely disordered cell, which is what
the module is named for. A wrapper around output that cannot be reproduced from
the definition is worse than the gap.

**`analysis.diffusion.neb.full_path_mapper`** — the add-on calls
`StructureGraph.with_local_env_strategy`, renamed upstream to
`from_local_env_strategy`, so every migration-graph entry point raises against
the pymatgen installed here.

**`analysis.magnetism.heisenberg`** — fitting exchange couplings needs
spin-polarised energies and no calculator matverse ships is spin-polarised.

Three different reasons — upstream is wrong, upstream is stale, and nothing here
can verify it — and none of them is "we did not get to it". Each was found by
checking output against a value computed independently, which is the same
discipline the contract probes apply to matverse's own claims.

## v0.1.24

**65 of 142 in-scope pymatgen modules, 45.8%.**

### Interstitials and antisites

`mv.pp.defects` built vacancies and substitutions. Both are a site you can point
at. The two it could not build are the ones that are not: an **interstitial**
has to be *found* — it is a hole, and the Voronoi construction locates them —
and an **antisite** is the cross product of the species present, which for a
quaternary is more combinations than anyone enumerates by hand.

On LiFePO₄: 24 antisites and 12 lithium interstitials. The Fe-on-Li antisite
among them is *the* defect that blocks the one-dimensional lithium channel that
cathode conducts through.

Both route through `pymatgen-analysis-defects`, which also picks the supercell
itself — targeting a minimum image distance rather than a fixed multiple — so
`supercell=` does not apply to those two kinds. Vacancies and substitutions
still need no extra package.

A supercell window nothing can satisfy now says so. It used to report "no defect
was generated", which blames the chemistry for what is an argument problem: the
generator targets an image distance as well as an atom count, so a small
`max_atoms` can leave no legal cell. The message names the count, the window and
the first real failure.

### One of the three add-ons does not work

`pymatgen-analysis-diffusion` 2025.11.15 calls
`StructureGraph.with_local_env_strategy`, which pymatgen renamed to
`from_local_env_strategy`. Against the installed pymatgen 2026.5.4 every entry
point that builds a migration graph raises `AttributeError`.

That is worth stating plainly, because these three add-ons were installed
earlier on this branch *to count toward coverage* and never wired into a code
path. One of them could not have been. `MigrationGraph` is the capability worth
having — it finds every symmetry-distinct hop and whether they percolate, where
`mv.neb.hop_endpoints` finds one — and it stays in the gap list until the
upstream package catches up rather than being wrapped into a function that
raises on every call.

`pymatgen-analysis-alloys` is installed and its `AlloyPair.from_structures`
signature does not match its documentation either; it is unwrapped for now.

## v0.1.23

**64 of 142 in-scope pymatgen modules, 45.1%** — of which 39.5% was new
capability and the rest a correction to the map, flagged below rather than
quietly banked.

### A classification the map had inconsistent

Two families sat in the gap list only because matverse uses *one* module from
each: `mv.elec.cohp` reads ICOHPLIST through `io.lobster.outputs`, and `mv.dft`
writes inputs through `io.vasp.sets`. The other fifteen lobster modules and two
vasp ones are more file formats, input writers and a parallel "future" API —
which is exactly the parsing surface the other thirty-three `io` families are
already exempted for, on a policy the README and the migration guide both state.

Leaving them in TODO implied matverse intended to wrap them. It does not, and
saying so is the honest classification; the alternative mistake — marking a
whole family covered because one module in it is used — is the one this file was
written to prevent. pymatgen's own 2D plotters joined `vis` in the same bucket,
and `transformations.transformation_abc` is an abstract base class.

**78 real gaps remain**, the largest clusters being `analysis.defects` (9),
`analysis.diffusion` (7), `analysis.chemenv` (6) and `analysis.compatibility`
(4).

### `mv.mag.jahn_teller`

Magnetic ordering is one consequence of a partly filled d shell; distortion is
the other, and it is structural rather than magnetic. A degenerate electronic
ground state in an octahedral site lowers its energy by distorting the
octahedron — which is why LaMnO₃ is orthorhombic rather than cubic and why
manganese spinel cathodes fade on cycling.

LaMnO₃ and LaNiO₃ come out strong and SrTiO₃ inactive: Mn³⁺ and Ni³⁺ both put an
electron in a doubly degenerate e_g level, and Ti⁴⁺ is d⁰ with no degeneracy to
lift. `jahn_teller_species` names the ion responsible, because knowing a
material distorts is not useful without knowing which site is doing it; the
ligand bond lengths, which are the distortion rather than a label for it, stay
in `uns`.

### Two capabilities deliberately not shipped, with the reason

**Exchange couplings.** `pymatgen.analysis.magnetism.heisenberg` maps a set of
ordered magnetic structures and their energies onto a Heisenberg model. matverse
has the orderings and the energies, so the fit is available — but **EMT is not
spin-polarised**, so every ordering matverse can compute has the same energy and
the couplings that come out are meaningless. There is no calculator shipped here
that can verify the function, so it is not shipped either. With DFT energies it
would work, and that is a different claim from "it works".

**Freysoldt and Kumagai image-charge corrections.** `mv.thermo.defect_formation`
already records `image_charge_correction: False` and says so in its notes; the
corrections need the electrostatic potential from a real DFT run, which is the
`mv.dft` boundary. Upgrading a declared limitation to a capability is worth
doing and is not doable offline.

## v0.1.22

Where the candidates come from. **63 of 162 in-scope pymatgen modules, 38.9%**,
up from 37.0%.

Every other namespace assumes a list of candidates already exists. These three
make it, from the statistics of what has been synthesised rather than from a
calculation.

**`mv.gen.predict_substitutions`** ranks the swaps you did not think of, using
the data-mined ionic substitution model of Hautier et al. — how often two
species replace one another across the ICSD, rather than whether their radii
match. From LiFePO₄ it reaches LiVPO₄ and LiTiPO₄, both real olivine analogues,
without being told anything about olivines.

It needs oxidation states and refuses without them, because the model is defined
over ionic species: Fe²⁺ and Fe³⁺ substitute differently and that distinction is
the point. A probability here is a prior from what has been made before; it says
nothing about stability in this structure, which is what the hull is for.

**`mv.gen.predict_dopants`** asks the same statistics a narrower question and
gets fluorine as the n-type choice for LiFePO₄ — the doping strategy that
material is actually treated with. n-type versus p-type is arithmetic on
oxidation states, not a calculation of where the level lands; the top choice
goes to `obs` and the full ranking stays in `uns`, because the second and third
are usually the interesting ones.

**`mv.pp.predict_volume`** predicts the equilibrium volume from tabulated bond
lengths, within 3% on three published cathodes, and deposits a rescaled variant.
A candidate built by substitution keeps the cell it was built from, which can be
several percent from where the new composition wants to sit — starting a
relaxation there costs steps and can land in a different minimum.
`obs['volume_scale']` records how far the cell moved, so a suspicious rescaling
is visible rather than silent.

## v0.1.21

Four more of the gaps the coverage map named. **60 of 162 in-scope pymatgen
modules, 37.0%**, up from 33.3%.

### `mv.prop.quasiharmonic`, and a number that had to be recomputed

The quasi-harmonic Debye model gets a material off the zero-kelvin hull: compute
the energy at several volumes, let the Debye model supply a vibrational free
energy at each, and minimise the Gibbs free energy over volume at every
temperature.

pymatgen's `QuasiharmonicDebyeApprox` reports a Gruneisen parameter that matches
experiment — 1.91 for copper against a measured 1.96, 2.17 for silver against
2.4 — and a set of optimum volumes that do not. Its volume minimum moves
**twelve times too little**: 4.3e-6 /K for copper against a measured 5.0e-5.

So the expansion is computed from the thermodynamic identity instead:

    alpha_V = gamma C_V / (B V)

with the model's own Gruneisen parameter, the bulk modulus fitted from the same
E(V) points, and the Debye heat capacity rather than the Dulong-Petit constant.
That gives 4.5e-5 for copper and 5.1e-5 for silver — within 15% of measurement
for the metals EMT reproduces.

The discrepancy was findable because the bulk modulus from `mv.prop.eos` and the
Gruneisen parameter from the Debye model sit on one object under names that say
what they are. That is the second time on this branch that two quantities which
had to agree, and did not, located a defect neither number showed alone.

Aluminium is the usual outlier and is pinned as one: EMT gives it a Gruneisen
parameter of 0.83 against a measured 2.2, so its expansion is held to a factor
of two rather than to 25%.

### `mv.prop.cost` and `mv.prop.supply_risk`

Two screening axes that need no calculator and end more projects than any energy
does. Raw-material cost from elemental prices, and the Herfindahl-Hirschman
indices for how concentrated the world's production and reserves are.

Cost will not tell you what a synthesis route costs — it is elements only. It
will tell you that platinum oxide is four orders of magnitude dearer than iron
oxide, and no process optimisation closes that. Supply concentration is a
**different** risk from price: cobalt and the rare earths are affordable and
concentrated, which is exactly what makes them awkward.

`mv.prop.cost` needs `bibtexparser`, which pymatgen uses to read the citations
in its price table and does not require itself; it is declared as the `cost`
extra and reports the install command rather than a traceback.

### `mv.prop.neutron`

Powder neutron diffraction on the same grid convention as the X-ray pattern.
Not redundant with it: neutrons scatter off nuclei rather than electrons, so
scattering lengths do not follow atomic number and light atoms show up here and
nowhere else. Copper's tallest X-ray line is at 43.4 degrees and its tallest
neutron line at 38.5 — the same allowed reflections, reweighted. For a lithium
cathode this is the pattern that locates the lithium.

### The alias collision check fired again

`mv.md.sweep` already claimed "thermal expansion", and it measures the quantity
directly by running dynamics at several temperatures. Both functions compute the
same thing by different routes and the registry cannot give one name to two
functions, so the direct measurement keeps the plain alias and the model is
reached as "quasiharmonic". Third time this mechanism has caught a name landing
on the wrong function.

## v0.1.20

### "Is pymatgen covered?" now has a number that cannot drift

Asked directly, the honest answer was no — and the figures quoted earlier in
this branch were wrong. "15 of 48 analysis modules" counted only the top level
of `pymatgen.analysis`, where the real tree has 110 leaves. Three pymatgen
add-ons were installed, counted toward coverage, and never wired into a single
code path.

`matverse._coverage` classifies every public pymatgen module into one of five
buckets — WRAPPED, NATIVE, INTERNAL, NOT_A_GOAL, TODO — and
`tests/test_pymatgen_coverage.py` enforces it:

- an unclassified module fails the test, so TODO cannot quietly shrink
- a module claimed as WRAPPED must actually be **reached by an import**,
  resolved at runtime rather than matched textually
- a matverse function named in the map must actually be registered
- the covered count may not fall

**56 of 162 in-scope modules, 34.6%.** 107 gaps are listed by name.

Three things the enforcement caught immediately, each of which would have
inflated the number:

**22 modules are re-export shims.** pymatgen 2026.5 moved much of `analysis`
into `core`, leaving three-line stubs behind. Counting both names doubles the
denominator *and* files the real module under TODO while the stub reads as
covered. Shims are detected by reading the source, so the next reorganisation is
absorbed rather than mismeasured.

**A package `__init__.py` can hold real API.** pymatgen's correction schemes live
in `analysis/compatibility/__init__.py`; skipping package inits made the classes
matverse calls invisible to the count.

**The two supported pymatgen versions ship different trees** — 162 modules in
scope on 2026.5, 134 on 2025.10 — so a count from one is not a ratchet for the
other. The number is reported on both and asserted on one.

### `mv.thermo.corrections` — the gap that made hulls wrong

The largest real gap, and it affects correctness rather than convenience.
Materials Project energies arrive **already corrected**; energies you computed
yourself and read back through `mv.dft.read_outputs` do not. Putting them on one
hull is the same class of error as mixing EMT with PBE, except that nothing
about the column names says so.

`mv.thermo.corrections` applies the published schemes — MP2020, the aqueous
variant, the legacy set, MIT — and deposits the result as **its own level**:
`energy_pbe` in, `energy_pbe-mp2020` out, with `uns['levels']['pbe-mp2020']`
recording the scheme and what it was corrected from. That is the level-of-theory
rule applied one step further along, and it makes a hull built on the wrong
column a visible mistake.

The magnitudes are not small. Against the ~0.05 eV/atom threshold a screen calls
"close to the hull":

| | correction | why |
|---|---|---|
| Al₂O₃ | −2.06 eV | 3 × −0.687, the oxide anion correction, once per oxygen |
| Fe₂O₃ | −6.57 eV | the same anion term plus 4.5 eV of +U correction on two irons |
| Cu | 0 | an elemental metal has no anion to correct and no U |

`run_type` is inferred by MP's own rule — a transition metal from its table
together with oxygen or fluorine — and recorded next to the result rather than
assumed.

The `produces` slots interpolate `level` and `scheme` rather than `key_added`,
because those are the two parameters the default output name is built from. A
template naming `key_added` resolves to nothing on the call everybody makes,
which is what probing the claim showed.

## v0.1.19

### The four functions that "would raise on every dataset"

v0.1.18 left piezoelectric constants, NMR shieldings, SLME and XAS out, on the
argument that they are ingestion rather than calculation and would therefore
raise on any dataset a user could build. The first half of that was right and
the conclusion did not follow, which the library itself already demonstrated:
`mv.elec.bands` takes pymatgen band structures as an argument, `mv.exp.attach`
takes measured curves, `mv.elec.transport` takes a bands object. Taking a result
somebody else computed is a pattern matverse already has.

What these functions do is the step *after* the calculation — reduce a tensor to
the parameters a spectrum is described by, check it against the crystal
symmetry, turn a dielectric function into an efficiency. That step is arithmetic
with conventions in it, and conventions are what gets a result quoted wrongly.

**`mv.prop.nmr`** and **`mv.prop.efg`** reduce per-atom shielding and electric
field gradient tensors on the sites axis. Both Haeberlen and Herzfeld-Berger
parameters are reported, because both are in use and they disagree about what
"anisotropy" names: Haeberlen's ζ is σ₃₃ − σ_iso, and the reduced anisotropy
most spectrometer software prints is 3/2 of it. A single column called
`anisotropy` would be wrong for half its readers. The quadrupolar coupling
constant is looked up from the element, since a quadrupole moment belongs to the
isotope rather than to the calculation.

**`mv.prop.piezoelectric`** checks a tensor against the structure's point group,
converts it to the IEEE frame, and reports the largest longitudinal response
over all directions. Validated on α-quartz: given the measured d₁₁ = 2.3 pC/N
and d₁₄ = −0.67 pC/N, it recovers 2.297 pC/N without being told where to look,
because for class 32 the basal-plane response goes as d₁₁cos 3θ and its maximum
is d₁₁ itself. Components depend on how somebody oriented the cell; the maximum
does not. The same tensor on fcc copper reports `piezo_symmetry_valid = False` —
inversion symmetry forbids piezoelectricity, so a non-zero tensor there is an
error rather than a discovery.

**`mv.prop.dielectric`** and **`mv.prop.slme`** take a dielectric function onto
the existing grid convention, derive the absorption coefficient from it
(α = 2Ek/ħc — derived, so ε stays recoverable), and compute the spectroscopic
limited maximum efficiency under AM1.5G. Calibrated against Shockley-Queisser: a
step absorption edge with no indirect gap gives 33.94% at a 1.34 eV gap, and the
limit falls away on both sides.

SLME is reported **as a percentage with the unit recorded**. pymatgen's `slme()`
returns the same number with no unit in its docstring, which is the kind of
thing that becomes a factor of 100 three functions downstream.

The reason to compute it at all shows up in one table: four model absorbers with
the *same* 1.34 eV gap and the same Shockley-Queisser ceiling of 33.94% score
33.94, 30.79, 7.61 and 0.86. A screen ranked on band gap cannot tell them apart.

### XAS, and what is actually left out

XAS did not get a function, and this time the reason survives inspection: an XAS
spectrum is a curve on an energy grid, which is what `uns['grids']` and an
`obsm` block already are. `mv.exp.attach(md, 'xas', spectra, energies)` stores
one today, and `mv.prop.compare_grids` compares a measured edge against a
computed one. A wrapper would add a name, not a capability.

What genuinely remains outside is generating these quantities, which needs a DFT
code, and reading them back from specific output formats — the boundary
`mv.dft` already draws.

## v0.1.18

### `mv.calc.relax` never moved the lattice

Adding `mv.prop.eos` gave a second, independent route to the bulk modulus — a
curvature in volume against `mv.prop.elastic`'s curvature in strain. The two are
the same quantity, and they disagreed by **9–12%, always in the same direction**.
Both were converged: the EOS value moved by 0.2% between a ±4% and a ±10% strain
series, and the elastic value by 0.2% between a 0.2% and a 2% strain.

The cause was not in either function. `mv.calc.relax` built a bare
`BFGS(atoms)`, with no cell filter, so it relaxed the atomic positions and never
the lattice. For a high-symmetry cell that is not a weak relaxation — it is *no*
relaxation, because the forces on an fcc metal vanish by symmetry. The optimiser
converged immediately, changed nothing, reported `relax_converged = True`, and
deposited the input geometry under the name `relaxed_emt`.

Everything downstream inherited it. Elastic constants were second derivatives
taken about a geometry under residual tensile stress, which softens them; phonon
frequencies, Debye temperatures and any energy entering a hull were computed at
whatever lattice constant the input happened to carry.

`cell=True` is now the default, via ASE's `FrechetCellFilter`, and is skipped
for molecules. Five of the seven bundled fcc metals moved closer to experiment:

| bulk modulus (GPa) | before | after | experiment |
|---|---|---|---|
| Cu | 123 | 135 | 140 |
| Ni | 157 | 176 | 180 |
| Ag | 93 | 100 | 101 |
| Au | 159 | 174 | 180 |
| Al | 35 | 40 | 76 |

With the cell relaxed, the two routes agree to **1.1% at worst and under 0.4%
for six of seven**.

The part worth keeping is how it was found. Neither number looked wrong on its
own — 123 GPa for copper is a plausible answer from a cheap potential, and no
test asserted otherwise. **The disagreement between two quantities that had to
match was the only visible symptom**, and it was visible only because both sit
on one object under names that say what they are. Pass `cell=False` when the
lattice is meant to be held, as for a slab.

Also worth recording: no existing test failed when this was fixed. Nothing in
the suite had asserted that relaxation relaxes anything, so
`tests/test_equation_of_state.py` now pins it directly.

### `mv.prop.eos` — the equation of state

Compress each cell over a series of volume scale factors, fit a Birch-Murnaghan
(or Vinet, Murnaghan, Poirier-Tarantola) equation of state, and deposit the bulk
modulus, its pressure derivative, the equilibrium volume and the RMS misfit. The
energy-volume curves land in `obsm` on a shared **scale-factor** grid rather
than a volume grid, because materials of different size have no common volume
axis while the strain series they were computed on is common by construction.

`B0'` earns its place beyond the bulk modulus: aluminium comes out at 1.95
against a measured 4.4, so EMT has the wrong *shape* for aluminium's
energy-volume curve rather than merely the wrong curvature at one point.

### `mv.prop.dimensionality` — 0D, 1D, 2D or 3D

Classify the bonded network so a screen can ask for exfoliable candidates
directly. This is the archetypal property `X` cannot reach: graphite and diamond
have the same composition matrix, the same `var` and the same element counts,
and differ by whether the bonds close in three directions.

Validated against textbook answers — graphite 2D, MoS2 2D with two layers per
cell, fcc copper 3D, well-separated I2 0D. The near-neighbour strategy is
recorded next to the answer for the same reason `mv.env.coordination` records
it, and the stakes are higher here: the classification turns entirely on whether
a long contact counts as a bond, which is exactly what a van der Waals gap is.

### What was not built, and why

The roadmap for this batch listed piezoelectric constants, NMR shieldings, SLME
and XAS alongside the equation of state. Those four are **ingestion, not
calculation**: pymatgen's `PiezoTensor`, `ChemicalShielding` and SLME all take a
DFT-computed tensor or dielectric function as *input*, and there is nothing for
a cheap potential to compute. They belong behind `mv.dft`, on the same boundary
as `read_outputs`, and pretending otherwise would have produced four functions
that raise on every dataset a user can actually build.

### A note on `uns['levels']`

`set_level` replaces its entry rather than merging, so per-call parameters
written by one operation are overwritten by the next at the same level —
`mv.prop.elastic` after `mv.calc.relax` leaves `strain` where `fmax` and `cell`
had been. Per-operation parameters are kept in order by `uns['provenance']`,
which is where they belong and where `mv.provenance(md)` reads them from, so
nothing is lost. The level entry should probably hold only level-defining
fields; that is not changed here.

## v0.1.17

### The contract-verified rate was measuring the test suite

matverse reports a **contract-verified rate** rather than a field-coverage
score, on the argument that the audit metric in common use credits a registry
field for being *present* rather than *true*. That argument only holds if the
probe reaches the whole registry, and it had stopped doing so: the battery grew
alongside the first six namespaces, the library grew to twenty-nine, and the
probe list did not follow. **165 of 491 claims were probed — 33.6% — and the
reported rate was 100%.**

A rate is only as good as its denominator. 100% of a third of the claims is the
same kind of statement the audit metric was criticised for.

The battery now covers everything: **433 claims verified by execution**, up from
165. Of the 132 entries that make a claim, 118 are probed and 14 are exempt —
each named with its reason (needs the network, needs a real VASP or LOBSTER
output, needs a tool that is not installed). `test_no_claim_goes_unprobed` fails
if an entry ships a claim that nothing checks, so this cannot silently reopen.

### Eleven claims were false, and the harness was hiding some of them

Probing the other two thirds deleted seven more claims and corrected five. The
ones worth reading:

`mv.screen.filter requires obs['{column}']` was not wrong so much as
**unsayable**. Its columns are encoded in the *keys* of `**criteria` —
`e_above_hull_emt__lt` is a column and an operator joined by `__` — so no
parameter holds a column name for a slot template to interpolate. This is the
second boundary this line of work has found, and it is a different one from the
`pymatgen` case: there the library had no named state to point at, here the
state is named but the name never becomes an argument.

`mv.utils.resume`, `mv.utils.job_status` and `mv.dft.status` each claimed to
require the thing they exist to report on. All three answer correctly when it is
absent — no column means every row is still to do — which is the right behaviour
and makes the claim false.

Three failures were in the **harness**, not the registry, and the last is the
one that matters:

- `probe_call(..., name=...)` — the lookup argument shadowed a real parameter of
  `mv.pp.supercell`, `mv.screen.filter` and others. Passing `name='big'` sent it
  to the probe, the entry came back empty, and the call was **silently not
  probed at all**. A harness that reports nothing wrong because it tested
  nothing is worse than no harness.
- The entry was looked up by bare function name, so `mv.pl.pareto` resolved to
  `mv.screen.pareto` and reported a claim the function never made.
- A `produces` claim on a call that returns a *new* dataset — one row per
  ordering, per slab, per fragment — was checked against the input object, where
  it correctly was not present.

### A contract slot can now say which object it lands on

The one place the omicverse contract vocabulary needed extending, and it is a
gap rather than a disagreement. On a library where every operation takes one
object, naming a container is enough. matverse has operations that take two:

```python
produces={"sites.obs": ["coordination_number"]}      # mv.env.coordination
```

`mv.mag.ground_state` is the case that forces it. It writes four columns onto
the parent and a fifth back onto the orderings, and an unqualified `obs` said
all five arrived together. `mv.env.coordination`, `mv.calc.forces` and
`mv.surf.wulff` had the same problem — and in every one of them **the prose note
already said which object**, while the machine-readable claim did not.
`mv.calc.forces` went further and said the fields "cannot name" those slots
"because they describe one object only", which is exactly the limitation now
lifted.

A qualifier is checked against the signature at import: one that names no
parameter resolves to no object and could never be probed, so it is refused
rather than allowed to become an unverifiable claim.

### A claim can now be undecided rather than failed

`mv.elec.transport` needs BoltzTraP2, which does not build here. Counting its
three claims as failures would report the environment as a defect in the
registry, and counting them as passes would be a lie. They are reported
separately and excluded from the rate's denominator.

## v0.1.16

### Molecules were never out of scope

The design note said molecules were out of scope "by construction". That was
wrong, and checking it took one line: a `Molecule` has a composition, so `X`
and `var` build for water exactly as they do for a crystal — H:2, O:1. The only
thing that failed was a decoder that assumed a lattice.

So molecules live on the same axes as everything else, and **one object holds
both**. `obs['is_periodic']` tells them apart; `volume` and `density` become
NaN for a molecule rather than absent, so a dataset mixing a catalyst with its
adsorbates stays one table. `mv.pp.qc`, `mv.feat.element_stats` and even
`mv.calc.energy` needed no changes at all.

**`mv.mol`** adds what genuinely differs — a point group rather than a space
group, covalent bonds rather than a coordination polyhedron, fragments rather
than defects. Validated against textbook answers: C2v for water, Td for
methane, C3v for ammonia, with group orders 4, 24 and 6.

The methane row is the one worth reading. `can_be_polar` comes out **False**,
and not as a tendency: Td contains operations that map any candidate dipole
onto its own negative, so symmetry forbids it outright. A dipole survives only
in C1, Cs, Cn and Cnv.

`fragments` breaks every acyclic bond and keeps the pieces, conserving atoms on
every cut. It records the **unreduced** formula next to the reduced one,
because pymatgen's reduced formula applies the diatomic convention — a single
hydrogen atom reads `H2` and a hydroxyl reads `H2O2`, which is right for a
stoichiometry and actively misleading for a fragment.

### `mv.pl.structure`

"Crystal Toolkit and VESTA do that better" was a reason not to compete, not a
reason to have nothing. Draws either kind of material, interactively through
py3Dmol when it is installed and through matplotlib when it is not. It is the
quick look you take twenty times a day to catch a slab built upside down.

### `mv.utils.submit` and `mv.utils.job_status`

matverse still runs no DFT and is not a workflow manager — atomate2, quacc and
AiiDA do that. What was missing was the link back: `submit` shells out to
`sbatch` and records the job id **on the object**, so "which job is computing
this dataset" is answerable from the data rather than from shell history.
`job_status` reads `squeue`, falling back to `sacct` for jobs that have
finished, because `squeue` forgets a job shortly after it ends.

`mv.records` is now exported, since reading the submissions back needs it.

### Another alias that pointed at the wrong function

`mv.utils.slurm_script` claimed `submit` and `sbatch`. It writes a script and
stops. This is the second time the registry's collision check has caught a
misdirected alias — the first was `mv.dft.read_dos` claiming `band structure` —
which is the machinery working rather than a coincidence.

## v0.1.15

### Two namespaces for what composition cannot reach

matverse wrapped 15 of pymatgen's 48 `analysis` modules, and the two biggest
holes were the two questions a composition vector provably cannot answer.

**`mv.env` — local and coordination environments.** Near-neighbour algorithms,
ChemEnv's polyhedron classifier, and the bond network. Validated on olivine
LiFePO₄: Li and Fe come out six-coordinate, P four-coordinate, and the
**continuous symmetry measure** tells the structural story a coordination
number cannot — 0.14 for the rigid phosphate tetrahedron against 2.37 for the
heavily distorted FeO₆ octahedron. That contrast is why the olivine framework
does not collapse when lithium leaves it.

Results land where their shape belongs: per-atom to the sites axis, the bond
network to `obsp` (the atoms × atoms slot a kNN graph occupies in single-cell
analysis), per-material summaries back to `obs` where a screen can reach them.

**`mv.elec` — electronic structure.** Previously `mv.dft` could read a density
of states and nothing else.

A band structure is *bands × k-points* per material and ragged in the number of
bands, so it gets **its own axis**: one row per band per spin per material, `X`
its energy along the path, `var` the k-points. Structurally a cells × genes
matrix again.

Two materials generally have different k-paths — Cu gets 102 points and Al 95 —
so bands are resampled onto a normalised path fraction. That is the same move
that puts two diffraction patterns on one 2θ grid, and the axis must be read as
"fraction along this material's own path", never as a wavevector.

Energies are stored relative to the Fermi level, because an absolute eigenvalue
means nothing across codes or even across two runs of one code.

`kpath`, `bands`, `read_bands`, `band_features`, `dos_fingerprint`, `cohp` and
`transport`. Registry 121 → 132, all of them exercised in a notebook.

**`mv.iface` — interfaces between two materials.** A pairing is not a
material, so rows are *pairs*, with `obs['film']` and `obs['substrate']`
pointing back at the parent — the same derived-axis-with-foreign-keys shape as
the sites and bands axes.

`match` runs Zur and McGill's epitaxial search and reports the strain each match
would need; `reactivity` reads off whether the contact survives; `build` returns
the interface cells as an ordinary materials object, one per termination.

The pairing is **ordered**: copper grown on aluminium is a different problem
from aluminium grown on copper, because the substrate is the fixed lattice and
the film is what has to stretch.

Validated on Ni–Al, where `reactivity` finds **Ni + 3 Al → Al₃Ni at
−0.45 eV/atom** — the intermetallic that actually forms, and the reason
diffusion barriers exist in turbine coatings. This is the question that decides
whether a solid electrolyte works: lithium metal and most sulfide electrolytes
are each stable and destroy each other on contact, which no hull computed on
either one alone can see.

**`mv.disorder` — partial occupancy.** Everything else in matverse assumes a
structure is ordered; a large fraction of real materials are not, and a
first-principles code cannot take a fractionally occupied cell as input.

`describe` reports where the disorder is and the ideal **configurational
entropy**, which comes out exactly at `k_B ln m` for an equiatomic mixture of
*m* species. That number is the argument for high-entropy alloys: at a
synthesis temperature of 1500 K a five-component equiatomic alloy gets
−0.21 eV/atom, four times the 50 meV a screen typically calls "close to the
hull". A screen that ignores `-TS` rejects exactly the phases the field exists
to study.

`orderings` produces ordered approximants, and admits when its ranking is
arbitrary: Ewald energies need oxidation states, and without them every
arrangement scores zero — honest for a metallic alloy and wrong for an oxide.

`sqs` **refuses** when ATAT is absent rather than handing back the ordered
ground state, which is the opposite of a solid solution and would be a
plausible-looking answer to a different question.

`dope` catches a silent failure in pymatgen: `DopingTransformation` enumerates
with enumlib and returns an **empty list** rather than raising when it is
missing, so a doping study runs to completion and produces nothing. matverse
reports which of the two things happened.

**`mv.transform` — pymatgen's transformations, on the object.** There are
around forty-five `Transformation` classes. Wrapping forty-five of them as
forty-five matverse functions would be a bad trade: most are one line, the
registry would be mostly noise, and every pymatgen release would need another
wrapper.

So there is one function that takes the transformation **by name**, looked up
in pymatgen at call time — which means a transformation added upstream is
available the day it lands:

```python
mv.transform.apply(md, 'PrimitiveCellTransformation')
mv.transform.apply(md, 'CubicSupercellTransformation', min_length=10)
```

The result becomes a structure **variant**, so the input survives and "which
structure was this computed on" stays answerable. A transformation that fails
on one row leaves that row alone and records `False` — a dataset where
something applies to some materials and not others is the normal case, not the
exceptional one.

`expand` keeps every result of a one-to-many transformation as its own row;
`chain` applies a sequence; `oxidation_states` is the missing prerequisite
behind three of pymatgen's more confusing errors (`Element has no attribute
oxi_state!`, `Valences cannot be assigned!`, and Ewald ranking that silently
scores everything zero).

### A migration guide for pymatgen users

`tutorials/from_pymatgen.ipynb` answers the question a pymatgen user actually
has, which is not "what do I learn instead" but "what does putting my
structures in an object buy me". It is explicit about what matverse does *not*
do — molecules, visualisation, running anything — so nobody goes looking.

### Structures carrying numpy site properties could not be stored

`mv.iface.build` returns pymatgen `Interface` objects, which carry the
interface-normal vector as a numpy array in their site properties. matverse
serialises structures to JSON so the object survives `write_h5ad`, and `json`
refused, with a bare `TypeError: Object of type ndarray is not JSON
serializable` naming neither the structure nor the property.

The encoder now converts arrays and numpy scalars, and refuses anything it
genuinely cannot store with a message that names the offending property type
and says what to do about it. Any structure from any source was affected, not
just interfaces.

### `mv.pl.set_style` follows omicverse's `plot_set`

Same shape: emoji status lines, inline retina format, font support including
the Arial download, scalar `figsize` promoted to a square, warning suppression,
accelerator detection, an ASCII logo printed once with the version and tutorial
link, and a closing line.

One item differs on purpose. Where omicverse reports scanpy's verbosity,
matverse reports **which calculators this installation can actually run** —
because it ships one working calculator and dispatches the rest to whatever you
installed, so that answer differs on every machine and decides what the session
can do.

`quiet=` still works and still silences everything, because v0.1.15's
predecessor is on PyPI and a rename is not worth breaking a working notebook
over.

### A wrong alias, caught by the registry

`mv.dft.read_dos` claimed the alias `band structure`. It reads a density of
states. The collision check refused to let `mv.elec.bands` register the name it
should have owned all along, which is the alias machinery doing exactly what it
was built for.

## v0.1.14

### Every registered function is called in a notebook

The registry had 121 entries and the tutorials exercised 82 of them. A registry
entry is a promise that a function is part of the public surface; if the
documentation never calls it, nothing checks the promise still holds — the
signature drifts, the slot it writes gets renamed, and the first person to find
out is a user.

So it is now **enforced**: `_scripts/build_notebooks.py` fails the build if any
registered function appears in no notebook, and the docs CI job runs it. 121 of
121, across 11 executed notebooks — 227 code cells and 36 figures.

Two new tutorials carry most of the remainder:

- **[Getting data in and out](tutorials/data_io.ipynb)** — every door into the
  object and back out, against a **live** OQMD over OPTIMADE rather than a
  described one. It queries all 15 binary Al–Ni entries and finds that 7 are
  duplicates, which is a better argument for `mv.pp.dedup` than any fixture.
- **[Infrastructure](tutorials/infrastructure.ipynb)** — units, checkpoints,
  corpora larger than memory, Slurm scripts, and the DFT hand-off.

The rest went where they belong: `free_energy` after the phonons,
`calc.committee` next to model uncertainty, `screen.rank` and `thermo.reaction`
into screening, `feat.embed` into chemical space, `pp.supercell`/`pp.strain`
into defects, batched MD engines into dynamics.

### Bugs this found

**You could not continue working on a saved object.** h5ad stores
`uns['provenance']` as a list of strings and reads it back as a numpy array,
which has no `.append` — so the *first operation* on any reloaded object raised
`AttributeError`. Every tutorial ended by saying the object survives a round
trip; it did, and then broke. `record()` now normalises whatever it finds, and
a test loads a written object and keeps working on it.

**`mv.data.from_optimade` failed on machines with working network.** It used
`urllib`, which verifies TLS against the interpreter's system certificate
bundle. On a cluster that bundle is routinely stale, and every HTTPS call died
with `CERTIFICATE_VERIFY_FAILED` on a node whose network was fine. It now
prefers `requests`, which ships `certifi` — and which arrives with pymatgen, so
it is always present.

**An unreadable cache entry was fatal.** `mv.datasets.fetch` let a cache file
written by a different anndata version take down the call. A cache exists to
make things faster; one that can make a working call fail is worse than none.
It now warns and re-fetches.

**"Check your filter" was the wrong advice.** Materials Project's OPTIMADE
mirror answers `200` with `data_returned=0` to *any* query, including an empty
one. A user sent to debug their filter will not find the problem there, so the
error now distinguishes a provider serving nothing from a filter matching
nothing.

## v0.1.13

### Four tutorials for the physics

The tutorials covered 46% of the public API and skipped four namespaces
entirely — `mv.md`, `mv.neb`, `mv.surf` and `mv.mag` — which between them are
diffusion, catalysis, dynamics and magnetism. The library could do all of it,
validated against literature in the test suite, and the documentation showed
none of it.

- **[Defects and diffusion](tutorials/defects_and_diffusion.ipynb)** — vacancy
  enumeration, formation energy against Fermi level, and a migration barrier of
  **0.754 eV** against a literature ~0.70 for fcc copper. Ends by combining
  formation and migration into a diffusion coefficient and reading the
  Arrhenius slope.
- **[Surfaces and adsorption](tutorials/surfaces_and_adsorption.ipynb)** —
  slabs, surface energies in the literature ordering **(111) < (100) < (110)**,
  the Wulff shape, and oxygen binding 170 meV more strongly in the hollow site
  than on top of an atom.
- **[Dynamics](tutorials/dynamics.ipynb)** — equilibration you can see, thermal
  expansion from motion alone, and a melt-quench with the failure mode that
  caught eight universal potentials in 2026.
- **[Magnetic ordering](tutorials/magnetic_ordering.ipynb)** — enumerate the
  spin states before the hull. It ends by refusing to name a ground state,
  because `magnetic_spread` comes out at exactly zero: EMT has no notion of
  spin, and the object says so in a number rather than producing a confident
  arbitrary answer.

### `mv.md.run` keeps the temperature trace

It sampled the instantaneous temperature every `sample_every` steps and threw
all of it away except the mean. A mean cannot distinguish a run that
equilibrated from one still drifting at the last step, and that distinction is
the difference between a result and an artefact — so the trace is now deposited
as `obsm['md_temperature_trace_{level}']` against a time grid in picoseconds.

A rerun of different length discards the earlier trace rather than failing.
Unlike a diffraction pattern, whose axis the caller chooses, a trace axis falls
out of `steps` and `sample_every`, so two runs of different length produce
curves that genuinely cannot share an axis.

### The NEB used a method ASE recommends against

`mv.neb.barrier` built its band with ASE's default tangent estimate. ASE's own
warning describes that default as "an unpublished, custom implementation ...
not recommended as it frequently results in very poor bands", which is not
something to leave on and hope for. It now uses the improved-tangent
formulation of Henkelman and Jónsson (2000). The copper barrier is unchanged at
0.754 eV, which is the reassuring outcome.

## v0.1.12

### The tutorials are executed notebooks

Four more tutorials — screening, chemical space, beyond one number, models and
campaigns — were prose with unexecuted code fences. They are now notebooks that
run when the documentation is built, which means the code in them is **tested**
rather than merely written. A tutorial whose examples only ever lived in a
markdown fence rots silently; this one fails the build.

`_scripts/build_notebooks.py` builds all five and exits non-zero if any cell
raised. Between them the tutorials now carry **27 figures**, up from 11 — a
number in a table and the same number in a plot are not interchangeable, and
several of the corrections below were only obvious once the result was drawn. `scale_and_dft` stays prose deliberately: it is about corpora larger
than memory and jobs handed to VASP, and a notebook of it would be a page of
code nobody could execute.

Executing them found five things the prose had asserted and the code did not do.

**Forces on a perfect cubic cell are zero.** The per-atom section computed
forces on unrelaxed fcc, L1₂ and B2 cells and presented the result as ragged
per-atom data. Every value was exactly zero — by symmetry, not by accident. The
notebook now rattles the structures first, which is both what makes the example
work and what force-training sets for machine-learned potentials actually are.

**`harmonize` had nothing to fit on.** The database-reconciliation example split
one library into two halves by alternating rows, so the two "databases" shared
no composition and the fit was silently skipped. It now duplicates every
material into both databases, and the fitted offsets recover the ones put in to
within 4e-17.

**`compare_grids` reports a count, not a fraction.** The text called the
`overlap` column a fraction of the grid. It is the number of grid points both
curves cover.

**The leakage diagnostic did not say what the tutorial claimed.** The prose
asserted that holding out a structure type roughly doubles the error against a
random split. On a 28-material library it does the opposite, and the reason is
in the table: the `prototype` spread is exactly zero across three seeds, because
the library has two structure types and only one way to partition it. The
notebook now reads the table honestly — "too small and too homogeneous to run
this diagnostic" — rather than quoting numbers from a library that no longer
exists.

**EMT cannot find a stable alloy.** Its energy zero is the pure element at
equilibrium, so every intermetallic is above the hull by construction. The
screening notebook now says so, along with the reminder that L1₂ Al₃Ni being
unstable says nothing about Al₃Ni, which is a real phase in a different
structure.

### `mv.set_level`

Now exported. Every operation that computes at a level records one, but a user
who computes a column themselves had no supported way to type it — which made
the level system a thing the library did to you rather than a thing you could
use. `mv.compare_levels` works on the result like any other level.

### `mv.datasets.load` no longer needs pytest

It located pymatgen's bundled structures by reading
`pymatgen.util.testing.STRUCTURES_DIR`. That module imports pytest at module
scope, so `mv.datasets.load('battery_cathodes')` — the first call in the
getting-started tutorial — raised `ModuleNotFoundError: No module named
'pytest'` on any installation without the test tooling, which is most of them.
Every local environment here happened to have pytest, so nothing caught it.

The new docs CI job did, on its first run, which is the argument for the job.
The path is now found from the pymatgen package directory; the constant was
only a `Path` join, and there was never anything to gain by importing a test
helper to get it. There is a test that blocks the pytest import and pins this.

### Fixes surfaced by executing the tutorials

- `mv.data.from_structures` and `mv.multi.sites` named their rows *after*
  handing the frame to AnnData, so every dataset construction emitted an
  `ImplicitModificationWarning` about coercing an integer index — and that
  warning landed in the output of the first cell of every notebook.
- `mv.opt.suggest` raised a bare `KeyError` naming the column when
  `predicted=` or `uncertainty=` named one that did not exist. It now says what
  the column was for and lists the available ones.
- The documentation pointed at `github.com/matverse/matverse`, which does not
  exist. Every "edit this page" and "download" button was dead.

## v0.1.11

### `mv.pl.set_style`

One call at the top of a notebook, and every later plot matches: resolution,
font size, spines, figure size. It touches only `rcParams`, so anything set
afterwards still wins and a figure built by hand is unaffected.

This is the counterpart of `ov.plot_set` in omicverse, and it exists for the
same reason — styling nine plots one at a time is how a notebook ends up with
nine different-looking plots.

### The getting-started notebook, rewritten

The first version read as an essay with code in it. It now follows the shape
omicverse's notebooks use, which readers of that ecosystem already know:

- a title, then the scientific context, with the structures **cited** — the
  olivine cathode is [Padhi et al. (1997)](https://doi.org/10.1149/1.1837571),
  not an anonymous CIF;
- a first cell that is imports plus `mv.pl.set_style()`;
- an explicit *Loading a dataset* section naming where the data comes from,
  rather than a call whose provenance the reader has to take on trust;
- short cells, mostly one operation, ending with the object so it renders;
- brief markdown between cells saying what the next step is for.

The pipeline it runs is unchanged, and it is still executed rather than
illustrative.

## v0.1.10

### `mv.datasets` — real materials to work on

Every tutorial built its structures in code, which meant the documentation
taught the library's own test fixtures rather than the domain. This ships real
published structures instead: LiFePO₄ as it is actually reported in P2₁/c,
Li₁₀GeP₂S₁₂ as the 58-atom solid electrolyte people actually study.

- `load` — five curated sets, each a coherent scenario rather than a grab-bag,
  and each saying what it is *for*: `battery_cathodes`, `solid_electrolytes`,
  `oxides`, `semiconductors`, `simple`.
- `metals` — the seven fcc metals from published room-temperature lattice
  parameters. EMT is the only calculator that ships working and is parameterised
  for exactly these plus H, C, N and O, so they are the materials on which every
  example runs end to end without downloading a model.
- `fetch` — Materials Project or any OPTIMADE provider, cached to disk so the
  second call is free. Honours `MATVERSE_DATA`, which is the one to set on a
  cluster: a home directory is small, NFS-backed and shared, and a downloaded
  corpus does not belong there.

The bundled structures are **read from the set pymatgen already ships** rather
than re-distributed. pymatgen is a hard dependency so they are always present,
which keeps the package small and the provenance of the crystallographic data
clear.

### A notebook that actually runs

`tutorials/getting_started.ipynb` is generated by a script and **executed**, so
the outputs in the documentation are real rather than illustrative — myst-nb is
configured with `execution_mode = "off"`, and a notebook committed without
outputs renders as a page of code and no results.

It loads real cathodes, computes their diffraction patterns, feeds one back as a
measured pattern and identifies it, then switches to the metals for the
calculator path — because the cathodes contain Fe, P and V, which EMT is not
parameterised for. That constraint is in the notebook rather than designed
around, since it is the constraint a reader will hit.

## v0.1.9

Tier 2 of the gap analysis: the four capabilities the survey called important
rather than essential. **115 functions across 20 namespaces.**

### `mv.mag` — magnetic order, and why a hull needs it

Every energy computed before this assumed the magnetic configuration handed to
the calculator was the right one. For anything containing iron, cobalt, nickel,
manganese or chromium that is usually wrong, and the ferromagnetic and
antiferromagnetic states can differ by hundreds of meV per atom — so a hull
built from whichever ordering the input file carried is a hull of the wrong
quantity, systematically.

`orderings` enumerates the candidates, `ground_state` collapses them onto the
parent keeping the lowest, and `magnetic_spread` records how far apart they
were. That last number is the one people skip and the one that says whether the
guess would have mattered.

pymatgen's antiferromagnetic strategies call out to **enumlib**, Fortran
executables that are not pip-installable and absent from most environments —
and it raises rather than returning the ferromagnetic state it could have
produced. matverse falls back to generating sign assignments on the magnetic
sublattice directly, which is coarser and dependency-free, and records that it
did.

The winning energy is written under the ordinary `energy_per_atom_<level>` name,
so `mv.thermo.hull` needs no special case — it sees a normal column that happens
to be the magnetic ground state.

### `mv.prop.thermal_conductivity`

κ_L by the Slack model from the phonon and elastic data already computed, with
the Debye temperature and Grüneisen parameter it needs. Debye temperatures land
within about 15% of the literature on fcc metals — Au 161 against 165, Ag 239
against 225 — and the ordering is right.

An order-of-magnitude model, not a solution of the Boltzmann transport equation,
and it says so. Matbench Discovery weights thermal conductivity at 40% of its
combined score precisely because it is the property cheap methods get wrong, so
a screen ranked on this is a shortlist for phono3py rather than an answer.

### `mv.dft.read_dos`

The density of states onto a Fermi-referenced grid, plus the scalars a screen
filters on: gap, whether it is direct, band edges, and the density of states at
the Fermi level. Cheap, because `mv.dft` already parses the same files.

The level records that a PBE gap is roughly half the experimental one — which is
why `is_metal` from a PBE run is trustworthy while a small gap is not. PBE turns
narrow-gap semiconductors into metals, so the false negatives all point the same
way.

### `mv.thermo.defect_formation`

Defect enumeration existed; the thermodynamics did not. Formation energy depends
on where the Fermi level sits and on the chemical potential of whatever was
added or removed, so it is a **line rather than a number** — stored on the grid
axis against the Fermi level, with the lowest charge state at each point giving
the stable charge.

Two things it refuses to do quietly. Without a chemical potential it produces
NaN and warns, because what an added or removed atom costs is not derivable from
the defective cell alone. And no image-charge correction is applied — a charged
defect in a periodic cell interacts with its own images — so the result is
declared uncorrected and points at doped and pydefect for the real thing.

### Registry

115 entries, 354 contract claims, **contract-verified rate 141/141**.

## v0.1.8

The three capabilities a 2026 survey of the field named as must-haves and this
library did not have. **107 functions across 19 namespaces.**

### `mv.md` — molecular dynamics

Everything before this described a structure sitting still at zero kelvin.
Diffusion, ionic conductivity, thermal expansion, melting and the structure of a
glass are all properties of a system that is moving, which is why this is the
module the rest of the missing half hangs off.

- `run` — NVT or NPT at one temperature, depositing mean energy, mean
  temperature, MSD and diffusivity. **Trajectories are not stored**: observables
  are computed as the run goes, because a library that materialises a trajectory
  per candidate stops working at a few hundred materials.
- Per-element diffusivity goes to a **layer**, since it is materials × elements
  — the same shape as `X`. For an ionic conductor the number that matters is the
  mobile species' diffusivity; averaging lithium with its framework describes
  nothing.
- `sweep` — a temperature series stored on the **condition axis**, using the same
  grid mechanism as a diffraction pattern. Nothing new was needed to express
  "the same property at different conditions". Gives thermal expansion and an
  Arrhenius activation energy.
- `conductivity` — Nernst–Einstein, with its assumption stated.
- `melt_quench` — with the fixed-volume quench that survives the 2026 finding
  that all eight tested universal potentials produce catastrophically
  under-dense amorphous structures under a naive NPT quench.

**The thermostat now tells you when it failed.** A weakly coupled thermostat
takes tens of picoseconds to reach its target, so a short run samples a
temperature that is not the one requested — 69 K for a requested 300 K in one
test. The achieved temperature was always recorded; a run that misses by more
than 20% now warns, because every observable belongs to the temperature actually
reached.

### `mv.neb` — migration barriers

`hop_endpoints` builds the endpoints for a vacancy-mediated hop; `barrier` runs
climbing-image NEB and records the forward and reverse barriers, the reaction
energy, and the energy profile on the grid axis.

Validated against physics: vacancy migration in fcc copper comes out at
**0.762 eV against a literature ~0.70**, the forward and reverse barriers match
for a symmetric hop, and the profile is a clean symmetric arc.

Two things had to be right for that, and both were wrong first:

- **The destination must be the nearest periodic image of the vacancy**, not its
  stored coordinates. Those differ whenever the shortest hop crosses a cell
  boundary, and using the stored ones sent the atom the long way round — a
  2.55 Å hop became a 7.66 Å traverse, and the NEB measured the cost of dragging
  an atom through the lattice. `hop_endpoints` now verifies the displacement it
  built matches the hop it reported.
- **Interpolation defaults to IDPP, not linear.** A straight-line path drives the
  hopping atom through its neighbours; the band starts with overlapping atoms and
  converges, if at all, on several eV of repulsion.

Every surrogate level records the caveat that machine-learned potentials soften
the potential energy surface and under-predict barriers — roughly 0.07–0.08 eV
MAE without transition metals, about 0.20 eV with them. The error has a sign, so
a screen ranked on these promotes candidates DFT would reject.

### `mv.surf` — surfaces, shapes and adsorption

This used to sit outside a screening library because a surface energy meant a
DFT slab per facet. Universal potentials changed the arithmetic.

- `slabs` — every distinct low-index facet and termination, returned as a new
  dataset with `obs['parent']` back to the material.
- `surface_energy` — requires the bulk object rather than guessing it, because
  a wrong reference shifts every value by a constant: invisible in a ranking,
  fatal in a Wulff construction. Recovers the literature ordering for copper,
  (111) < (100) < (110).
- `wulff` — the equilibrium crystal shape, with per-facet area fractions on the
  facets and the shape summary on the material.
- `adsorption_sites` / `adsorption_energy` — enumerate atop, bridge and hollow
  rather than guessing one, relax all, keep the minimum. That is the AdsorbML
  protocol, and it reproduces hollow-binds-strongest for oxygen on Cu(111). The
  adsorbate reference is a required argument with no default, since an energy
  against half an H₂ and one against an isolated H atom differ by about 2.3 eV
  and both conventions are in use.

### Fixed

`mv.calc.relax` always wrote `relaxed_<level>`, the same name every time, so
relaxing two variants at one level silently overwrote the first. NEB needs two
relaxed endpoints, which is how it surfaced. `key_added` now names the output.

Miller indices were spelled two ways inside `mv.surf` — `'111'` on the facets and
`'1.1.1'` in the Wulff summary. Both are now `'1_1_1'`, with underscores because
a bare join makes `(1, -1, 1)` into `'1-11'`.

## v0.1.7

### Fixed

`mv.data.from_ase` put per-atom arrays — magnetic moments, charges, tags — into
`uns['sites']` as one record per material. That was wrong twice over, and the
two failures are the same two the substrate has taught before:

- **It did not subset.** `uns` is untouched by `md[mask]`, so filtering a dataset
  left every surviving row pointing at another material's atoms.
- **It did not save.** A list of dicts is not writable to `h5ad` at all, so any
  object built with `from_ase` failed on `write_h5ad`.

No test caught it because none combined `from_ase` with a save or a subset.
Three now do.

Per-atom arrays are now attached as **site properties on the structure itself**.
They travel with it, serialise with it, and appear as columns the moment someone
calls `mv.multi.sites` — no extra plumbing, and alignment is automatic rather
than maintained. The function's note about a site axis being "the open design
problem" was also stale: `mv.multi.sites` shipped in v0.1.2.

### Documented

The Developer guide gains the rule that decides where a result lives: **put it
in `uns` only if it is not aligned to an axis.** `uns` is a first-class part of
the model and holds much more than it gets credit for — scalars, arrays,
DataFrames and nested dicts of those all serialise — and matverse uses it
heavily and correctly for levels, grids, screens and provenance, none of which
are per-material. What it will not do is subset, so anything of length `n_obs`
belongs in `obs`, `obsm` or `obsp`.

## v0.1.6

The last concrete gaps the roadmap named. **96 functions across 16 namespaces.**

- **`mv.dft` is no longer VASP-only.** `code='espresso'` writes Quantum ESPRESSO
  input. Pseudopotentials are named, not shipped: which set a run used is part of
  the level of theory — SSSP and PSLibrary disagree for the same functional — so
  guessing a filename would put a silent choice into a result the object claims
  to record.
- **`mv.pp.defects`** enumerates vacancies and substitutions in a supercell,
  one per symmetry-inequivalent site. Without that deduplication a 32-atom
  elemental supercell yields 32 identical vacancies and wastes a calculator on
  31 of them. Defect *formation energies* need charge states, a chemical
  potential and a finite-size correction, which is what doped and pydefect exist
  for; this builds the structures.
- **`mv.thermo.pourbaix`** gives distance from aqueous stability at a pH and
  potential. A material on the solid-state hull can still dissolve, which is why
  it is a separate question rather than a column of the same one. Needs
  Materials Project's fitted ion energies and cannot be computed from a
  candidate set alone.
- **`mv.feat.embed`** takes a pretrained model's latent vectors, through the
  same registration interface as `mv.calc.register_calculator`. matverse ships
  no embedder — weights are hundreds of megabytes with their own licences — and
  the docstring notes the 2026 finding that plain CGCNN and ALIGNN generalise
  better out of distribution than fine-tuned foundation embeddings.

### Fixed

`mv.pp.defects` silently skipped its own symmetry deduplication.
`SpacegroupAnalyzer` was imported inside the calling function, so the helper
raised `NameError`, hit a bare `except`, and fell back to enumerating every
site. The test that caught it asserts the physics — one distinct vacancy in an
elemental fcc supercell — rather than a stored count.

## v0.1.5

Working past memory.

### Scale

Alexandria is 5.06M entries and OMat24 is roughly 110M calculations. A
constructor that takes a list rules both out before anything else does.

- **`mv.data.from_iterable`** builds a dataset from a stream in blocks,
  concatenating as it goes. The element axis is unioned across blocks, so a block
  containing an element no earlier block had widens the axis rather than failing.
- **`mv.data.from_ase_file`** reads extended-XYZ and ASE databases with ASE's own
  iterator, so a hundred-million-frame training corpus can be sampled without
  being read.
- **`mv.utils.map_chunks`** applies an expensive operation block by block and
  merges back the obs columns, feature blocks and structure variants each block
  produced. `skip_if=` skips blocks already finished, so a re-run after a killed
  job continues rather than restarts, and `checkpoint_to=` writes between blocks.
- **`mv.structures(md, rows=...)`** decodes a window rather than everything.
  Decoding is what costs at scale — five million serialised structures are a few
  gigabytes of strings and several times that as objects.

`map_chunks` deliberately does **not** merge `uns`. A per-block `uns` entry is a
statement about that block, and quietly keeping the last one would be wrong: a
screen's criteria and a hull's reference count both mean something different per
block than they do overall.

This is chunking, not laziness. The object is still materialised; a zarr-backed
`obs` with on-demand structure resolution is the next step.

## v0.1.4

First principles at the boundary, phase equilibria beyond the hull, and lattice
dynamics. **88 functions across 16 namespaces**, contract-verified rate 142/142.

### `mv.dft` — inputs out, results back in

matverse runs no DFT and submits no jobs. Workflow management has three good
answers already — atomate2 with jobflow-remote, quacc, AiiDA — and a fourth would
be a liability.

What is not solved is the boundary: a screen lives in one object, DFT lives in a
directory tree, and the correspondence is normally maintained by a naming
convention and someone's memory.

```python
mv.dft.write_inputs(md, 'runs/', preset='relax')   # one directory per row
# ... sbatch, atomate2, quacc, a week ...
mv.dft.read_outputs(md, 'runs/', level='pbe')      # back onto the same rows
```

Each directory carries a manifest recording which row it came from, so a
directory renamed by a workflow manager still resolves — the usual reason a
hand-rolled harvest attaches results to the wrong material. Presets (`relax`,
`static`, `bands`, `scan`, `hse`) record what each reproduces, so a run tagged
`scan` arrives as r2SCAN and `mv.thermo.hull` will not mix it with PBE.

Rows whose run is missing or unconverged get NaN and a reason rather than being
dropped: which candidates failed is a result, and a systematically failing corner
of composition space is worth seeing.

### `mv.prop.phonon` — the check a hull cannot make

Gamma-point frozen phonons on a supercell, giving the phonon density of states on
a shared frequency grid, the zero-point energy, and a count of imaginary modes.

A composition can sit on the convex hull and still be a structure that will not
hold together. Copper's stable phase is fcc; bcc copper has the same composition
and is dynamically unstable, and only a phonon calculation says so. Generated
structures fail this far more often than they fail the hull.

`mv.prop.free_energy` derives the harmonic vibrational free energy, entropy and
heat capacity from that DOS, which is where a hull built at 0 K starts to become
a hull at temperature.

Both are validated against physics rather than against a stored number: the
zero-point energy of copper (0.029 eV/atom against a literature 0.03) and the
Dulong–Petit limit (heat capacity converging to 3k_B per atom above the Debye
temperature). A regression test that compares only to last week's output cannot
tell a refactoring from a sign error.

### `mv.prop.elastic`

Elastic stiffness by finite strains with Voigt–Reuss–Hill moduli, Poisson ratio
and a Born stability flag.

### `mv.thermo` — beyond the hull distance

- `reaction` balances a reaction between compositions in the dataset and computes
  its energy. A reaction energy is not a synthesis route: it says a product is
  downhill, not that anything gets there.
- `chempot_limits` reports the chemical potential window over which each stable
  phase remains on the hull — the conditions it could be grown under — and says
  plainly when a closed hull makes that window a statement about the dataset
  rather than about chemistry.

### Registry

88 entries, 250 contract claims, **contract-verified rate 142/142**.

## v0.1.3

The namespaces the design named and the library did not have: plotting,
supervised models, design campaigns, and the infrastructure underneath them.
**80 functions across 15 namespaces**, all decorated, all probed.

### `mv.pl` — plotting

Every function draws onto an axis and returns it, and none calls `plt.show`; a
library that shows figures cannot be used to build one.

- **`periodic_table`** — the display for `rank_elements_groups`, in the way a dot
  plot is the display for differential expression. A bar chart of 118 categories
  is unreadable and throws away the structure a chemist reads a periodic table
  for.
- `rank_elements_groups`, `hull` (which labels itself when the hull is closed),
  `parity` (with error bars when an uncertainty is recorded, and a warning on the
  plot when the two levels reproduce different methods), `pareto`, `embedding`,
  `spectra`, `provenance`.

### `mv.model` — supervised prediction, honestly split

A prediction is a level of theory: it gets a record saying what it was trained
on and how it was split, so a predicted number and a DFT number cannot be
averaged together by accident.

**`mv.model.split` defaults to grouping by composition.** Random train/test
splits are the field's most common silent methodological failure — a materials
dataset is full of near-duplicates, so a random split puts relatives on both
sides and reports a number that will not survive a genuinely new material.
Strategies: `composition`, `prototype` (anonymised formula plus space group),
`element` (hold out everything containing one element), and `random`, which is
recorded as `leaky: True`.

**`mv.model.cross_validate`** scores under several strategies at once and reports
`leakage_mae` — the gap between the grouped number and the random one. On the
test library that gap is 1.7, which is the size of the leak a random split would
have hidden.

### `mv.opt` — design campaigns

Pool-based active learning, which is the right shape for materials: the search
space is a list of structures, not a box of real numbers.

`start` / `suggest` / `observe` / `history`, with greedy, uncertainty, UCB,
expected-improvement and random acquisition. Batches can be diversified by
farthest-point selection, because ten highest-scoring candidates are often ten
variations on one idea and computing all ten answers one question.

Methods needing an uncertainty **refuse** when none exists rather than pretending
sigma is zero.

### `mv.utils` — units, checkpoints, cluster

- `convert` / `set_units` / `check_units`. eV, meV, kJ/mol, kcal/mol, Rydberg,
  Hartree; angstrom, nm, pm, bohr. Conversions deposit beside the original
  rather than overwriting, and `check_units` fills in what matverse itself
  produced.
- `resume` reports which rows an operation has not filled, so a screen killed by
  a walltime limit continues rather than restarts. `checkpoint` writes and
  records.
- `slurm_script` writes a batch script rather than submitting it — submitting is
  a side effect on a shared machine.
- `summary` renders what an object contains, including warnings about a closed
  hull or a non-commercial level.

### `mv.prop.elastic`

Elastic stiffness by finite strains at any level, with Voigt–Reuss–Hill bulk,
shear and Young's moduli, Poisson ratio, and a Born stability flag. On relaxed
EMT metals it recovers the right ordering (Ni > Cu > Al).

### `mv.data` — OPTIMADE

`from_optimade` queries any OPTIMADE-compliant provider with one filter
expression; eight base URLs ship, and any endpoint works via `base_url=`. One
protocol against roughly twenty providers beats twenty bespoke clients, which is
why this is now the primary connector.

`from_optimade_response` parses an already-fetched payload — separate on purpose,
because parsing is deterministic and testable while fetching is neither.

### Registry: derived names versus claimed ones

Adding `mv.pl` surfaced a real distinction the registry had collapsed.
`mv.pl.hull` mirroring `mv.thermo.hull` is a convention worth keeping, the way
scanpy pairs `pl` with `tl` — but both derive the bare name `hull`.

The rule is now:

- an **explicit alias** is a claim, and two functions claiming one still raises;
- a **derived name** is not, so an explicit alias outranks it, and two derived
  names that collide **withdraw** the key from exact lookup rather than awarding
  it to whichever module imported first.

`mv.describe('rank_elements_groups')` now answers "names more than one function;
say which", which is the honest response to an ambiguous question.

### Fixed

- A list of dicts in `uns` cannot be written to `h5ad` — anndata turns it into an
  object array and h5py refuses it. `uns['checkpoints']` and the campaign's
  rounds now use an ordered, writable record store. This is the same class of
  failure as structures in `uns` in v0.1.1, found the same way: by trying to
  save.
- `mv.model.cross_validate` swallowed fit failures and returned an empty table,
  which reads as "the model scored nothing" rather than "nothing was fitted". It
  now re-raises the underlying error.
- `mv.prop.compare_grids` gained an overlap count alongside cosine and RMSE.

### Registry

**80 entries, 223 contract claims, contract-verified rate 128/128.** Four claims
were adjusted rather than kept when probing showed them unresolvable or untrue,
including two on `mv.opt` whose slot names are chosen at `mv.opt.start` and so
have nothing in a later call's arguments to resolve against — recorded in the
function's notes as a third place the contract vocabulary runs out.

## v0.1.2

The second and third axes, measured data on the same footing as computed data,
and two things the design document had claimed but never built.

### Grid-shaped results, and one convention instead of two

A curve — a diffraction pattern, a density of states, a phonon spectrum — is
`materials × grid`. It goes into `obsm` as `'<quantity>_<level>'`, with the
shared axis recorded once in `uns['grids'][quantity]`.

That resolves an open problem rather than adding a feature. The design had
array-shaped results using a `layer` for the level of theory and scalars using a
name suffix, and listed the split as a wart forced by `obs` having no layers.
Putting grids in `obsm` means the suffix works everywhere: `obs['energy_pbe']`
and `obsm['xrd_pbe']` read the same way, and a measured pattern is
`obsm['xrd_experiment']` rather than a different kind of thing.

It also removed a dependency. Grids no longer need a modality, so they no longer
need MuData.

- `mv.prop.xrd` — powder patterns, broadened onto a common grid
- `mv.prop.rdf` — radial distribution functions, a structural fingerprint that
  separates polymorphs without needing dscribe
- `mv.prop.compare_grids` — cosine and RMSE between two levels, over the points
  where both are defined, with the overlap recorded

### The sites axis

Per-atom results — forces, charges, moments — are ragged: the number of atoms
differs per material. v0.1 stored them as records in `uns` and called it the open
design problem, which was accurate.

`mv.multi.sites(md)` returns a second `AnnData` whose rows are atoms, with
`obs['material']` as the foreign key. Per-atom results become a matrix again.

- `mv.calc.forces` writes `sites.obsm['forces_{level}']`
- `mv.multi.aggregate` reduces a per-atom column back onto the material axis, so
  a per-site result becomes something a screen can filter on
- `mv.multi.to_mudata` assembles both into one MuData, and nothing requires it

`X` on the sites object is the one-hot element, so `var` is the same periodic
table the parent carries and `mv.tl.rank_elements_groups` runs unchanged on
atoms — "which elements carry the largest forces" needed no new function.

### Experiment is a level of theory

This turned out to need no new machinery, which is the argument for having typed
the level of theory in the first place.

- `mv.exp.measure` — a measured scalar becomes a level, so
  `mv.compare_levels(md, 'band_gap')` puts PBE, HSE06 and the spectrometer in one
  table without anyone deciding which is the band gap
- `mv.exp.attach` — a measured curve, resampled onto the computed grid
- `mv.exp.match_xrd` — rank every candidate against one measured pattern. It
  records that it scored against this object's candidates and nothing else, so a
  high score reads as "the best of what you gave it" rather than "identified"

### `mv.pp.harmonize`

Formation energies from Materials Project, OQMD and Alexandria carry systematic
offsets from differing pseudopotentials, cutoffs and correction schemes. That is
a batch effect with compositional structure, and `harmonize` fits it the way the
field already does by hand — as per-element reference offsets, by least squares
on the compositions two databases share.

It recovers an injected offset exactly and drives the cross-database residual to
zero on synthetic anchors. It cannot repair a disagreement that is not linear in
composition, reports the residual so the size of what is left is visible, and
warns rather than silently doing nothing when the databases share no composition.

### `mv.gen`

- `mv.gen.validate` — validity, uniqueness, novelty and stability using
  LeMat-GenBench's definitions rather than a variant, with every parameter it
  used recorded in `uns['gen_validate']['definitions']`. Stability is reported as
  **not assessed** rather than zero when no level is given or when the hull was
  closed, because a hull over a dataset's own compositions cannot say whether
  anything is stable.
- `mv.gen.substitute` — element substitution enumeration with an optional
  charge-balance filter. Substitution within a known structure type is what
  several generative models were found to be doing implicitly, so it is the
  baseline worth beating.

### Registry

52 entries, 157 contract claims, **contract-verified rate 101/101**.

A second limitation of the contract vocabulary surfaced, and is recorded rather
than worked around: `mv.calc.forces` writes into the `sites` object passed as its
second argument, and `produces` describes slots on one object only. Those writes
are documented in prose and go unprobed. Together with the route-conditional
`requires` on `mv.tl.cluster`, that is two places the vocabulary has run out.

### Fixed

- `mv.prop.compare_grids` let a single undefined point turn a whole comparison
  into NaN. A measurement covering a narrower range than the calculation is the
  normal case, so comparison now runs over the overlap and records how many
  points it used.

### Known caveat

Attach a measurement at its own resolution or better. Diffraction peaks are
narrow, and resampling onto a grid coarser than the peak width discards them
permanently — no later interpolation brings them back. There is a test pinning
this so nobody "fixes" it.

## v0.1.1

A screening pipeline that works end to end, and the substrate decisions that
make the rest of the roadmap possible.

### The composition matrix

`X` is now materials × elements and `var` is the periodic table, built at
construction rather than deposited by a featuriser — AnnData ties `X`'s width to
`var`, so it cannot be widened in place, and composition is intrinsic to a
material rather than derived from it.

What it unlocks:

- `mv.tl.pca`, `neighbors`, `cluster` on chemical space
- `mv.tl.rank_elements_groups` — which elements distinguish one group of
  materials from another, the question that follows every screen
- `mv.tl.novelty` — distance to the nearest known composition
- `mv.feat.element_stats` as a matrix product of `X` and `var`, rather than a
  featuriser that re-derives the periodic table

`build_X=False` restores the width-zero `X` of v0.1.0.

### Two claims that had never held

Both were fixed by moving structures from `uns` to `obsm`, and both now have
tests.

- **Subsetting silently misaligned structures.** `uns` does not subset with the
  object, so `md[mask]` kept every structure while dropping rows and each
  surviving row pointed at the wrong one — the exact failure this substrate
  exists to prevent.
- **The object could not be written to `h5ad`.** A list of pymatgen objects in
  `uns` is not something anndata can serialise, so the central interoperability
  claim failed on contact. Structures are now JSON strings in an `obsm` frame,
  which is aligned by construction and writes without special handling.

### Levels of theory

`uns['calc']` became `uns['levels']` and gained three fields that 2026 made
load-bearing:

- `reference` — what the level reproduces. A model trained on OMat24 targets
  PBE+U, one trained on MatPES targets r2SCAN; `surrogate: True` alone no longer
  distinguishes them.
- `license` — MACE-MP and MACE-MPA are MIT, MACE-OMAT and MACE-MATPES are ASL and
  forbid commercial use, UMA's licence excludes several countries.
  `mv.calc.check_licenses(md)` reads it back.
- `uncertainty` — where `_std` came from. `mv.calc.committee` produces one and
  says plainly that it is uncalibrated.

`mv.thermo.hull` now raises `LevelMismatch` rather than building a hull from two
levels whose references disagree. `mv.compare_levels` lines one quantity up
across every level that computed it.

### The hull can be absolute

v0.1.0 built the hull over the dataset's own compositions, which makes
`e_above_hull` a statement about which candidate is lowest rather than whether
any is stable. `references=` now accepts competing phases — a list of entries or
another matverse object — and `mv.thermo.references_from_mp` fetches them.
`uns['phase_diagram']['closed_system']` records which kind of number you have,
and a closed hull warns.

Also new: `formation_energy_{level}` when elemental references are present, and
`decomposes_to_{level}` — *what* a material decomposes into, not just how far
above the hull it sits.

### New namespaces

- **`mv.pp`** absorbs `mv.struct` and adds `qc`, `filter_materials`,
  `filter_elements`, `normalize_composition`, `dedup`, `rattle` and `strain`.
  `mv.struct` remains as re-exports.
- **`mv.tl`** — the borrowed analysis layer described above.

`mv.pp.dedup` blocks on `(reduced formula, space group)` before running
`StructureMatcher` inside each block; an all-pairs comparison is quadratic and
unusable past a few thousand candidates.

`mv.screen` gains `pareto` for multi-objective screens, and `filter` now
excludes NaN — a candidate whose calculation failed to converge has not met the
criterion, and silently admitting it is how a broken run reaches a shortlist.

### The registry and its probe

39 entries, 115 contract claims, all carrying a description and a runnable
example. The registry is vendored into matverse rather than imported, so the core
dependency list stays at six packages.

Claims are verified by execution: `produces` by running the call and looking,
`requires` by deleting the slot and confirming failure. The current
**contract-verified rate is 68/68**, and four claims were deleted rather than
repaired when they failed:

| deleted claim | why |
|---|---|
| `feat.element_stats requires var['Z']` | takes whatever numeric columns `var` has |
| `thermo.hull requires levels[{level}]` | only read when `references=` is given |
| `tl.cluster requires obsp['connectivities']` | true of the leiden route, not kmeans |
| `pp.strain produces structures[{name}]` | template was unresolvable; the default is now a real value |

The third is a finding rather than a defect: `requires` has one field per
function, not one per dispatch route, so a route-conditional dependency is
expressible only in prose a tool cannot check.

### Fixed

- `mv.screen.pareto` reduced domination over the wrong axis, marking points that
  dominated others as dominated themselves.
- `mv.pp.standardize` no longer aborts a whole dataset when spglib fails on one
  disordered cell.

### Compatibility

- Requires Python 3.10.
- `uns['calc']` → `uns['levels']`; objects written by v0.1.0 need the key renamed.
- `uns['structures']` → `obsm['structures']`, serialised as JSON.
- `mv.feat.composition` is now `mv.feat.element_stats`, with the old name kept as
  an alias. It deposits `obsm['X_element_stats']`, not `obsm['X_composition']`.

## v0.1.0

The initial skeleton: six namespaces, 17 functions, and the two conventions that
survived — operations deposit rather than return, and a result carries its level
of theory in the slot name.

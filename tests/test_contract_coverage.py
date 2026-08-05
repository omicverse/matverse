"""Every contract claim in the registry, probed — and a check that none escapes.

``test_contracts.py`` verifies the claims it probes. This file answers the
question that one cannot: *how much of the registry is probed at all*. Before
it existed, 165 of 491 claims were — the rest were asserted, and an asserted
claim is what the audit metric already credits. A library that reports a
contract-verified rate over a third of its claims is reporting a number about
its test suite rather than about its registry.

``test_no_claim_goes_unprobed`` is the part that has to keep working. Six
namespaces were added in one week and the battery did not follow; the check is
there so the next one cannot repeat it silently.
"""

from __future__ import annotations

import warnings

import pytest

import matverse as mv
from matverse._probe import ProbeReport, probe_call

from _contract_cases import (NAMES_PROBED_IN_TEST_CONTRACTS, UNPROBEABLE,
                             cases)


@pytest.fixture(scope="module")
def probe_report(tmp_path_factory) -> ProbeReport:
    """Run every case once; the tests below read the same report."""
    tmp = tmp_path_factory.mktemp("contract_probes")
    report = ProbeReport()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for func, factory, args, kwargs in cases(tmp):
            kwargs = dict(kwargs)
            returns = kwargs.pop("returns", "self")
            report.results += probe_call(func, factory, *args,
                                         returns=returns, **kwargs)
    return report


@pytest.fixture(scope="module")
def probed_names(probe_report) -> set[str]:
    return ({r.function for r in probe_report.results}
            | set(NAMES_PROBED_IN_TEST_CONTRACTS))


class TestClaims:
    def test_every_claim_probed_is_verified(self, probe_report):
        assert probe_report.results, "no claims were probed"
        assert not probe_report.failed, "\n" + probe_report.summary()

    def test_the_rate_is_reported(self, probe_report):
        print("\n" + probe_report.summary())
        assert probe_report.rate == 1.0

    def test_a_missing_backend_is_undecided_rather_than_failed(
            self, probe_report):
        """BoltzTraP2 does not build here, so mv.elec.transport cannot be
        decided either way. Counting it as a failure would report the
        environment as a defect in the registry."""
        for result in probe_report.untestable:
            assert "not installed" in result.detail


class TestCoverage:
    def test_no_claim_goes_unprobed(self, probed_names):
        """Every entry that makes a contract claim is probed by execution.

        A new function may not ship a claim that nothing checks. If a claim
        genuinely cannot be decided offline, name it in UNPROBEABLE with the
        reason — that is a smaller and more honest thing to write down than a
        rate computed over whatever happened to be easy.
        """
        unprobed = []
        for entry in mv.registry.entries():
            name = entry["public_name"]
            if not (entry["requires"] or entry["produces"]):
                continue                        # nothing to verify
            if name in probed_names or name in UNPROBEABLE:
                continue
            unprobed.append(name)
        assert not unprobed, (
            "these entries make contract claims that no probe checks:\n  "
            + "\n  ".join(sorted(unprobed))
            + "\nAdd a case to tests/_contract_cases.py, or name it in "
              "UNPROBEABLE with the reason it cannot be decided offline.")

    def test_the_unprobeable_list_is_not_a_dumping_ground(self, probed_names):
        """Anything listed as unprobeable must still exist, and must not in
        fact be probed — a stale exemption hides a claim nobody checks."""
        for name, reason in UNPROBEABLE.items():
            assert mv.registry.get(name) is not None, (
                f"{name} is exempted but no longer registered")
            assert reason.strip(), f"{name} is exempted without a reason"
            assert name not in probed_names, (
                f"{name} is exempted but is probed after all; remove it")

    def test_most_of_the_registry_is_covered(self, probed_names):
        claimed = [e for e in mv.registry.entries()
                   if e["requires"] or e["produces"]]
        covered = [e for e in claimed if e["public_name"] in probed_names]
        print(f"\nentries making claims: {len(claimed)}; "
              f"probed: {len(covered)}; "
              f"exempt: {len(UNPROBEABLE)}")
        assert len(covered) / len(claimed) > 0.85


class TestQualifiedContainers:
    """The one place the omicverse contract vocabulary needed extending."""

    def test_a_two_object_call_says_which_object_it_writes_to(self):
        entry = mv.registry.get("mv.env.coordination")
        assert "sites.obs" in entry["produces"], (
            "coordination deposits on the sites object, and the claim has to "
            "say so or it points an agent at the wrong one")

    def test_one_call_can_deposit_on_both_of_its_objects(self):
        """mv.mag.ground_state writes four columns to the parent and one back
        onto the orderings. An unqualified 'obs' said they arrived together."""
        entry = mv.registry.get("mv.mag.ground_state")
        assert set(entry["produces"]) == {"md.obs", "orderings_.obs"}
        assert entry["produces"]["orderings_.obs"] == ["is_ground_state_{level}"]

    def test_a_qualifier_must_name_a_real_parameter(self):
        """An unresolvable qualifier could never be probed, so it is refused at
        import rather than becoming a claim nothing can check."""
        from matverse._registry import register_function

        with pytest.raises(ValueError, match="not a parameter"):
            @register_function(
                aliases=["a nonexistent qualifier"], category="test",
                description="Deposits somewhere that does not exist.",
                produces={"nowhere.obs": ["x"]},
                examples=["never called"],
            )
            def _bad(md, sites):
                ...

    def test_rendering_shows_the_object(self):
        text = mv.describe("mv.env.summarise")
        assert "md.obs['mean_coordination']" in text
        assert "sites.obs['coordination_number']" in text


#: Package names as they appear in an UNPROBEABLE reason, mapped to the module
#: that has to be importable for that reason to still be true.
_REASON_PACKAGES = {
    "dscribe": "dscribe", "openbabel": "openbabel", "matminer": "matminer",
    "pydefect": "pydefect", "smol": "smol", "smact": "smact",
    "pyxtal": "pyxtal", "hiphive": "hiphive", "phono3py": "phono3py",
    "gpaw": "gpaw", "pycalphad": "pycalphad", "ifermi": "ifermi",
    "mudata": "mudata", "mp-api": "mp_api", "matgl": "matgl",
}


def test_an_exclusion_reason_naming_an_absent_package_stays_true():
    """An entry excluded from probing is unexamined by construction, so the
    reason had better keep being true.

    mv.elec.transport was excluded because BoltzTraP2 was 'absent in this
    environment'. Installing IFermi pulled BoltzTraP2 in, the reason stopped
    being true, and nothing noticed — which is how a tutorial calling that
    same function with the bands-axis object where a list of BandStructures
    belongs survived. The ImportError was raised first and the cell's except
    swallowed it.

    This fails when a reason claims a package is missing and it imports. The
    fix is to reword the reason or to start probing the entry, and which one
    depends on whether the thing actually runs — an installed package that
    raises on use is still a valid exclusion, but for a different reason than
    the one recorded.
    """
    import importlib.util

    stale = {}
    for name, reason in UNPROBEABLE.items():
        lowered = reason.lower()
        for token, module in _REASON_PACKAGES.items():
            if token not in lowered:
                continue
            if not any(word in lowered for word in
                       ("not installed", "is absent", "absent in", "not present",
                        "neither is present")):
                continue
            if importlib.util.find_spec(module) is not None:
                stale[name] = f"{token} imports here, but the reason says it does not"
    assert not stale, (
        f"these exclusion reasons have gone stale: {stale}. Reword them, or "
        f"probe the entry now that its backend is available")

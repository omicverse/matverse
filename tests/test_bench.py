"""matverse-bench: does it run, and does it discriminate?

Two things need proving about a benchmark, and only one of them is usually
checked. The first is that it is passable — otherwise every arm scores zero
because a criterion was written wrong and the number looks like a hard
benchmark. The second is that it **fails** when it should, which is the one that
matters: a grader that passes everything measures nothing.

So most of this file feeds the grader wrong answers on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matverse as mv  # noqa: E402
from bench import fixtures, grader, run as runner, tasks  # noqa: E402


class TestTaskSpecifications:
    def test_every_task_names_a_real_fixture(self):
        for task in tasks.TASKS:
            assert task.fixture in fixtures.BUILDERS, task.id

    def test_every_task_has_checks_and_a_layer(self):
        for task in tasks.TASKS:
            assert task.checks, task.id
            assert task.layer in ("single", "compose", "pipeline"), task.id

    def test_no_prompt_names_an_api_path(self):
        """Goal, not API. A prompt that says 'call mv.thermo.hull' measures
        whether a model can read, not whether it can work."""
        for task in tasks.TASKS:
            assert "mv." not in task.prompt, task.id
            assert "()" not in task.prompt, task.id

    def test_task_ids_are_unique(self):
        ids = [t.id for t in tasks.TASKS]
        assert len(set(ids)) == len(ids)

    def test_all_three_layers_are_populated(self):
        for layer in ("single", "compose", "pipeline"):
            assert tasks.by_layer(layer), layer

    def test_every_task_has_a_reference_solution(self):
        from bench import reference
        for task in tasks.TASKS:
            assert task.id in reference.SOLUTIONS, task.id


class TestGraderIsPassable:
    def test_the_reference_solutions_pass_everything(self):
        results = runner.run()
        scores = grader.grade(results)
        assert scores["n_passed"] == scores["n_tasks"], \
            "\n" + grader.report(results)

    def test_every_layer_is_reported(self):
        results = runner.run()
        scores = grader.grade(results)
        assert set(scores["by_layer"]) == {"single", "compose", "pipeline"}


class TestGraderDiscriminates:
    """The half that matters: it must fail when it should."""

    def test_doing_nothing_fails_everything(self):
        results = runner.run(solver=lambda task, md, wd: md)
        scores = grader.grade(results)
        assert scores["n_passed"] == 0
        assert scores["accuracy"] == 0.0

    def test_returning_nothing_is_an_error_not_a_pass(self):
        results = runner.run(selection=[tasks.by_id("stability")],
                             solver=lambda task, md, wd: None)
        assert not results[0].passed
        assert "no object" in results[0].error

    def test_a_crash_is_recorded_not_swallowed(self):
        def explode(task, md, wd):
            raise RuntimeError("simulated agent failure")

        results = runner.run(selection=[tasks.by_id("stability")],
                             solver=explode)
        assert not results[0].passed
        assert "simulated agent failure" in results[0].error

    def test_a_flag_of_the_wrong_shape_fails(self):
        """find-broken asks for exactly six valid rows out of seven."""
        def mark_everything_valid(task, md, wd):
            md.obs["is_valid"] = np.ones(md.n_obs, dtype=bool)
            return md

        results = runner.run(selection=[tasks.by_id("find-broken")],
                             solver=mark_everything_valid)
        assert not results[0].passed
        assert "7 true" in results[0].checks[0].detail

    def test_a_sign_error_in_the_hull_fails(self):
        """The check a presence test cannot make."""
        def negate(task, md, wd):
            from bench import reference
            md = reference.solve(task.id, md, wd)
            md.obs["e_above_hull_emt"] = \
                -md.obs["e_above_hull_emt"].to_numpy(dtype=float) - 0.1
            return md

        results = runner.run(selection=[tasks.by_id("stability")],
                             solver=negate)
        assert not results[0].passed
        assert "outside" in results[0].summary()

    def test_a_constant_offset_fails_the_reconciliation(self):
        """Both a real reconciliation and a constant shift produce a column.
        Only one makes the databases agree."""
        def constant_shift(task, md, wd):
            values = md.obs["energy_per_atom_dft"].to_numpy(dtype=float)
            batch = md.obs["database"].astype(str).to_numpy()
            offset = float(np.mean(values[batch == "oqmd"])
                           - np.mean(values[batch == "mp"]))
            md.obs["energy_corrected"] = np.where(batch == "oqmd",
                                                  values - offset, values)
            return md

        results = runner.run(selection=[tasks.by_id("reconcile-databases")],
                             solver=constant_shift)
        assert not results[0].passed
        # The column check passes; the agreement check is what catches it.
        assert results[0].checks[0].passed
        assert not results[0].checks[1].passed
        assert "disagreement" in results[0].checks[1].detail

    def test_a_partly_computed_column_fails(self):
        def half_finished(task, md, wd):
            from bench import reference
            md = reference.solve(task.id, md, wd)
            values = md.obs["energy_per_atom_emt"].to_numpy(dtype=float)
            values[:2] = np.nan
            md.obs["energy_per_atom_emt"] = values
            return md

        results = runner.run(selection=[tasks.by_id("relax-and-rank")],
                             solver=half_finished)
        assert not results[0].passed
        assert "finite" in results[0].summary()

    def test_not_saving_the_file_fails(self):
        def forget_to_save(task, md, wd):
            mv.pp.standardize(md)
            mv.pp.describe(md)
            mv.calc.energy(md, level="emt")
            mv.prop.xrd(md, two_theta=(10, 80), step=0.05)
            return md                                  # no write_h5ad

        results = runner.run(selection=[tasks.by_id("full-report")],
                             solver=forget_to_save)
        assert not results[0].passed
        assert "not written" in results[0].summary()

    def test_a_task_needs_every_check(self):
        """No partial credit: half a screen is not half a result."""
        def only_the_hull(task, md, wd):
            mv.pp.qc(md)
            md = mv.pp.filter_materials(md)
            mv.calc.relax(md, level="emt", fmax=0.05)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mv.thermo.hull(md, level="emt", source="relaxed_emt")
            return md                                  # never screened

        results = runner.run(selection=[tasks.by_id("screen-for-stability")],
                             solver=only_the_hull)
        assert not results[0].passed
        assert sum(c.passed for c in results[0].checks) >= 2


class TestGraderAcceptsDifferentSpellings:
    def test_a_differently_named_column_still_passes(self):
        """The name is not part of the specification unless the task said so."""
        def unusual_names(task, md, wd):
            from matverse._core import structures
            S = structures(md)
            md.obs["chemical_formula"] = [s.composition.reduced_formula
                                          for s in S]
            md.obs["natoms"] = [len(s) for s in S]
            md.obs["cell_volume"] = [float(s.volume) for s in S]
            md.obs["mass_density"] = [float(s.density) for s in S]
            return md

        results = runner.run(selection=[tasks.by_id("describe-library")],
                             solver=unusual_names)
        assert results[0].passed, results[0].summary()

    def test_a_hand_rolled_answer_scores_the_same(self):
        """The trajectory is not graded. An agent that computes a duplicate
        flag itself passes exactly as one that calls mv.pp.dedup."""
        def by_hand(task, md, wd):
            from matverse._core import structures
            seen, flags = {}, []
            for i, s in enumerate(structures(md)):
                key = (s.composition.reduced_formula, round(s.volume, 3))
                flags.append(key in seen)
                seen.setdefault(key, i)
            md.obs["duplicate"] = flags
            return md

        results = runner.run(selection=[tasks.by_id("find-duplicates")],
                             solver=by_hand)
        assert results[0].passed, results[0].summary()


class TestGraderPurity:
    def test_the_grader_makes_no_model_calls(self):
        """The property that makes a pass arguable from the code.

        Checked by reading the source, which is crude and is the only check
        that cannot itself be fooled by a mock.
        """
        source = (Path(__file__).resolve().parent.parent
                  / "bench" / "grader.py").read_text()
        for forbidden in ("openai", "anthropic", "requests", "urllib",
                          "httpx", "llm", "gpt", "claude", "completion"):
            assert forbidden not in source.lower(), \
                f"grader.py mentions {forbidden!r}"

    def test_the_report_names_the_failures(self):
        results = runner.run(solver=lambda task, md, wd: md)
        text = grader.report(results)
        assert "0/12" in text or "0.0%" in text
        assert "FAIL" in text

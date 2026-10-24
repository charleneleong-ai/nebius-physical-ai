import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from nebius_physical_ai.autopilot import (
    ACTIONS,
    MENU,
    RECOVERY_FAILURES,
    REQUESTS,
    Action,
    Candidate,
    CycleLog,
    Planner,
    app,
    axis_of,
    diagnosis,
    judge,
    learning_request,
    next_step,
    read_jsonl,
    results_from_log,
)
from nebius_physical_ai.benchmark import Cell
from nebius_physical_ai.task import TaskSpec
from nebius_physical_ai.verdicts import closed_families

N = 96


def cell(suite: str, group: str, wins: int, failures: dict[str, int] | None = None, flips: int = 0) -> Cell:
    """A suite scored on N paired scenes: the first `wins` succeed, then `flips` more scenes than the reference."""
    outcomes = [i < wins + flips for i in range(N)]
    return Cell(suite, group, wins + flips, N, 12, 1.0, outcomes=outcomes, failures=failures or {})


@pytest.fixture
def ref() -> dict[str, Cell]:
    return {
        "held_out_objects": cell("held_out_objects", "held_out", 57, {"dropped": 11, "pushed": 8, "no_contact": 14}),
        "clean_many": cell("clean_many", "diagnostic", 88, {"dropped": 4}),
        "real_1_pomade_tin_many": cell("real_1_pomade_tin_many", "diagnostic", 87, {"pushed": 6}),
        "held_out_task_basket": cell("held_out_task_basket", "held_out_tasks", 6, {"no_contact": 81}),
        "held_out_task_ring": cell("held_out_task_ring", "held_out_tasks", 0, {"no_contact": 96}),
    }


class FixedChooser:
    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer

    def choose(self, diag: dict, eligible: list[Candidate], closed: list[str], requests: list) -> tuple[str, str]:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer, "because"


@pytest.mark.parametrize(
    ("suite", "group", "axis"),
    [("held_out_task_ring", "held_out_tasks", "tasks"), ("held_out_objects", "held_out", "objects"),
     ("real_1_pomade_tin_many", "diagnostic", "environments"), ("clean_top_down_many", "diagnostic", "environments"),
     ("seen_vocabulary", "held_out", None), ("clean_many", "diagnostic", None)],
)  # fmt: skip
def test_axis_of(suite: str, group: str, axis: str | None) -> None:
    assert axis_of(cell(suite, group, 50)) == axis


def test_closed_families_needs_every_row_discarded(tmp_path: Path) -> None:
    rows = [("v7", "DISCARD"), ("synthesis", "DISCARD"), ("synthesis", "DISCARD"), ("dagger", "DISCARD"),
            ("dagger", "PROMOTE")]  # fmt: skip
    path = tmp_path / "verdicts.yaml"
    path.write_text(json.dumps({"experiments": [{"family": f, "verdict": v, "result": "x"} for f, v in rows]}))
    assert set(closed_families(path)) == {"v7", "synthesis"}


class TestPlanner:
    def test_ranks_by_reach_per_hour_and_explains_exclusions(self, ref: dict[str, Cell]) -> None:
        closed = Action("shelved_recipe", "shelved", frozenset({"objects"}), RECOVERY_FAILURES, "true", 1.0)
        ranked = Planner(
            {"shelved": "437"}, {"task_round:ring"}, budget_hours=5.5, chooser=None, menu=(*MENU, closed)
        ).candidates(ref)
        by_name = {c.action.name: c for c in ranked}
        assert ranked[0].action.name == "task_round:basket" and by_name["task_round:basket"].reach == 177
        assert "closed" in by_name["shelved_recipe"].excluded
        assert "already judged" in by_name["task_round:ring"].excluded
        assert "budget" in by_name["dagger_round"].excluded
        assert [c.excluded is None for c in ranked] == sorted([c.excluded is None for c in ranked], reverse=True)

    @pytest.mark.parametrize(
        ("answer", "picked"),
        [("dagger_round", "dagger_round"), ("real_cotrain", "task_round:basket"),
         ("rm -rf /", "task_round:basket"), (httpx.ConnectError("down"), "task_round:basket")],
    )  # fmt: skip
    def test_chooser_cannot_leave_the_eligible_set(
        self, ref: dict[str, Cell], answer: str | Exception, picked: str
    ) -> None:
        _, pick, _, by = Planner({}, set(), 8.0, FixedChooser(answer)).choose(ref, diagnosis(ref))
        assert pick.action.name == picked
        # only an answer that was actually used may be attributed to Nemotron on the published page
        assert by == ("nemotron" if picked == answer else "ranking")

    def test_nothing_eligible(self, ref: dict[str, Cell]) -> None:
        families = {a.family: "closed" for a in ACTIONS.values()}
        _, pick, why, _ = Planner(families, set(), 8.0, None).choose(ref, diagnosis(ref))
        assert pick is None and "nothing" in why


class TestJudge:
    def test_transfer_counts_but_the_trained_task_does_not(self, ref: dict[str, Cell]) -> None:
        new = dict(ref) | {"held_out_task_basket": cell("held_out_task_basket", "held_out_tasks", 6, flips=60)}
        assert judge(ACTIONS["task_round:basket"], ref, new).verdict == "KEEP"
        new["held_out_task_ring"] = cell("held_out_task_ring", "held_out_tasks", 0, flips=20)
        result = judge(ACTIONS["task_round:basket"], ref, new)
        assert result.verdict == "PROMOTE" and "held_out_task_ring" in result.why

    def test_any_suite_worse_discards(self, ref: dict[str, Cell]) -> None:
        new = dict(ref) | {
            "held_out_task_ring": cell("held_out_task_ring", "held_out_tasks", 0, flips=30),
            "clean_many": cell("clean_many", "diagnostic", 70),
        }
        result = judge(ACTIONS["task_round:basket"], ref, new)
        assert result.verdict == "DISCARD" and "clean_many" in result.why


class TestCycleLog:
    def test_folds_events_and_tracks_tried_and_in_flight(self, tmp_path: Path) -> None:
        log = CycleLog(tmp_path / "cycles.jsonl")
        log.append("planned", 1, action="task_round:basket", refs=["r3.json"])
        log.append("launched", 1, log="x.log")
        assert log.cycles().in_flight() == [1] and log.cycles().tried(["r3.json"]) == set()
        log.append("judged", 1, verdict="KEEP", why="within noise")
        with log.path.open("a") as f:
            f.write('{"event": "launch')  # an append cut off mid-write
        cycles = log.cycles()
        assert (cycles[1]["event"], cycles[1]["action"], cycles[1]["verdict"]) == (
            "judged",
            "task_round:basket",
            "KEEP",
        )
        assert cycles.in_flight() == [] and cycles.tried(["r3.json"]) == {"task_round:basket"}
        assert cycles.tried(["r4.json"]) == set()  # a new reference reopens every action

    def test_only_a_planned_cycle_can_be_dropped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        CycleLog().append("planned", 1, action="task_round:ring", refs=["r3.json"])
        CycleLog().append("planned", 2, action="dagger_round", refs=["r3.json"])
        CycleLog().append("launched", 2, log="x.log")
        assert CliRunner().invoke(app, ["drop", "2", "--why", "x"]).exit_code != 0
        assert CliRunner().invoke(app, ["drop", "1", "--why", "cycle 1 showed no transfer"]).exit_code == 0
        assert CycleLog().cycles()[1]["event"] == "dropped"


def test_menu_rejects_unknown_failure_modes() -> None:
    with pytest.raises(ValueError, match="no_cntact"):
        Action("typo", "x", frozenset({"objects"}), frozenset({"no_cntact"}), "true", 1.0)


class TestPlanCommand:
    @pytest.fixture
    def bench(self, tmp_path: Path, ref: dict[str, Cell], monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        (tmp_path / "verdicts.yaml").write_text(
            json.dumps({"experiments": [{"family": "synthesis", "verdict": "DISCARD", "result": "437"}]})
        )
        path = tmp_path / "bench.json"
        path.write_text(json.dumps({"label": "r3", "cells": [vars(c) for c in ref.values()]}))
        return path

    @pytest.mark.parametrize(
        ("args", "action", "chosen_by"),
        [
            ([], "task_round:basket", "ranking"),
            (["--action", "dagger_round", "--why", "test"], "dagger_round", "human"),
        ],
    )
    def test_logs_who_chose(self, bench: Path, args: list[str], action: str, chosen_by: str) -> None:
        result = CliRunner().invoke(app, ["plan", "--bench", str(bench), "--verdicts", "verdicts.yaml", *args])
        assert result.exit_code == 0, result.output
        planned = CycleLog().cycles()[1]
        assert (planned["action"], planned["chosen_by"]) == (action, chosen_by)

    def test_human_cannot_pick_a_closed_family(self, bench: Path) -> None:
        args = ["plan", "--bench", str(bench), "--verdicts", "verdicts.yaml", "--action", "failure_synthesis"]
        result = CliRunner().invoke(app, args)
        assert result.exit_code != 0 and not CycleLog().cycles()


def record(verdict: str, perception: float = 0.89, successes: int = 0, n: int = 0) -> dict:
    """A Preflight result.json, as npai preflight-job writes it."""
    rehearsal = {"successes": successes, "n": n} if n else {}
    return {"verdict": verdict, "scene": {"perception": perception}, "rehearsal": rehearsal}


class TestLearningRequests:
    @pytest.mark.parametrize(
        ("result", "task", "axes", "unmet"),
        [(record("GO", successes=31, n=32), TaskSpec(), None, None),
         (record("RETRY", successes=8, n=32), TaskSpec("yellow", "red"), ["objects"], []),  # gallery new_task
         (record("ASK_HUMAN", perception=0.24), TaskSpec("yellow", "red"), ["objects"], ["perception"]),  # real_photo
         (record("RETRY", successes=25, n=32), TaskSpec(), ["environments"], []),  # tight_gap
         ({"verdict": "ASK_HUMAN", "scene": None}, TaskSpec(), [], ["perception"]),  # mat not located
         # the gate passed without rehearsing: a pass, not a refusal, so nothing is filed
         (record("PERCEIVED", successes=31, n=32), TaskSpec(), None, None)],
    )  # fmt: skip
    def test_sorts_refusals_into_trainable_axes_and_the_rest(
        self, result: dict[str, Any], task: TaskSpec, axes: list[str] | None, unmet: list[str] | None
    ) -> None:
        request = learning_request("scene", result, task)
        assert (request.axes, request.unmet) == (axes, unmet) if axes is not None else request is None

    def test_a_recipe_answering_a_request_ranks_first(self, ref: dict[str, Cell]) -> None:
        tight_gap = learning_request("tight_gap", record("RETRY", successes=25, n=32), TaskSpec())
        ranked = Planner({}, set(), 8.0, None, [tight_gap]).candidates(ref)
        assert [c.action.name for c in ranked[:2]] == ["dagger_round", "real_cotrain"]
        assert ranked[0].answers == ["tight_gap"] and not ranked[2].answers

    def test_a_closed_family_reopens_only_for_requests_and_ranks_after_open_ones(self, ref: dict[str, Cell]) -> None:
        new_colour = learning_request("new_task", record("RETRY", successes=8, n=32), TaskSpec("yellow", "red"))
        closed = {"shelved": "437/576, worse on three suites"}
        menu = (*MENU, Action("shelved_recipe", "shelved", frozenset({"objects"}), RECOVERY_FAILURES, "true", 1.0))
        quiet = {c.action.name: c for c in Planner(closed, set(), 8.0, None, menu=menu).candidates(ref)}
        assert "closed" in quiet["shelved_recipe"].excluded
        ranked = Planner(closed, set(), 8.0, None, [new_colour], menu=menu).candidates(ref)
        names = [c.action.name for c in ranked]
        # reopened only by the unanswered refusal, and behind every open recipe despite the cheapest hours on the menu
        assert names.index("dagger_round") < names.index("shelved_recipe")
        shelved = ranked[names.index("shelved_recipe")]
        assert shelved.excluded is None and shelved.reopened == closed["shelved"]

    def test_intake_files_each_refused_scene_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        for name, result in {"centre": record("GO", successes=31, n=32), "new_task": record("RETRY", successes=8, n=32),
                             "real_photo": record("ASK_HUMAN", perception=0.24)}.items():  # fmt: skip
            (tmp_path / "gallery" / name).mkdir(parents=True)
            (tmp_path / "gallery" / name / "result.json").write_text(json.dumps(result))
        scenes = tmp_path / "scenes.json"
        scenes.write_text(json.dumps({n: {"pick": "yellow", "place": "red"} for n in ("new_task", "real_photo")}))
        for _ in range(2):
            assert CliRunner().invoke(app, ["intake", "gallery", "--scenes", str(scenes)]).exit_code == 0
        assert sorted(r["source"] for r in read_jsonl(REQUESTS)) == ["new_task", "real_photo"]


class TestRunLoop:
    @pytest.mark.parametrize(
        ("log", "results"),
        [("[07:24] task round basket: TASK_ROUND_DONE: logs/a/bench_tasks.json logs/a/bench_stack.json",
          ["logs/a/bench_tasks.json", "logs/a/bench_stack.json"]),
         ("[08:30] round 7: serving next: /ckpt\n[08:30] round 7: DAGGER_ROUND_DONE: logs/dagger/r7/bench.json",
          ["logs/dagger/r7/bench.json"]),
         ("[05:08] round 7: fine-tuning\n[08:30] round 7: DAGGER_ROUND_DONE; serving next: x", ["logs/dagger/r7/bench.json"]),
         ("[23:14] cotrain-r7: COTRAIN_DONE: logs/cotrain-r7/bench.json", ["logs/cotrain-r7/bench.json"]),
         ("[05:08] round 7: fine-tuning", None)],
    )  # fmt: skip
    def test_results_come_from_the_runs_last_line(self, log: str, results: list[str] | None) -> None:
        assert results_from_log(log) == results

    @pytest.mark.parametrize(
        ("events", "flight", "launched", "step"),
        [([("planned", 1), ("launched", 1)], (1, "finished", ["b.json"]), 0, ("judge", 1)),
         ([("planned", 1), ("launched", 1)], (1, "dead", []), 0, ("fail", 1)),
         ([("planned", 1), ("launched", 1)], (1, "running", []), 0, ("wait", 1)),
         ([("planned", 1), ("judged", 1, "PROMOTE"), ("planned", 2)], None, 0, ("await_approval", 1)),
         ([("planned", 1), ("judged", 1, "KEEP"), ("planned", 2)], None, 0, ("launch", 2)),
         ([("planned", 1), ("judged", 1, "KEEP"), ("planned", 2), ("dropped", 2)], None, 0, ("plan", None)),
         ([("planned", 1), ("judged", 1, "DISCARD")], None, 2, ("stop", None))],
    )  # fmt: skip
    def test_next_step(
        self,
        tmp_path: Path,
        events: list[tuple[str, ...]],
        flight: tuple[int, str, list[str]] | None,
        launched: int,
        step: tuple[str, int | None],
    ) -> None:
        log = CycleLog(tmp_path / "cycles.jsonl")
        for event, cycle, *verdict in events:
            log.append(event, cycle, action="dagger_round", **({"verdict": verdict[0]} if verdict else {}))
        assert next_step(log.cycles(), flight, launched, max_cycles=2) == step


class TestMenuMatchesItsScripts:
    """failure_synthesis named a stage dagger_round.sh does not have, and two cycles each burned a GPU wait to find
    out. A recipe is a string until something runs it, so check the two couplings a string can get wrong."""

    @staticmethod
    def script_of(action: Action) -> tuple[Path, str]:
        found = re.search(r"bash (scripts/\S+\.sh)", action.command)
        assert found, f"{action.name}: no script in {action.command!r}"
        path = Path(found.group(1))
        assert path.exists(), f"{action.name}: {path} does not exist"
        return path, path.read_text()

    @pytest.mark.parametrize("action", MENU, ids=lambda a: a.name)
    def test_every_variable_a_recipe_sets_is_one_its_script_reads(self, action: Action) -> None:
        path, source = self.script_of(action)
        for name in re.findall(r"\b([A-Z][A-Z0-9_]*)=", action.command):
            assert f"${{{name}" in source or f"${name}" in source, f"{action.name}: {path} never reads ${name}"

    def test_a_stage_a_recipe_names_must_be_one_its_script_branches_on(self) -> None:
        # the deleted entry, verbatim: STAGE=recover is a dagger.py subcommand, and dagger_round.sh branches on
        # harvest and aggregate, so it skipped building the dataset and died on a Hub 404
        source = Path("scripts/dagger_round.sh").read_text()
        assert set(re.findall(r'"\$STAGE" = (\w+)', source)) == {"harvest", "aggregate"}
        for action in MENU:
            if stage := re.search(r"\bSTAGE=(\S+)", action.command):
                _, source = self.script_of(action)
                assert stage.group(1) in re.findall(r'"\$STAGE" = (\w+)', source), action.name

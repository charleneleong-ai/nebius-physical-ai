"""Autopilot: the closed loop that widens what the policy can do, one tested change at a time.

Each cycle reads the served policy's benchmark, sorts its failures by generalisation axis (new objects, new tasks, new
environments) and failure mode, and proposes one training recipe from a fixed menu of tested scripts. Nemotron on
Token Factory chooses among the eligible candidates and writes why; it can't leave the menu and it never scores a
result. The run launches detached with a heartbeat (each script gates its own GPU stage on `autoresearch-wait`), the
candidate is judged by the paired gate on the cycle's target suites with no suite allowed to get worse (G3), and
promotion still needs a human. A family verdicts.yaml has closed comes back only for open requests it was never tried
on, ranked after open recipes. Every step is an event in logs/autopilot/cycles.jsonl.

Preflight feeds it: every scene it doesn't clear (RETRY or ASK_HUMAN) becomes a learning request naming the axis
training could fix, and recipes that answer open requests rank first. A refusal training can't fix (the twin could not
be trusted: a perception problem) is recorded as unmet rather than turned into a training run.

    npai autopilot intake ~/gallery --scenes ~/gallery/scenes.json   # Preflight's refusals become learning requests
    npai autopilot plan --bench logs/dagger/r3/bench.json --bench logs/held-out-tasks/bench.json
    npai autopilot launch 1                         # detached on this GPU host; --adopt LOG records a running job
    npai autopilot judge 1 logs/multitask-basket/bench_tasks.json logs/multitask-basket/bench_stack.json
    npai autopilot approve 1                        # a human promotes it
    npai autopilot status
    npai autopilot run --bench ... --bench ...       # unattended: judge, launch, plan; stops at a PROMOTE
"""

import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from nebius_physical_ai.benchmark import Cell, describe, load_cells, promotion
from nebius_physical_ai.pipeline import (
    EXPLAIN_MAX_TOKENS,
    LLM_TIMEOUT_S,
    TOKEN_FACTORY_KEY_ENV,
    TOKEN_FACTORY_MODELS,
    TOKEN_FACTORY_URL,
    parse_json_object,
    reply_text,
)
from nebius_physical_ai.task import DEFAULT_TASK, TaskSpec
from nebius_physical_ai.trust.verdict import TrustPolicy
from nebius_physical_ai.twin.rollout import FailureWatch
from nebius_physical_ai.verdicts import VERDICTS, closed_families

LOG = Path("logs/autopilot/cycles.jsonl")
REQUESTS = LOG.parent / "requests.jsonl"
ENVIRONMENT_PREFIXES = ("real_", "in_path_", "clean_top_down", "azimuth_", "clean_calibrated")
CANDIDATES_SHOWN = 3  # Nemotron chooses among the best few by reach per GPU hour, never from the whole menu
GPU_GATE = "autoresearch-wait --label autopilot --gpu-free-gb 70 --ram-free-gb 20 --stable 120 --timeout 43200"
console = Console(width=140)


def axis_of(cell: Cell) -> str | None:
    """Which generalisation axis a suite measures; None for seen scenes, which are only guarded against regression."""
    if cell.group == "held_out_tasks":
        return "tasks"
    if cell.suite.startswith("held_out_objects"):
        return "objects"
    if cell.group == "appearance" or cell.suite.startswith(ENVIRONMENT_PREFIXES):
        return "environments"
    return None


@dataclass(frozen=True)
class Action:
    """A tested training recipe: what it can fix, what it costs, and the command that runs it from the repo root.

    Every command gates its own GPU stage (DAgger's TRAIN_WHEN, co-training's WAIT_FOR, task_round.sh's wait), so
    CPU work such as harvesting or recording demos starts right away."""

    name: str
    family: str  # its family in verdicts.yaml; closed when every row there is DISCARD
    axes: frozenset[str]
    fixes: frozenset[str]  # FailureWatch categories it addresses
    command: str
    hours: float
    trains: str | None = None  # the suite it trains on directly: a gain there is not transfer

    def __post_init__(self) -> None:
        if unknown := self.fixes - set(FailureWatch.FAILURES):
            raise ValueError(f"{self.name}: unknown failure modes {sorted(unknown)}")

    def reach(self, cells: dict[str, Cell]) -> int:
        """Failed rehearsals on this action's axes that fail the way it addresses: the most it could win back."""
        return sum(
            n for c in cells.values() if axis_of(c) in self.axes for mode, n in c.failures.items() if mode in self.fixes
        )


RECOVERY_FAILURES = frozenset({"dropped", "pushed", "holding"})
CONTACT_FAILURES = RECOVERY_FAILURES | {"no_contact"}
HELD_OUT_TASKS = ("basket", "ring")
# The recipe itself is cotrain.sh's defaults; what only the caller knows is which lineage to fine-tune and compare
# against, which the DAgger state carries so an approved promotion feeds the next cycle. Mapping the names here rather
# than sourcing the file inside cotrain.sh is deliberate: state.env's BASE is dagger's demos, cotrain's is a checkpoint.
# TWIN is the round's aggregate (base demos plus its recoveries upweighted), not state.env's DATA, which is the base
# demos alone: cycle 5 fine-tuned the round-3 policy on data without the recoveries it was trained on and forgot them
# (clean_many 88 -> 64, real_1_pomade_tin 87 -> 62, both p=0.000). cotrain.sh stops if the aggregate is not on disk.
COTRAIN_CMD = (
    ". logs/dagger/state.env && NAME=cotrain-r$ROUND BASE=$POLICY TWIN=$HOME/data/dagger_agg_r${ROUND}_v30 "
    f"REF_BENCH=$REF_BENCH WAIT_FOR='{GPU_GATE}' bash scripts/cotrain.sh"
)
MENU = (
    Action("dagger_round", "dagger", frozenset({"objects", "environments"}), RECOVERY_FAILURES,
           f"AUTOPILOT=1 TRAIN_WHEN='{GPU_GATE}' bash scripts/dagger_round.sh", hours=6.0),
    *(Action(f"task_round:{task}", "multitask", frozenset({"tasks"}), CONTACT_FAILURES,
             f"TASK={task} bash scripts/task_round.sh", hours=5.0, trains=f"held_out_task_{task}")
      for task in HELD_OUT_TASKS),
    # failure_synthesis is not here: it needs `dagger.py synthesise`, which lives on the unmerged synthesis branch
    # (efca8d3). The entry that was here passed STAGE=recover, a `dagger.py` subcommand and not one of
    # dagger_round.sh's stages, so it skipped harvest and aggregate and trained on a dataset nothing had built:
    # cycles 6 and 7 each waited hours for the GPU and then died in 13 seconds. Restore it with the branch.
    Action("real_cotrain", "cotrain", frozenset({"environments"}), CONTACT_FAILURES, COTRAIN_CMD, hours=6.0),
)  # fmt: skip
ACTIONS = {a.name: a for a in MENU}


def merged_cells(benches: list[Path]) -> dict[str, Cell]:
    return {suite: cell for path in benches for suite, cell in load_cells(path)[1].items()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete row; an append cut off mid-write loses that row, not the file."""
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@dataclass
class LearningRequest:
    """A scene Preflight would not clear, as demand for training: the axes a recipe could fix, and what none can."""

    source: str
    verdict: str
    instruction: str
    axes: list[str]
    unmet: list[str]
    reasons: list[str]


def learning_request(source: str, record: dict[str, Any], task: TaskSpec) -> LearningRequest | None:
    """None unless Preflight refused; otherwise why it did, sorted into trainable axes and the rest."""
    # Only a refusal is a request. PERCEIVED means the gate passed without rehearsing, and a failed run has no
    # verdict at all; filing either gave a request with no axes, which no action can answer and which still counted
    # itself on the Autopilot page.
    if (verdict := record.get("verdict")) not in ("RETRY", "ASK_HUMAN"):
        return None
    axes, unmet, reasons = [], [], []
    scene, rehearsal = record.get("scene"), record.get("rehearsal") or {}
    if not task.in_distribution:
        axes.append("objects")  # the trained skill, asked of objects it was never shown
        reasons.append(f"asked {task.instruction!r}, outside the one task it was trained on")
    if scene is None or scene["perception"] < TrustPolicy.min_fidelity:
        unmet.append("perception")
        if scene is None:
            why = record.get("explanation", "").removeprefix(f"{verdict}: ") or "perception failed"
            reasons.append(f"no twin was built ({why}); recognising the objects fixes that, not policy training")
        else:
            doubts = [w for w in scene.get("warnings", []) if "outside what it was trained on" not in w]
            reasons.append(
                f"a twin was built from the photo, but its perception scores {scene['perception']:.2f}, below "
                f"{TrustPolicy.min_fidelity:.2f}"
                + (f" ({'; '.join(doubts)})" if doubts else "")
                + ", so a rehearsal on it isn't trusted; better recognition or measured object models fix that, not "
                "policy training"
            )
    elif rehearsal.get("n") and task.in_distribution:
        axes.append("environments")  # the trained task, failing in this arrangement
        reasons.append(f"only {rehearsal['successes']} of {rehearsal['n']} rehearsals succeeded in this layout")
    if not (axes or unmet):  # refused for something training cannot fix, e.g. markers the camera never saw
        return None
    return LearningRequest(source, verdict, task.instruction, axes, unmet, reasons)


def diagnosis(cells: dict[str, Cell]) -> dict[str, dict[str, Any]]:
    """Per axis: rehearsals, successes and how the rest failed."""
    axes: dict[str, dict[str, Any]] = {}
    for cell in cells.values():
        row = axes.setdefault(axis_of(cell) or "seen", {"successes": 0, "n": 0, "failures": Counter(), "suites": []})
        row["successes"] += cell.successes
        row["n"] += cell.n
        row["failures"].update(cell.failures)
        row["suites"].append(f"{cell.suite} {cell.successes}/{cell.n}")
    return {axis: {**row, "failures": dict(row["failures"])} for axis, row in axes.items()}


@dataclass
class Candidate:
    action: Action
    reach: int
    excluded: str | None = None  # why it may not be chosen, if it may not
    answers: list[str] = field(default_factory=list)  # open learning requests on its axes
    reopened: str | None = None  # a closed family's last result, when open requests bring it back

    @property
    def score(self) -> float:
        return self.reach / self.action.hours

    def record(self) -> dict[str, Any]:
        return {"action": self.action.name, "reach": self.reach, "hours": self.action.hours, "excluded": self.excluded,
                "answers": self.answers, "reopened": self.reopened}  # fmt: skip


class Chooser:
    """Nemotron on Token Factory, asked to pick one candidate and say why. Its answer must name an eligible candidate."""

    def __init__(self, api_key: str, model: str = "nemotron-3-ultra") -> None:
        self.api_key, self.model = api_key, TOKEN_FACTORY_MODELS[model]["model_id"]

    def choose(
        self, diag: dict[str, Any], eligible: list[Candidate], closed: dict[str, str], requests: list[LearningRequest]
    ) -> tuple[str, str]:
        menu = [
            {**c.record(), "axes": sorted(c.action.axes), "fixes": sorted(c.action.fixes),
             "judged_on": [s for axis in sorted(c.action.axes) for s in diag.get(axis, {}).get("suites", [])]}
            for c in eligible
        ]  # fmt: skip
        asked = [{"scene": r.source, "asked": r.instruction, "axes": r.axes, "why": r.reasons} for r in requests]
        prompt = (
            "You run the training loop of a robot policy (GR00T N1.7 on an SO-101 arm, rehearsed in a MuJoCo twin). "
            "The goal is generalisation to new objects, new tasks and new environments without losing seen ones.\n"
            f"Benchmark of the served policy, by axis: {json.dumps(diag)}\n"
            f"Closed experiment families and the result that closed each: {json.dumps(closed)}\n"
            "An action marked `reopened` belongs to a closed family and is back only because it answers open requests; "
            "choose it only if those failures differ from the ones that closed it, and say why.\n"
            f"Learning requests, scenes Preflight refused to clear for real users: {json.dumps(asked)}\n"
            f"Eligible actions, ranked by failed rehearsals they could win back per GPU hour: {json.dumps(menu)}\n"
            "Each action is judged only on its `judged_on` suites (paired gain, and no suite anywhere may get worse), "
            "and `answers` names the requests it addresses; prefer answering requests when the evidence is close.\n"
            'Choose exactly one. Answer as JSON: {"choice": "<action name>", "rationale": "<2-3 sentences: the '
            'failure pattern it targets and what result on its judged_on suites would count as success>"}'
        )
        with httpx.Client(timeout=LLM_TIMEOUT_S) as client:
            resp = client.post(
                f"{TOKEN_FACTORY_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": EXPLAIN_MAX_TOKENS, "temperature": 0.2, "response_format": {"type": "json_object"}},
            )  # fmt: skip
        resp.raise_for_status()
        answer = parse_json_object(reply_text(resp.json(), "Nemotron plan"), "Nemotron plan")
        return str(answer["choice"]), str(answer["rationale"])


@dataclass
class Planner:
    """Ranks the menu against the served policy's failures, then lets Nemotron choose among the top eligible few."""

    closed: dict[str, str]  # family -> the result that closed it
    tried: set[str]
    budget_hours: float
    chooser: Chooser | None
    requests: list[LearningRequest] = field(default_factory=list)
    menu: tuple[Action, ...] = MENU  # injectable so a test can rank a menu of its own rather than whatever ships

    def candidates(self, cells: dict[str, Cell]) -> list[Candidate]:
        ranked = []
        for action in self.menu:
            reach = action.reach(cells)
            answers = [r.source for r in self.requests if action.axes & set(r.axes)]
            # a closed family comes back only for refusals it was never tried on, and ranks after open recipes
            reopened = self.closed.get(action.family) if answers else None
            why = (
                f"family '{action.family}' closed in verdicts.yaml" if action.family in self.closed and not answers
                else "already judged against this reference" if action.name in self.tried
                else f"{action.hours:.0f} h is over the {self.budget_hours:.0f} h budget" if action.hours > self.budget_hours
                else "no failures it addresses" if not reach
                else None
            )  # fmt: skip
            ranked.append(Candidate(action, reach, why, answers, reopened))
        # demand first: a recipe that answers a real refusal outranks one that only moves the benchmark
        return sorted(ranked, key=lambda c: (c.excluded is not None, not c.answers, c.reopened is not None, -c.score))

    def choose(
        self, cells: dict[str, Cell], diag: dict[str, Any]
    ) -> tuple[list[Candidate], Candidate | None, str, str]:
        """The ranking, the pick, why, and who decided it - "nemotron" only when Nemotron's answer was used."""
        ranked = self.candidates(cells)
        eligible = [c for c in ranked if c.excluded is None][:CANDIDATES_SHOWN]
        if not eligible:
            return (
                ranked,
                None,
                "nothing on the menu is eligible: every family that could help is closed or tried",
                "ranking",
            )
        if self.chooser is None:
            rationale = "the top-ranked candidate (Nemotron not asked)"
        else:
            try:
                name, rationale = self.chooser.choose(diag, eligible, self.closed, self.requests)
                if (pick := next((c for c in eligible if c.action.name == name), None)) is not None:
                    return ranked, pick, rationale, "nemotron"
                rationale = f"Nemotron chose '{name}', which is not an eligible candidate; taking the top-ranked one"
            except (httpx.HTTPError, KeyError, ValueError, RuntimeError) as err:
                rationale = f"Nemotron unavailable ({type(err).__name__}); taking the top-ranked candidate"
        top = eligible[0]
        # the fallback is the ranking's pick, whether or not a chooser was configured: say so rather than claim
        # Nemotron decided, because the cycle log is what the published Space attributes the decision from
        return (
            ranked,
            top,
            f"{rationale}. {top.action.name} could win back up to {top.reach} failed rehearsals.",
            "ranking",
        )


@dataclass
class Judgement:
    verdict: str  # PROMOTE (pending a human), KEEP (within noise) or DISCARD (a suite got worse)
    why: str
    suites: dict[str, str] = field(default_factory=dict)  # suite -> score against the reference, with paired evidence


def judge(action: Action, ref: dict[str, Cell], new: dict[str, Cell]) -> Judgement:
    """G3: pooled paired gain on the action's target suites (minus the one it trains on), and no shared suite worse."""
    shared = [s for s in ref if s in new]
    suites = {s: f"{ref[s].successes}/{ref[s].n} -> {describe(new[s], ref[s])}" for s in shared}
    target = [s for s in shared if axis_of(ref[s]) in action.axes and s != action.trains]
    gained, why = promotion(new, ref, pool=target or None)
    if why.startswith("worse on"):
        return Judgement("DISCARD", why, suites)
    if not target:
        return Judgement("KEEP", "no target suites in common with the reference", suites)
    return Judgement("PROMOTE" if gained else "KEEP", f"target {', '.join(target)}: {why}", suites)


def utc_now(fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    return time.strftime(fmt, time.gmtime())


class Cycles(dict[int, dict[str, Any]]):
    """Every cycle by number, each folded to what it is now: a mapping, plus the questions the loop asks of it."""

    def tried(self, refs: list[str]) -> set[str]:
        """Actions already judged against the same reference benchmark: rerunning one only re-measures it."""
        return {c["action"] for c in self.values() if c.get("refs") == refs and "verdict" in c}

    def in_flight(self) -> list[int]:
        return [n for n, c in self.items() if c["event"] == "launched"]

    def promoted(self) -> list[int]:
        return [n for n, c in self.items() if c["event"] == "judged" and c.get("verdict") == "PROMOTE"]

    def planned(self) -> list[int]:
        return sorted(n for n, c in self.items() if c["event"] == "planned")

    def action(self, cycle: int) -> Action:
        """The recipe a cycle ran, refused by name when the menu no longer carries it (a deleted entry, say)."""
        if (name := self[cycle]["action"]) not in ACTIONS:
            raise typer.BadParameter(f"cycle {cycle} ran '{name}', which is no longer on the menu")
        return ACTIONS[name]

    def require(self, cycle: int, event: str) -> dict[str, Any]:
        """The cycle, refused unless it is in `event`: each command that moves one is valid from a single state."""
        if (state := self.get(cycle, {}).get("event")) != event:
            raise typer.BadParameter(f"cycle {cycle} is {state or 'unknown'}, not {event}")
        return self[cycle]


class CycleLog:
    """Append-only events; a cycle is the fold of its events, its state the latest event (planned, launched, judged,
    approved)."""

    def __init__(self, path: Path = LOG) -> None:
        self.path = path

    def append(self, event: str, cycle: int, **data: Any) -> None:
        row = {"event": event, "cycle": cycle, "time": utc_now(), **data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def cycles(self) -> Cycles:
        folded = Cycles()
        for e in read_jsonl(self.path):
            stamp = {"planned_at": e["time"]} if e["event"] == "planned" else {}
            folded.setdefault(e["cycle"], {}).update(e, **stamp)
        return folded


DONE = re.compile(r"[A-Z][A-Z_]*_DONE[:;]([^\n]*)")  # every recipe script ends with `<NAME>_DONE: <result paths>`
STALL_S = 3 * 3600  # a run whose log stays silent this long counts as dead (GPU waits check in every 30 min)
TAIL_BYTES = 65536


def cycle_pattern(cycle: int) -> str:
    return f"[c]ycle_{cycle}.sh"  # the bracket keeps pgrep from matching its own command line


def results_from_log(text: str) -> list[str] | None:
    """The candidate's result files once its run has finished, from the run's own last line; None while it runs.

    DAgger rounds started before the done line carried a path print none: their file follows from the round number."""
    if not (match := DONE.search(text)):
        return None
    if paths := re.findall(r"\S+\.json", match.group(1)):
        return paths
    rounds = re.findall(r"round (\d+):", text)
    return [f"logs/dagger/r{rounds[-1]}/bench.json"] if rounds else None


def read_tail(log: Path) -> str:
    """The end of a run's log, where its done line is; the whole file only when an old DAgger line needs its round."""
    with log.open("rb") as f:
        f.seek(max(0, log.stat().st_size - TAIL_BYTES))
        tail = f.read().decode(errors="replace")
    return log.read_text(errors="replace") if "DAGGER_ROUND_DONE;" in tail else tail


def observe(cycles: Cycles) -> tuple[int, str, list[str]] | None:
    """The cycle in flight, if any, as (cycle, "finished" | "dead" | "running", result files)."""
    if not (busy := cycles.in_flight()):
        return None
    n, log = busy[0], Path(cycles[busy[0]]["log"])
    if not log.exists():
        return n, "dead", []
    if results := results_from_log(read_tail(log)):
        return n, "finished", results
    silent = time.time() - log.stat().st_mtime > STALL_S
    running = (
        cycles[n].get("adopted")
        or subprocess.run(["pgrep", "-f", cycle_pattern(n)], capture_output=True, check=False).returncode == 0
    )
    return n, "dead" if silent or not running else "running", []


def next_step(
    cycles: Cycles, flight: tuple[int, str, list[str]] | None, launched: int, max_cycles: int
) -> tuple[str, int | None]:
    """The loop's one move: judge, fail, wait, stop at a PROMOTE for a human, launch what is planned, or plan."""
    if flight is not None:
        n, state, _ = flight
        return {"finished": "judge", "dead": "fail", "running": "wait"}[state], n
    if promoted := cycles.promoted():
        return "await_approval", promoted[0]
    if launched >= max_cycles:
        return "stop", None
    if planned := cycles.planned():
        return "launch", planned[0]
    return "plan", None


def end_cycle(cycle: int, was: str, why: str) -> None:
    """Close a cycle in `was` without a verdict: one only planned is dropped, one that had launched failed."""
    log = CycleLog()
    log.cycles().require(cycle, was)
    becomes = "dropped" if was == "planned" else "failed"
    log.append(becomes, cycle, why=why)
    console.print(f"cycle {cycle}: {becomes} ({why})")


def abandon(cycle: int, why: str) -> None:
    """Close a cycle the loop cannot act on at all, from whichever state it is in."""
    end_cycle(cycle, CycleLog().cycles().get(cycle, {}).get("event", "planned"), why)


def detach(command: list[str], log: Path) -> None:
    """Start `command` in its own session (PPID 1 once this CLI exits), output appended to `log`."""
    with log.open("ab") as sink:
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, start_new_session=True)


app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__.split("\n\n")[0])
BenchOpt = Annotated[list[Path], typer.Option("--bench", help="The served policy's result files (benchmark.py run)")]


def print_plan(diag: dict[str, dict[str, Any]], ranked: list[Candidate]) -> None:
    table = Table("axis", "rehearsals", "failures")
    for axis, row in diag.items():
        table.add_row(axis, f"{row['successes']}/{row['n']}", ", ".join(f"{k} {v}" for k, v in row["failures"].items()))
    console.print(table)
    table = Table("action", "reach", "hours", "reach/h", "answers", "status")
    for c in ranked:
        table.add_row(c.action.name, str(c.reach), f"{c.action.hours:.0f}", f"{c.score:.1f}", ", ".join(c.answers),
                      c.excluded or "eligible")  # fmt: skip
    console.print(table)


@app.command()
def plan(
    bench: BenchOpt,
    budget_hours: Annotated[float, typer.Option(help="Most GPU hours one cycle may spend")] = 8.0,
    offline: Annotated[bool, typer.Option(help="Take the top-ranked candidate without asking Nemotron")] = False,
    action: Annotated[str | None, typer.Option(help="A human's pick from the menu instead (logged as such)")] = None,
    why: Annotated[str, typer.Option(help="The human's reason, with --action")] = "",
    verdicts: Path = VERDICTS,
) -> None:
    """Diagnose the served policy and propose the next cycle's one action."""
    log, cells, refs = CycleLog(), merged_cells(bench), [str(p) for p in bench]
    cycles, diag, key = log.cycles(), diagnosis(cells), os.environ.get(TOKEN_FACTORY_KEY_ENV)
    chooser = None if offline or action or not key else Chooser(key)
    requests = [LearningRequest(**r) for r in read_jsonl(REQUESTS)]
    planner = Planner(closed_families(verdicts), cycles.tried(refs), budget_hours, chooser, requests)
    if action is None:
        ranked, pick, rationale, chosen_by = planner.choose(cells, diag)
    else:
        ranked = planner.candidates(cells)
        if (pick := next((c for c in ranked if c.action.name == action), None)) is None or pick.excluded:
            raise typer.BadParameter(f"{action} is {pick.excluded if pick else 'not on the menu'}")
        rationale, chosen_by = why or "a human's choice", "human"
    print_plan(diag, ranked)
    if pick is None:
        console.print(rationale)
        raise typer.Exit(1)
    if chosen_by == "ranking" and not offline:
        rationale = f"{TOKEN_FACTORY_KEY_ENV} not set; {rationale}"
    cycle = max(cycles, default=0) + 1
    log.append("planned", cycle, action=pick.action.name, chosen_by=chosen_by, refs=refs, rationale=rationale,
               reach=pick.reach, answers=pick.answers, candidates=[c.record() for c in ranked])  # fmt: skip
    console.print(f"cycle {cycle}: [bold]{pick.action.name}[/bold] - {rationale}")


@app.command()
def launch(
    cycle: int,
    adopt: Annotated[Path | None, typer.Option(help="Record a job already running this action, by its log")] = None,
    dry_run: bool = False,
) -> None:
    """Start a planned cycle detached with a heartbeat; one cycle in flight at a time."""
    log = CycleLog()
    cycles = log.cycles()
    cycles.require(cycle, "planned")
    action = cycles.action(cycle)
    if dry_run:
        console.print(action.command)
        return
    if busy := cycles.in_flight():
        raise typer.BadParameter(f"cycle {busy[0]} is still running; judge it first")
    if adopt is not None:
        log.append("launched", cycle, log=str(adopt), adopted=True)
        console.print(f"cycle {cycle}: adopted the running job logging to {adopt}")
        return
    script = LOG.parent / f"cycle_{cycle}.sh"
    out = LOG.parent / f"cycle_{cycle}_{utc_now('%Y%m%dT%H%M%SZ')}.log"
    script.write_text(
        f"#!/usr/bin/env bash\n# autopilot cycle {cycle}: {action.name}\ncd {Path.cwd()}\n{action.command}\n"
    )
    detach(["bash", str(script)], out)
    log.append("launched", cycle, log=str(out))
    console.print(f"cycle {cycle}: {action.name} launched, log {out}")
    # non-interactive shells often lack ~/.local/bin on PATH, where `uv tool install` puts the heartbeat
    if heartbeat := shutil.which("autoresearch-heartbeat") or shutil.which(
        "autoresearch-heartbeat", path=str(Path.home() / ".local/bin")
    ):
        detach([heartbeat, "--label", f"autopilot-{cycle}", "--pattern", cycle_pattern(cycle), "--log", str(out),
                "--stall-after", "3600"], LOG.parent / "heartbeat.log")  # fmt: skip
    else:
        console.print("[yellow]autoresearch-heartbeat not found: the run has no check-ins[/yellow]")


@app.command(name="judge")
def judge_cycle(
    cycle: int,
    results: Annotated[list[Path], typer.Argument(help="The candidate's result files, on the reference's suites")],
) -> None:
    """Score a finished cycle against its reference with the G3 gate; PROMOTE still waits for `approve`."""
    log = CycleLog()
    cycles = log.cycles()
    planned = cycles.require(cycle, "launched")
    action = cycles.action(cycle)
    result = judge(action, merged_cells([Path(p) for p in planned["refs"]]), merged_cells(results))
    log.append("judged", cycle, results=[str(p) for p in results], **asdict(result))
    for suite, line in result.suites.items():
        console.print(f"  {suite}: {line}")
    console.print(f"cycle {cycle}: [bold]{result.verdict}[/bold] ({result.why})")


@app.command()
def approve(cycle: int, note: str = "") -> None:
    """A human's promotion of a cycle the gate passed."""
    log = CycleLog()
    if (judged := log.cycles().require(cycle, "judged"))["verdict"] != "PROMOTE":
        raise typer.BadParameter(f"cycle {cycle} was judged {judged['verdict']}, not PROMOTE")
    log.append("approved", cycle, note=note)
    console.print(f"cycle {cycle}: {judged['action']} approved for serving")


@app.command()
def drop(cycle: int, why: Annotated[str, typer.Option(help="What changed since it was planned")]) -> None:
    """Withdraw a planned cycle that new evidence has overtaken; the log keeps it and the reason."""
    end_cycle(cycle, "planned", why)


@app.command()
def intake(
    results: Annotated[Path, typer.Argument(help="Preflight job outputs: <scene>/result.json (npai preflight-job)")],
    scenes: Annotated[Path | None, typer.Option(help="Each scene's asked task, {scene: {pick, place}}")] = None,
) -> None:
    """File every scene Preflight did not clear as a learning request; each scene is filed once."""
    tasks = json.loads(scenes.read_text()) if scenes else {}
    filed = {r["source"] for r in read_jsonl(REQUESTS)}
    new = []
    for path in sorted(results.glob("*/result.json")):
        name = path.parent.name
        task = TaskSpec(**tasks[name]) if name in tasks else DEFAULT_TASK
        if name not in filed and (request := learning_request(name, json.loads(path.read_text()), task)):
            new.append(request)
    REQUESTS.parent.mkdir(parents=True, exist_ok=True)
    with REQUESTS.open("a") as f:
        f.writelines(json.dumps(asdict(r)) + "\n" for r in new)
    for r in new:
        console.print(f"{r.source} ({r.verdict}): axes {r.axes or '-'}; unmet {r.unmet or '-'}; {'; '.join(r.reasons)}")
    console.print(f"{len(new)} new learning requests -> {REQUESTS}")


@app.command()
def fail(cycle: int, why: Annotated[str, typer.Option(help="Why the run is over without a result")]) -> None:
    """Close a launched cycle whose run died or stalled without a result; the log keeps it and the reason."""
    end_cycle(cycle, "launched", why)


def act_on(step: str, cycle: int, flight: tuple[int, str, list[str]] | None) -> int:
    """Take a step that is about one cycle (next_step always names one for these), and report launches.

    A cycle the loop cannot act on is closed rather than raised through: next_step always returns the lowest
    planned cycle, so raising here would make one unrunnable cycle kill the daemon on every restart and block
    every cycle behind it."""
    try:
        match step:
            case "judge":
                judge_cycle(cycle, [Path(p) for p in flight[2]] if flight else [])
            case "fail":
                fail(cycle, why="the run ended or fell silent without a result; see its log")
            case "launch":
                launch(cycle)
                return 1
    except typer.BadParameter as err:
        abandon(cycle, f"the loop could not act on it: {err}")
    return 0


@app.command()
def run(
    bench: BenchOpt,
    every_min: Annotated[float, typer.Option(help="Minutes between checks while a cycle runs")] = 10.0,
    max_cycles: Annotated[int, typer.Option(help="Most cycles this run may launch")] = 2,
    budget_hours: float = 8.0,
    verdicts: Path = VERDICTS,
) -> None:
    """Drive the loop unattended: judge finished cycles, launch planned ones, plan the next. It stops when a cycle is
    judged PROMOTE (a human approves, then restarts it against the new reference), after max_cycles launches, or when
    nothing on the menu is eligible."""
    launched = 0
    while True:
        cycles = CycleLog().cycles()
        flight = observe(cycles)
        step, n = next_step(cycles, flight, launched, max_cycles)
        console.print(f"[{utc_now()}] autopilot: {step} {n or ''}".rstrip())
        match step:
            case "judge" | "fail" | "launch":
                launched += act_on(step, n, flight)
            case "plan":
                try:
                    plan(bench, budget_hours=budget_hours, verdicts=verdicts)
                except typer.Exit:
                    return
            case "await_approval" | "stop":
                return
            case "wait":
                time.sleep(every_min * 60)
            case _:  # next_step returns a closed set; a new one must not quietly become a sleep
                raise AssertionError(f"unknown step {step!r}")


@app.command()
def status() -> None:
    """Every cycle: its action, state and verdict."""
    table = Table("cycle", "action", "state", "verdict", "why")
    for n, c in sorted(CycleLog().cycles().items()):
        table.add_row(str(n), c["action"], c["event"], c.get("verdict", ""), c.get("why") or c.get("rationale", ""))
    console.print(table)

"""Render the Autopilot Space (a static page) from the loop's own files: the cycle log, the reference benchmarks the
latest cycle was planned against, verdicts.yaml, and rollout videos of the suites it targets.

    uv run python deploy/autopilot_space/build.py OUT --videos ~/gallery/suites    # then publish.sh OUT
"""

import html
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from nebius_physical_ai.autopilot import LOG, REQUESTS, CycleLog, diagnosis, merged_cells, read_jsonl
from nebius_physical_ai.verdicts import closed_families, load_verdicts

HERE = Path(__file__).parent
LINKS = {
    "Hugging Face collection": "https://huggingface.co/collections/chaleong/preflight-rehearse-before-the-robot-moves-6ad409a5fccdebd3c38eea7e",
    "Preflight Space": "https://huggingface.co/spaces/chaleong/preflight",
    "All experiments": "https://huggingface.co/spaces/chaleong/preflight-experiments",
    "Our checkpoint": "https://huggingface.co/chaleong/preflight-gr00t-so101-dagger-r3",
    "Code (GitHub)": "https://github.com/charleneleong-ai/nebius-physical-ai",
    "Experiments (W&B report)": "https://wandb.ai/chaleong/preflight/reports/Preflight:-SO-101-generalisation-experiments--VmlldzoxODA5MDQ3Mg==",
}
DEMOS = (  # (suite, video, caption): what the first cycles target
    ("held_out_task_ring", "failure.mp4", "New task, ring over a peg: the policy never reaches for the ring (0 of 96)"),
    ("held_out_task_basket", "failure.mp4", "New task, eye drops into a basket: mostly no contact (6 of 96)"),
    ("held_out_objects", "failure.mp4", "New objects: grasps that drop or push (57 of 96)"),
    ("clean_many", "success.mp4", "Seen scenes, which every cycle must not get worse on (88 of 96)"),
)
AXIS_NOTE = {"objects": "colours and sizes never seen", "tasks": "tasks never trained", "seen": "regression guard",
             "environments": "cameras and real objects"}  # fmt: skip
STEPS = (  # title, one line, the detail a hover or click reveals
    (
        "Diagnose",
        "failures by axis and mode, from the served policy's benchmark",
        (
            "Reads the served policy's result files (8 suites x 96 paired rehearsals), puts each suite on an axis (objects, "
            "tasks, environments, or seen scenes as the guard) and counts failures by mode: no_contact, pushed, dropped, "
            "holding, fell_off. It also reads the learning requests intake filed from Preflight's refusals."
        ),
    ),
    (
        "Propose",
        "rank a fixed menu of tested recipes; Nemotron picks one of the top three",
        (
            "Five tested recipes are ranked by failed rehearsals each could win back per GPU hour. Recipes that answer a "
            "Preflight refusal rank first; a closed family stays out unless a refusal reopens it. The top three eligible go to "
            "Nemotron 3 Ultra on Nebius Token Factory, which picks one and says what result would count as success. A choice "
            "off the list falls back to the ranking."
        ),
    ),
    (
        "Train",
        "detached on the A100, gated on free GPU memory, heartbeat every 30 min",
        (
            "The chosen script starts detached (PPID 1) behind autoresearch-wait, which holds until 70 GB of GPU memory and "
            "20 GB of RAM have stayed free for two minutes, so other jobs on the host are never displaced. A heartbeat checks "
            "in every 30 minutes and flags a run whose log has stopped growing."
        ),
    ),
    (
        "Benchmark",
        "the same 96 paired scenes per suite, seed 42",
        (
            "The candidate rehearses exactly the scenes the reference did: 12 scenes x 8 perturbations per suite at seed 42. "
            "Every rehearsal therefore pairs with the reference's on the identical scene, which is what makes the gate paired."
        ),
    ),
    (
        "Gate (G3)",
        "paired gain on the target suites, no suite worse",
        (
            "McNemar's test on the scenes where the two policies disagree. The target suites pooled must be better and no suite "
            "anywhere significantly worse (p < 0.05); a gain on the suite a recipe trains on directly doesn't count. The "
            "verdict (PROMOTE, KEEP or DISCARD) goes into the cycle log and verdicts.yaml."
        ),
    ),
    (
        "Human",
        "approves promotion; the new policy becomes the next reference",
        (
            "Only npai autopilot approve promotes a candidate. Until then Preflight keeps serving the current policy and the "
            "next cycle plans against the same reference, so nothing the agent does reaches the robot unreviewed."
        ),
    ),
)
e = html.escape


def so_far(cycles: dict[int, dict[str, Any]], requests: list[dict[str, Any]]) -> tuple[str, ...]:
    """What each step has actually done, in the same order as STEPS, so the diagram reports rather than describes."""
    judged = [c for c in cycles.values() if "verdict" in c]
    tally = Counter(c["verdict"] for c in judged)
    running = [n for n, c in cycles.items() if c["event"] == "launched"]
    answered = {request for c in cycles.values() for request in c.get("answers") or []}
    chosen = [f"cycle {n}: {c['action']} ({c.get('chosen_by', 'ranking')})" for n, c in sorted(cycles.items())]
    flight = (
        f"cycle {running[0]} ({cycles[running[0]]['action']}) started {cycles[running[0]]['planned_at']}"
        if running
        else "nothing running"
    )
    return (
        f"{len(requests)} learning requests filed from Preflight's refusals, {len(answered)} picked up by a cycle.",
        f"{len(cycles)} cycles planned - {'; '.join(chosen)}.",
        f"{len(cycles) - len(running)} runs finished, {flight}.",
        f"{len(judged)} candidates benchmarked on the same 576 paired rehearsals as the served policy.",
        f"{len(judged)} judged: " + (", ".join(f"{n} {v}" for v, n in sorted(tally.items())) or "none yet")
        + f". Closed families: {', '.join(sorted(closed_families())) or 'none'}.",
        (f"{tally.get('PROMOTE', 0)} promotions approved, so the policy Preflight serves is still the one a human "
         "last signed off."),
    )  # fmt: skip


def bars(diag: dict[str, dict[str, Any]]) -> str:
    """One row per axis: success rate as a bar, with the count and the failure mix."""
    rows = []
    for axis, row in sorted(diag.items(), key=lambda kv: kv[1]["successes"] / kv[1]["n"]):
        pct = 100 * row["successes"] / row["n"]
        mix = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(row["failures"].items(), key=lambda kv: -kv[1]))
        rows.append(
            f'<div class="bar"><div class="label"><b>{e(axis)}</b><span>{e(AXIS_NOTE.get(axis, ""))}</span></div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="num">{row["successes"]}/{row["n"]} ({pct:.0f}%)</div><div class="mix">failures: {e(mix)}</div></div>'
        )
    return "\n".join(rows)


def cycle_cards(cycles: dict[int, dict[str, Any]]) -> str:
    cards = []
    who = {"nemotron": "proposed by Nemotron", "human": "chosen by a human", "ranking": "top of the ranking"}
    for n, c in sorted(cycles.items(), reverse=True):
        verdict = c.get("verdict") or {"launched": "RUNNING", "planned": "QUEUED"}.get(c["event"], c["event"].upper())
        moved = [f"<li>{e(s.replace('_', ' '))}: {e(line)}</li>" for s, line in c.get("suites", {}).items()]
        cards.append(
            f'<div class="card"><h3>Cycle {n}: {e(c["action"])} <span class="pill {e(verdict)}">{e(verdict)}</span></h3>'
            f'<div class="stat">{e(who.get(c.get("chosen_by", ""), ""))} · planned '
            f"{e(c.get('planned_at', c['time'])[:16].replace('T', ' '))} UTC"
            f" · could win back up to {c.get('reach', '?')} failed rehearsals</div>"
            f"<p>{e(c.get('rationale', ''))}</p>"
            + (f"<p><b>Judged:</b> {e(c['why'])}</p><ul class='moved'>{''.join(moved)}</ul>" if "verdict" in c else "")
            + (f"<p><b>Dropped:</b> {e(c['why'])}</p>" if c["event"] == "dropped" else "")
            + "</div>"
        )
    return "\n".join(cards) or "<p>No cycles yet.</p>"


def request_cards(requests: list[dict[str, Any]], cycles: dict[int, dict[str, Any]], frames: set[str]) -> str:
    """Each refused scene: its photo, why Preflight refused it, the axis training could fix, and the cycle answering it."""
    cards = []
    for r in requests:
        answering = [f"cycle {n}" for n, c in sorted(cycles.items())
                     if r["source"] in c.get("answers", []) and c["event"] != "dropped"]  # fmt: skip
        fix = ", ".join(r["axes"]) or "nothing a training recipe can fix"
        status = ", ".join(answering) or ("unmet: " + ", ".join(r["unmet"]) if not r["axes"] else "waiting for a cycle")
        img = f'<img src="{e(r["source"])}.jpg" alt="{e(r["source"])}">' if r["source"] in frames else ""
        cards.append(
            f'<div class="request">{img}<div><h3>{e(r["source"].replace("_", " "))} '
            f'<span class="pill {e(r["verdict"])}">{e(r["verdict"])}</span></h3>'
            f'<div class="stat">asked: {e(r["instruction"])}</div><p>{e("; ".join(r["reasons"]))}.</p>'
            f'<div class="stat">trainable axis: <b>{e(fix)}</b> · {e(status)}</div></div></div>'
        )
    return "\n".join(cards) or "<p>No refusals filed yet.</p>"


def menu_table(cycles: dict[int, dict[str, Any]]) -> str:
    latest = cycles[max(cycles)]["candidates"] if cycles else []
    rows = "".join(
        f"<tr><td>{e(c['action'])}</td><td>{c['reach']}</td><td>{c['hours']:.0f}</td>"
        f"<td>{c['reach'] / c['hours']:.1f}</td><td>{e(c['excluded'] or ('reopened, closed by: ' + c['reopened'] if c.get('reopened') else 'eligible'))}</td></tr>"
        for c in latest
    )
    return f"<table><tr><th>recipe</th><th>reach</th><th>GPU h</th><th>reach / h</th><th>status</th></tr>{rows}</table>"


def ledger(verdicts: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{e(r['label'])}</td><td><span class='pill {e(r['verdict'])}'>{e(r['verdict'])}</span></td>"
        f"<td>{e(str(r['result']))}</td></tr>"
        for r in verdicts["experiments"]
    )
    return f"<table><tr><th>experiment</th><th>verdict</th><th>result</th></tr>{rows}</table>"


def main(
    out: Annotated[Path, typer.Argument(help="Directory for index.html and its videos")],
    videos: Annotated[Path, typer.Option(help="Suite rollout videos, <suite>/<kind>.mp4 (deploy/suite_cards.py)")],
    gallery: Annotated[Path | None, typer.Option(help="Preflight's gallery, for the refused scenes' photos")] = None,
    log: Path = LOG,
    requests_log: Path = REQUESTS,
) -> None:
    cycles = CycleLog(log).cycles()
    if not cycles:
        raise typer.BadParameter(f"no cycles in {log}")
    diag = diagnosis(merged_cells([Path(p) for p in cycles[max(cycles)]["refs"]]))
    out.mkdir(parents=True, exist_ok=True)
    demos = []
    for suite, kind, caption in DEMOS:
        if (src := videos / suite / kind).exists():
            shutil.copy(src, out / f"{suite}_{kind}")
            demos.append(f'<figure><video src="{suite}_{kind}" autoplay controls muted loop playsinline preload="metadata"></video>'
                         f"<figcaption>{e(caption)}</figcaption></figure>")  # fmt: skip
    requests, frames = read_jsonl(requests_log), set()
    for r in requests:
        if gallery is not None and (frame := gallery / f"{r['source']}.jpg").exists():
            shutil.copy(frame, out / frame.name)
            frames.add(r["source"])
    page = (HERE / "template.html").read_text()
    fills = {
        "LINKS": "".join(f'<a href="{u}"{" class=primary" if i == 0 else ""}>{e(k)}</a>' for i, (k, u) in enumerate(LINKS.items())),
        "STEPS": "".join(f'<details class="step"><summary><b>{i}. {e(t)}</b>{e(d)}</summary><p>{e(more)}</p>'
                         f'<p class="sofar"><b>So far</b> {e(s)}</p></details>'
                         for i, ((t, d, more), s) in enumerate(zip(STEPS, so_far(cycles, requests), strict=True), 1)),
        "BARS": bars(diag), "DEMOS": "".join(demos), "CYCLES": cycle_cards(cycles), "REQUESTS": request_cards(requests, cycles, frames), "MENU": menu_table(cycles),
        "CLOSED": ", ".join(sorted(closed_families())) or "none", "LEDGER": ledger(load_verdicts()),
        "STATUS": time.strftime("%d %b %Y %H:%M UTC", time.gmtime()),
    }  # fmt: skip
    for key, value in fills.items():
        page = page.replace(f"{{{{{key}}}}}", value)
    (out / "index.html").write_text(page)
    for name in ("README.md", "thumbnail.png"):
        shutil.copy(HERE / name, out / name)
    print(f"{len(cycles)} cycles, {len(demos)} videos -> {out / 'index.html'}")


if __name__ == "__main__":
    typer.run(main)

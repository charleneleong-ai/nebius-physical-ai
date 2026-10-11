"""Build docs/artifact.html, the one-page summary published as a Claude Artifact.

Numbers come from the benchmark JSONs on the A100 (`logs/bench_*.json`, `logs/dagger/r*/bench.json`) and the W&B runs;
update DATA at each milestone, rerun, and republish. Every chart is inline SVG drawn to one scale per chart.
"""

from pathlib import Path

import typer

SUITES = [
    "held_out_objects",
    "seen_vocabulary",
    "clean_many",
    "clean_top_down_many",
    "real_1_pomade_tin_many",
    "in_path_pomade_tin_many",
]
SCORES = {  # successes of 96 per suite, seed 42
    "v5": [52, 75, 81, 83, 74, 73],
    "v6 targeted demos": [54, 68, 76, 78, 73, 72],
    "DAgger r1": [47, 76, 82, 77, 78, 76],
    "DAgger r2": [50, 76, 86, 85, 86, 81],
    "DAgger r3": [61, 80, 86, 88, 85, 81],
    "FLUX LoRA (fair)": [3, 11, 9, 17, 6, 4],
}
PAIRED_R2 = [(12, 14, 0.85), (8, 7, 1.0), (7, 2, 0.18), (6, 4, 0.75), (14, 2, 0.004), (13, 5, 0.10)]  # (+, -, p) vs v5
FAILURE_KINDS = ["no_contact", "pushed", "dropped", "holding", "fell_off", "knocked_place"]
FAILURES = {
    "v5": [57, 34, 24, 4, 8, 11],
    "DAgger r1": [45, 40, 33, 13, 9, 0],
    "DAgger r2": [33, 31, 26, 18, 3, 1],
    "DAgger r3": [24, 32, 17, 13, 9, 0],
    "FLUX LoRA": [358, 115, 22, 24, 0, 7],
}
FUNNELS = {  # rollouts, succeeded, recoverable failures, kept, recovered
    "Round 1 (v5)": [800, 456, 330, 207, 85],
    "Round 2 (v5)": [800, 494, 296, 189, 72],
    "Round 3 (r2)": [800, 574, 197, 146, 59],
}
FUNNEL_STAGES = ["rollouts", "succeeded", "recoverable", "kept", "expert recovered"]
OPENLOOP = {
    "fair LoRA": [0.23, 1.08, 1.75, 2.71],
    "zero-shot FLUX": [0.37, 2.26, 4.34, 8.19],
    "hold pose": [0.53, 2.47, 4.66, 8.24],
}
HORIZONS = [1, 8, 16, 32]
RL_RESIDUAL = [
    (6, 0.45),
    (25, 0.50),
    (43, 0.45),
    (61, 0.54),
    (92, 0.63),
    (123, 0.60),
    (141, 0.73),
    (166, 0.59),
    (190, 0.69),
    (215, 0.71),
    (240, 0.76),
    (264, 0.72),
    (289, 0.64),
    (313, 0.79),
    (350, 0.79),
    (375, 0.74),
    (399, 0.82),
]
RL_GRASP = [
    (31, 0),
    (86, 0.04),
    (166, 0),
    (252, 0.01),
    (375, 0.03),
    (424, 0.06),
    (497, 0.01),
    (589, 0.02),
    (657, 0.05),
    (743, 0.12),
    (792, 0.06),
    (836, 0.09),
    (866, 0.06),
]
EXPERT_MIX = 0.73  # zero-residual expert on the training mix: half fresh (100%), half snapshots (45.9%)


def svg(width: int, height: int, body: str, label: str) -> str:
    return (
        f'<div class="scroll"><svg viewBox="0 0 {width} {height}" width="100%" style="min-width:520px" '
        f'role="img" aria-label="{label}">{body}</svg></div>'
    )


def pooled_bars() -> str:
    rows, x0, scale = [], 170, 0.75
    for i, (name, s) in enumerate(SCORES.items()):
        total, y = sum(s), 10 + i * 30
        cls = "b-good" if total > 438 else "b-bad" if total < 400 else "b-base"
        rows.append(
            f'<text x="{x0 - 8}" y="{y + 14}" text-anchor="end" class="lab">{name}</text>'
            f'<rect x="{x0}" y="{y}" width="{total * scale:.1f}" height="20" class="{cls}"/>'
            f'<text x="{x0 + total * scale + 6:.1f}" y="{y + 14}" class="lab">{total}</text>'
        )
    ref = x0 + 438 * scale
    rows.append(
        f'<line x1="{ref}" y1="4" x2="{ref}" y2="{14 + 30 * len(SCORES)}" class="ref"/>'
        f'<text x="{ref + 4}" y="{26 + 30 * len(SCORES)}">v5 = 438</text>'
        f'<text x="{x0 + 576 * scale}" y="{26 + 30 * len(SCORES)}" text-anchor="end">of 576</text>'
    )
    return svg(640, 40 + 30 * len(SCORES), "".join(rows), "Pooled successes of 576 per candidate")


def dumbbells() -> str:
    x0, x = 200, lambda v: 200 + (v - 40) * 6.5  # 40..96 successes
    rows = [
        f'<text x="{x(v)}" y="12" text-anchor="middle">{v}</text>'
        f'<line x1="{x(v)}" y1="18" x2="{x(v)}" y2="{24 + 34 * len(SUITES)}" class="grid"/>'
        for v in (40, 60, 80, 96)
    ]
    for i, suite in enumerate(SUITES):
        y, a, b = 40 + i * 34, SCORES["v5"][i], SCORES["DAgger r2"][i]
        plus, minus, p = PAIRED_R2[i]
        cls = "d-sig" if p < 0.05 else "d-up" if b > a else "d-down"
        rows.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" class="lab">{suite}</text>'
            f'<line x1="{x(a)}" y1="{y}" x2="{x(b)}" y2="{y}" class="{cls}" stroke-width="3"/>'
            f'<circle cx="{x(a)}" cy="{y}" r="5" class="dot-base"/><circle cx="{x(b)}" cy="{y}" r="5" class="{cls}-dot"/>'
            f'<text x="{x(max(a, b)) + 14}" y="{y + 4}">+{plus}/−{minus}  p={p}</text>'
        )
    return svg(720, 50 + 34 * len(SUITES), "".join(rows), "Per-suite successes, v5 to DAgger round 2")


def failure_stacks() -> str:
    x0, width, rows = 120, 440, []
    for i, (name, counts) in enumerate(FAILURES.items()):
        y, total, cursor = 10 + i * 34, sum(counts), x0
        rows.append(f'<text x="{x0 - 8}" y="{y + 15}" text-anchor="end" class="lab">{name}</text>')
        for k, c in enumerate(counts):
            w = width * c / total
            if w:
                rows.append(f'<rect x="{cursor:.1f}" y="{y}" width="{w:.1f}" height="22" class="f{k}"/>')
                if w > 26:
                    rows.append(
                        f'<text x="{cursor + w / 2:.1f}" y="{y + 15}" text-anchor="middle" class="on">{c}</text>'
                    )
            cursor += w
        rows.append(f'<text x="{x0 + width + 8}" y="{y + 15}" class="lab">{total} failed</text>')
    legend = "".join(
        f'<rect x="{x0 + k * 92}" y="{20 + 34 * len(FAILURES)}" width="10" height="10" class="f{k}"/>'
        f'<text x="{x0 + k * 92 + 14}" y="{29 + 34 * len(FAILURES)}">{kind}</text>'
        for k, kind in enumerate(FAILURE_KINDS)
    )
    return svg(
        680, 44 + 34 * len(FAILURES), "".join(rows) + legend, "Failure labels as a share of each checkpoint's failures"
    )


def funnels() -> str:
    col, rows = 220, []
    for j, (name, counts) in enumerate(FUNNELS.items()):
        x0 = 10 + j * col
        rows.append(f'<text x="{x0}" y="14" class="lab strong">{name}</text>')
        for k, c in enumerate(counts):
            y, w = 26 + k * 30, 190 * c / 800
            rows.append(
                f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="20" class="fn{k}"/>'
                f'<text x="{x0 + w + 4:.1f}" y="{y + 14}" class="lab">{c}</text>'
            )
    legend = "".join(
        f'<rect x="{10 + k * 128}" y="186" width="10" height="10" class="fn{k}"/>'
        f'<text x="{24 + k * 128}" y="195">{s}</text>'
        for k, s in enumerate(FUNNEL_STAGES)
    )
    return svg(670, 206, "".join(rows) + legend, "DAgger funnels for rounds 1 to 3")


def lines(
    series: dict[str, list[tuple[float, float]]],
    xmax: float,
    ymax: float,
    xlabel: str,
    ylabel: str,
    ref: tuple[float, str] | None,
    label: str,
    yfmt: str = "{:.0f}",
) -> str:
    x0, y0, w, h = 50, 180, 560, 160
    px = lambda v: x0 + w * v / xmax
    py = lambda v: y0 - h * v / ymax
    body = [f'<line x1="{x0}" y1="{y0}" x2="{x0 + w}" y2="{y0}" class="axis"/>']
    for t in (0, ymax / 2, ymax):
        body.append(
            f'<text x="{x0 - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{yfmt.format(t)}</text>'
            f'<line x1="{x0}" y1="{py(t):.1f}" x2="{x0 + w}" y2="{py(t):.1f}" class="grid"/>'
        )
    if ref:
        body.append(
            f'<line x1="{x0}" y1="{py(ref[0]):.1f}" x2="{x0 + w}" y2="{py(ref[0]):.1f}" class="ref"/>'
            f'<text x="{x0 + w}" y="{py(ref[0]) - 6:.1f}" text-anchor="end">{ref[1]}</text>'
        )
    for k, (name, pts) in enumerate(series.items()):
        path = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in pts)
        last = pts[-1]
        body.append(
            f'<polyline points="{path}" class="l{k}" fill="none" stroke-width="2.5"/>'
            f'<circle cx="{px(last[0]):.1f}" cy="{py(last[1]):.1f}" r="4" class="l{k}-dot"/>'
            f'<rect x="{x0 + 10 + k * 170}" y="196" width="12" height="4" class="l{k}-dot"/>'
            f'<text x="{x0 + 28 + k * 170}" y="201">{name}</text>'
        )
    body.append(
        f'<text x="{x0 + w}" y="218" text-anchor="end">{xlabel}</text>'
        f'<text x="{x0 - 40}" y="{y0 - h - 10}">{ylabel}</text>'
    )
    return svg(640, 226, "".join(body), label)


def openloop_chart() -> str:
    series = {name: list(zip(HORIZONS, errs, strict=True)) for name, errs in OPENLOOP.items()}
    return lines(series, 32, 9, "steps ahead", "mean |action error|", None, "Open-loop action error by horizon")


def rl_chart() -> str:
    pct = lambda pts: [(a, 100 * b) for a, b in pts]
    return lines(
        {"residual on expert": pct(RL_RESIDUAL), "shaped PPO (grasp)": pct(RL_GRASP)},
        900,
        100,
        "thousand env steps",
        "training success %",
        (100 * EXPERT_MIX, "scripted expert, same mix: 73%"),
        "RL expert training success",
    )


def build(template: Path, out: Path) -> None:
    html = template.read_text()
    for key, chart in {
        "POOLED": pooled_bars,
        "DUMBBELLS": dumbbells,
        "FAILURES": failure_stacks,
        "FUNNELS": funnels,
        "OPENLOOP": openloop_chart,
        "RL": rl_chart,
    }.items():
        html = html.replace(f"<!--{key}-->", chart())
    out.write_text(html)


def main(template: Path = Path("docs/artifact.template.html"), out: Path = Path("docs/artifact.html")) -> None:
    build(template, out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)

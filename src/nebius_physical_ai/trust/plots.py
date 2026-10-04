from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebius_physical_ai.trust.stress import StressReport
from nebius_physical_ai.trust.verdict import TrustPolicy, Verdict

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2; validated light-mode pair
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


class StressPlots:
    """PNG figures for PRs and the README; light surface, recessive axes, identity never by colour alone."""

    def __init__(self, report: StressReport, out_dir: Path) -> None:
        self.report, self.out_dir = report, out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, name: str) -> Path:
        path = self.out_dir / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path

    def predicted_vs_true(self) -> Path:
        cases, verdicts = self.report.cases, [a.verdict for a in self.report.assess(calibrated=False)]
        p = np.array([c.prediction.result.p for c in cases])
        true = np.array([c.true_rate for c in cases])
        raw = self.report.calibration()["raw"]
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
            ax.text(0.62, 0.55, "perfect prediction", color=MUTED, ha="center", fontsize=8, rotation=41)
            for verdict, color, marker in ((Verdict.GO, BLUE, "o"), (Verdict.RETRY, ORANGE, "^")):
                mask = np.array([v is verdict for v in verdicts])
                ax.scatter(
                    p[mask],
                    true[mask],
                    s=64,
                    color=color,
                    marker=marker,
                    edgecolor=SURFACE,
                    lw=2,
                    zorder=3,
                    label=f"{verdict.value} ({mask.sum()} scenes)",
                )
            ax.set(
                xlim=(-0.03, 1.03),
                ylim=(-0.05, 1.05),
                xlabel="Preflight: rehearsed success rate (32 variants)",
                ylabel="Hidden world: real success rate (8 attempts)",
            )
            ax.set_title(
                f"Predicted vs real success, 20 scenes\ncorrelation {np.corrcoef(p, true)[0, 1]:.2f} · "
                f"Brier {raw.brier:.3f} · ECE {raw.ece:.3f}",
                loc="left",
                fontsize=10,
            )
            ax.legend(frameon=False, loc="upper left")
            return self.save(fig, "predicted_vs_true")

    def coverage_curve(self) -> Path:
        """Success rate of executed attempts vs fraction executed, as the GO threshold on the interval's lower bound moves."""
        cases = self.report.cases
        lows = np.array([c.prediction.result.wilson()[0] for c in cases])
        true = np.array([c.true_rate for c in cases])
        points = [(float((lows >= t).mean()), float(true[lows >= t].mean())) for t in np.unique(lows)[::-1]]
        chosen_go = lows >= TrustPolicy().go_lower
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            ax.plot(*zip(*points), color=BLUE, lw=2, marker="o", ms=4, zorder=2)
            ax.axhline(true.mean(), color=MUTED, lw=1, ls="--")
            ax.text(0.02, true.mean() + 0.015, f"execute everything: {true.mean():.0%}", color=MUTED, fontsize=8)
            ax.scatter(
                [chosen_go.mean()], [true[chosen_go].mean()], s=110, color=BLUE, edgecolor=SURFACE, lw=2, zorder=3
            )
            ax.annotate(
                f"GO if lower bound ≥ {TrustPolicy().go_lower}\n{true[chosen_go].mean():.0%} success on {chosen_go.sum()} scenes",
                (chosen_go.mean(), true[chosen_go].mean()),
                xytext=(0.22, 0.4),
                textcoords="data",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
                fontsize=8,
                color=INK,
            )
            ax.set(
                xlim=(0, 1.02),
                ylim=(0, 1.05),
                xlabel="Fraction of scenes executed",
                ylabel="Real success rate of executed scenes",
            )
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title("Trading coverage for reliability", loc="left", fontsize=10)
            return self.save(fig, "coverage_curve")

    @staticmethod
    def sweep_distribution(rows: list[dict[str, Any]], out_dir: Path) -> Path:
        """Per-scene rehearsed p with 95% interval, sorted; ordinary vs hard layouts."""
        rows = sorted(rows, key=lambda r: r["p"])
        x = np.arange(len(rows))
        p = np.array([r["p"] for r in rows])
        lo, hi = np.array([r["ci95"] for r in rows]).T
        hard = np.array([r.get("hard", r.get("kind") == "hard") for r in rows])
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(7, 3.6))
            for mask, color, marker, label in (
                (~hard, BLUE, "o", "ordinary layout"),
                (hard, ORANGE, "^", "hard layout"),
            ):
                ax.errorbar(
                    x[mask],
                    p[mask],
                    yerr=[p[mask] - lo[mask], hi[mask] - p[mask]],
                    fmt=marker,
                    color=color,
                    ms=7,
                    mec=SURFACE,
                    mew=1.5,
                    elinewidth=1.5,
                    capsize=0,
                    label=label,
                )
            margin = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
            go = TrustPolicy().go_lower
            ax.axhline(go, color=MUTED, lw=0.8, ls=":")
            ax.text(
                1.01,
                go,
                "GO needs the\nlower bound\nabove this",
                color=MUTED,
                fontsize=8,
                va="center",
                transform=margin,
            )
            ax.set(
                ylim=(-0.03, 1.05),
                xticks=[],
                xlabel="Scenes, sorted by predicted success",
                ylabel="Rehearsed success rate",
            )
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title(
                "Rehearsal spreads scenes across the verdict range (20 scenes × 32 variants, 95% intervals)",
                loc="left",
                fontsize=10,
            )
            ax.legend(frameon=False, loc="upper left")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "sweep_distribution.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            return path

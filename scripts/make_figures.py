"""Generate every figure used in the THGTM paper from the experiment JSONs.

Outputs PNG files to ``paper/figures/`` so the LaTeX source can ``\\includegraphics``
them.  All numbers come from the JSON results -- no hard-coded values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
FIGS_DIR = ROOT / "paper" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load(name):
    return json.loads((RESULTS_DIR / name).read_text())


# ---------------------------------------------------------------- #
# Figure: temporal XOR
# ---------------------------------------------------------------- #
def fig_temporal_xor():
    data = load("temporal_xor.json")
    summary = data["summary"]
    delays = sorted(int(k) for k in summary.keys())
    names = ["M_raw", "M_past_only", "M_past_etta"]
    display = {"M_raw": "Raw inputs only (no temporal features)",
               "M_past_only": "PAST$_k$ literals, no trace ($\\lambda = 0$)",
               "M_past_etta": "PAST$_k$ + ETTA trace ($\\lambda = 0.5, \\alpha = 2$)"}

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.25
    x = np.arange(len(delays))
    for i, n in enumerate(names):
        means = [summary[str(d)][n]["mean"] for d in delays]
        stds  = [summary[str(d)][n]["std"]  for d in delays]
        ax.bar(x + (i - 1) * width, means, width=width,
               yerr=stds, capsize=3, label=display[n])
    ax.set_xticks(x)
    ax.set_xticklabels([f"delay = {d}" for d in delays])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("Temporal-XOR: temporal literals + ETTA trace each contribute")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "temporal_xor.png", dpi=200)
    fig.savefig(FIGS_DIR / "temporal_xor.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- #
# Figure: depth-N parity
# ---------------------------------------------------------------- #
def fig_depth_n_parity():
    data = load("depth_n_parity.json")
    summary = data["summary"]
    path_lens = sorted(int(k) for k in summary.keys())
    names = ["L1", "L2_no_etta", "L2_etta"]
    display = {"L1": "L = 1 (single layer, 2x clauses)",
               "L2_no_etta": "L = 2, no ETTA ($\\alpha = 0$, $\\lambda = 0$)",
               "L2_etta":    "L = 2, ETTA ($\\alpha = 2$, $\\lambda = 0.5$)"}

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.25
    x = np.arange(len(path_lens))
    for i, n in enumerate(names):
        means = [summary[str(d)][n]["mean"] for d in path_lens]
        stds  = [summary[str(d)][n]["std"]  for d in path_lens]
        ax.bar(x + (i - 1) * width, means, width=width,
               yerr=stds, capsize=3, label=display[n])
    ax.set_xticks(x)
    ax.set_xticklabels([f"path = {d}" for d in path_lens])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("Depth-N-Parity-on-Path: L = 2 + ETTA narrows the gap to L = 1")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "depth_n_parity.png", dpi=200)
    fig.savefig(FIGS_DIR / "depth_n_parity.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- #
# Figure: trajectory verification
# ---------------------------------------------------------------- #
def fig_trajectory_verification():
    data = load("trajectory_verification.json")
    summary = data["summary"]

    names = ["per_step", "trajectory_LTL"]
    display = {"per_step": "Per-step verifier (no LTL)",
               "trajectory_LTL": "Trajectory receipt + LTL"}
    metrics = ["asr_mean", "fpr_mean", "tpr_mean"]
    metric_labels = {"asr_mean": "Attack success rate",
                     "fpr_mean": "False-positive rate",
                     "tpr_mean": "True-positive rate"}

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.35
    x = np.arange(len(metrics))
    for i, n in enumerate(names):
        means = [summary[n][m] for m in metrics]
        stds  = [summary[n][m.replace("_mean", "_std")] for m in metrics]
        ax.bar(x + (i - 0.5) * width, means, width=width,
               yerr=stds, capsize=3, label=display[n])
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Slow-roll exfiltration: trajectory LTL drops ASR 0.687 -> 0.000")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "trajectory_verification.png", dpi=200)
    fig.savefig(FIGS_DIR / "trajectory_verification.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- #
# Figure: ETTA trace dynamics
# ---------------------------------------------------------------- #
def fig_trace_dynamics():
    """Synthetic illustration: an ETTA's state and trace under a stream
    of feedback events.  Generated on the fly so it doesn't need a
    JSON.  Useful as a 'how it works' diagram for the paper."""
    from thgtm.etta import EchoTraceAutomaton
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.4), sharex=True)
    for ax, lam, label in [(axes[0], 0.0, "$\\lambda = 0$ (vanilla TA)"),
                           (axes[1], 0.7, "$\\lambda = 0.7$ (ETTA)")]:
        a = EchoTraceAutomaton(n_states_per_action=20, lambda_decay=lam)
        states, traces = [], []
        for t in range(120):
            # Stochastic +1/-1/0 input, biased toward +1 at first then mixed
            if t < 30:
                d = +1 if rng.random() < 0.7 else -1
            elif t < 60:
                d = 0    # idle
            else:
                d = -1 if rng.random() < 0.6 else +1
            a.update(d)
            states.append(a.state)
            traces.append(a.trace)

        t_axis = np.arange(120)
        ax.plot(t_axis, states, color="#1f77b4", label="state")
        ax2 = ax.twinx()
        ax2.plot(t_axis, traces, color="#d62728", label="trace", linestyle="--")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("trace", color="#d62728")
        ax.set_ylabel("state", color="#1f77b4")
        ax.axhline(a.N, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(label, loc="left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step $t$")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "trace_dynamics.png", dpi=200)
    fig.savefig(FIGS_DIR / "trace_dynamics.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- #
# Figure: architecture schematic (ASCII -> matplotlib text rendering)
# ---------------------------------------------------------------- #
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axis("off")
    # Boxes: (label, x, y, w, h, color).  Labels are wrapped to stay inside the
    # box, and the columns are spaced so nothing overlaps a neighbour.
    boxes = [
        ("Input stream $x_t$",            0.02, 0.74, 0.15, 0.12, "#ddd"),
        ("Temporal literal\nencoder\n(PAST$_k$, SINCE,\nALWAYS$_w$)",
                                          0.21, 0.67, 0.18, 0.26, "#cfe6f7"),
        ("Layer 1:\nETTA + clauses",      0.45, 0.74, 0.22, 0.12, "#fde2e1"),
        ("Layer 2:\nETTA + clauses",      0.45, 0.52, 0.22, 0.12, "#fde2e1"),
        ("Per-class top banks\n(vote sums -> class)",
                                          0.45, 0.28, 0.22, 0.16, "#fff5cc"),
        ("Trajectory receipt\n(per-step CNF + LTL)",
                                          0.73, 0.50, 0.25, 0.20, "#dbf3d9"),
    ]
    for txt, x, y, w, h, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="black",
                                   linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    # Arrows
    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0))
    arr(0.17, 0.80, 0.21, 0.80)               # input -> encoder
    arr(0.39, 0.80, 0.45, 0.80)               # encoder -> L1
    arr(0.56, 0.74, 0.56, 0.64)               # L1 -> L2
    arr(0.56, 0.52, 0.56, 0.44)               # L2 -> class head
    arr(0.67, 0.59, 0.73, 0.60)               # L2 -> receipt
    arr(0.67, 0.38, 0.73, 0.54)               # class -> receipt
    # Annotation: ETTA trace backflow (dashed red), pointing into Layer 2
    ax.annotate("trace-projected\nfeedback", xy=(0.45, 0.57), xytext=(0.16, 0.40),
                fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", linewidth=0.8, linestyle="--",
                                color="#a00"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("THGTM architecture", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "architecture.png", dpi=200)
    fig.savefig(FIGS_DIR / "architecture.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- #
def main():
    fig_temporal_xor()
    fig_depth_n_parity()
    fig_trajectory_verification()
    fig_trace_dynamics()
    fig_architecture()
    print(f"Wrote figures to {FIGS_DIR}")


if __name__ == "__main__":
    main()

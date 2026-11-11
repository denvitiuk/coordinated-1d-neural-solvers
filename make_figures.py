"""
Рисунки для статьи и README (PNG + PDF в results/figures/).

    python make_figures.py

Нужны веса и данные из опытов 4–6 (results/*.pt). Числа для рис. 2 читаются из логов
results/nn_coordination_*.log, для рис. 4 — взяты из логов (указаны в коде).
Подписи — на английском (рисунки идут в статью).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import nn_coordination_grids as ng  # noqa: E402

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Палитра (проверена валидатором: 3 категориальных слота, все пары; бирюзовый < 3:1 —
# поэтому у всех серий есть прямые подписи и разные маркеры).
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a6a5", "#e34948", "#a8302f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   {OUT / name}.png / .pdf")


def fig_section():
    """Рис. 1: разрез атмосферы при сильной химии — эталон, сеть, разница."""
    d = torch.load("results/nn_column_chem_section_300.pt")
    ref, phi = d["phi_ref"].numpy(), d["phi"].numpy()
    diff = phi - ref
    err = float(((d["phi"] - d["phi_ref"]).norm() / d["phi_ref"].norm()))
    vmax = float(ref.max())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    ext = (0, 1, 0, 1)
    for ax, field, title in ((axes[0], ref, "Reference: Newton column"),
                             (axes[1], phi, "Learned column")):
        im = ax.imshow(field, origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(title, loc="left")
        ax.contour(field, levels=[0.05, 0.1, 0.2, 0.4], origin="lower", extent=ext, colors="white",
                   linewidths=0.6, alpha=0.8)
    cb = fig.colorbar(im, ax=axes[:2], shrink=0.85, pad=0.01)
    cb.set_label("concentration φ")
    lim = float(abs(diff).max())
    im2 = axes[2].imshow(diff, origin="lower", extent=ext, cmap=DIV,
                         norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="equal")
    axes[2].set_title(f"Difference (error {err:.1%})", loc="left")
    cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.85, pad=0.02)
    cb2.set_label("network − reference")
    for ax in axes:
        ax.set_xlabel("x (horizontal)")
        ax.grid(False)
    axes[0].set_ylabel("z (height)")
    axes[0].annotate("stack", xy=(0.3, 0.08), xytext=(0.42, 0.32), color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.suptitle("Air pollution with strong chemistry (54 % of the balance), 100 × 100 grid",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "fig1_pollution_section")


def log_means(path):
    """Средние ошибки по сеткам из лога («средняя 0.56%» в v2 или «средняя 1.23e-05» в v3)."""
    out = []
    for m in re.finditer(r"средняя ([0-9.e+-]+)(%?)", Path(path).read_text()):
        v = float(m.group(1))
        out.append(v if m.group(2) else 100 * v)      # всё в процентах
    return out


def fig_grids():
    """Рис. 2: ошибка не растёт с сеткой (опыты 4 и 5), сети v2 и v3; числа — из логов."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    n2 = [48 ** 2, 98 ** 2, 198 ** 2]
    n3 = [24 ** 3, 48 ** 3, 96 ** 3]
    series = (
        (n2, "results/nn_coordination_grids.log", BLUE, "o", "--", "2D, v2"),
        (n3, "results/nn_coordination_3d.log", ORANGE, "s", "--", "3D, v2"),
        (n2, "results/nn_coordination_grids_v3.log", BLUE, "o", "-", "2D, v3 (residual refinement)"),
        (n3, "results/nn_coordination_3d_v3.log", ORANGE, "s", "-", "3D, v3 (residual refinement)"),
    )
    for n, log, color, mk, ls, label in series:
        e = log_means(log)[-3:]
        hollow = ls == "--"
        ax.plot(n, e, color=color, lw=2, ls=ls, marker=mk, ms=8, mew=2,
                mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE, label=label)
        ax.annotate(label.split(" (")[0], (n[-1], e[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    e2v2, e3v3 = log_means(series[0][1])[-3:], log_means(series[3][1])[-3:]
    for x, y, lab in zip(n2, e2v2, ("50²", "100²", "200²")):
        ax.annotate(lab, (x, y), xytext=(0, 10), textcoords="offset points", ha="center", color=INK2, fontsize=8)
    for x, y, lab in zip(n3, e3v3, ("24³", "48³", "96³")):
        ax.annotate(lab, (x, y), xytext=(0, -16), textcoords="offset points", ha="center", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.2e3, 3e6)
    ax.set_ylim(5e-5, 3)
    ax.set_xlabel("number of unknowns")
    ax.set_ylabel("mean relative error of φ, %")
    ax.set_title("One 1D network (lines of 24–96 nodes), no retraining", loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=0, marker="o", ms=8, label="2D, 9 problems per grid"),
               Line2D([], [], color=ORANGE, lw=0, marker="s", ms=8, label="3D, 6 problems per grid"),
               Line2D([], [], color=INK2, lw=2, ls="--", label="v2: hybrid network"),
               Line2D([], [], color=INK2, lw=2, ls="-", label="v3: + residual refinement")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.76), fontsize=9, handlelength=2.6)
    save(fig, "fig2_grid_independence")


def fig_frequency():
    """Рис. 3: урок про частоты — FNO против гибрида на sin kπx."""
    n = 48
    gen = torch.Generator().manual_seed(77)
    c, _, _, _ = ng.sample(256, n, gen)
    x = torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)
    z = torch.zeros(256, dtype=torch.float64)
    nets = {}
    fno = ng.LineFNO()
    fno.load_state_dict(torch.load("results/nn_line_fno.pt"))
    nets["Sine-mode FNO (v1)"] = (fno.float().eval(), torch.float32)
    hyb = ng.LineHybrid()
    hyb.load_state_dict(torch.load("results/nn_line_hybrid.pt"))
    nets["Hybrid: prior + network (v2)"] = (hyb.double().eval(), torch.float64)
    ks = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for (name, (net, dt)), color, mk in zip(nets.items(), (ORANGE, BLUE), ("s", "o")):
        errs = []
        for k in ks:
            f = torch.sin(math.pi * k * x).expand(256, n).clone()
            with torch.no_grad():
                w = net.zero_bc(c.to(dt), f.to(dt)).double()
            errs.append(100 * ng.rel(w, ng.exact_line(c, f, z, z)).mean().item())
        ax.plot(ks, errs, color=color, lw=2, marker=mk, ms=4, markevery=6, label=name)
        ax.annotate(name.split(" (")[0], (ks[-1], errs[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("frequency k of the source f = sin kπx")
    ax.set_ylabel("error of the 1D network, %")
    ax.set_title("Coordination needs accuracy at every frequency", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig3_frequency_lesson")


def fig_amplification():
    """Рис. 4: ошибка решения ≈ ошибке подзадачи (опыты 1, 2, 6)."""
    # опыт 2, запуск 2: ε сети на своих входах -> ошибка φ (results/nn_coordination_run2_boundary.log)
    e2 = ([31, 25, 14, 8.88, 0.91], [28.3, 23.0, 14.8, 8.93, 0.77])
    # опыт 6, v2a: ошибка сети на истинных колонках -> ошибка φ, 50×50 (σ₀ = 5, 50, 300)
    e6 = ([3.79, 12.10, 22.27], [1.93, 9.98, 21.24])
    # опыт 1: искусственная ошибка ε = 0.1, 1, 10 % (усиление 0.82 шум / 0.28 гладкая), 400×400
    e1x = [0.1, 1, 10, 0.1, 1, 10]
    e1y = [0.082, 0.82, 8.33, 0.028, 0.28, 2.73]
    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
    line = [0.05, 50]
    ax.plot(line, line, color=INK2, lw=1, ls="--", label="error of φ = subproblem error")
    ax.scatter(e1x, e1y, s=55, color=AQUA, marker="^", edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="exp. 1: synthetic perturbation")
    ax.scatter(*e2, s=55, color=BLUE, marker="o", edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="exp. 2: 1D network, diffusion with k(x, y)")
    ax.scatter(*e6, s=55, color=ORANGE, marker="s", edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="exp. 6: chemistry column")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 50)
    ax.set_ylim(0.01, 50)
    ax.set_xlabel("subproblem (1D solver) error, %")
    ax.set_ylabel("error of the coordinated solution φ, %")
    ax.set_title("Coordination does not amplify subproblem errors", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig4_error_amplification")


if __name__ == "__main__":
    print("Рисунки:")
    fig_section()
    fig_grids()
    fig_frequency()
    fig_amplification()

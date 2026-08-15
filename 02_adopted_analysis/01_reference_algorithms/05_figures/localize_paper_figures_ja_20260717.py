from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figs"
RES = ROOT / "results"


def japanese_font() -> FontProperties:
    candidates = [
        ROOT / "fonts/NotoSansCJKjp-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf"),
    ]
    for path in candidates:
        if path.exists():
            mpl.font_manager.fontManager.addfont(path)
            prop = FontProperties(fname=path)
            mpl.rcParams["font.family"] = prop.get_name()
            mpl.rcParams["axes.unicode_minus"] = False
            return prop
    raise FileNotFoundError("Noto Sans CJK JPフォントが見つかりません。")


JP = japanese_font()


def _cdf(values: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)
    probability = mass[order] / mass.sum()
    return values[order], np.cumsum(probability)


def make_baseline_figures() -> None:
    data = np.load(RES / "monthly_D0_policy_arrays.npz")
    t = data["decision_times"]
    strategies = [
        ("PCMV", data["pcmv_glide"], data["xg_pc"], data["pcmv_pmf"][-1]),
        ("DOMV", data["domv_glide"], data["xg_pc"], data["domv_pmf"][-1]),
        ("cTCMV", data["ctcmv_glide"], data["xg_tc"], data["ctcmv_pmf"][-1]),
        ("dTCMV", data["dtcmv_glide"], data["xg_tc"], data["dtcmv_pmf"][-1]),
        ("CP", data["cp_glide"], data["xg_tc"], data["cp_pmf"][-1]),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for name, glide, _, _ in strategies:
        ax.plot(t, glide, label=name, linewidth=1.8)
    ax.set_xlabel("加入後経過年数")
    ax.set_ylabel("確率質量加重リスク資産比率")
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_all_strategies_glidepaths_D0_N480.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for name, _, values, mass in strategies:
        x, y = _cdf(values, mass)
        ax.plot(x, y, label=name, linewidth=1.8)
    ax.set_xlabel("終端DC富")
    ax.set_ylabel("累積確率")
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_all_strategies_terminal_cdf_D0_N480.png", dpi=180)
    plt.close(fig)


def _weighted_quantile(grid: np.ndarray, mass: np.ndarray, q: float) -> float:
    cdf = np.cumsum(mass / mass.sum())
    return float(np.interp(q, cdf, grid))


def make_heatmap(policy_key: str, pmf_key: str, name: str, output: str) -> None:
    data = np.load(RES / "monthly_D0_policy_arrays.npz")
    t = data["decision_times"]
    grid = data["xg_tc"]
    policy = data[policy_key]
    pmf = data[pmf_key][:-1]
    lower = np.array([_weighted_quantile(grid, row, 0.005) for row in pmf])
    upper = np.array([_weighted_quantile(grid, row, 0.995) for row in pmf])
    image = policy.T.copy()
    for n in range(len(t)):
        image[(grid < lower[n]) | (grid > upper[n]), n] = np.nan

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    mesh = ax.pcolormesh(t, grid, image, shading="auto", vmin=0, vmax=1, cmap="viridis")
    ax.plot(t, lower, linewidth=1.2, label="0.5%包絡")
    ax.plot(t, upper, linewidth=1.2, label="99.5%包絡")
    ax.set_title(f"到達可能状態における{name}リスク資産比率")
    ax.set_xlabel("加入後経過年数")
    ax.set_ylabel("DC残高")
    ax.set_ylim(0, 160)
    ax.legend(loc="upper left")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("リスク資産比率")
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.09, right=0.90)
    fig.savefig(FIG / output, dpi=180)
    plt.close(fig)


def make_common_state_figure() -> None:
    data = pd.read_csv(RES / "selected_common_state_rolling_D0_N480.csv")
    selected = [
        (30, "q50", "30年・中央値状態"),
        (30, "q90", "30年・高残高状態"),
        (35, "q10", "35年・低残高状態"),
    ]
    colors = {"cTCMV": "#4c9f45", "dTCMV": "#f28e2b"}
    ypos = {"cTCMV": 1, "dTCMV": 0}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharex=True)
    for ax, (year, state, label) in zip(axes, selected):
        part = data[(data["year"] == year) & (data["state"] == state)]
        xstate = float(part["starting_x"].iloc[0])
        for strategy in ["cTCMV", "dTCMV"]:
            row = part[part["strategy"] == strategy].iloc[0]
            y = ypos[strategy]
            ax.hlines(y, row["q05"], row["q95"], color=colors[strategy], linewidth=2.3)
            ax.vlines([row["q05"], row["q95"]], y - 0.10, y + 0.10, color=colors[strategy], linewidth=1.3)
            ax.scatter(row["mean"], y, s=34, facecolor="white", edgecolor="black", zorder=3)
            ax.text(row["mean"], y + 0.18, f'{row["mean"]:.1f}', ha="center", va="bottom", fontsize=8)
            ax.text(row["q05"], y - 0.18, f'{row["q05"]:.1f}', ha="center", va="top", fontsize=7)
            ax.text(row["q95"], y - 0.18, f'{row["q95"]:.1f}', ha="center", va="top", fontsize=7)
        ax.set_yticks([1, 0], ["cTCMV", "dTCMV"])
        ax.set_title(f"{label}\n($x={xstate:.2f}$)", fontsize=9.5)
        ax.set_xlabel("条件付き終端富")
        ax.grid(axis="x", alpha=0.25)
        ax.set_ylim(-0.55, 1.55)
    xmin = float(data["q05"].min()) - 5
    xmax = float(data["q95"].max()) + 8
    for ax in axes:
        ax.set_xlim(xmin, xmax)
    fig.suptitle("共通状態ローリング比較：q05--q95区間と平均", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "fig_common_state_rolling_selected_D0_N480.png", dpi=220)
    plt.close(fig)


def localize_strict_clip_figure() -> None:
    target = FIG / "fig_strict_vs_clipped_glidepaths_by_strategy_N480.png"
    source = FIG / "fig_strict_vs_clipped_glidepaths_by_strategy_N480_source_en.png"
    if not source.exists():
        shutil.copyfile(target, source)
    original = Image.open(source).convert("RGB")
    w, h = original.size
    # Plot rectangles in the original 1890 x 1475 raster. Fractions preserve
    # the construction if an otherwise identical raster is exported at a
    # different resolution.
    rectangles = [
        (118 / 1890, 154 / 1475, 933 / 1890, 712 / 1475),
        (1050 / 1890, 154 / 1475, 1864 / 1890, 712 / 1475),
        (118 / 1890, 818 / 1475, 933 / 1890, 1374 / 1475),
        (1050 / 1890, 818 / 1475, 1864 / 1890, 1374 / 1475),
    ]
    panels = [original.crop((int(x0*w), int(y0*h), int(x1*w), int(y1*h)))
              for x0, y0, x1, y1 in rectangles]

    def calibrated_extent(panel: Image.Image) -> tuple[float, float, float, float]:
        array = np.asarray(panel)
        green = ((array[:, :, 1] > 100)
                 & (array[:, :, 1] > array[:, :, 0] * 1.2)
                 & (array[:, :, 1] > array[:, :, 2] * 1.2))
        orange = ((array[:, :, 0] > 200) & (array[:, :, 1] > 60)
                  & (array[:, :, 1] < 180) & (array[:, :, 2] < 80))
        orange[:, int(panel.width * 0.28):] = False
        y_green = float(np.median(np.where(green)[0]))
        # The initial flat segment equals one. Use its uppermost orange pixels;
        # the dTCMV curve leaves the plateau early, so a median would be biased.
        y_one = float(np.percentile(np.where(orange)[0], 5))
        y_range = 0.4 * panel.height / (y_green - y_one)
        y_top = 1.0 + y_one * y_range / panel.height
        y_bottom = y_top - y_range
        return -2.0, 42.0, y_bottom, y_top

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7))
    for ax, panel, title in zip(axes.flat, panels, ["PCMV", "DOMV", "cTCMV", "dTCMV"]):
        extent = calibrated_extent(panel)
        ax.imshow(panel, extent=extent, aspect="auto")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("加入後経過年数")
        ax.set_ylabel("確率質量加重リスク資産比率")
        ax.set_xlim(-2, 42)
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks(np.arange(0, 41, 10))
        lower_tick = np.ceil(extent[2] * 10) / 10
        ax.set_yticks(np.arange(lower_tick, 1.01, 0.1))
    handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=2.4, label="厳密制約解"),
        Line2D([0], [0], color="#ff7f0e", linewidth=2.4, label="無制約クリップ近似"),
        Line2D([0], [0], color="#2ca02c", linewidth=2.4, linestyle="--", label="CP 60%ベンチマーク"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=3,
               frameon=False, fontsize=15, handlelength=2.0, columnspacing=2.0)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.91, wspace=0.16, hspace=0.24)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def _box(ax, x, y, w, h, edge, face, title, subtitle="", title_size=13.5, subtitle_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                linewidth=1.6, edgecolor=edge, facecolor=face))
    ax.text(x + 0.022, y + h * 0.64, title, ha="left", va="center", fontsize=title_size,
            weight="bold", color="#172033")
    if subtitle:
        ax.text(x + 0.022, y + h * 0.30, subtitle, ha="left", va="center",
                fontsize=subtitle_size, color="#354052")


def make_architecture_figure() -> None:
    png = FIG / "fig_three_layer_mvs_replacement.png"
    pdf = FIG / "fig_three_layer_mvs_replacement.pdf"
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    mv_edge, mv_face = "#4267a9", "#eef4ff"
    mvs_edge, mvs_face = "#d97706", "#fff5e8"
    neutral_edge, neutral_face = "#596579", "#f7f8fa"
    ax.text(0.32, 0.973, "MV構造", ha="center", va="center", fontsize=14, weight="bold", color=mv_edge)
    ax.text(0.82, 0.973, "選択可能なMVS置換", ha="center", va="center", fontsize=14, weight="bold", color=mvs_edge)
    left_x, left_w, right_x, right_w, height = 0.045, 0.555, 0.705, 0.255, 0.145
    ys = [0.785, 0.575, 0.365, 0.155]
    _box(ax, left_x, ys[0], left_w, height, neutral_edge, neutral_face,
         "設計基準：PCMV", "目標，効率フロンティア，下方リスク予算")
    _box(ax, left_x, ys[1], left_w, height, mv_edge, mv_face,
         "第1層・標準デフォルト：cTCMV", "安定したMV均衡を標準運用として維持")
    _box(ax, left_x, ys[2], left_w, height, mv_edge, mv_face,
         "第2層・定期見直し：厳密制約DOMV（MV）", "見直し時点で残存期間を再最適化")
    _box(ax, right_x, ys[2], right_w, height, mvs_edge, mvs_face,
         "DOMV--MVS", "統制可能な場合にMVを置換", title_size=14.5, subtitle_size=9.0)
    _box(ax, left_x, ys[3], left_w, height, mv_edge, mv_face,
         "第3層・個別化オーバーレイ：厳密制約dTCMV（MV）", "総年金富から個別リスク予算を設定")
    _box(ax, right_x, ys[3], right_w, height, mvs_edge, mvs_face,
         "dTCMV--MVS", "統制可能な場合にMVを置換", title_size=14.5, subtitle_size=9.0)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((left_x + left_w/2, y0 - 0.012),
                                     (left_x + left_w/2, y1 + height + 0.012),
                                     arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#536071"))
    for y in (ys[2], ys[3]):
        ax.add_patch(FancyArrowPatch((left_x + left_w + 0.018, y + height/2),
                                     (right_x - 0.018, y + height/2),
                                     arrowstyle="<->", mutation_scale=14, linewidth=1.8, color=mvs_edge))
    ax.text(0.82, 0.765, "第四層ではない\n説明可能な較正，非凹性の管理，\n下方リスク限度を満たす場合のみ使用",
            ha="center", va="center", fontsize=10.0, color="#6b4c20",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffaf2", edgecolor="#e5b46a", linewidth=1.2))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.025)
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_baseline_figures()
    make_heatmap("ctcmv_policy", "ctcmv_pmf", "cTCMV", "policy_heatmap_cTCMV_N480_reachable.png")
    make_heatmap("dtcmv_policy", "dtcmv_pmf", "dTCMV", "policy_heatmap_dTCMV_N480_reachable.png")
    make_common_state_figure()
    localize_strict_clip_figure()
    make_architecture_figure()
    print("論文図を日本語化しました。")

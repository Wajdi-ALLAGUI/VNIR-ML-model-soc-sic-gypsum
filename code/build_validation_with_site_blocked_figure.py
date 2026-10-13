"""Build a four-validation RF comparison including site-blocked GroupKFold."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from package_paths import output_root

ROOT = output_root()
PREDICTIONS = ROOT / "diagnostics" / "predictions_validation_all.csv"
OUT = ROOT / "revision_submission" / "figures"
STEM = "Figure8_Validation_Including_SiteBlocked"

TARGETS = ("SOC", "SIC", "Gypsum")
PATHS = ("A_Global", "B1_Management", "B2_Topography")
VALIDATIONS = ("GroupKFold_Plot", "GroupKFold_Oasis", "KFold", "LOOCV")
PATH_LABELS = {
    "A_Global": "Global",
    "B1_Management": "Management stratified",
    "B2_Topography": "Topography stratified",
}
VALIDATION_LABELS = {
    "GroupKFold_Plot": "Plot-blocked\nGroupKFold",
    "GroupKFold_Oasis": "Site-blocked\nGroupKFold",
    "KFold": "5-fold\nKFold",
    "LOOCV": "LOOCV",
}
COLORS = {
    "A_Global": "#3566A7",
    "B1_Management": "#2E8B57",
    "B2_Topography": "#D97A24",
}
EDGES = {
    "A_Global": "#1F3E73",
    "B1_Management": "#1F6840",
    "B2_Topography": "#8E4C16",
}


def pooled_metrics(group: pd.DataFrame) -> pd.Series:
    observed = group["Observed"].to_numpy(dtype=float)
    predicted = group["Predicted"].to_numpy(dtype=float)
    residual = observed - predicted
    denominator = np.sum((observed - observed.mean()) ** 2)
    return pd.Series(
        {
            "R2": 1.0 - np.sum(residual**2) / denominator,
            "RMSE": np.sqrt(np.mean(residual**2)),
        }
    )


def load_best_rf_results() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS)
    selected = predictions[
        (predictions["Scenario"] == "CR")
        & (predictions["ModelFamily"] == "RF")
        & predictions["Path"].isin(PATHS)
        & predictions["Validation"].isin(VALIDATIONS)
        & predictions["Target"].isin(TARGETS)
    ].copy()
    pooled = (
        selected.groupby(
            ["Path", "Validation", "Mode", "Target"], as_index=False
        )
        .apply(pooled_metrics, include_groups=False)
        .reset_index(drop=True)
    )
    best = (
        pooled.sort_values(
            ["Path", "Validation", "Target", "R2", "RMSE"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["Path", "Validation", "Target"], as_index=False)
        .first()
    )
    expected = len(PATHS) * len(VALIDATIONS) * len(TARGETS)
    if len(best) != expected:
        raise ValueError(f"Expected {expected} comparison rows, found {len(best)}.")
    return best


def build_figure(best: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "text.color": "#000000",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    width_in = 185.0 / 25.4
    fig = plt.figure(figsize=(width_in, 8.8))
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=(4.0, 0.92, 4.0, 0.92, 4.0, 0.92),
        hspace=0.18,
    )
    axes: list[plt.Axes] = []
    delta_axes: list[plt.Axes] = []
    for panel_index in range(3):
        if panel_index == 0:
            axis = fig.add_subplot(grid[panel_index * 2, 0])
        else:
            axis = fig.add_subplot(
                grid[panel_index * 2, 0], sharex=axes[0], sharey=axes[0]
            )
        delta_axis = fig.add_subplot(grid[panel_index * 2 + 1, 0], sharex=axis)
        axes.append(axis)
        delta_axes.append(delta_axis)

    x = np.arange(len(VALIDATIONS), dtype=float)
    bar_width = 0.22
    offsets = (-0.27, 0.0, 0.27)

    for panel_index, (ax, delta_ax, target) in enumerate(
        zip(axes, delta_axes, TARGETS)
    ):
        ax.axvspan(0.50, 1.50, color="#F2F3F5", zorder=0)
        target_rows = best[best["Target"] == target]
        deltas: dict[str, dict[str, tuple[float, float]]] = {}
        for path_index, path in enumerate(PATHS):
            path_rows = target_rows[target_rows["Path"] == path].set_index("Validation")
            path_rows = path_rows.loc[list(VALIDATIONS)]
            values = path_rows["R2"].to_numpy(dtype=float)
            plot_baseline = values[0]
            site_baseline = values[1]
            deltas[path] = {
                "KFold": (values[2] - plot_baseline, values[2] - site_baseline),
                "LOOCV": (values[3] - plot_baseline, values[3] - site_baseline),
            }
            bars = ax.bar(
                x + offsets[path_index],
                values,
                width=bar_width,
                color=COLORS[path],
                edgecolor=EDGES[path],
                linewidth=0.9,
                label=PATH_LABELS[path],
                zorder=3,
            )
            for bar, value in zip(bars, values):
                if value >= 0:
                    y = value + 0.026
                    va = "bottom"
                else:
                    y = value - 0.030
                    va = "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va=va,
                    fontsize=8.2,
                    color="#000000",
                    clip_on=False,
                )

        ax.axhline(0, color="#000000", linewidth=0.9, zorder=2)
        ax.set_ylim(-0.66, 1.16)
        ax.set_yticks(np.arange(-0.6, 1.01, 0.2))
        ax.set_ylabel("R²")
        ax.set_title(target, loc="left", fontweight="bold", pad=6)
        ax.grid(axis="y", color="#DDE1E7", linewidth=0.7, zorder=1)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_color("#000000")
        ax.tick_params(axis="both", colors="#000000", width=0.8)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.text(
            -0.085,
            1.02,
            f"{chr(ord('a') + panel_index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#000000",
        )

        # Compact aligned delta rows preserve both requested baselines without
        # crowding the absolute R-squared labels above the bars.
        delta_ax.set_ylim(-0.48, 1.48)
        delta_ax.set_yticks((1.0, 0.0), ("ΔP", "ΔS"))
        delta_ax.set_ylabel("ΔR²", rotation=90, labelpad=9, va="center")
        delta_ax.tick_params(axis="y", length=0, pad=4, labelsize=8.5)
        delta_ax.tick_params(axis="x", colors="#000000", width=0.8)
        for validation_x in (2.0, 3.0):
            delta_ax.axvspan(
                validation_x - 0.47,
                validation_x + 0.47,
                color="#F7F7F7",
                zorder=0,
            )
            for path_index, path in enumerate(PATHS):
                cell_center = validation_x + offsets[path_index]
                delta_ax.axvspan(
                    cell_center - 0.125,
                    cell_center + 0.125,
                    color=COLORS[path],
                    alpha=0.07,
                    zorder=1,
                )
            for separator in (-0.135, 0.135):
                delta_ax.plot(
                    [validation_x + separator, validation_x + separator],
                    [-0.45, 1.45],
                    color="#FFFFFF",
                    linewidth=1.2,
                    zorder=2,
                )
        delta_ax.plot([1.53, 3.47], [0.5, 0.5], color="#D9DDE3", linewidth=0.6)
        delta_ax.plot([2.5, 2.5], [-0.45, 1.45], color="#D9DDE3", linewidth=0.6)
        for validation_index, validation in ((2, "KFold"), (3, "LOOCV")):
            for path_index, path in enumerate(PATHS):
                delta_plot, delta_site = deltas[path][validation]
                text_x = validation_index + offsets[path_index]
                delta_ax.text(
                    text_x,
                    1.0,
                    f"{delta_plot:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="#000000",
                )
                delta_ax.text(
                    text_x,
                    0.0,
                    f"{delta_site:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="#000000",
                )
        for spine in delta_ax.spines.values():
            spine.set_visible(False)
        delta_ax.grid(False)

    for delta_ax in delta_axes[:-1]:
        delta_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    delta_axes[-1].set_xticks(x, [VALIDATION_LABELS[value] for value in VALIDATIONS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.3,
    )
    fig.text(
        0.5,
        0.948,
        "Delta rows under 5-fold KFold and LOOCV: ΔP vs plot-blocked; ΔS vs site-blocked",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#000000",
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.092, top=0.90)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    best = load_best_rf_results()
    source = best.copy()
    source["Path_Label"] = source["Path"].map(PATH_LABELS)
    source["Validation_Label"] = source["Validation"].map(
        {key: value.replace("\n", " ") for key, value in VALIDATION_LABELS.items()}
    )
    source.to_csv(OUT / f"{STEM}_source.csv", index=False)

    fig = build_figure(best)
    fig.savefig(OUT / f"{STEM}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{STEM}.pdf", facecolor="white")
    plt.close(fig)
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()

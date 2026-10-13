from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


from package_paths import output_root

ROOT = output_root()
OUT = ROOT / "revision_submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["SOC", "SIC", "Gypsum"]
VALIDATIONS = ["GroupKFold_Plot", "KFold", "LOOCV"]
VALIDATION_LABELS = {
    "GroupKFold_Plot": "GroupKFold",
    "KFold": "KFold",
    "LOOCV": "LOOCV",
}
PATHS = ["A_Global", "B1_Management", "B2_Topography"]
PATH_LABELS = {
    "A_Global": "Global",
    "B1_Management": "Management stratified",
    "B2_Topography": "Topography stratified",
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#000000",
    "muted": "#000000",
    "grid": "#E6E8F0",
    "axis": "#000000",
}
COLORS = {
    "blue": {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
}
NEUTRAL = {"light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}


def use_submitted_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "text.color": "#000000",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def rebalance_typography(fig: plt.Figure) -> None:
    data_axes = [
        ax
        for ax in fig.axes
        if ax.get_visible() and ax.axison and ax.get_label() != "<colorbar>"
    ]
    for ax in fig.axes:
        ax.tick_params(axis="both", colors="#000000", labelcolor="#000000", labelsize=9, width=0.8)
        ax.xaxis.label.set_color("#000000")
        ax.yaxis.label.set_color("#000000")
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.title.set_color("#000000")
        ax.title.set_fontsize(10)

        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_color("#000000")
            label.set_fontsize(9)

        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_color("#000000")
                spine.set_linewidth(0.8)

        for text in ax.texts:
            content = text.get_text().strip()
            panel_match = re.fullmatch(r"(?:\(([a-z])\)|([A-Z])|([a-z])\))", content)
            if panel_match:
                letter = next(group for group in panel_match.groups() if group is not None)
                text.set_text(f"{letter.lower()})")
                text.set_color("#000000")
                text.set_fontsize(10)
            else:
                text.set_color("#000000")
                text.set_fontsize(9)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#000000")
                text.set_fontsize(9)
            legend.get_title().set_color("#000000")
            legend.get_title().set_fontsize(9)

    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_color("#000000")
            text.set_fontsize(9)
        legend.get_title().set_color("#000000")
        legend.get_title().set_fontsize(9)

    for text in fig.texts:
        text.set_color("#000000")
        content = text.get_text().strip()
        panel_match = re.fullmatch(r"(?:\(([a-z])\)|([A-Z])|([a-z])\))", content)
        if panel_match:
            letter = next(group for group in panel_match.groups() if group is not None)
            text.set_text(f"{letter.lower()})")
            text.set_fontsize(10)
        else:
            text.set_fontsize(9)

    if len(data_axes) > 1:
        for index, ax in enumerate(data_axes):
            has_panel_label = any(
                re.fullmatch(
                    r"(?:\([a-z]\)|[A-Z]|[a-z]\))",
                    text.get_text().strip(),
                )
                for text in ax.texts
            )
            if not has_panel_label:
                ax.text(
                    -0.10,
                    1.03,
                    f"{chr(ord('a') + index)})",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#000000",
                    zorder=20,
                )


def save_figure(fig: plt.Figure, stem: str) -> None:
    # Figure 1 is a workflow diagram without axes; retain its submitted text hierarchy.
    if stem != "Figure1":
        rebalance_typography(fig)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded_box(ax, xy, width, height, face, edge, linewidth=1.0, radius=0.025):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    return box


def box_text(ax, cx, cy, title, subtitle="", title_size=12.5, subtitle_size=9.2, title_color="#102238", subtitle_color="#4E5C70"):
    ax.text(cx, cy + (0.010 if subtitle else 0), title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color)
    if subtitle:
        ax.text(cx, cy - 0.022, subtitle, ha="center", va="center", fontsize=subtitle_size, color=subtitle_color, linespacing=1.25)


def arrow(ax, start, end, color="#727E8D"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.0, color=color))


def checkmark(ax, x, y):
    ax.scatter([x], [y], s=260, color="#1E7F58", zorder=8)
    ax.text(x, y, "✓", ha="center", va="center", fontsize=13, color="white", fontweight="bold", zorder=9)


def numbered_heading(ax, y, number, text):
    ax.scatter([0.025], [y], s=250, color="#283747", zorder=5)
    ax.text(0.025, y, str(number), ha="center", va="center", color="white", fontsize=9.5, fontweight="bold", zorder=6)
    ax.text(0.047, y, text, ha="left", va="center", fontsize=11.5, fontweight="bold", color="#283747")


def figure1_workflow() -> None:
    """Recreate the submitted workflow diagram, changing only the requested methodology text."""
    # Match the physical canvas used by the submitted 1430 x 1301 px diagram.
    # Keeping that aspect and scale prevents long workflow labels from crowding.
    fig, ax = plt.subplots(figsize=(14.30, 13.01))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, (0.145, 0.925), 0.71, 0.060, "#EEEEEE", "#BEBEBE", 0.9, 0.012)
    box_text(
        ax,
        0.50,
        0.956,
        "Soil dataset and VNIR spectra",
        "216 samples, 18 plots, 6 oases, 3 study areas | 400–2500 nm, 2101 bands | SOC, SIC, gypsum (%)",
        11.7,
        8.1,
    )
    arrow(ax, (0.50, 0.925), (0.50, 0.884))

    numbered_heading(ax, 0.895, 1, "Spectral scenarios")
    spec_y, spec_h, spec_w = 0.793, 0.074, 0.245
    xs = [0.055, 0.3775, 0.700]
    for x in xs[:2]:
        rounded_box(ax, (x, spec_y), spec_w, spec_h, "#EAF0F7", "#B8CAE0", 0.9, 0.011)
    rounded_box(ax, (xs[2] - 0.004, spec_y - 0.004), spec_w + 0.008, spec_h + 0.008, "#DDEEE5", "#C7E3D4", 1.0, 0.012)
    rounded_box(ax, (xs[2], spec_y), spec_w, spec_h, "#E1F0E8", "#1E7F58", 1.5, 0.010)
    box_text(ax, xs[0] + spec_w / 2, spec_y + spec_h / 2, "RAW", "raw reflectance", 11.5, 8.7, title_color="#7C8794", subtitle_color="#7C8794")
    box_text(ax, xs[1] + spec_w / 2, spec_y + spec_h / 2 + 0.002, "RAW + SG + MSC", "Savitzky–Golay 1st derivative\n(window 11, order 3) + MSC\nfitted on the calibration fold", 10.9, 7.9, title_color="#7C8794", subtitle_color="#7C8794")
    box_text(ax, xs[2] + spec_w / 2, spec_y + spec_h / 2, "CR", "continuum removal,\nper-spectrum convex hull", 11.5, 8.7)
    checkmark(ax, xs[2] + spec_w - 0.012, spec_y + spec_h - 0.010)

    line_y = 0.765
    for x in [v + spec_w / 2 for v in xs]:
        ax.plot([x, x], [spec_y, line_y], color="#8A96A4", linewidth=0.8)
    ax.plot([xs[0] + spec_w / 2, xs[2] + spec_w / 2], [line_y, line_y], color="#8A96A4", linewidth=0.8)
    arrow(ax, (0.50, line_y), (0.50, 0.720))

    numbered_heading(ax, 0.735, 2, "Calibration paths")
    path_y, path_h, path_w = 0.650, 0.057, 0.245
    for x in xs:
        rounded_box(ax, (x, path_y), path_w, path_h, "#F5E8D3", "#D5B98D", 0.9, 0.010)
    box_text(ax, xs[0] + path_w / 2, path_y + path_h / 2, "Path A – Global", "all 216 samples", 11.2, 8.5)
    box_text(ax, xs[1] + path_w / 2, path_y + path_h / 2, "Path B1 – Management", "modern | traditional", 11.2, 8.5)
    box_text(ax, xs[2] + path_w / 2, path_y + path_h / 2, "Path B2 – Topography", "upslope | midslope | downslope", 11.2, 8.5)
    line_y = 0.620
    for x in [v + path_w / 2 for v in xs]:
        ax.plot([x, x], [path_y, line_y], color="#8A96A4", linewidth=0.8)
    ax.plot([xs[0] + path_w / 2, xs[2] + path_w / 2], [line_y, line_y], color="#8A96A4", linewidth=0.8)
    arrow(ax, (0.50, line_y), (0.50, 0.575))

    numbered_heading(ax, 0.590, 3, "Model families – each fitted as single- and multi-target")
    model_y, model_h = 0.505, 0.055
    rounded_box(ax, (0.145, model_y), 0.33, model_h, "#EAF0F7", "#B8CAE0", 0.9, 0.010)
    rounded_box(ax, (0.515 - 0.004, model_y - 0.004), 0.34 + 0.008, model_h + 0.008, "#DDEEE5", "#C7E3D4", 1.0, 0.011)
    rounded_box(ax, (0.515, model_y), 0.34, model_h, "#E1F0E8", "#1E7F58", 1.5, 0.010)
    box_text(ax, 0.31, model_y + model_h / 2, "PLSR", title_size=11.7, title_color="#7C8794")
    box_text(ax, 0.685, model_y + model_h / 2, "Random forest", title_size=11.7)
    checkmark(ax, 0.835, model_y + model_h - 0.009)
    ax.text(0.5, 0.482, "PLSR only in Path A; random forest in all paths", ha="center", va="center", fontsize=8.0, style="italic", color="#4E5C70")
    arrow(ax, (0.50, 0.468), (0.50, 0.424))

    numbered_heading(ax, 0.438, 4, "Validation schemes")
    val_y, val_h, val_w = 0.333, 0.076, 0.205
    val_xs = [0.050, 0.285, 0.520, 0.755]
    # Selected plot-blocked validation.
    rounded_box(ax, (val_xs[0] - 0.004, val_y - 0.004), val_w + 0.008, val_h + 0.008, "#DDEEE5", "#C7E3D4", 1.0, 0.011)
    rounded_box(ax, (val_xs[0], val_y), val_w, val_h, "#E1F0E8", "#1E7F58", 1.5, 0.010)
    for x in val_xs[1:]:
        rounded_box(ax, (x, val_y), val_w, val_h, "#EAF0F7", "#B8CAE0", 0.9, 0.010)
    box_text(ax, val_xs[0] + val_w / 2, val_y + val_h / 2, "Plot-blocked GroupKFold —", "18 plots, 6 folds", 9.3, 8.0)
    checkmark(ax, val_xs[0] + val_w - 0.010, val_y + val_h - 0.009)
    box_text(ax, val_xs[1] + val_w / 2, val_y + val_h / 2, "Site-blocked GroupKFold —", "6 oases, 6 folds", 9.0, 8.0, title_color="#7C8794", subtitle_color="#7C8794")
    box_text(ax, val_xs[2] + val_w / 2, val_y + val_h / 2, "KFold", "5-fold random,\nshuffle, seed 42", 10.8, 8.0, title_color="#7C8794", subtitle_color="#7C8794")
    box_text(ax, val_xs[3] + val_w / 2, val_y + val_h / 2, "LOOCV", "216 folds", 10.8, 8.0, title_color="#7C8794", subtitle_color="#7C8794")
    line_y = 0.307
    for x in [v + val_w / 2 for v in val_xs]:
        ax.plot([x, x], [val_y, line_y], color="#8A96A4", linewidth=0.8)
    ax.plot([val_xs[0] + val_w / 2, val_xs[-1] + val_w / 2], [line_y, line_y], color="#8A96A4", linewidth=0.8)
    arrow(ax, (0.50, line_y), (0.50, 0.266))

    numbered_heading(ax, 0.278, 5, "Evaluation")
    rounded_box(ax, (0.105, 0.196), 0.79, 0.052, "#EDE1F5", "#BDA0D5", 0.9, 0.010)
    box_text(ax, 0.50, 0.222, "Pooled held-out predictions", "predictions of all folds concatenated, metrics computed once", 10.7, 8.0)
    arrow(ax, (0.50, 0.196), (0.50, 0.166))
    rounded_box(ax, (0.105, 0.090), 0.365, 0.056, "#EDE1F5", "#BDA0D5", 0.9, 0.010)
    rounded_box(ax, (0.530, 0.090), 0.365, 0.056, "#EDE1F5", "#BDA0D5", 0.9, 0.010)
    box_text(ax, 0.2875, 0.118, "Performance metrics", "R², RMSE, MAE, Bias,\nLin's CCC, RPD", 10.4, 7.8)
    box_text(ax, 0.7125, 0.118, "Validation bias", "ΔR² = R²(KFold or LOOCV)\n− R²(Plot-blocked GroupKFold)", 10.4, 7.8)

    rounded_box(ax, (0.055, 0.026), 0.022, 0.018, "#E1F0E8", "#1E7F58", 1.2, 0.005)
    ax.text(0.087, 0.035, "selected at this step", ha="left", va="center", fontsize=7.8)
    rounded_box(ax, (0.300, 0.026), 0.022, 0.018, "#EAF0F7", "#B8CAE0", 0.9, 0.005)
    ax.text(0.332, 0.035, "alternative run and reported, not carried into interpretation", ha="left", va="center", fontsize=7.8)
    save_figure(fig, "Figure1")


def load_soil_data() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "diagnostics" / "working_data_with_study_area.csv")
    out = pd.DataFrame(
        {
            "Sample_ID": raw["identifiant"].astype(str),
            "Plot": raw["sites"].astype(str),
            "Management": pd.to_numeric(raw["OasisType"], errors="raise").astype(int).map({1: "Modern", 2: "Traditional"}),
            "Topography": pd.to_numeric(raw["Slope_Category"], errors="raise").astype(int).map({1: "Upslope", 2: "Midslope", 3: "Downslope"}),
            "Depth": pd.to_numeric(raw["Depth  "], errors="raise").astype(int),
            "SOC": pd.to_numeric(raw["SOC %"], errors="coerce"),
            "SIC": pd.to_numeric(raw["SIC %"], errors="coerce"),
            "Gypsum": pd.to_numeric(raw["Gypsum %"], errors="coerce"),
        }
    )
    out["Depth_Label"] = pd.Categorical(out["Depth"].astype(str), categories=["0", "5", "10", "30", "60", "90"], ordered=True)
    out["Management"] = pd.Categorical(out["Management"], categories=["Modern", "Traditional"], ordered=True)
    out["Topography"] = pd.Categorical(out["Topography"], categories=["Upslope", "Midslope", "Downslope"], ordered=True)
    return out


def sig_text(p: float) -> str:
    if not np.isfinite(p):
        return "n.a."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def figure2_distributions() -> None:
    use_submitted_theme()
    df = load_soil_data()
    tests = pd.read_csv(ROOT / "mixed_models" / "mixed_effects_omnibus_tests.csv")
    pmap = {(r.Target, r.Term): float(r.P_Value) for r in tests.itertuples(index=False)}
    topo_order = ["Upslope", "Midslope", "Downslope"]
    mgmt_order = ["Modern", "Traditional"]
    depth_order = ["0", "5", "10", "30", "60", "90"]
    mgmt_palette = {"Modern": "#3566A7", "Traditional": "#5AA493"}
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.8), sharex=True)
    for row_idx, target in enumerate(TARGETS):
        for col_idx, topo in enumerate(topo_order):
            ax = axes[row_idx, col_idx]
            sub = df[df["Topography"] == topo].copy()
            sns.boxplot(data=sub, x="Depth_Label", y=target, hue="Management", order=depth_order, hue_order=mgmt_order, palette=mgmt_palette, width=0.62, fliersize=0, linewidth=1.0, ax=ax)
            for patch in ax.patches:
                patch.set_alpha(0.16)
            sns.stripplot(data=sub, x="Depth_Label", y=target, hue="Management", order=depth_order, hue_order=mgmt_order, palette=mgmt_palette, dodge=True, size=3.0, alpha=0.56, linewidth=0, ax=ax)
            for mgmt_idx, management in enumerate(mgmt_order):
                means = sub[sub["Management"] == management].groupby("Depth_Label", observed=False)[target].mean().reindex(depth_order)
                offset = -0.18 if mgmt_idx == 0 else 0.18
                ax.plot(np.arange(len(depth_order)) + offset, means.to_numpy(float), color=mgmt_palette[management], marker="o", markersize=3.6, linewidth=1.7, zorder=6)
            if target == "Gypsum":
                ax.set_yscale("symlog", linthresh=0.5)
                ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            if row_idx == 0:
                ax.set_title(f"{topo}\n(n = {sub['Sample_ID'].nunique()})", fontweight="bold", fontsize=14, fontfamily="serif", pad=14)
            else:
                ax.set_title("")
            panel = chr(ord("A") + row_idx * 3 + col_idx)
            ax.text(-0.16, 1.08, panel, transform=ax.transAxes, fontsize=14, fontweight="bold", fontfamily="serif", va="top", ha="left", color="#000000")
            ax.set_xlabel("Depth (cm)" if row_idx == 2 else "")
            ax.set_ylabel(f"{target} (%)" if col_idx == 0 else "")
            if col_idx == 0:
                ax.yaxis.label.set_fontsize(14)
                ax.yaxis.label.set_fontweight("bold")
                ax.yaxis.label.set_fontfamily("serif")
            if row_idx == 2:
                ax.xaxis.label.set_fontsize(13)
                ax.xaxis.label.set_fontweight("bold")
                ax.xaxis.label.set_fontfamily("serif")
            ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8)
            ax.grid(False, axis="x")
            if ax.legend_:
                ax.legend_.remove()
            if col_idx == 2:
                txt = (
                    f"Depth {sig_text(pmap.get((target, 'C(Depth)'), np.nan))}\n"
                    f"Topography {sig_text(pmap.get((target, 'C(Topography)'), np.nan))}\n"
                    f"Management {sig_text(pmap.get((target, 'C(Management)'), np.nan))}"
                )
                ax.text(0.98, 0.96, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#000000", fontfamily="serif", bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#BFC7D5", "linewidth": 0.9})
    handles = [Line2D([0], [0], color=mgmt_palette[x], marker="o", linewidth=1.7, markersize=4.5, label=x) for x in mgmt_order]
    fig.legend(handles, mgmt_order, loc="upper center", ncol=2, frameon=False, title="Management", bbox_to_anchor=(0.5, 0.995), prop={"family": "serif", "size": 10})
    fig.legends[0].get_title().set_fontfamily("serif")
    fig.legends[0].get_title().set_fontsize(11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93), h_pad=1.1, w_pad=0.8)
    save_figure(fig, "Figure2")


def figure3_pca() -> None:
    use_submitted_theme()
    df = load_soil_data()
    cr = pd.read_csv(ROOT / "verification" / "computed_continuum_removal_400_2500.csv")
    spectral_cols = sorted([c for c in cr.columns if re.fullmatch(r"X\d+", str(c))], key=lambda c: int(c[1:]))
    X = SimpleImputer(strategy="median").fit_transform(cr[spectral_cols].to_numpy(float))
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    pca_df = df[["Sample_ID", "Topography", "Management", "Depth_Label"]].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharex=True, sharey=True)
    topo_palette = {"Upslope": COLORS["blue"]["base"], "Midslope": COLORS["gold"]["base"], "Downslope": COLORS["olive"]["base"]}
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Topography", style="Management", palette=topo_palette, s=50, edgecolor="white", linewidth=0.5, ax=axes[0])
    depth_order = ["0", "5", "10", "30", "60", "90"]
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Depth_Label", hue_order=depth_order, palette=sns.color_palette("viridis", n_colors=6), s=48, edgecolor="white", linewidth=0.5, ax=axes[1])
    for ax, label in zip(axes, ["Topography and management", "Depth class"]):
        ax.axhline(0, color=NEUTRAL["light"], linewidth=0.8)
        ax.axvline(0, color=NEUTRAL["light"], linewidth=0.8)
        ax.set_title(label, loc="left", fontweight="semibold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.legend(title="", loc="best", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, "Figure3")


def pooled_metrics(group: pd.DataFrame) -> pd.Series:
    y = group["Observed"].to_numpy(float)
    p = group["Predicted"].to_numpy(float)
    residual = p - y
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return pd.Series({"R2": r2, "RMSE": rmse})


def revised_validation_best() -> pd.DataFrame:
    pred = pd.read_csv(ROOT / "diagnostics" / "predictions_validation_all.csv")
    pred = pred[(pred["Scenario"] == "CR") & (pred["ModelFamily"] == "RF") & pred["Validation"].isin(VALIDATIONS)].copy()
    agg = pred.groupby(["Path", "Validation", "Mode", "Target"], as_index=False).apply(pooled_metrics, include_groups=False)
    return (
        agg.sort_values(["Path", "Validation", "Target", "R2", "RMSE"], ascending=[True, True, True, False, True])
        .groupby(["Path", "Validation", "Target"], as_index=False)
        .first()
    )


def figure4_validation() -> None:
    use_submitted_theme()
    best = revised_validation_best()
    palette = {"Global": "#3566A7", "Management stratified": "#2E8B57", "Topography stratified": "#D97A24"}
    edge = {"Global": "#1F3E73", "Management stratified": "#1F6840", "Topography stratified": "#8E4C16"}
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.2), sharey=True)
    x = np.arange(3)
    width = 0.23
    y_min, y_max = -0.25, 1.10  # Submitted limits; all revised values remain inside.
    label_offset = max(0.025, (y_max - y_min) * 0.035)
    for ax, target in zip(axes, TARGETS):
        sub = best[best["Target"] == target]
        lookup = sub.set_index(["Path", "Validation"])
        validation_peaks = {v: pd.to_numeric(sub[sub["Validation"] == v]["R2"], errors="coerce").max() for v in VALIDATIONS[1:]}
        for path_idx, path in enumerate(PATHS):
            display = PATH_LABELS[path]
            offset = (path_idx - 1) * width
            values = [float(lookup.loc[(path, v), "R2"]) for v in VALIDATIONS]
            bars = ax.bar(x + offset, values, width=width, color=palette[display], edgecolor=edge[display], linewidth=1.0, label=display, zorder=3)
            baseline = values[0]
            for vi, validation in enumerate(VALIDATIONS[1:], start=1):
                value = values[vi]
                peak = validation_peaks[validation]
                y_text = peak + label_offset * (0.70 + 1.15 * path_idx)
                ax.text(bars[vi].get_x() + bars[vi].get_width() / 2, y_text, f"{value - baseline:+.2f}", ha="center", va="bottom", fontsize=10, color="#000000", clip_on=False)
        ax.axhline(0, color="#000000", linewidth=0.8)
        ax.set_title(target, fontweight="bold", fontfamily="serif", fontsize=12)
        ax.set_xticks(x, [VALIDATION_LABELS[v] for v in VALIDATIONS])
        ax.set_xlabel("")
        ax.set_ylabel("R²" if target == "SOC" else "")
        if target == "SOC":
            ax.yaxis.label.set_fontweight("bold")
            ax.yaxis.label.set_fontfamily("serif")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8)
        ax.grid(False, axis="x")
        ax.text(-0.10, 1.05, chr(ord("A") + TARGETS.index(target)), transform=ax.transAxes, fontsize=11, fontweight="bold", fontfamily="serif", va="top", ha="left", color="#000000")
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.995), title="Modeling scheme")
    fig.text(0.5, 0.885, "Labels above KFold and LOOCV bars show Delta R² versus the same path under GroupKFold.", ha="center", va="top", fontsize=10, color="#000000")
    fig.tight_layout(rect=(0, 0.02, 1, 0.80), w_pad=1.8)
    save_figure(fig, "Figure4")


def figure5_shap() -> None:
    use_submitted_theme()
    shap = pd.read_csv(ROOT / "shap" / "shap_top5_best_RF_CR_RAW_SG_MSC.csv")
    summary = pd.read_csv(ROOT / "diagnostics" / "summary_metrics_pooled_validation.csv")
    metrics = summary[(summary["Path"] == "A_Global") & (summary["Stratum"] == "ALL") & (summary["Validation"] == "GroupKFold_Plot") & (summary["ModelFamily"] == "RF") & (summary["Phase"] == "Validation")]
    metric_lookup = metrics.set_index(["Scenario", "Target", "Mode"])
    mode_colors = {"direct": COLORS["blue"]["base"], "multi": COLORS["olive"]["base"]}
    mode_edges = {"direct": COLORS["blue"]["dark"], "multi": COLORS["olive"]["dark"]}
    mode_labels = {"direct": "RF direct", "multi": "RF multi-target"}
    submitted_xmax = {
        ("CR", "SOC"): 0.00575,
        ("CR", "SIC"): 0.01150,
        ("CR", "Gypsum"): 0.225,
        ("RAW_SG_MSC", "SOC"): 0.00575,
        ("RAW_SG_MSC", "SIC"): 0.01475,
        ("RAW_SG_MSC", "Gypsum"): 0.185,
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.2), sharex=False, sharey=False)
    axis_changes = []
    for row_idx, scenario in enumerate(["CR", "RAW_SG_MSC"]):
        scenario_label = "CR" if scenario == "CR" else "RAW + SG + MSC"
        for col_idx, target in enumerate(TARGETS):
            ax = axes[row_idx, col_idx]
            sub = shap[(shap["Scenario"] == scenario) & (shap["Target"] == target)].sort_values("MeanAbsSHAP", ascending=True).copy()
            mode = str(sub["Mode"].iloc[0])
            sub["Wavelength_Label"] = sub["Wavelength_nm"].round().astype(int).astype(str) + " nm"
            bars = ax.barh(sub["Wavelength_Label"], sub["MeanAbsSHAP"], color=mode_colors[mode], edgecolor=mode_edges[mode], linewidth=0.85)
            old_limit = submitted_xmax[(scenario, target)]
            actual_max = float(sub["MeanAbsSHAP"].max())
            xmax = old_limit if actual_max <= old_limit else actual_max * 1.25
            ax.set_xlim(0, xmax)
            if xmax > old_limit + 1e-12:
                axis_changes.append({"Figure": "Figure5", "Panel": chr(ord("A") + row_idx * 3 + col_idx), "Scenario": scenario_label, "Target": target, "Submitted_xmax": old_limit, "Revised_xmax": xmax, "Reason": "Revised SHAP value exceeds submitted axis range"})
            for bar, value, rank in zip(bars, sub["MeanAbsSHAP"], sub["Rank"]):
                ax.text(value + xmax * 0.025, bar.get_y() + bar.get_height() / 2, f"{int(rank)}", va="center", ha="left", fontsize=10, color="#000000")
            row = metric_lookup.loc[(scenario, target, mode)]
            ax.set_title(f"{target} - {mode_labels[mode]} (R²={row['R2']:.2f}, RMSE={row['RMSE']:.2f})", fontweight="semibold", fontsize=11)
            ax.set_xlabel("Mean absolute SHAP")
            ax.set_ylabel(f"{scenario_label}\nWavelength (nm)" if col_idx == 0 else "")
            ax.grid(True, axis="x", color=TOKENS["grid"], linewidth=0.65, alpha=0.8)
            ax.grid(False, axis="y")
            ax.text(-0.13, 1.05, chr(ord("A") + row_idx * 3 + col_idx), transform=ax.transAxes, fontsize=11, fontweight="bold", fontfamily="serif", va="top", ha="left", color="#000000")
    handles = [Patch(facecolor=mode_colors["direct"], edgecolor=mode_edges["direct"], label="RF direct"), Patch(facecolor=mode_colors["multi"], edgecolor=mode_edges["multi"], label="RF multi-target")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.020))
    fig.tight_layout(rect=(0, 0.055, 1, 0.98), w_pad=1.4, h_pad=1.35)
    save_figure(fig, "Figure5")
    pd.DataFrame(axis_changes).to_csv(ROOT / "revision_submission" / "figure_axis_changes.csv", index=False)


def selected_cr_predictions() -> pd.DataFrame:
    pred = pd.read_csv(ROOT / "diagnostics" / "predictions_validation_all.csv")
    selected_modes = {"SOC": "multi", "SIC": "direct", "Gypsum": "direct"}
    parts = []
    for target, mode in selected_modes.items():
        part = pred[(pred["Scenario"] == "CR") & (pred["Path"] == "A_Global") & (pred["Stratum"] == "ALL") & (pred["Validation"] == "GroupKFold_Plot") & (pred["ModelFamily"] == "RF") & (pred["Mode"] == mode) & (pred["Target"] == target)].copy()
        part["Selected_Mode"] = mode
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def diagnostic_figures() -> None:
    use_submitted_theme()
    pred = selected_cr_predictions()
    palette = {"SOC": "#3566A7", "SIC": "#D97A24", "Gypsum": "#2E8B57"}

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.1))
    for ax, target in zip(axes, TARGETS):
        sub = pred[pred["Target"] == target]
        y = sub["Observed"].to_numpy(float)
        p = sub["Predicted"].to_numpy(float)
        lo = min(float(y.min()), float(p.min()))
        hi = max(float(y.max()), float(p.max()))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad
        ax.scatter(y, p, s=25, alpha=0.65, color=palette[target], edgecolor="white", linewidth=0.35)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#000000", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        mode = str(sub["Selected_Mode"].iloc[0])
        r2 = 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        rmse = np.sqrt(np.mean((y - p) ** 2))
        ax.set_title(f"{target} - RF {mode}", fontweight="semibold")
        ax.set_xlabel(f"Observed {target} (%)")
        ax.set_ylabel(f"Predicted {target} (%)" if target == "SOC" else "")
        ax.text(0.04, 0.96, f"R² = {r2:.2f}\nRMSE = {rmse:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#000000", "linewidth": 0.7})
        ax.grid(True, color=TOKENS["grid"])
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "Reviewer1_Observed_vs_Predicted")

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))
    for ax, target in zip(axes, TARGETS):
        sub = pred[pred["Target"] == target]
        p = sub["Predicted"].to_numpy(float)
        residual = sub["Observed"].to_numpy(float) - p
        ax.scatter(p, residual, s=25, alpha=0.65, color=palette[target], edgecolor="white", linewidth=0.35)
        ax.axhline(0, color="#000000", linestyle="--", linewidth=1.0)
        mode = str(sub["Selected_Mode"].iloc[0])
        ax.set_title(f"{target} - RF {mode}", fontweight="semibold")
        ax.set_xlabel(f"Predicted {target} (%)")
        ax.set_ylabel("Residual (observed − predicted)" if target == "SOC" else "")
        ax.grid(True, color=TOKENS["grid"])
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "Reviewer1_Residual_vs_Predicted")


def main() -> None:
    figure1_workflow()
    figure2_distributions()
    figure3_pca()
    figure4_validation()
    figure5_shap()
    diagnostic_figures()
    records = []
    for path in sorted(OUT.glob("*")):
        if path.suffix.lower() in {".png", ".pdf"}:
            records.append({"File": path.name, "Purpose": "Revised data in submitted visual design" if path.stem.startswith("Figure") else "New Reviewer 1 diagnostic", "Format": path.suffix.lower().lstrip(".")})
    pd.DataFrame(records).to_csv(ROOT / "revision_submission" / "figure_manifest.csv", index=False)


if __name__ == "__main__":
    main()

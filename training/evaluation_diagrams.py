from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Config
# =========================
INPUT_XLSX = "model_eval_data_dc.xlsx"
SCENARIO_NAME = "deadly_corridor"
OUTPUT_DIR = Path(f"{SCENARIO_NAME}_charts")
SCENARIO_NAME_DISPLAY = SCENARIO_NAME.replace("_", " ").title()

# Set this explicitly if you know the exact model name.
# Otherwise the script tries to find something containing "random" or "baseline".
BASELINE_MODEL = None
BASELINE_REGEX = re.compile(r"(random|baseline)", re.IGNORECASE)

TOP_K_FOR_DISTRIBUTION = 5
N_BOOT = 5000
CI = 95
RNG = np.random.default_rng(42)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================
ARCH_MARKER = {
    "CNN": "o",
    "LateFusion": "s",
    "FiLM": "^",
    "Rainbow": "P",
    "Random": "D",
}

def get_algorithm_facecolor(algorithm: str, status_color: str) -> str:
    """
    Fill encodes algorithm:
      - PPO        -> open marker
      - Q-Learning -> filled marker
      - Random/N/A -> shaded marker
    """
    if algorithm == "PPO":
        return "white"
    elif algorithm == "Q-Learning":
        return status_color
    else:
        return "#c7c7c7"   # shaded fill for Random / N/A

def bootstrap_mean_ci(x, n_boot=N_BOOT, ci=CI):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(np.mean(x)), float(lo), float(hi)

def bootstrap_delta_ci(x, y, n_boot=N_BOOT, ci=CI):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    bx = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    by = RNG.choice(y, size=(n_boot, len(y)), replace=True).mean(axis=1)
    d = bx - by
    lo = np.percentile(d, (100 - ci) / 2)
    hi = np.percentile(d, 100 - (100 - ci) / 2)
    return float(np.mean(x) - np.mean(y)), float(lo), float(hi)

def ecdf(arr):
    arr = np.sort(np.asarray(arr, dtype=float))
    arr = arr[np.isfinite(arr)]
    y = np.arange(1, len(arr) + 1) / len(arr)
    return arr, y

def detect_long_format(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    model_candidates = ["model", "model_type", "agent", "name"]
    score_candidates = ["score", "episode_score", "reward", "return", "total_reward"]

    model_col = next((cols[c] for c in model_candidates if c in cols), None)
    score_col = next((cols[c] for c in score_candidates if c in cols), None)

    if model_col is not None and score_col is not None:
        out = df[[model_col, score_col]].copy()
        out.columns = ["model", "score"]
        out = out.dropna(subset=["model", "score"])
        out["model"] = out["model"].astype(str).str.strip()
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        out = out.dropna(subset=["score"])
        return out

    return None

def detect_wide_format(df):
    df = df.copy()
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Drop obvious index-like columns
    drop_like = {"episode", "trial", "run", "id", "seed"}
    keep_cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in drop_like:
            continue
        keep_cols.append(c)
    df = df[keep_cols]

    numeric_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, len(df) * 0.5):
            numeric_cols.append(c)

    if len(numeric_cols) >= 2:
        out = df[numeric_cols].copy()
        out = out.melt(var_name="model", value_name="score")
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        out = out.dropna(subset=["score"])
        out["model"] = out["model"].astype(str).str.strip()
        return out

    return None

def load_episode_level_data(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)
    candidates = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)

        long_df = detect_long_format(df)
        if long_df is not None and len(long_df) > 0:
            candidates.append((sheet, long_df))

        wide_df = detect_wide_format(df)
        if wide_df is not None and len(wide_df) > 0:
            candidates.append((sheet, wide_df))

    if not candidates:
        raise ValueError(
            "Could not detect episode-level model evaluation data.\n"
            "Expected either:\n"
            "  - long format with columns like model + score\n"
            "  - wide format with one numeric column per model"
        )

    # Pick the largest parsed candidate
    sheet, long_df = max(candidates, key=lambda x: len(x[1]))
    print(f"Using sheet: {sheet}")
    return long_df

def choose_baseline(models):
    if BASELINE_MODEL is not None and BASELINE_MODEL in models:
        return BASELINE_MODEL
    matches = [m for m in models if BASELINE_REGEX.search(m)]
    if matches:
        return matches[0]
    raise ValueError(
        "Could not infer baseline model. Set BASELINE_MODEL explicitly near the top of the script."
    )

status_handles = [
    Line2D([0], [0], marker='o', color='w', label='Best model',
           markerfacecolor='#d62728', markeredgecolor='#d62728', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Baseline',
           markerfacecolor='#1f77b4', markeredgecolor='#1f77b4', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Other models',
           markerfacecolor='#9e9e9e', markeredgecolor='#9e9e9e', markersize=8),
]

arch_handles = [
    Line2D([0], [0], marker='o', color='black', label='CNN',
           markerfacecolor='white', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='s', color='black', label='LateFusion',
           markerfacecolor='white', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='^', color='black', label='FiLM',
           markerfacecolor='white', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='P', color='black', label='Rainbow',
           markerfacecolor='white', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='D', color='black', label='Random',
           markerfacecolor='white', markersize=8, linestyle='None'),
]

algo_handles = [
    Line2D([0], [0], marker='o', color='black', label='PPO (open)',
           markerfacecolor='white', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='o', color='black', label='Q-Learning (filled)',
           markerfacecolor='black', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='o', color='black', label='Random / N/A (shaded)',
           markerfacecolor='#c7c7c7', markersize=8, linestyle='None'),
]

# =========================
# Load + summarize
# =========================
scores_wide = pd.read_excel(INPUT_XLSX, sheet_name="model_eval_data")
meta = pd.read_excel(INPUT_XLSX, sheet_name="model_metadata")

df = (
    scores_wide
    .melt(id_vars=["Episode"], var_name="model", value_name="score")
    .dropna(subset=["score"])
    .merge(meta, on="model", how="left", validate="many_to_one")
)

missing_meta = df.loc[df["display_name"].isna(), "model"].unique().tolist()
if missing_meta:
    raise ValueError(f"Missing metadata for models: {missing_meta}")

baseline_model = meta.loc[meta["is_baseline"] == True, "model"].iloc[0]

summary_rows = []
for model, g in df.groupby("model"):
    x = g["score"].to_numpy(dtype=float)
    mean_, lo, hi = bootstrap_mean_ci(x)
    summary_rows.append({
        "model": model,
        "n": len(x),
        "mean": mean_,
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
    })

summary = pd.DataFrame(summary_rows).merge(meta, on="model", how="left")
summary = summary.sort_values("mean", ascending=False).reset_index(drop=True)
best_model = summary.iloc[0]["model"]
summary["is_best"] = summary["model"].eq(best_model)

baseline_scores = df.loc[df["model"] == baseline_model, "score"].to_numpy(dtype=float)

delta_rows = []
for model, g in df.groupby("model"):
    x = g["score"].to_numpy(dtype=float)
    delta_mean, delta_lo, delta_hi = bootstrap_delta_ci(x, baseline_scores)
    delta_rows.append({
        "model": model,
        "delta_mean": delta_mean,
        "delta_low": delta_lo,
        "delta_high": delta_hi,
    })

delta_df = (
    pd.DataFrame(delta_rows)
    .merge(meta, on="model", how="left")
    .sort_values("delta_mean", ascending=False)
    .reset_index(drop=True)
)

summary.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)
delta_df.to_csv(OUTPUT_DIR / "delta_vs_baseline.csv", index=False)

print("Baseline:", baseline_model)
print("Best:", best_model)

# =========================
# 1) Ranked lollipop chart
# =========================
plot_df = summary.sort_values("mean", ascending=True).reset_index(drop=True)
y = np.arange(len(plot_df))

colors = []
for m in plot_df["model"]:
    if m == baseline_model:
        colors.append("#1f77b4")   # baseline
    elif m == best_model:
        colors.append("#d62728")   # best
    else:
        colors.append("#bdbdbd")   # others

fig, ax = plt.subplots(figsize=(12, max(6, 0.50 * len(plot_df))))
for i, row in plot_df.iterrows():
    xerr_left = row["mean"] - row["ci_low"]
    xerr_right = row["ci_high"] - row["mean"]

    color = colors[i]
    marker = ARCH_MARKER.get(row["architecture"], "o")
    facecolor = get_algorithm_facecolor(row["algorithm"], color)

    ax.errorbar(
        x=row["mean"],
        y=i,
        xerr=np.array([[xerr_left], [xerr_right]]),
        fmt=marker,
        markersize=7,
        capsize=4,
        elinewidth=2,
        capthick=1.5,
        color=color,
        mfc=facecolor,
        mec=color,
        mew=1.8,
        alpha=0.95,
    )

ax.set_yticks(y)
ax.set_yticklabels(plot_df["display_name"])
ax.set_xlabel("Average episode score")
ax.set_title(f"{SCENARIO_NAME_DISPLAY} — Ranked model performance (95% bootstrap CI)")

for i, row in plot_df.iterrows():
    if row["model"] in {baseline_model, best_model}:
        ax.text(row["mean"] + 3, i + 0.14, row["display_name"], fontsize=9)

ax.grid(axis="x", alpha=0.25)

leg1 = ax.legend(
    handles=status_handles,
    title="Status",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    borderaxespad=0.0,
    frameon=False,
)
ax.add_artist(leg1)

leg2 = ax.legend(
    handles=arch_handles,
    title="Architecture",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.66),
    borderaxespad=0.0,
    frameon=False,
)
ax.add_artist(leg2)

leg3 = ax.legend(
    handles=algo_handles,
    title="Algorithm",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.28),
    borderaxespad=0.0,
    frameon=False,
)
ax.add_artist(leg3)

plt.subplots_adjust(right=0.66)
plt.savefig(
    OUTPUT_DIR / "01_ranked_lollipop.png",
    dpi=220,
    bbox_inches="tight",
    bbox_extra_artists=(leg1, leg2, leg3),
)
plt.close()

# =========================
# 2) Delta-to-baseline forest plot
# =========================
plot_df = delta_df.sort_values("delta_mean", ascending=True).reset_index(drop=True)
y = np.arange(len(plot_df))

colors = []
for m in plot_df["model"]:
    if m == baseline_model:
        colors.append("#1f77b4")
    elif m == best_model:
        colors.append("#d62728")
    else:
        colors.append("#7f7f7f")

fig, ax = plt.subplots(figsize=(12, max(6, 0.50 * len(plot_df))))
ax.axvline(0, ls="--", lw=1.2, color="black", alpha=0.8)

for i, row in plot_df.iterrows():
    xerr_left = row["delta_mean"] - row["delta_low"]
    xerr_right = row["delta_high"] - row["delta_mean"]

    color = colors[i]
    marker = ARCH_MARKER.get(row["architecture"], "o")
    facecolor = get_algorithm_facecolor(row["algorithm"], color)

    ax.errorbar(
        x=row["delta_mean"],
        y=i,
        xerr=np.array([[xerr_left], [xerr_right]]),
        fmt=marker,
        markersize=7,
        capsize=4,
        elinewidth=2,
        capthick=1.5,
        color=color,
        mfc=facecolor,
        mec=color,
        mew=1.8,
        alpha=0.95,
    )

ax.set_yticks(y)
ax.set_yticklabels(plot_df["display_name"])
ax.set_xlabel(f"Mean score difference vs baseline ({meta.loc[meta['model'] == baseline_model, 'display_name'].iloc[0]})")
ax.set_title(f"{SCENARIO_NAME_DISPLAY} — Delta to baseline (95% bootstrap CI)")

for i, row in plot_df.iterrows():
    if row["model"] in {baseline_model, best_model}:
        ax.text(row["delta_mean"] + 3, i + 0.14, row["display_name"], fontsize=9)

ax.grid(axis="x", alpha=0.25)

leg1 = ax.legend(
    handles=status_handles,
    title="Status",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    borderaxespad=0.0,
    frameon=False,
)
ax.add_artist(leg1)

leg2 = ax.legend(
    handles=arch_handles,
    title="Architecture",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.66),
    borderaxespad=0.0,
    frameon=False,
)
ax.add_artist(leg2)

leg3 = ax.legend(
    handles=algo_handles,
    title="Algorithm",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.28),
    borderaxespad=0.0,
    frameon=False,
)
ax.add_artist(leg3)

plt.subplots_adjust(right=0.66)
plt.savefig(
    OUTPUT_DIR / "02_delta_to_baseline_forest.png",
    dpi=220,
    bbox_inches="tight",
    bbox_extra_artists=(leg1, leg2, leg3),
)
plt.close()

# =========================
# 3) Distribution plot (baseline vs top models)
# =========================
top_models = summary["model"].head(TOP_K_FOR_DISTRIBUTION).tolist()
selected_models = []
if baseline_model not in selected_models:
    selected_models.append(baseline_model)
for m in top_models:
    if m not in selected_models:
        selected_models.append(m)

dist_df = df[df["model"].isin(selected_models)].copy()
order = (
    dist_df.groupby("model")["score"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

data = [dist_df.loc[dist_df["model"] == m, "score"].to_numpy(dtype=float) for m in order]

name_map = meta.set_index("model")["display_name"].to_dict()
tick_names = [name_map[m] for m in order]

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(data, tick_labels=tick_names, patch_artist=True, showfliers=False)

for patch, m in zip(bp["boxes"], order):
    if m == baseline_model:
        patch.set_facecolor("#1f77b4")
    elif m == best_model:
        patch.set_facecolor("#d62728")
    else:
        patch.set_facecolor("#d9d9d9")

ax.set_ylabel("Episode score")
ax.set_title(f"{SCENARIO_NAME_DISPLAY} — Score distributions for baseline vs top models")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_distribution_boxplot.png", dpi=220, bbox_inches="tight")
plt.close()

# =========================
# 4) ECDF plot
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
for m in order:
    x = dist_df.loc[dist_df["model"] == m, "score"].to_numpy(dtype=float)
    xx, yy = ecdf(x)

    if m == baseline_model:
        color = "#1f77b4"
        lw = 2.5
    elif m == best_model:
        color = "#d62728"
        lw = 2.5
    else:
        color = "#7f7f7f"
        lw = 1.4

    label = meta.loc[meta["model"] == m, "display_name"].iloc[0]
    ax.plot(xx, yy, label=label, linewidth=lw, color=color)

ax.set_xlabel("Episode score")
ax.set_ylabel("ECDF")
ax.set_title(f"{SCENARIO_NAME_DISPLAY} — ECDF of episode scores")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_ecdf.png", dpi=220, bbox_inches="tight")
plt.close()

# =========================
# 5) Pairwise superiority heatmap
#    P(mean_A > mean_B) via bootstrap
# =========================
heat_models = summary["model"].head(min(8, len(summary))).tolist()
if baseline_model not in heat_models:
    heat_models = [baseline_model] + heat_models
heat_models = list(dict.fromkeys(heat_models))

boots = {}
for m in heat_models:
    x = df.loc[df["model"] == m, "score"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    boots[m] = RNG.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)

M = np.zeros((len(heat_models), len(heat_models)))
for i, a in enumerate(heat_models):
    for j, b in enumerate(heat_models):
        M[i, j] = np.mean(boots[a] > boots[b])

name_map = meta.set_index("model")["display_name"].to_dict()
heat_names = [name_map[m] for m in heat_models]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(M, vmin=0.0, vmax=1.0)
ax.set_xticks(np.arange(len(heat_models)))
ax.set_yticks(np.arange(len(heat_models)))
ax.set_xticklabels(heat_names, rotation=35, ha="right")
ax.set_yticklabels(heat_names)
ax.set_title(f"{SCENARIO_NAME_DISPLAY} — Pairwise superiority\nP(mean_A > mean_B)")

for i in range(len(heat_models)):
    for j in range(len(heat_models)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_pairwise_superiority_heatmap.png", dpi=220, bbox_inches="tight")
plt.close()

print(f"Saved charts to: {OUTPUT_DIR.resolve()}")
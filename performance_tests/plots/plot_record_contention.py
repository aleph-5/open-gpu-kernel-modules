"""
plot_record_contention.py - Figures for record_contention_cost.

Outputs (saved to midsem-review/figures/):
  contention_time_vs_pages.pdf   - side-by-side line charts: elapsed vs pages by stream count
  contention_ratio_heatmap.pdf   - heatmap: hot_set / disjoint elapsed time ratio
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -- paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "..", "operation_microbenchmarks", "record_contention_cost.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "midsem-review", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -- load & aggregate ----------------------------------------------------------
df  = pd.read_csv(CSV_PATH)
agg = (
    df.groupby(["condition", "streams", "pages_per_stream"])["elapsed_us"]
    .agg(mean="mean", std="std")
    .reset_index()
)

stream_counts = sorted(agg["streams"].unique())
page_counts   = sorted(agg["pages_per_stream"].unique())

# -- style ---------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# blue gradient per stream count: lightest = 1 stream, darkest = 8 streams
palette = dict(zip(stream_counts, sns.color_palette("Blues_d", len(stream_counts))[::-1]))

# -- Figure 1 - side-by-side line charts ---------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

for ax, condition, title in zip(axes,
                                ["disjoint", "hot_set"],
                                ["Disjoint pages", "Hot set (same pages)"]):
    sub = agg[agg["condition"] == condition]
    for s in stream_counts:
        row = sub[sub["streams"] == s].sort_values("pages_per_stream")
        ax.plot(row["pages_per_stream"], row["mean"] / 1000,
                marker="o", linewidth=1.5, color=palette[s], label=f"{s} stream{'s' if s > 1 else ''}")
        ax.fill_between(row["pages_per_stream"],
                        (row["mean"] - row["std"]) / 1000,
                        (row["mean"] + row["std"]) / 1000,
                        alpha=0.10, color=palette[s])
    ax.set_title(title, pad=6)
    ax.set_xlabel("Pages per stream")
    ax.set_xscale("log", base=2)
    ax.set_xticks(page_counts)
    ax.set_xticklabels([str(p) for p in page_counts])
    sns.despine(ax=ax)

axes[0].set_ylabel("Elapsed time (ms)")
axes[1].legend(frameon=False, loc="upper left", title="Streams")

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "contention_time_vs_pages.pdf")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# -- Figure 2 - hot_set / disjoint ratio heatmap -------------------------------
disjoint_mean = agg[agg["condition"] == "disjoint"].set_index(["streams", "pages_per_stream"])["mean"]
hotset_mean   = agg[agg["condition"] == "hot_set" ].set_index(["streams", "pages_per_stream"])["mean"]
ratio         = (hotset_mean / disjoint_mean).reset_index()
ratio.columns = ["streams", "pages_per_stream", "ratio"]
pivot = ratio.pivot(index="streams", columns="pages_per_stream", values="ratio")
pivot = pivot.loc[sorted(pivot.index, reverse=True)]   # highest streams at top

fig2, ax2 = plt.subplots(figsize=(6, 3.0))
sns.heatmap(
    pivot,
    ax=ax2,
    cmap="Blues",
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    linecolor="#dddddd",
    cbar_kws={"label": "Hot-set / disjoint ratio", "shrink": 0.85},
)
ax2.set_xlabel("Pages per stream")
ax2.set_ylabel("Streams")
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "contention_ratio_heatmap.pdf")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")

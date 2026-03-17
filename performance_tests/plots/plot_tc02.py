"""
plot_tc02.py - Figures for tc02_performance_overhead.

Outputs (saved to midsem-review/figures/):
  tc02_overhead_grouped.pdf   - grouped bar chart: overhead % by workload x pages
  tc02_overhead_heatmap.pdf   - heatmap: overhead % (workload x pages)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# -- paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "..", "results.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "midsem-review", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -- load & aggregate ----------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["wall_overhead_pct"] = pd.to_numeric(df["wall_overhead_pct"], errors="coerce")

# average overhead over iteration counts for each (workload, pages) combo
on = df[df["tracking"] == "ON"].groupby(["workload", "pages"])["wall_overhead_pct"].mean().reset_index()
pivot = on.pivot(index="workload", columns="pages", values="wall_overhead_pct")
pivot = pivot.loc[["READ", "MIXED", "WRITE"]]   # fixed row order

pages     = sorted(on["pages"].unique())
workloads = ["READ", "MIXED", "WRITE"]

# -- style ---------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# three blue shades, light -> dark (READ is lightest, WRITE darkest)
WL_COLORS = {
    "READ":  "#9ecae1",
    "MIXED": "#3182bd",
    "WRITE": "#08306b",
}

# -- Figure 1 - Grouped bar chart ----------------------------------------------
n_groups = len(pages)
n_bars   = len(workloads)
x        = np.arange(n_groups)
width    = 0.22

fig1, ax1 = plt.subplots(figsize=(7, 4))

for i, wl in enumerate(workloads):
    vals = [pivot.loc[wl, p] for p in pages]
    bars = ax1.bar(x + (i - 1) * width, vals, width,
                   label=wl, color=WL_COLORS[wl], edgecolor="white", linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels([str(p) for p in pages])
ax1.set_xlabel("Pages")
ax1.set_ylabel("Overhead (%)")
ax1.legend(title="Workload", frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.12), ncol=3)
sns.despine(ax=ax1)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "tc02_overhead_grouped.pdf")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# -- Figure 2 - Heatmap --------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 2.8))

sns.heatmap(
    pivot,
    ax=ax2,
    cmap="Blues",
    annot=True,
    fmt=".0f",
    linewidths=0.4,
    linecolor="#dddddd",
    cbar_kws={"label": "Overhead (%)", "shrink": 0.85},
)

ax2.set_xlabel("Pages")
ax2.set_ylabel("")
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "tc02_overhead_heatmap.pdf")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")

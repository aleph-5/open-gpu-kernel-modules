"""
plot_duplicate_fault.py - Figures for duplicate_fault_skip_cost.

Outputs (saved to midsem-review/figures/):
  dup_fault_time_vs_pages.pdf   - line chart: first vs duplicate fault time
  dup_fault_ratio_vs_pages.pdf  - bar chart: first/duplicate speedup ratio
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -- paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "..", "operation_microbenchmarks", "duplicate_fault_skip_cost.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "midsem-review", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -- load & aggregate ----------------------------------------------------------
df  = pd.read_csv(CSV_PATH)
agg = df.groupby("page_count").agg(
    first_mean=("first_fault_ms",     "mean"),
    first_std =("first_fault_ms",     "std"),
    dup_mean  =("duplicate_fault_ms", "mean"),
    dup_std   =("duplicate_fault_ms", "std"),
).reset_index()
pages = agg["page_count"].tolist()

# -- style ---------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
C_FIRST = "#2166ac"
C_DUP   = "#92c5de"

# -- Figure 1 - first vs duplicate fault time ----------------------------------
fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))

ax1.plot(pages, agg["first_mean"], marker="o", linewidth=1.5,
         color=C_FIRST, label="First fault")
ax1.fill_between(pages,
                 agg["first_mean"] - agg["first_std"],
                 agg["first_mean"] + agg["first_std"],
                 alpha=0.12, color=C_FIRST)

ax1.plot(pages, agg["dup_mean"], marker="s", linewidth=1.5,
         linestyle="--", color=C_DUP, label="Duplicate fault")
ax1.fill_between(pages,
                 agg["dup_mean"] - agg["dup_std"],
                 agg["dup_mean"] + agg["dup_std"],
                 alpha=0.12, color=C_DUP)

ax1.set_xscale("log", base=2)
ax1.set_xticks(pages)
ax1.set_xticklabels([str(p) for p in pages])
ax1.set_xlabel("Pages")
ax1.set_ylabel("Kernel time (ms)")
ax1.legend(frameon=False, loc="upper left")
sns.despine(ax=ax1)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "dup_fault_time_vs_pages.pdf")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# -- Figure 2 - speedup ratio (first / duplicate) -----------------------------
ratio = (agg["first_mean"] / agg["dup_mean"]).tolist()

fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))

bars = ax2.bar(
    [str(p) for p in pages],
    ratio,
    color=sns.color_palette("Blues_d", len(pages))[::-1],
    edgecolor="#2166ac",
    linewidth=0.6,
    width=0.5,
)

for bar, val in zip(bars, ratio):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01,
             f"{val:.2f}x", ha="center", va="bottom",
             fontsize=8.5, color="black")

ax2.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--", label="Equal cost")
ax2.set_xlabel("Pages")
ax2.set_ylabel("First / duplicate fault time")
# zoom in around 1.0 - the key result is that ratios are near unity
lo = min(min(ratio) - 0.15, 0.6)
hi = max(max(ratio) + 0.15, 1.3)
ax2.set_ylim(lo, hi)
sns.despine(ax=ax2)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "dup_fault_ratio_vs_pages.pdf")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")

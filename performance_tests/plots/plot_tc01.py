"""
plot_tc01.py - Generate figures for tc01_stress_testing analysis.

Uses the real WRITE-workload rows from results.csv (same scenario: all pages
faulted on every tracked run) to illustrate how baseline vs. tracked kernel
time and the resulting overhead scale with page count.

Outputs (saved to midsem-review/figures/):
  tc01_time_vs_pages.pdf
  tc01_overhead_vs_pages.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -- paths --------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "..", "general_performance","performance_overhead.csv")
OUT_DIR     = os.path.join(SCRIPT_DIR, "..", "..", "midsem-review", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -- load & filter -------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# tc01 is a pure WRITE workload: use WRITE rows, average across iterations
write = df[df["workload"] == "WRITE"].copy()
write["wall_overhead_pct"] = pd.to_numeric(write["wall_overhead_pct"], errors="coerce")

# average over the three iteration counts (10 / 50 / 200) for each page count
agg = (
    write.groupby(["pages", "tracking"])
    .agg(
        kernel_avg_ms=("kernel_avg_ms", "mean"),
        kernel_std_ms=("kernel_std_ms", "mean"),
        wall_overhead_pct=("wall_overhead_pct", "mean"),
    )
    .reset_index()
)

off = agg[agg["tracking"] == "OFF"].set_index("pages")
on  = agg[agg["tracking"] == "ON" ].set_index("pages")
pages = sorted(off.index.tolist())

# -- style ---------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
C_OFF = "#2166ac"   # deep blue - baseline
C_ON  = "#92c5de"   # light blue - tracked

# -- Figure 1 - Kernel time vs page count -------------------------------------
fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))

ax1.plot(pages, off.loc[pages, "kernel_avg_ms"],
         marker="o", linewidth=1.5, color=C_OFF, label="Baseline")
ax1.fill_between(pages,
                 off.loc[pages, "kernel_avg_ms"] - off.loc[pages, "kernel_std_ms"],
                 off.loc[pages, "kernel_avg_ms"] + off.loc[pages, "kernel_std_ms"],
                 alpha=0.12, color=C_OFF)

ax1.plot(pages, on.loc[pages, "kernel_avg_ms"],
         marker="s", linewidth=1.5, color=C_ON, linestyle="--", label="Tracked")
ax1.fill_between(pages,
                 on.loc[pages, "kernel_avg_ms"] - on.loc[pages, "kernel_std_ms"],
                 on.loc[pages, "kernel_avg_ms"] + on.loc[pages, "kernel_std_ms"],
                 alpha=0.12, color=C_ON)

ax1.set_xlabel("Pages")
ax1.set_ylabel("Kernel time (ms)")
ax1.set_xticks(pages)
ax1.set_xticklabels([str(p) for p in pages])
ax1.legend(frameon=False)
sns.despine(ax=ax1)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "tc01_time_vs_pages.pdf")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# -- Figure 2 - Overhead % vs page count --------------------------------------
overhead_vals = on.loc[pages, "wall_overhead_pct"].values

fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))

bars = ax2.bar(
    [str(p) for p in pages],
    overhead_vals,
    color=sns.color_palette("Blues_d", len(pages))[::-1],
    edgecolor="#2166ac",
    linewidth=0.6,
    width=0.5,
)

for bar, val in zip(bars, overhead_vals):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 15,
        f"{val:.0f}%",
        ha="center", va="bottom", fontsize=8.5, color="black",
    )

ax2.set_xlabel("Pages")
ax2.set_ylabel("Overhead (%)")
ax2.set_ylim(0, max(overhead_vals) * 1.18)
sns.despine(ax=ax2)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "tc01_overhead_vs_pages.pdf")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")

"""
plot_single_fault.py - Figures for single_fault_record_cost.

Outputs (saved to midsem-review/figures/):
  single_fault_boxplot.pdf       - box plot: no-tracking vs tracked kernel time
  single_fault_overhead_hist.pdf - histogram: per-fault overhead distribution
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# -- paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "..", "operation_microbenchmarks", "single_fault_record_cost.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "midsem-review", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_PAGES = 4096

# -- load ----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["overhead_per_fault_us"] = (df["with_tracking_ms"] - df["no_tracking_ms"]) * 1000 / NUM_PAGES

# reshape for box plot
long = pd.DataFrame({
    "Kernel time (ms)": pd.concat([df["no_tracking_ms"], df["with_tracking_ms"]], ignore_index=True),
    "Condition":        ["No tracking"] * len(df) + ["Tracked"] * len(df),
})

# -- style ---------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
C_OFF = "#2166ac"
C_ON  = "#92c5de"

# -- Figure 1 - box plot -------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))

sns.boxplot(data=long, x="Condition", y="Kernel time (ms)", hue="Condition",
            palette={"No tracking": C_OFF, "Tracked": C_ON},
            width=0.45, linewidth=1.2, fliersize=3, legend=False, ax=ax1)

ax1.set_xlabel("")
ax1.set_ylabel("Kernel time (ms)")
sns.despine(ax=ax1)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "single_fault_boxplot.pdf")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# -- Figure 2 - per-fault overhead histogram -----------------------------------
mean_val   = df["overhead_per_fault_us"].mean()
median_val = df["overhead_per_fault_us"].median()

fig2, ax2 = plt.subplots(figsize=(5.0, 3.5))

ax2.hist(df["overhead_per_fault_us"], bins=20,
         color=C_ON, edgecolor=C_OFF, linewidth=0.6)

ax2.axvline(mean_val,   color=C_OFF,    linewidth=1.4, linestyle="--", label=f"Mean {mean_val:.1f} µs")
ax2.axvline(median_val, color="#444444", linewidth=1.4, linestyle=":",  label=f"Median {median_val:.1f} µs")

ax2.set_xlabel("Per-fault overhead (µs)")
ax2.set_ylabel("Count")
ax2.legend(frameon=False)
sns.despine(ax=ax2)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "single_fault_overhead_hist.pdf")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")

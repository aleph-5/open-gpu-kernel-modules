"""
plot_table_lifecycle.py - Figures for table_lifecycle_cost.

Outputs (saved to midsem-review/figures/):
  lifecycle_init_destroy.pdf   - line chart: init and destroy time vs page count
  lifecycle_reinit_heatmap.pdf - heatmap: reinit time (dirty_pages x read_pages)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -- paths ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MB_DIR     = os.path.join(SCRIPT_DIR, "..", "operation_microbenchmarks")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "midsem-review", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -- load & aggregate ----------------------------------------------------------
init_df    = pd.read_csv(os.path.join(MB_DIR, "table_init_cost.csv"))
destroy_df = pd.read_csv(os.path.join(MB_DIR, "table_destroy_cost.csv"))
reinit_df  = pd.read_csv(os.path.join(MB_DIR, "table_reinit_cost.csv"))

init_agg = init_df.groupby("page_count")["init_time_us"].agg(mean="mean", std="std").reset_index()
dest_agg = destroy_df.groupby("page_count")["destroy_time_us"].agg(mean="mean", std="std").reset_index()
pages    = init_agg["page_count"].tolist()

reinit_agg = reinit_df.groupby(["dirty_pages", "read_pages"])["reinit_time_us"].mean().reset_index()
pivot      = reinit_agg.pivot(index="dirty_pages", columns="read_pages", values="reinit_time_us")
pivot      = pivot.loc[sorted(pivot.index, reverse=True)]

# -- style ---------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
C_INIT    = "#2166ac"
C_DESTROY = "#92c5de"

# -- Figure 1 - init vs destroy time ------------------------------------------
fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))

ax1.plot(pages, init_agg["mean"], marker="o", linewidth=1.5,
         color=C_INIT, label="Init (start_track)")
ax1.fill_between(pages,
                 init_agg["mean"] - init_agg["std"],
                 init_agg["mean"] + init_agg["std"],
                 alpha=0.12, color=C_INIT)

ax1.plot(pages, dest_agg["mean"], marker="s", linewidth=1.5,
         linestyle="--", color=C_DESTROY, label="Destroy (stop_track)")
ax1.fill_between(pages,
                 dest_agg["mean"] - dest_agg["std"],
                 dest_agg["mean"] + dest_agg["std"],
                 alpha=0.12, color=C_DESTROY)

ax1.set_xscale("log", base=2)
ax1.set_xticks(pages)
ax1.set_xticklabels([str(p) for p in pages])
ax1.set_xlabel("Pages")
ax1.set_ylabel("Time ($\mu$s)")
ax1.legend(frameon=False, loc="upper left")
sns.despine(ax=ax1)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "lifecycle_init_destroy.pdf")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# -- Figure 2 - reinit heatmap ------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 3.8))

sns.heatmap(
    pivot,
    ax=ax2,
    cmap="Blues",
    annot=True,
    fmt=".0f",
    linewidths=0.4,
    linecolor="#dddddd",
    cbar_kws={"label": "Reinit time ($\mu$s)", "shrink": 0.85},
)
ax2.set_xlabel("Read pages (M)")
ax2.set_ylabel("Dirty pages (N)")
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "lifecycle_reinit_heatmap.pdf")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

df = pd.read_csv("overhead_results_windowed_4.csv")
df = df.rename(columns={"queries_per_tracking_interval": "queries_per_epoch"})

# 4GB only
df = df[df["benchmark"].str.endswith("_4g")].copy()
df["overhead_pct"] = (
    df["overhead_pct"].astype(str).str.replace("+", "", regex=False).astype(float)
)

BENCH_ORDER = [
    "int_set_4k_4g",
    "int_set_seq_4g",
    "gemm_4g",
    "2mm_4g",
    "bicg_4g",
    "mvt_4g",
    "needle_4g",
]
BENCH_LABELS = {
    "int_set_4k_4g":  "4K-stride",
    "int_set_seq_4g": "Sequential",
    "gemm_4g":        "GEMM",
    "2mm_4g":         "2MM",
    "bicg_4g":        "BICG",
    "mvt_4g":         "MVT",
    "needle_4g":      "Needle",
}
QUERY_COLORS = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c"}
QUERY_MARKERS = {1: "o", 5: "s", 10: "^"}

# For each (benchmark, queries_per_epoch): mean and std across sleep intervals
stats = (
    df.groupby(["benchmark", "queries_per_epoch"])["overhead_pct"]
    .agg(["mean", "std"])
    .reset_index()
)

x = np.arange(len(BENCH_ORDER))
labels = [BENCH_LABELS[b] for b in BENCH_ORDER]

fig, ax = plt.subplots(figsize=(9, 4.5))

for q in [1, 5, 10]:
    sub = stats[stats["queries_per_epoch"] == q].set_index("benchmark")
    means = [sub.loc[b, "mean"] if b in sub.index else np.nan for b in BENCH_ORDER]
    stds  = [sub.loc[b, "std"]  if b in sub.index else np.nan for b in BENCH_ORDER]
    ax.errorbar(
        x, means,
        yerr=stds,
        label=f"{q} epoch{'s' if q > 1 else ''}/interval",
        color=QUERY_COLORS[q],
        marker=QUERY_MARKERS[q],
        markersize=6,
        linewidth=1.5,
        capsize=4,
        capthick=1.2,
        elinewidth=1.0,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Overhead (%)", fontsize=11)
ax.set_xlabel("Benchmark (4 GB working set)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(title="Epoch frequency", fontsize=9, title_fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_xlim(-0.5, len(BENCH_ORDER) - 0.5)

fig.tight_layout()
fig.savefig("overhead_query_freq_4g.pdf", bbox_inches="tight")
print("Saved overhead_query_freq_4g.pdf")

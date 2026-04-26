import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

df = pd.read_csv("overhead_results.csv")

# 512MB only
df = df[df["benchmark"].str.endswith("_512m")].copy()
df["overhead_pct"] = (
    df["overhead_pct"].astype(str).str.replace("+", "", regex=False).astype(float)
)

BENCH_ORDER = [
    "int_set_4k_512m",
    "int_set_seq_512m",
    "gemm_512m",
    "2mm_512m",
    "bicg_512m",
    "mvt_512m",
    "needle_512m",
]
BENCH_LABELS = {
    "int_set_4k_512m":  "4K-stride",
    "int_set_seq_512m": "Sequential",
    "gemm_512m":        "GEMM",
    "2mm_512m":         "2MM",
    "bicg_512m":        "BICG",
    "mvt_512m":         "MVT",
    "needle_512m":      "Needle",
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
ax.set_xlabel("Benchmark (512 MB working set)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(title="Epoch frequency", fontsize=9, title_fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_xlim(-0.5, len(BENCH_ORDER) - 0.5)

fig.tight_layout()
fig.savefig("overhead_query_freq.pdf", bbox_inches="tight")
print("Saved overhead_query_freq.pdf")

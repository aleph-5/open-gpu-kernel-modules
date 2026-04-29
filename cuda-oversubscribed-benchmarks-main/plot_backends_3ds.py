#!/usr/bin/env python3
"""Backend comparison figures for XArray, Vector, and Nested BM only.

Produces two PDF figures:
    backend_stride_3ds.pdf - 4K-stride overhead
    backend_gemm_3ds.pdf   - GEMM overhead
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

OUT_DIR = Path('.')

SIZES     = ['512m', '4g',  '8g',   '16g',  '32g',   '40g',   '48g',   '64g']
XLABELS   = ['512\nMB', '4\nGB', '8\nGB', '16\nGB', '32\nGB', '40\nGB', '48\nGB', '64\nGB']
X         = list(range(len(SIZES)))
OVERSUB_X = 5.5

BACKENDS = ['XArray', 'Vector', 'Nested BM']

COLORS = {
    'XArray':    '#7EB8D4',
    'Vector':    '#F4A27A',
    'Nested BM': '#84C49E',
}
MARKERS = {
    'XArray':    'o',
    'Vector':    's',
    'Nested BM': 'D',
}

# None = OOM / no data
STRIDE_DATA = {
    'XArray':    [36.6,  272.0,  462.1,  680.8,  981.1,  1046.0, 1038.7, 942.0],
    'Vector':    [57.5,  372.3,  659.9,  1108.2, 1537.6, 1688.3, 1730.7, 1547.1],
    'Nested BM': [48.6,  263.6,  456.8,  689.3,  914.0,  1001.2, 922.9,  921.0],
}

GEMM_DATA = {
    'XArray':    [26.3,  24.7,  17.1,  13.6,  266.8, 256.1, 259.3, 266.2],
    'Vector':    [26.0,  19.3,  15.1,  13.0,  207.6, 258.8, 213.1, 261.6],
    'Nested BM': [23.9,  25.4,  20.9,  13.4,  None,  None,  None,  None],
}


# Shared helpers

def shade_oversub(ax, ylim_top):
    ax.axvspan(OVERSUB_X, len(SIZES) - 0.5, alpha=0.10, color='grey', zorder=0)
    ax.axvline(OVERSUB_X, color='grey', linewidth=0.9, linestyle='--', zorder=1)
    ax.text(OVERSUB_X + 0.12, ylim_top * 0.95,
            'oversubscribed', fontsize=7, color='dimgrey', va='top')


def set_xaxis(ax):
    ax.set_xticks(X)
    ax.set_xticklabels(XLABELS, fontsize=8)
    ax.set_xlim(-0.5, len(SIZES) - 0.5)


def grid_pct(ax):
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.grid(axis='y', alpha=0.30, linewidth=0.6)
    ax.set_axisbelow(True)


def save_fig(fig, stem):
    path = OUT_DIR / (stem + '.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  saved {path}')
    plt.close(fig)


def plot_lines(ax, data):
    for name in BACKENDS:
        ys_raw = data[name]
        xs = [i for i, v in enumerate(ys_raw) if v is not None]
        ys = [v for v in ys_raw if v is not None]
        ax.plot(xs, ys, marker=MARKERS[name], color=COLORS[name],
                label=name, linewidth=1.8, markersize=6, zorder=3)


# Figure 1: 4K-stride

def fig_stride_3ds():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    plot_lines(ax, STRIDE_DATA)

    ylim_top = 2000
    ax.set_ylim(0, ylim_top)
    shade_oversub(ax, ylim_top)
    set_xaxis(ax)
    grid_pct(ax)
    ax.set_ylabel('Overhead (%)', fontsize=9)
    ax.set_xlabel('Working-set size', fontsize=9)
    ax.set_title('4K-stride overhead by backend', fontsize=9, pad=4)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    save_fig(fig, 'backend_stride_3ds')


# Figure 2: GEMM

def fig_gemm_3ds():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    plot_lines(ax, GEMM_DATA)

    ylim_top = 310
    ax.set_ylim(0, ylim_top)
    shade_oversub(ax, ylim_top)
    set_xaxis(ax)
    grid_pct(ax)
    ax.set_ylabel('Overhead (%)', fontsize=9)
    ax.set_xlabel('Working-set size', fontsize=9)
    ax.set_title('GEMM overhead by backend', fontsize=9, pad=4)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    save_fig(fig, 'backend_gemm_3ds')


if __name__ == '__main__':
    print('Generating 3-backend figures ...')
    fig_stride_3ds()
    fig_gemm_3ds()
    print('Done.')

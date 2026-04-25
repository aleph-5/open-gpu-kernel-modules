#!/usr/bin/env python3
"""Generate backend comparison figures for the end-semester report.

Produces three PDF figures:
    backend_stride.pdf  - 4K-stride overhead by backend vs. working-set size
    backend_gemm.pdf    - GEMM overhead by backend vs. working-set size
    backend_ops.pdf     - insert + for-each latency bar charts (log scale)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT_DIR = Path('.')

SIZES     = ['512m', '4g',  '8g',   '16g',  '32g',   '40g',   '48g',   '64g']
XLABELS   = ['512\nMB', '4\nGB', '8\nGB', '16\nGB', '32\nGB', '40\nGB', '48\nGB', '64\nGB']
X         = list(range(len(SIZES)))
OVERSUB_X = 5.5

COLORS = {
    'XArray':      '#7EB8D4',  # pastel blue
    'Vector':      '#F4A27A',  # pastel coral
    'Nested BM':   '#84C49E',  # pastel mint
    'Vec.-U':      '#E8909A',  # pastel rose
    'Chunked':     '#B8A8D8',  # pastel lavender
    'Linked List': '#D4C07A',  # pastel gold
}
MARKERS = {
    'XArray':      'o',
    'Vector':      's',
    'Nested BM':   'D',
    'Vec.-U':      '^',
    'Chunked':     'v',
    'Linked List': 'X',
}

# None = OOM / no data
STRIDE_ALL = {
    'XArray':      [36.6,    272.0,   462.1,  680.8,   981.1,   1046.0,  1038.7, 942.0],
    'Vector':      [57.5,    372.3,   659.9,  1108.2,  1537.6,  1688.3,  1730.7, 1547.1],
    'Nested BM':   [48.6,    263.6,   456.8,  689.3,   914.0,   1001.2,  922.9,  921.0],
    'Vec.-U':      [900.8,   27730.1, None,   None,    None,    None,    None,   None],
    'Chunked':     [247.9,   16951.0, None,   None,    None,    None,    None,   None],
    'Linked List': [917.4,   None,    None,   None,    None,    None,    None,   None],
}

GEMM_ALL = {
    'XArray':      [26.3,    24.7,    17.1,   13.6,   266.8,   256.1,   259.3,  266.2],
    'Vector':      [26.0,    19.3,    15.1,   13.0,   207.6,   258.8,   213.1,  261.6],
    'Nested BM':   [23.9,    25.4,    20.9,   13.4,   None,    None,    None,   None],
    'Chunked':     [30.3,    103.1,   137.3,  287.3,  None,    None,    None,   None],
    'Vec.-U':      [-93.3,   380.8,   551.3,  None,   None,    None,    None,   None],
    'Linked List': [161.2,   1147.3,  None,   None,   None,    None,    None,   None],
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


def plot_lines_and_scatter(ax, data, line_backends, scatter_backends, ylim_top):
    """Draw full lines for line_backends; clipped scatter + arrows for scatter_backends."""
    for name in line_backends:
        ys_raw = data[name]
        xs = [i for i, v in enumerate(ys_raw) if v is not None]
        ys = [v for v in ys_raw if v is not None]
        ax.plot(xs, ys, marker=MARKERS[name], color=COLORS[name],
                label=name, linewidth=1.8, markersize=6, zorder=3)

    for name in scatter_backends:
        ys_raw = data[name]
        pairs  = [(i, v) for i, v in enumerate(ys_raw) if v is not None]
        inside = [(x, y) for x, y in pairs if y <= ylim_top * 0.97]
        above  = [(x, y) for x, y in pairs if y  > ylim_top * 0.97]

        label_added = False
        if inside:
            xs_in, ys_in = zip(*inside)
            ax.scatter(xs_in, ys_in, marker=MARKERS[name], color=COLORS[name],
                       s=55, zorder=4, edgecolors='k', linewidths=0.5,
                       label=f'{name} (limited)')
            label_added = True
        if above:
            xs_ab = [x for x, _ in above]
            cap   = ylim_top * 0.95
            ax.scatter(xs_ab, [cap] * len(xs_ab), marker=MARKERS[name],
                       color=COLORS[name], s=55, zorder=5,
                       edgecolors='k', linewidths=0.5,
                       **({}  if label_added else
                          {'label': f'{name} (limited, ↑ off-chart)'}))
            for xc in xs_ab:
                ax.annotate('', xy=(xc, cap), xytext=(xc, cap * 0.86),
                            arrowprops=dict(arrowstyle='->', color=COLORS[name], lw=1.2))


# Figure 1: 4K-stride overhead by backend

def fig_backend_stride():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    plot_lines_and_scatter(ax, STRIDE_ALL,
                           line_backends=['XArray', 'Vector', 'Nested BM'],
                           scatter_backends=['Vec.-U', 'Chunked', 'Linked List'],
                           ylim_top=2000)

    ylim_top = 2000
    ax.set_ylim(0, ylim_top)
    shade_oversub(ax, ylim_top)
    set_xaxis(ax)
    grid_pct(ax)
    ax.set_ylabel('Overhead (%)', fontsize=9)
    ax.set_xlabel('Working-set size', fontsize=9)
    ax.set_title('4K-stride overhead by backend', fontsize=9, pad=4)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    fig.tight_layout()
    save_fig(fig, 'backend_stride')


# Figure 2: GEMM overhead by backend

def fig_backend_gemm():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Chunked has valid data up to 16 GB (max 287%) so include it as a line
    plot_lines_and_scatter(ax, GEMM_ALL,
                           line_backends=['XArray', 'Vector', 'Nested BM', 'Chunked'],
                           scatter_backends=['Vec.-U', 'Linked List'],
                           ylim_top=310)

    ylim_top = 310
    ax.set_ylim(-120, ylim_top)
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    shade_oversub(ax, ylim_top)
    set_xaxis(ax)
    grid_pct(ax)
    ax.set_ylabel('Overhead (%)', fontsize=9)
    ax.set_xlabel('Working-set size', fontsize=9)
    ax.set_title('GEMM overhead by backend', fontsize=9, pad=4)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    fig.tight_layout()
    save_fig(fig, 'backend_gemm')


# Figure 3: per-operation latency bar charts

def fig_backend_ops():
    backends  = ['XArray', 'Vector', 'Vec.-U*', 'Chunked', 'Nested BM', 'Linked List*']
    bkeys     = ['XArray', 'Vector', 'Vec.-U',  'Chunked', 'Nested BM', 'Linked List']
    insert_ns = [145,      867,      65010,      196990,    69,          149581]
    foreach_s = [0.189,    0.052,    8.542,      204.234,   0.047,       20.047]

    x          = np.arange(len(backends))
    bar_colors = [COLORS[k] for k in bkeys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.8))

    ax1.bar(x, insert_ns, width=0.6, color=bar_colors,
            edgecolor='k', linewidth=0.5, zorder=3)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:.2f}'))
    ax1.set_ylabel('Insert avg (ns)', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(backends, fontsize=7.5, rotation=20, ha='right')
    ax1.set_title('Insert latency', fontsize=9, pad=4)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax1.set_axisbelow(True)

    ax2.bar(x, foreach_s, width=0.6, color=bar_colors,
            edgecolor='k', linewidth=0.5, zorder=3)
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax2.set_ylabel('For-each avg (s)', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(backends, fontsize=7.5, rotation=20, ha='right')
    ax2.set_title('For-each latency', fontsize=9, pad=4)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.suptitle(
        'Per-operation latency across backends  (* = 512 MB measurement)',
        fontsize=8.5, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'backend_ops')


if __name__ == '__main__':
    print('Generating backend comparison figures ...')
    fig_backend_stride()
    fig_backend_gemm()
    fig_backend_ops()
    print('Done.')

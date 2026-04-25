#!/usr/bin/env python3
"""Generate dirty-tracking overhead figures from overhead_results_final.csv.

Produces three PDF figures ready for \includegraphics in the LaTeX report:
    overhead_write.pdf - int_set 4K-stride vs sequential
    overhead_polybench.pdf - GEMM, 2MM, BICG, MVT
    overhead_needle.pdf - Rodinia Needle (wall-time + overhead %)
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Constants

CSV_FILE = 'overhead_results_final.csv'
OUT_DIR  = Path('.')

SIZES   = ['512m', '4g', '8g', '16g', '32g', '40g', '48g', '64g']
XLABELS = ['512\nMB', '4\nGB', '8\nGB', '16\nGB', '32\nGB', '40\nGB', '48\nGB', '64\nGB']
X       = list(range(len(SIZES)))

# Between index 5 (40 GB, last in-GPU) and index 6 (48 GB, first oversubscribed)
OVERSUB_X = 5.5

PREFIXES = ['int_set_4k', 'int_set_seq', 'gemm', '2mm', 'bicg', 'mvt', 'needle']

COLORS = {
    'int_set_4k':  '#d62728',
    'int_set_seq': '#ff7f0e',
    'gemm':        '#1f77b4',
    '2mm':         '#17becf',
    'bicg':        '#2ca02c',
    'mvt':         '#9467bd',
    'needle':      '#8c564b',
}

LABELS = {
    'int_set_4k':  'int_set (4K stride)',
    'int_set_seq': 'int_set (sequential)',
    'gemm':        'GEMM',
    '2mm':         '2MM',
    'bicg':        'BICG',
    'mvt':         'MVT',
    'needle':      'Needle',
}

MARKERS = {
    'int_set_4k':  'o',
    'int_set_seq': 's',
    'gemm':        'D',
    '2mm':         '^',
    'bicg':        'v',
    'mvt':         'P',
    'needle':      'X',
}

# Data loading

def load_data(csv_file):
    data = {p: {} for p in PREFIXES}
    with open(csv_file, newline='') as f:
        for row in csv.DictReader(f):
            bname = row['benchmark']
            for pfx in PREFIXES:
                if bname.startswith(pfx + '_'):
                    sz = bname[len(pfx) + 1:]
                    data[pfx][sz] = {
                        'off_avg': float(row['off_avg_s']),
                        'off_std': float(row['off_std_s']),
                        'on_avg':  float(row['on_avg_s']),
                        'on_std':  float(row['on_std_s']),
                        'ovhd':    float(row['overhead_pct'].lstrip('+')),
                    }
                    break
    return data


def get_xy(data, bench):
    """Return (x_indices, overhead_pct) for the given benchmark."""
    xs, ys = [], []
    for i, sz in enumerate(SIZES):
        if sz in data.get(bench, {}):
            xs.append(i)
            ys.append(data[bench][sz]['ovhd'])
    return xs, ys

# Shared axis helpers 

def shade_oversub(ax, ylim_top):
    ax.axvspan(OVERSUB_X, len(SIZES) - 0.5, alpha=0.10, color='grey', zorder=0)
    ax.axvline(OVERSUB_X, color='grey', linewidth=0.9, linestyle='--', zorder=1)
    ax.text(OVERSUB_X + 0.12, ylim_top * 0.97,
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

# Figure 1: write-intensive workloads

def fig_write(data):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for bench in ['int_set_4k', 'int_set_seq']:
        xs, ys = get_xy(data, bench)
        ax.plot(xs, ys,
                marker=MARKERS[bench], color=COLORS[bench],
                label=LABELS[bench], linewidth=1.8, markersize=6, zorder=3)

    ylim_top = 2000
    ax.set_ylim(0, ylim_top)
    shade_oversub(ax, ylim_top)
    set_xaxis(ax)
    grid_pct(ax)

    ax.set_ylabel('Overhead (%)', fontsize=9)
    ax.set_xlabel('Working-set size', fontsize=9)
    ax.set_title('Overhead: write-intensive benchmarks (int_set)', fontsize=9, pad=4)
    ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    save_fig(fig, 'overhead_write')

# Figure 2: Polybench workloads

def fig_polybench(data):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for bench in ['gemm', '2mm', 'bicg', 'mvt']:
        xs, ys = get_xy(data, bench)
        ax.plot(xs, ys,
                marker=MARKERS[bench], color=COLORS[bench],
                label=LABELS[bench], linewidth=1.8, markersize=6, zorder=3)

    ylim_top = 310
    ax.set_ylim(0, ylim_top)
    shade_oversub(ax, ylim_top)
    set_xaxis(ax)
    grid_pct(ax)

    ax.set_ylabel('Overhead (%)', fontsize=9)
    ax.set_xlabel('Working-set size', fontsize=9)
    ax.set_title('Overhead: Polybench kernels (GEMM, 2MM, BICG, MVT)', fontsize=9, pad=4)
    ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    save_fig(fig, 'overhead_polybench')

# Figure 3: Needle

def fig_needle(data):
    bench = 'needle'
    bdata = data[bench]

    xs    = [i for i, sz in enumerate(SIZES) if sz in bdata]
    off_t = [bdata[SIZES[i]]['off_avg'] for i in xs]
    off_e = [bdata[SIZES[i]]['off_std'] for i in xs]
    on_t  = [bdata[SIZES[i]]['on_avg']  for i in xs]
    on_e  = [bdata[SIZES[i]]['on_std']  for i in xs]
    ovhd  = [bdata[SIZES[i]]['ovhd']    for i in xs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))

    # Left: wall times with error bars
    ax1.errorbar(xs, off_t, yerr=off_e,
                 label='Tracking OFF', marker='s', color='steelblue',
                 linewidth=1.6, capsize=3, markersize=5, zorder=3)
    ax1.errorbar(xs, on_t, yerr=on_e,
                 label='Tracking ON', marker='o', color='tomato',
                 linewidth=1.6, capsize=3, markersize=5, zorder=3)

    ylim_top1 = max(on_t) * 1.15
    shade_oversub(ax1, ylim_top1)
    ax1.set_ylim(0, ylim_top1)
    set_xaxis(ax1)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%gs'))
    ax1.grid(axis='y', alpha=0.30, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.set_ylabel('Wall time (s)', fontsize=9)
    ax1.set_xlabel('Working-set size', fontsize=9)
    ax1.set_title('Wall time', fontsize=9)
    ax1.legend(fontsize=8)

    # Right: overhead %
    ax2.plot(xs, ovhd,
             marker=MARKERS[bench], color=COLORS[bench],
             linewidth=1.8, markersize=6, zorder=3)

    ylim_top2 = max(ovhd) * 1.15
    shade_oversub(ax2, ylim_top2)
    ax2.set_ylim(0, ylim_top2)
    set_xaxis(ax2)
    grid_pct(ax2)
    ax2.set_ylabel('Overhead (%)', fontsize=9)
    ax2.set_xlabel('Working-set size', fontsize=9)
    ax2.set_title('Overhead (%)', fontsize=9)

    fig.suptitle('Rodinia Needle (Needleman--Wunsch)', fontsize=9, y=1.01)
    fig.tight_layout()
    save_fig(fig, 'overhead_needle')

# Main

def main():
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        sys.exit(f'ERROR: {CSV_FILE} not found. Run from the report directory.')

    print(f'Loading {CSV_FILE} ...')
    data = load_data(csv_path)

    print('Generating figures ...')
    fig_write(data)
    fig_polybench(data)
    fig_needle(data)
    print('Done.')


if __name__ == '__main__':
    main()

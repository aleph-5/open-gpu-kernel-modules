#!/usr/bin/env python3
"""Generate Rodinia Needle overhead figure limited to 16 GB (in-GPU regime).

Reads overhead_results_final-vector.csv and produces overhead_needle_16g.pdf.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV_FILE = 'overhead_results_final-vector.csv'
OUT_FILE = 'overhead_needle_16g.pdf'

SIZES  = ['512m', '4g', '8g', '16g']
LABELS = ['512\nMB', '4\nGB', '8\nGB', '16\nGB']

def load_needle(csv_file):
    data = {}
    with open(csv_file, newline='') as f:
        for row in csv.DictReader(f):
            bname = row['benchmark']
            if bname.startswith('needle_'):
                sz = bname[len('needle_'):]
                if sz in SIZES:
                    data[sz] = {
                        'off_avg': float(row['off_avg_s']),
                        'off_std': float(row['off_std_s']),
                        'on_avg':  float(row['on_avg_s']),
                        'on_std':  float(row['on_std_s']),
                        'ovhd':    float(row['overhead_pct'].lstrip('+')),
                    }
    return data

def main():
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        sys.exit(f'ERROR: {CSV_FILE} not found. Run from cuda-oversubscribed-benchmarks-main/.')

    data = load_needle(csv_path)

    xs    = list(range(len(SIZES)))
    off_t = [data[sz]['off_avg'] for sz in SIZES]
    off_e = [data[sz]['off_std'] for sz in SIZES]
    on_t  = [data[sz]['on_avg']  for sz in SIZES]
    on_e  = [data[sz]['on_std']  for sz in SIZES]
    ovhd  = [data[sz]['ovhd']    for sz in SIZES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))

    # Left: wall times with error bars
    ax1.errorbar(xs, off_t, yerr=off_e,
                 label='Tracking OFF', marker='s', color='steelblue',
                 linewidth=1.6, capsize=3, markersize=5, zorder=3)
    ax1.errorbar(xs, on_t, yerr=on_e,
                 label='Tracking ON', marker='o', color='tomato',
                 linewidth=1.6, capsize=3, markersize=5, zorder=3)

    ax1.set_ylim(0, max(on_t) * 1.15)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(LABELS, fontsize=8)
    ax1.set_xlim(-0.5, len(SIZES) - 0.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%gs'))
    ax1.grid(axis='y', alpha=0.30, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.set_ylabel('Wall time (s)', fontsize=9)
    ax1.set_xlabel('Working-set size', fontsize=9)
    ax1.set_title('Wall time', fontsize=9)
    ax1.legend(fontsize=8)

    # Right: overhead %
    ax2.plot(xs, ovhd,
             marker='X', color='#D4C07A',
             linewidth=1.8, markersize=6, zorder=3)

    ax2.set_ylim(0, max(ovhd) * 1.15)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(LABELS, fontsize=8)
    ax2.set_xlim(-0.5, len(SIZES) - 0.5)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax2.grid(axis='y', alpha=0.30, linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_ylabel('Overhead (%)', fontsize=9)
    ax2.set_xlabel('Working-set size', fontsize=9)
    ax2.set_title('Overhead (%)', fontsize=9)

    fig.suptitle('Rodinia Needle (Needleman--Wunsch) — up to 16 GB', fontsize=9, y=1.01)
    fig.tight_layout()

    fig.savefig(OUT_FILE, bbox_inches='tight')
    print(f'Saved {OUT_FILE}')
    plt.close(fig)

if __name__ == '__main__':
    main()

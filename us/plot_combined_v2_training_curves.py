"""Plot training accuracy & loss curves from Dflooding training history.

Reads:
    D:/experiment_flood_combined_v2/dynamic/ATT_combined5_2330/history_*.csv

Produces:
    D:/Combined_output_v2/2330_combined5_dynamic/plots/training_curves.png

Logic: highlight surviving top-3 repeats (those with a matching experiment_*.keras),
fade the rest. Plot train + val for loss and accuracy.
"""
from __future__ import annotations
import os
import re
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def platform_path(p: str) -> str:
    if os.name != 'nt' and len(p) >= 2 and p[1] == ':':
        return f'/mnt/{p[0].lower()}{p[2:]}'
    return p


EXP_DIR = Path(platform_path(os.getenv(
    'EXP_DIR',
    'D:/experiment_flood_combined_v2/dynamic/ATT_combined5_2330',
)))
OUT_DIR = Path(platform_path(os.getenv(
    'OUT_DIR',
    'D:/Combined_output_v2/2330_combined5_dynamic/plots',
)))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- detect surviving (top-3) repeats from experiment_*.keras ---
keras_pat = re.compile(r'experiment_(\d+)\.keras$')
top3 = sorted({int(m.group(1))
               for f in EXP_DIR.glob('experiment_*.keras')
               if (m := keras_pat.search(f.name))})
print(f"[INFO] EXP_DIR={EXP_DIR}")
print(f"[INFO] surviving top-3 repeats = {top3}")

# --- load all history files ---
hist_pat = re.compile(r'history_(\d+)\.csv$')
hist_files = sorted(EXP_DIR.glob('history_*.csv'),
                    key=lambda f: int(hist_pat.search(f.name).group(1)))
histories: dict[int, pd.DataFrame] = {}
for f in hist_files:
    r = int(hist_pat.search(f.name).group(1))
    df = pd.read_csv(f)
    histories[r] = df

print(f"[INFO] loaded {len(histories)} history files: "
      f"{sorted(histories.keys())[:6]}...")

# --- plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
metrics = [
    ('loss',     'val_loss',     'Loss (binary cross-entropy)', axes[0, 0]),
    ('accuracy', 'val_accuracy', 'Accuracy',                    axes[0, 1]),
    ('f1',       'val_f1',       'F1 score',                    axes[1, 0]),
    ('pr_auc',   'val_pr_auc',   'PR-AUC',                      axes[1, 1]),
]

for tr_col, va_col, title, ax in metrics:
    # Background: all repeats faded
    for r, df in histories.items():
        if r in top3:
            continue
        if tr_col in df.columns:
            ax.plot(df['epoch'], df[tr_col], color='#888888', alpha=0.15, lw=0.8)
        if va_col in df.columns:
            ax.plot(df['epoch'], df[va_col], color='#888888', alpha=0.10,
                    lw=0.8, ls='--')
    # Foreground: top-3 highlighted
    colors = ['#2E86AB', '#E63946', '#06A77D']
    for i, r in enumerate(top3):
        df = histories.get(r)
        if df is None:
            continue
        c = colors[i % len(colors)]
        if tr_col in df.columns:
            ax.plot(df['epoch'], df[tr_col], color=c, lw=1.6,
                    label=f'r{r} train', alpha=0.95)
        if va_col in df.columns:
            ax.plot(df['epoch'], df[va_col], color=c, lw=1.4, ls='--',
                    label=f'r{r} val', alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel(tr_col)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc='best')

fig.suptitle(
    f'2330 combined5 (dynamic flooding) — training curves '
    f'(top-3 repeats highlighted, others faded grey)',
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))

out = OUT_DIR / 'training_curves.png'
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"[DONE] saved {out}")

# --- also plot flooding_b trajectory (sanity check that b actually moved) ---
fig, ax = plt.subplots(figsize=(12, 4.5))
for r, df in histories.items():
    if 'flooding_b' not in df.columns:
        continue
    if r in top3:
        i = top3.index(r)
        c = ['#2E86AB', '#E63946', '#06A77D'][i]
        ax.plot(df['epoch'], df['flooding_b'], color=c, lw=1.4,
                label=f'r{r} (top-3)', alpha=0.9)
    else:
        ax.plot(df['epoch'], df['flooding_b'], color='#888888',
                alpha=0.2, lw=0.7)
ax.axhline(0.0, color='black', ls=':', lw=0.5)
ax.axhline(0.4, color='black', ls=':', lw=0.5)
ax.set_xlabel('epoch')
ax.set_ylabel('flooding_b')
ax.set_title('DynamicFlooding b trajectory across 27 repeats '
             '(top-3 colored; clip range [0, 0.4])')
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=8, frameon=False)
fig.tight_layout()
out_b = OUT_DIR / 'flooding_b_trajectory.png'
fig.savefig(out_b, dpi=150)
plt.close(fig)
print(f"[DONE] saved {out_b}")
